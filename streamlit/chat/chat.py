import streamlit as st
import openai
import os
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import tiktoken
from typing import Generator, List
import tempfile
import io
import re

# Set the environment variable to disable tokenizer parallelism and suppress the warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set page config
st.set_page_config(
    page_title="GPT-4o RAG Chat App",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 GPT-4o RAG Chat Assistant")
st.caption("Chat with your documents using AI")

# Initialize components
@st.cache_resource
def init_components():
    # Initialize OpenAI client
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set your OpenAI API key in Streamlit secrets or as an environment variable 'OPENAI_API_KEY'")
        st.stop()
    
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize tokenizer for chunk size estimation
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    return client, chroma_client, embedding_model, tokenizer

client, chroma_client, embedding_model, tokenizer = init_components()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# Use 'active_collection' to store the loaded ChromaDB collection object
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
# Use 'active_collection_name' to store the name of the collection
if "active_collection_name" not in st.session_state:
    st.session_state.active_collection_name = None

# Helper functions
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks"""
    # Use session state values if available, otherwise defaults
    chunk_size = chunk_size or getattr(st.session_state, 'chunk_size', 500)
    overlap = overlap or getattr(st.session_state, 'overlap', 50)
    
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def sanitize_collection_name(name: str) -> str:
    """Sanitizes a string to be a valid ChromaDB collection name."""
    if not name:
        return ""
    # Replace spaces and invalid characters with underscores
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    # Ensure it's not too long or short
    if len(name) < 3:
        st.warning(f"Collection name '{name}' is too short (min 3 chars).")
        return ""
    if len(name) > 63:
        st.warning(f"Collection name '{name}' is too long (max 63 chars).")
        return ""
    return name

def clean_text(text: str) -> str:
    """Cleans extracted text to fix common formatting issues."""
    if not text:
        return ""
    # Add a space after a period or comma if it's followed by a letter.
    # This helps fix issues like "word.Anotherword" or "123.Nextword".
    text = re.sub(r'([.,])([a-zA-Z])', r'\\1 \\2', text)
    # Replace multiple newline characters with two, preserving paragraphs.
    text = re.sub(r'\\n\\s*\\n', '\\n\\n', text)
    # Replace multiple spaces with a single space.
    text = re.sub(r' +', ' ', text)
    # Remove leading/trailing whitespace from each line
    lines = (line.strip() for line in text.splitlines())
    return "\\n".join(lines)

def add_to_collection(collection_name: str, documents: List):
    """Adds new documents to a specified ChromaDB collection."""
    if not documents:
        st.warning("No documents provided to add.")
        return None

    collection = chroma_client.get_or_create_collection(collection_name)
    
    # Get the current number of documents in the collection to create unique IDs
    doc_offset = collection.count()
    
    all_chunks, metadatas, ids = [], [], []
    for doc_idx, (filename, text) in enumerate(documents):
        if not text.strip(): continue
        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            all_chunks.append(chunk)
            metadatas.append({
                "filename": filename, 
                "chunk_index": chunk_idx, 
                "doc_index": doc_idx + doc_offset
            })
            ids.append(f"doc_{doc_offset + doc_idx}_chunk_{chunk_idx}")
    
    if all_chunks:
        try:
            embeddings = embedding_model.encode(all_chunks, show_progress_bar=False).tolist()
            collection.add(
                embeddings=embeddings, 
                documents=all_chunks, 
                metadatas=metadatas, 
                ids=ids
            )
            st.success(f"Added {len(documents)} documents to collection '{collection_name}'.")
        except Exception as e:
            st.error(f"Error adding to collection: {str(e)}")
            return None
    return collection

def search_knowledge_base(collection, query: str, n_results: int = None):
    """Searches a given collection for relevant documents."""
    if collection is None: 
        st.warning("Cannot search: No active collection.")
        return []
        
    n_results = n_results or getattr(st.session_state, 'n_results', 3)
    try:
        results = collection.query(
            query_embeddings=embedding_model.encode([query]).tolist(),
            n_results=n_results
        )
        return [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )
        ]
    except Exception as e:
        st.error(f"Error searching knowledge base: {e}")
        return []

def generate_response_with_rag(messages, use_rag=True) -> Generator[str, None, None]:
    """Generate response using RAG if enabled and a collection is active."""
    try:
        # Get the latest user message for RAG search
        if use_rag and messages and st.session_state.active_collection:
            latest_message = messages[-1]["content"]
            relevant_docs = search_knowledge_base(st.session_state.active_collection, latest_message)
            
            # Add context to the system message
            if relevant_docs:
                context = "\n\n".join([
                    f"Document: {doc['metadata']['filename']}\nContent: {doc['content']}"
                    for doc in relevant_docs
                ])
                
                system_message = {
                    "role": "system",
                    "content": f"""You are a helpful assistant with access to the following relevant information:

{context}

Use this information to answer the user's question. If the information isn't relevant or doesn't contain the answer, say so and provide a general response. Always cite the document name when using specific information."""
                }
                messages = [system_message] + messages
        
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
            temperature=getattr(st.session_state, 'temperature', 0.7),
            max_tokens=1000
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        yield "Sorry, I encountered an error while processing your request."

# Sidebar for document management and settings
with st.sidebar:
    st.header("📚 Knowledge Base Management")

    def set_active_collection(collection_name):
        """Callback function to set the active collection in session state."""
        if collection_name:
            try:
                st.session_state.active_collection = chroma_client.get_collection(collection_name)
                st.session_state.active_collection_name = collection_name
            except Exception as e:
                st.error(f"Failed to load collection '{collection_name}': {e}")
                st.session_state.active_collection = None
                st.session_state.active_collection_name = None
        else:
            st.session_state.active_collection = None
            st.session_state.active_collection_name = None

    # Get existing collections from ChromaDB
    try:
        existing_collections = [c.name for c in chroma_client.list_collections()]
    except Exception as e:
        st.error(f"Could not connect to ChromaDB. Please ensure it's running. Error: {e}")
        existing_collections = []

    # Dropdown to select an active collection
    selected_collection = st.selectbox(
        "Select an Active Collection",
        options=[""] + existing_collections,
        format_func=lambda x: "None" if x == "" else x,
        key="collection_selector",
        on_change=lambda: set_active_collection(st.session_state.collection_selector)
    )

    # Display active collection info
    if st.session_state.active_collection_name:
        st.success(f"Active Collection: **{st.session_state.active_collection_name}** ({st.session_state.active_collection.count()} docs)")
    else:
        st.info("No collection is active. Select one or create a new one.")

    st.markdown("---")

    # UI for creating a new collection and adding documents
    st.subheader("Create or Add to Collection")
    
    # Determine the target collection for the operation
    target_collection_name_input = st.text_input("Enter new collection name (or leave blank to add to active one)")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload documents to create or add to a knowledge base"
    )

    if st.button("Process and Add Documents"):
        # Determine the target collection, sanitizing if a new name is provided
        raw_collection_name = target_collection_name_input.strip()
        if raw_collection_name:
            target_collection = sanitize_collection_name(raw_collection_name)
            if target_collection != raw_collection_name:
                st.info(f"Collection name sanitized to: '{target_collection}'")
        else:
            target_collection = st.session_state.active_collection_name

        if not target_collection:
            st.error("Please enter a valid collection name or select an active one first.")
        elif not uploaded_files:
            st.warning("Please upload files to process.")
        else:
            with st.spinner(f"Processing documents for '{target_collection}'..."):
                # Document reading logic
                new_documents = []
                for file in uploaded_files:
                    if file.size > 10 * 1024 * 1024:
                        st.error(f"File {file.name} is too large (max 10MB)")
                        continue
                    if file.type == "text/plain":
                        text = str(file.read(), "utf-8")
                    elif file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = extract_text_from_docx(file)
                    else:
                        st.warning(f"Unsupported file type: {file.type}")
                        continue
                    
                    # Clean the extracted text before adding
                    cleaned_text = clean_text(text)

                    if cleaned_text.strip():
                        new_documents.append((file.name, cleaned_text))
                    else:
                        st.warning(f"No text could be extracted from {file.name}")
                
                if new_documents:
                    collection = add_to_collection(target_collection, new_documents)
                    # If this was a new collection, make it active
                    if raw_collection_name and collection:
                        set_active_collection(target_collection)
                        st.rerun()

    st.markdown("---")
    
    # RAG toggle
    use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=True, disabled=not st.session_state.active_collection)
    
    st.markdown("---")
    
    # Chat controls
    st.header("⚙️ Controls")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.active_collection_name:
        if st.button(f"Delete Collection '{st.session_state.active_collection_name}'", type="primary"):
            with st.spinner(f"Deleting collection '{st.session_state.active_collection_name}'..."):
                chroma_client.delete_collection(st.session_state.active_collection_name)
                set_active_collection(None) # Clear active collection from session
            st.rerun()
    
    st.markdown("---")
    st.markdown("### Settings")
    st.markdown("**Model:** GPT-4o")
    st.markdown("**Embedding:** all-MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, help="Size of text chunks for processing")
        overlap = st.slider("Chunk Overlap", 0, 200, 50, help="Overlap between chunks")
        n_results = st.slider("Number of Retrieved Documents", 1, 10, 3, help="How many relevant documents to retrieve")
        temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7, help="Creativity of responses")
        
        # Store in session state
        st.session_state.chunk_size = chunk_size
        st.session_state.overlap = overlap
        st.session_state.n_results = n_results
        st.session_state.temperature = temperature

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents or anything else..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare messages for API call
            api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            # Stream the response
            for chunk in generate_response_with_rag(api_messages, use_rag):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

with col2:
    # Show relevant documents for the last query if RAG is enabled
    if use_rag and st.session_state.messages and st.session_state.active_collection:
        st.subheader("📑 Relevant Sources")
        # Find the last user message to use as the query
        latest_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)

        if latest_query:
            relevant_docs = search_knowledge_base(st.session_state.active_collection, latest_query, n_results=3)
            
            if relevant_docs:
                for i, doc in enumerate(relevant_docs):
                    with st.expander(f"📄 {doc['metadata']['filename']} (Relevance: {1-doc['distance']:.2f})"):
                        st.text(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])
            else:
                st.info("No relevant sources found in the active collection for the last query.")