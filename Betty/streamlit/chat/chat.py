
# Workaround for ChromaDB/SQLite compatibility (works on both local and Streamlit Cloud)
try:
    import sys
    import importlib
    pysqlite3 = importlib.import_module('pysqlite3')
    sys.modules['sqlite3'] = pysqlite3
except ModuleNotFoundError:
    # Fallback to default sqlite3 if pysqlite3 is not available (e.g., local dev)
    pass

import streamlit as st
import openai
import os
import io
import re
from typing import Generator, List
try:
    from streamlit_mermaid import st_mermaid
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False

# Add parent directory to path to access shared utilities
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import configuration and utilities
from config.settings import ChatConfig, AppConfig
from utils.document_processor import document_processor
from utils.vector_store import chat_vector_store
from utils.feedback_manager import feedback_manager

# Initialize configuration
AppConfig.init_environment()

# Set page config
st.set_page_config(
    page_title=ChatConfig.PAGE_TITLE,
    page_icon=ChatConfig.PAGE_ICON,
    layout="wide"
)

# Title
st.title(f"{ChatConfig.PAGE_ICON} {ChatConfig.PAGE_TITLE}")
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
    return client

client = init_components()

# Use the configured vector store
vector_store = chat_vector_store

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# Use 'active_collection' to store the loaded ChromaDB collection object
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
# Use 'active_collection_name' to store the name of the collection
if "active_collection_name" not in st.session_state:
    st.session_state.active_collection_name = None
# Initialize session ID for feedback tracking
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
# Initialize feedback tracking
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

# Helper functions - now use shared utilities
def extract_text_from_pdf(file):
    """Extract text from PDF file using shared utilities"""
    file_bytes = io.BytesIO(file.read())
    return document_processor.extract_text_from_pdf(file_bytes)

def extract_text_from_docx(file):
    """Extract text from DOCX file using shared utilities"""
    file_bytes = io.BytesIO(file.read())
    return document_processor.extract_text_from_docx(file_bytes)

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks using shared utilities"""
    # Use session state values if available, otherwise defaults
    chunk_size = chunk_size or getattr(st.session_state, 'chunk_size', ChatConfig.DEFAULT_CHUNK_SIZE)
    overlap = overlap or getattr(st.session_state, 'overlap', ChatConfig.DEFAULT_OVERLAP)
    
    return document_processor.chunk_text(text, chunk_size, overlap)

def sanitize_collection_name(name: str) -> str:
    """Sanitizes a string to be a valid ChromaDB collection name."""
    import re
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
    """Cleans extracted text using shared utilities."""
    return document_processor.clean_text(text)

def add_to_collection(collection_name: str, documents: List):
    """Adds new documents to a specified ChromaDB collection using shared utilities."""
    if not documents:
        st.warning("No documents provided to add.")
        return None
    
    # Convert document list to file paths for the vector store
    # For uploaded files, we need to create temporary files
    import tempfile
    temp_files = []
    try:
        for filename, text in documents:
            if not text.strip():
                continue
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                temp_files.append(f.name)
        
        if temp_files:
            success = vector_store.add_documents_from_files(collection_name, temp_files, show_progress=False)
            if success:
                st.success(f"Added {len(documents)} documents to collection '{collection_name}'.")
                return vector_store.get_or_create_collection(collection_name)
            else:
                return None
        return None
    finally:
        # Clean up temporary files
        import os
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

def search_knowledge_base(collection, query: str, n_results: int = None):
    """Searches a given collection for relevant documents using shared utilities."""
    if collection is None: 
        st.warning("Cannot search: No active collection.")
        return []
    
    n_results = n_results or getattr(st.session_state, 'n_results', ChatConfig.DEFAULT_N_RESULTS)
    
    # Get collection name from the collection object
    try:
        collection_name = collection.name
        results = vector_store.search_collection(collection_name, query, n_results)
        # Add distance information for compatibility
        return [{**result, "distance": 0.0} for result in results]
    except Exception as e:
        st.error(f"Error searching knowledge base: {e}")
        return []

def detect_and_render_mermaid(content: str) -> bool:
    """
    Detect Mermaid diagrams in content and render them.
    Returns True if Mermaid diagrams were found and rendered.
    """
    if not MERMAID_AVAILABLE:
        st.warning("⚠️ Mermaid rendering not available. Install streamlit-mermaid to enable diagram visualization.")
        return False
    
    # Pattern to match standard Mermaid diagram blocks
    mermaid_pattern = r'```mermaid\s*\n(.*?)\n```'
    
    diagrams_found = False
    remaining_parts = []
    last_end = 0
    
    # Find all mermaid code blocks
    matches = list(re.finditer(mermaid_pattern, content, re.DOTALL | re.IGNORECASE))
    
    if not matches:
        # No mermaid diagrams found
        return False
    
    for match in matches:
        # Add content before this diagram
        if match.start() > last_end:
            text_before = content[last_end:match.start()].strip()
            if text_before:
                remaining_parts.append(text_before)
        
        # Extract and render the diagram
        diagram_code = match.group(1).strip()
        
        if diagram_code:  # Only render if there's actual content
            try:
                # Render the diagram with streamlit-mermaid
                st_mermaid(diagram_code, height=400)
                diagrams_found = True
                
                # Add a small expander with the code for reference
                with st.expander("📊 View Mermaid Code", expanded=False):
                    st.code(diagram_code, language="mermaid")
                    
            except Exception as e:
                st.error(f"❌ Error rendering Mermaid diagram: {e}")
                # Show the code as fallback
                with st.expander("⚠️ Mermaid Code (Failed to Render)", expanded=True):
                    st.code(diagram_code, language="mermaid")
                    st.info("💡 Try copying this code to a Mermaid live editor: https://mermaid.live/")
                diagrams_found = True  # Still count as found even if rendering failed
        
        last_end = match.end()
    
    # Add any remaining content after the last diagram
    if last_end < len(content):
        text_after = content[last_end:].strip()
        if text_after:
            remaining_parts.append(text_after)
    
    # Display remaining content as markdown if any
    for part in remaining_parts:
        if part.strip():
            st.markdown(part)
    
    return diagrams_found

def handle_feedback(message_index: int, feedback_type: str, user_message: str, assistant_message: str):
    """Handle user feedback for a specific message"""
    try:
        session_id = st.session_state.get('session_id', 'default_session')
        feedback_manager.record_feedback(
            session_id=session_id,
            user_message=user_message,
            betty_response=assistant_message,
            feedback_type=feedback_type,
            user_agent=st.context.headers.get('User-Agent', 'Unknown')
        )
        
        # Store feedback in session state to show confirmation
        if 'feedback_given' not in st.session_state:
            st.session_state.feedback_given = {}
        st.session_state.feedback_given[message_index] = feedback_type
        
        st.success(f"Thank you for your feedback! ({feedback_type.replace('_', ' ').title()})")
    except Exception as e:
        st.error(f"Error recording feedback: {e}")

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
            model=AppConfig.OPENAI_MODEL,
            messages=messages,
            stream=True,
            temperature=getattr(st.session_state, 'temperature', ChatConfig.DEFAULT_TEMPERATURE),
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
                st.session_state.active_collection = vector_store.client.get_collection(collection_name)
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
        existing_collections = vector_store.list_collections()
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
                # Document reading logic using shared utilities
                new_documents = []
                for file in uploaded_files:
                    if file.size > AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
                        st.error(f"File {file.name} is too large (max {AppConfig.MAX_FILE_SIZE_MB}MB)")
                        continue
                    
                    # Process file using shared document processor
                    text = document_processor.process_uploaded_file(file)
                    
                    if text.strip():
                        new_documents.append((file.name, text))
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
                vector_store.delete_collection(st.session_state.active_collection_name)
                set_active_collection(None) # Clear active collection from session
            st.rerun()
    
    st.markdown("---")
    st.markdown("### Settings")
    st.markdown("**Model:** GPT-4o")
    st.markdown("**Embedding:** all-MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider(
            "Chunk Size", 
            ChatConfig.MIN_CHUNK_SIZE, 
            ChatConfig.MAX_CHUNK_SIZE, 
            ChatConfig.DEFAULT_CHUNK_SIZE, 
            help="Size of text chunks for processing"
        )
        overlap = st.slider(
            "Chunk Overlap", 
            0, 
            ChatConfig.MAX_OVERLAP, 
            ChatConfig.DEFAULT_OVERLAP, 
            help="Overlap between chunks"
        )
        n_results = st.slider(
            "Number of Retrieved Documents", 
            ChatConfig.MIN_N_RESULTS, 
            ChatConfig.MAX_N_RESULTS, 
            ChatConfig.DEFAULT_N_RESULTS, 
            help="How many relevant documents to retrieve"
        )
        temperature = st.slider(
            "Response Temperature", 
            ChatConfig.MIN_TEMPERATURE, 
            ChatConfig.MAX_TEMPERATURE, 
            ChatConfig.DEFAULT_TEMPERATURE, 
            help="Creativity of responses"
        )
        
        # Store in session state
        st.session_state.chunk_size = chunk_size
        st.session_state.overlap = overlap
        st.session_state.n_results = n_results
        st.session_state.temperature = temperature

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Try to render Mermaid diagrams for assistant messages
            if message["role"] == "assistant":
                mermaid_rendered = detect_and_render_mermaid(message["content"])
                # If no Mermaid diagrams were found, display as normal markdown
                if not mermaid_rendered:
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant":
                col_feedback1, col_feedback2, col_feedback3 = st.columns([1, 1, 10])
                
                with col_feedback1:
                    if st.button("👍", key=f"thumbs_up_{i}", help="Good response"):
                        # Find the corresponding user message
                        user_message = ""
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_message = st.session_state.messages[i-1]["content"]
                        handle_feedback(i, "thumbs_up", user_message, message["content"])
                        st.rerun()
                
                with col_feedback2:
                    if st.button("👎", key=f"thumbs_down_{i}", help="Poor response"):
                        # Find the corresponding user message
                        user_message = ""
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            user_message = st.session_state.messages[i-1]["content"]
                        handle_feedback(i, "thumbs_down", user_message, message["content"])
                        st.rerun()
                
                # Show feedback status if given
                if i in st.session_state.feedback_given:
                    feedback_type = st.session_state.feedback_given[i]
                    if feedback_type == "thumbs_up":
                        st.success("✅ Feedback recorded: Helpful")
                    else:
                        st.info("✅ Feedback recorded: Needs improvement")
    
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
            
            # Try to render Mermaid diagrams in the final response
            mermaid_rendered = detect_and_render_mermaid(full_response)
            if not mermaid_rendered:
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