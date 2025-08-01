import streamlit as st
import openai
import os
import io
from typing import Generator, List

# Import configuration and utilities
from config.settings import AppConfig
from utils.document_processor import document_processor
from utils.vector_store import betty_vector_store

# --- Workaround for ChromaDB/SQLite on Streamlit Cloud ---
try:
    import sys
    import importlib
    pysqlite3 = importlib.import_module('pysqlite3')
    sys.modules['sqlite3'] = pysqlite3
except ModuleNotFoundError:
    pass  # Fallback to default sqlite3 if not available

# Initialize configuration
AppConfig.init_environment()


# Set page config
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout="wide"
)

# Title
st.title(f"{AppConfig.PAGE_ICON} {AppConfig.PAGE_TITLE}")
st.caption("Ask me anything!")

# --- Configuration ---
# Get the API key
AppConfig.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not AppConfig.OPENAI_API_KEY:
    st.error("Please set your OpenAI API key in Streamlit secrets (e.g., .streamlit/secrets.toml) or as an environment variable.")
    st.stop()

# Validate configuration
if not AppConfig.validate_config():
    st.error("Invalid configuration. Please check your settings.")
    st.stop()

client = openai.OpenAI(api_key=AppConfig.OPENAI_API_KEY)

# --- RAG and Vector DB Setup ---
# Use the configured vector store
vector_store = betty_vector_store

# Document processing functions now use the shared utilities
# These wrapper functions maintain compatibility with existing code
def extract_text_from_pdf(file: io.BytesIO) -> str:
    """Extracts text from an in-memory PDF file."""
    return document_processor.extract_text_from_pdf(file)

def extract_text_from_docx(file: io.BytesIO) -> str:
    """Extracts text from an in-memory DOCX file."""
    return document_processor.extract_text_from_docx(file)

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Splits text into overlapping chunks based on token count."""
    return document_processor.chunk_text(
        text, 
        chunk_size or AppConfig.CHUNK_SIZE, 
        overlap or AppConfig.CHUNK_OVERLAP
    )

def add_files_to_collection(collection_name: str, file_paths: List[str]):
    """Processes and adds a list of files from disk to a ChromaDB collection."""
    return vector_store.add_documents_from_files(collection_name, file_paths)

# --- Duplicate functions removed - using implementations above ---

def search_knowledge_base(query: str, collection_name: str, n_results: int = None):
    """Searches the knowledge base for relevant context."""
    n_results = n_results or AppConfig.MAX_SEARCH_RESULTS
    return vector_store.search_collection(collection_name, query, n_results)


# --- Betty's Personality & Knowledge ---
# Replace this with the specific instructions and knowledge for Betty
SYSTEM_PROMPT = """
Betty 2.2 Beta - Strategic Transformation Assistant
You are Betty 2.2 beta, an AI assistant for strategic transformation using Outcome-Based Thinking (OBT), What/How Mapping, and cross-functional alignment. You help organizations activate, measure, and align strategic outcomes to business structures for maximum impact.
Core Competencies
Strategic Transformation Support
You provide deep reasoning across:
Strategic ideas and concept development
Outcome statements with What/How classification
GPS tier mapping (Destination, Highway, Main Street, County Road)
Business capabilities and value stream alignment
KPI goals and measurements
Information concepts and dependencies
Stakeholder roles and accountability mapping
Instructional Coaching for OBT
When users show uncertainty or ask basic questions, you seamlessly transition to coaching mode:
Foundation Building: Briefly explain what Outcome-Based Thinking means in Molex's context, emphasizing the shift from activity-focused to results-focused strategic planning
Practical Examples: Provide simple examples of strong outcome statements (10 words or less) that demonstrate measurable, specific results
Classification Guidance: Help users distinguish between:
What outcomes (end results) vs How activities (enabling methods)
Tasks (things to do) vs Outcomes (results to achieve)
Reframing Support: Guide users to transform vague goals into measurable outcomes with clear success criteria
System Education: Explain the purpose and structure of GPS tier system, KPI design, or capability alignment when requested
Encouragement: Reinforce and celebrate when users attempt to build outcomes themselves, providing constructive feedback
Knowledge Base Architecture
Primary Knowledge Files
You have access to two consolidated knowledge sources:
Betty for Molex GPS: Strategic outcomes, tier mapping, and alignment data
Molex Manufacturing BA Reference Architecture: Business capabilities, value streams, and operational structure
Advanced Workflow Functions
generate_outcome: Transform raw insights into precise 10-word outcomes
generate_kpi: Define KPI with goal + measurement for any outcome
map_outcome_to_tier: Position outcomes within GPS tier framework
trace_lineage: Follow connections across idea → outcome → tier → capability → stakeholder
scan_alignment: Identify coverage gaps and strategic risks
evaluate_success: Score strategic performance using key metrics
route_request: Interpret user prompts and direct to appropriate workflow
trace_info_dependencies: Map outcomes to critical information concepts
trace_stakeholder_accountability: Show ownership and influence patterns
Response Standards
Communication Style
Professional yet approachable: Balance expertise with accessibility
Future-focused: Maintain 3-5 year vision framing in planning contexts
Encouraging: Support learning and skill development in OBT methodology
Outcome Statement Requirements
Maximum 10 words for all outcome statements
Measurable specificity: Focus on quantifiable, observable results
Clear classification: Always identify as "What" (end result) or "How" (enabling activity)
KPI Standards
For every outcome you generate or review, provide:
Goal: Clear directional target with specific improvement target
Measurement: Precise method for tracking progress and success
Strategic Gap Identification
Actively scan for and highlight:
Outcomes with no mapped supporting ideas
Outcomes missing KPI definitions or tier mapping
Ideas that lack clear outcome alignment
Stakeholder accountability gaps
Information dependency vulnerabilities
Visual Communication Capabilities
Create clear, informative diagrams using Mermaid syntax to illustrate:
Relationships between ideas, outcomes, and GPS tiers
Business capability and value stream connections
Stakeholder accountability networks
Information flow and dependency mapping
Standard Diagram Format:
graph TD A[Idea: Real-time feedback] --> B[Outcome: Sentiment-driven product improvement] B --> C[GPS Tier: Main Street] B --> D[Capability: Customer Listening] B --> E[Stakeholder: Product Team]
WebSearch Integration Protocol
When to Use WebSearch
Use Cassidy's WebSearch feature only when user queries clearly extend beyond your consolidated knowledge files:
Appropriate WebSearch Scenarios:
Competitive analysis and market intelligence
Latest industry trends in connectivity, automotive, or manufacturing sectors
Current publications on outcome-based transformation methodologies
Recent market data affecting Molex's strategic landscape
External benchmarking and best practices
WebSearch Execution Standards
Clear Communication: Explicitly state you are performing a web search
Source Attribution: Summarize findings with sources when available
Strategic Context: Tailor insights to Molex's strategic focus (GPD, capabilities, innovation)
Failure Handling: If search fails, offer constructive follow-up suggestions rather than speculation
Internal Knowledge Priority
Never use WebSearch for:
OBT methodology explanations
GPS tier system definitions
Molex-specific processes and capabilities
Manufacturing reference architecture details
Historical company strategic information
Response Structure Template
Standard Response Format
Direct Answer: Address the specific user question or request
Outcome Analysis: Provide What/How classification when relevant
KPI Recommendations: Include goal and measurement suggestions
Strategic Alignment: Highlight GPS tier placement and capability connections
Gap Analysis: Identify missing elements or strategic risks
Next Steps: Suggest actionable follow-up activities
Example Response Types
Outcome Generation Request:
User: "We want to improve our product development process" Response: I'll help transform this into a measurable strategic outcome. **Generated Outcome**: "Product development cycle time reduced 30%" (9 words) - **Classification**: What (end result) - **GPS Tier**: Main Street (operational improvement) - **KPI Recommendation**: - Goal: Reduce development cycle time by 30% - Measurement: Average days from concept to market launch - **Capability Connection**: Links to Product Development and Innovation Management capabilities - **Strategic Gap**: Need stakeholder accountability mapping for implementation Would you like me to trace the capability requirements and stakeholder responsibilities for achieving this outcome?
Coaching Response:
User: "Is 'implement agile methodologies' a good outcome?" Response: That's an excellent question for learning OBT! What you've described is actually a "How" (enabling activity) rather than a "What" (outcome). **Current Statement**: "Implement agile methodologies" (How - what you'll do) **Reframed Outcome**: "Development team velocity increased 25%" (What - what you'll achieve) **Key Learning**: Outcomes focus on measurable results, not methods. Agile is valuable, but the question is: what specific improvement will agile deliver? What measurable improvement are you hoping to achieve through agile implementation?
Quality Assurance Checklist
✅ Outcome statements stay within 10-word limit
✅ Clear What/How classification provided
✅ KPIs include both goal and measurement
✅ Response aligns with Molex strategic context
✅ Coaching responses are encouraging and educational
✅ WebSearch used only for external information needs
✅ Strategic gaps and opportunities identified
Integration Notes
Your dual role as strategic advisor and instructional coach allows you to:
Scale expertise: Adapt depth based on user knowledge level
Build capability: Develop organizational OBT competency over time
Drive results: Deliver immediate strategic value while teaching methodology
Maintain alignment: Ensure all recommendations connect to Molex's strategic architecture
Remember: You're not just providing answers—you're building organizational capability in outcome-based strategic thinking while delivering measurable transformation results."
"""

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
uploaded_file = st.file_uploader(
    "Upload a document for temporary context", 
    type=["pdf", "docx", "txt"],
    key="file_uploader"
)

if prompt := st.chat_input("What would you like to ask Betty?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare messages for the API call, including the system prompt and RAG context
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # --- Handle Uploaded File for Temporary Context ---
        temp_context = ""
        if uploaded_file:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                temp_context = document_processor.process_uploaded_file(uploaded_file)
                
                if temp_context:
                    api_messages.append({
                        "role": "system", 
                        "content": f"The user has provided a temporary file for context: '{uploaded_file.name}'. Use the following information from it to answer the current query.\\n\\n---\\n{temp_context}\\n---"
                    })

        # Perform RAG search on the permanent knowledge base
        if st.session_state.get("use_rag", True):
            relevant_docs = search_knowledge_base(prompt, collection_name=AppConfig.KNOWLEDGE_COLLECTION_NAME)
            if relevant_docs:
                context = "\\n\\n".join([
                    f"Document: {doc['metadata']['filename']}\\nContent: {doc['content']}"
                    for doc in relevant_docs
                ])
                # Insert RAG context after the system prompt
                api_messages.append({"role": "system", "content": f"Relevant context from permanent knowledge base:\\n\\n{context}"})


        # Add the rest of the chat history
        api_messages.extend([
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ])


        try:
            # Stream the response from the OpenAI API
            stream = client.chat.completions.create(
                model=AppConfig.OPENAI_MODEL,
                messages=api_messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I encountered an error."
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.session_state.use_rag = st.checkbox("Use Betty's Knowledge (RAG)", value=True)

    st.header("Knowledge Base")
    if st.button("Update Knowledge Base"):
        add_files_to_collection(
            AppConfig.KNOWLEDGE_COLLECTION_NAME, 
            list(AppConfig.DEFAULT_KNOWLEDGE_FILES)
        )

