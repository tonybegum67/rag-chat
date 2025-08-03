import streamlit as st
import openai
import anthropic
import os
import io
import sqlite3
import re
from typing import Generator, List

# Mermaid diagram support
try:
    from streamlit_mermaid import st_mermaid
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False

# Import configuration and utilities
from config.settings import AppConfig
from utils.document_processor import document_processor
from utils.vector_store import betty_vector_store
from utils.feedback_manager import feedback_manager

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

# Initialize session state early
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session ID for feedback tracking
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Initialize feedback state
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

# Enhanced Navigation Header
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="
            color: white;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 600;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        ">
            {AppConfig.PAGE_ICON} {AppConfig.PAGE_TITLE}
        </h1>
        <p style="
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            font-weight: 300;
        ">
            Strategic Transformation Assistant powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="padding-top: 1rem;">
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🏠 Betty Chat", 
                 use_container_width=True, 
                 type="primary",
                 help="Main chat interface"):
        st.rerun()

with col3:
    st.markdown("""
    <div style="padding-top: 1rem;">
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Admin Dashboard", 
                 use_container_width=True, 
                 type="secondary",
                 help="Access analytics and admin features"):
        st.switch_page("pages/admin_dashboard.py")

# Betty's Introduction and Sample Prompts
if not st.session_state.messages:
    st.markdown("---")
    
    # Betty's Description
    st.markdown("""
    ### 👋 Welcome! I'm Betty
    
    I'm an AI assistant designed to facilitate strategic transformation through **Outcome-Based Thinking (OBT)** and **What/How Mapping**. My role is to help organizations like Molex activate, measure, and align strategic outcomes with business structures for maximum impact.
    
    I assist in developing strategic ideas, creating measurable outcome statements, mapping these to the GPS tier framework, aligning them with business capabilities, and defining relevant KPIs. Additionally, I provide instructional coaching to enhance understanding and application of OBT methodology, building organizational capability while delivering strategic value.
    """)
    
    # Sample Prompts
    st.markdown("### 🚀 Try these sample prompts:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Transform Strategy", use_container_width=True):
            sample_prompt = "Help me transform 'improve customer satisfaction' into a measurable outcome statement with KPIs and GPS tier mapping"
            st.session_state.messages.append({"role": "user", "content": sample_prompt})
            st.rerun()
        
        st.caption("Transform vague goals into measurable outcomes")
    
    with col2:
        if st.button("🎯 Outcome Analysis", use_container_width=True):
            sample_prompt = "Analyze this statement: 'implement agile methodologies across development teams' - is this a What or How? Help me reframe it."
            st.session_state.messages.append({"role": "user", "content": sample_prompt})
            st.rerun()
        
        st.caption("Learn What vs How classification")
    
    with col3:
        if st.button("🏗️ GPS Mapping", use_container_width=True):
            sample_prompt = "Map the outcome 'product defect rates reduced by 50%' to the appropriate GPS tier and identify supporting business capabilities"
            st.session_state.messages.append({"role": "user", "content": sample_prompt})
            st.rerun()
        
        st.caption("Align outcomes with organizational structure")
    
    st.markdown("---")
    st.markdown("💬 **Or ask me anything about strategic transformation, OBT methodology, or Molex operations!**")

# --- Configuration ---
# Get the API key based on provider
if AppConfig.AI_PROVIDER == "claude":
    AppConfig.ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not AppConfig.ANTHROPIC_API_KEY:
        st.error("Please set your Anthropic API key in Streamlit secrets (e.g., .streamlit/secrets.toml) or as an environment variable.")
        st.stop()
    client = anthropic.Anthropic(api_key=AppConfig.ANTHROPIC_API_KEY)
else:
    AppConfig.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not AppConfig.OPENAI_API_KEY:
        st.error("Please set your OpenAI API key in Streamlit secrets (e.g., .streamlit/secrets.toml) or as an environment variable.")
        st.stop()
    client = openai.OpenAI(api_key=AppConfig.OPENAI_API_KEY)

# Validate configuration
if not AppConfig.validate_config():
    st.error("Invalid configuration. Please check your settings.")
    st.stop()

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
    """Searches the knowledge base for relevant context with optional reranking."""
    n_results = n_results or AppConfig.MAX_SEARCH_RESULTS
    if AppConfig.USE_RERANKING:
        return vector_store.search_collection_with_reranking(collection_name, query, n_results)
    else:
        return vector_store.search_collection(collection_name, query, n_results)

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
Manufacturing BA Reference Architecture: Business capabilities, value streams, and operational structure

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
Create clear, informative diagrams using Mermaid syntax ONLY when users request visual elements using keywords like:
- Primary triggers: diagram, chart, graph, visual, flowchart, map, show, illustrate, visualize, draw
- Strategic triggers: outcome mapping, tier structure, capability flow, stakeholder network, dependency map

When triggered, create diagrams to illustrate:
- Relationships between ideas, outcomes, and GPS tiers
- Business capability and value stream connections
- Stakeholder accountability networks
- Information flow and dependency mapping

Always use ```mermaid code blocks for diagrams.


Rewrite each outcome statement to focus solely on the desired end state or result, eliminating any reference to:
- Specific numbers or metrics
- Tools, solutions, or methods (the "how")
- Specific actions or processes
Express the outcome in a general, solution-agnostic way that describes what is accomplished, not how it is achieved.

- Original: "Robust impact analysis on changes"
Outcome: "A robust impact analysis is made on every change before it occurs."
- Original: "Execute behavior change plans"
Outcome: "Plans to facilitate changes in behavior are effectively executed."
- Original: "Standardized evaluation guides are consistently applied with broad oversight"
Outcome: "Changes are consistently evaluated to set standards."
- Original: "Comprehensive dashboards track all change management metrics"
Outcome: "Performance in managing changes is effectively tracked and measured."


For every outcome statement, ask: "Does this describe what is achieved, without specifying how, with what, or using which metric?" 
If not, revise to remove the 'how' and focus on the 'what'.


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
Remember: You're not just providing answers—you're building organizational capability in outcome-based strategic thinking while delivering measurable transformation results.
"""

# --- Feedback UI Functions ---
def display_feedback_buttons(message_index: int, user_message: str, betty_response: str):
    """Display thumbs up/down feedback buttons for a Betty response."""
    feedback_key = f"feedback_{message_index}"
    
    # Skip if feedback already given for this message
    if feedback_key in st.session_state.feedback_given:
        st.caption("✅ Thank you for your feedback!")
        return
    
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("👍", key=f"thumbs_up_{message_index}", help="This response was helpful"):
            # Record positive feedback
            feedback_manager.record_feedback(
                session_id=st.session_state.session_id,
                user_message=user_message,
                betty_response=betty_response,
                feedback_type="thumbs_up"
            )
            st.session_state.feedback_given.add(feedback_key)
            st.success("Thank you for the positive feedback! 🎉")
            st.rerun()
    
    with col2:
        if st.button("👎", key=f"thumbs_down_{message_index}", help="This response needs improvement"):
            # Record negative feedback
            feedback_manager.record_feedback(
                session_id=st.session_state.session_id,
                user_message=user_message,
                betty_response=betty_response,
                feedback_type="thumbs_down"
            )
            st.session_state.feedback_given.add(feedback_key)
            
            # Show optional feedback form
            with st.expander("Help us improve (optional)"):
                feedback_details = st.text_area(
                    "What could Betty do better?",
                    key=f"feedback_details_{message_index}",
                    placeholder="e.g., The outcome wasn't specific enough, missing KPI details, unclear GPS tier mapping..."
                )
                if st.button("Submit Details", key=f"submit_details_{message_index}"):
                    if feedback_details:
                        # Update the feedback with details
                        conversation_id = feedback_manager.generate_conversation_id(user_message, betty_response)
                        with sqlite3.connect(feedback_manager.db_path) as conn:
                            conn.execute("""
                                UPDATE feedback 
                                SET feedback_details = ? 
                                WHERE conversation_id = ? AND feedback_type = 'thumbs_down'
                            """, (feedback_details, conversation_id))
                        st.success("Thank you for the detailed feedback! This helps us improve Betty.")
            st.rerun()

# --- Session State Initialization ---
# (Moved earlier in the file to prevent AttributeError)

# --- Chat Interface ---

# Display chat messages from history
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
        
        # Add feedback buttons for Betty's responses
        if message["role"] == "assistant" and i > 0:  # Make sure there's a user message before this
            user_message = st.session_state.messages[i-1]["content"]
            display_feedback_buttons(i, user_message, message["content"])

# Accept user input
uploaded_file = st.file_uploader(
    "Upload a document for temporary context", 
    type=["pdf", "docx", "txt"],
    key="file_uploader"
)

# Check if there's a new message to process (either from chat input or sample prompts)
if prompt := st.chat_input("What would you like to ask Betty?"):
    # Add user message to chat history from chat input
    st.session_state.messages.append({"role": "user", "content": prompt})

# Check if the last message is from user and needs a response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # Get the last user message
    last_user_message = st.session_state.messages[-1]["content"]
    
    # Check if we already have a response for this message
    needs_response = True
    if len(st.session_state.messages) >= 2:
        # If there's already an assistant response after this user message, don't process again
        if len(st.session_state.messages) % 2 == 0:  # Even number means last was assistant
            needs_response = False
    
    if needs_response:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(last_user_message)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare the system prompt with all context
            system_prompt = SYSTEM_PROMPT
            
            # --- Handle Uploaded File for Temporary Context ---
            temp_context = ""
            if uploaded_file:
                with st.spinner(f"Reading {uploaded_file.name}..."):
                    temp_context = document_processor.process_uploaded_file(uploaded_file)
                    
                    if temp_context:
                        system_prompt += f"\n\nThe user has provided a temporary file for context: '{uploaded_file.name}'. Use the following information from it to answer the current query.\n\n---\n{temp_context}\n---"

            # Perform RAG search on the permanent knowledge base
            if st.session_state.get("use_rag", True):
                relevant_docs = search_knowledge_base(last_user_message, collection_name=AppConfig.KNOWLEDGE_COLLECTION_NAME)
                if relevant_docs:
                    context = "\n\n".join([
                        f"Document: {doc['metadata']['filename']}\nContent: {doc['content']}"
                        for doc in relevant_docs
                    ])
                    system_prompt += f"\n\nRelevant context from permanent knowledge base:\n\n{context}"

            # Prepare messages for the API call (no system messages in the array)
            api_messages = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            try:
                if AppConfig.AI_PROVIDER == "claude":
                    # Stream the response from the Claude API
                    with client.messages.stream(
                        model=AppConfig.CLAUDE_MODEL,
                        max_tokens=4000,
                        messages=api_messages,
                        system=system_prompt,  # Use consolidated system prompt
                    ) as stream:
                        for text in stream.text_stream:
                            full_response += text
                            message_placeholder.markdown(full_response + "▌")
                else:
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
                
                # Try to render Mermaid diagrams in the final response
                mermaid_rendered = detect_and_render_mermaid(full_response)
                if not mermaid_rendered:
                    message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "Sorry, I encountered an error."
                message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar for Controls ---
with st.sidebar:
    st.markdown("### 🎛️ App Controls")
    
    # Current Page Indicator
    st.markdown("#### 📍 Current Page")
    st.success("🏠 **Betty Chat** - Main Interface")
    
    st.markdown("---")
    
    # Chat Controls
    st.markdown("#### 💬 Chat Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.feedback_given = set()
        st.rerun()
    
    st.session_state.use_rag = st.checkbox(
        "🧠 Use Betty's Knowledge (RAG)", 
        value=True,
        help="Enable Betty to search her knowledge base for relevant context"
    )
    
    st.markdown("---")

    # Knowledge Base Section
    st.markdown("#### 📚 Knowledge Base")
    if st.button("🔄 Update Knowledge Base", use_container_width=True, type="secondary"):
        with st.spinner("Updating knowledge base..."):
            add_files_to_collection(
                AppConfig.KNOWLEDGE_COLLECTION_NAME, 
                list(AppConfig.DEFAULT_KNOWLEDGE_FILES)
            )
            st.success("Knowledge base updated!")
    
    st.markdown("---")
    
    # Analytics Section
    st.markdown("#### 📊 Analytics & Admin")
    st.info("📈 **Admin Dashboard**\n\nTo access analytics and feedback data, use the page selector at the top left of the screen and choose 'admin_dashboard'.")
    
    # Quick stats if available
    try:
        total_messages = len(st.session_state.messages)
        if total_messages > 0:
            st.metric("💬 Chat Messages", total_messages)
            
        feedback_count = len(st.session_state.get("feedback_given", set()))
        if feedback_count > 0:
            st.metric("👍 Feedback Given", feedback_count)
    except:
        pass
    
    st.markdown("---")
    
    # Help Section
    st.markdown("#### ❓ Need Help?")
    with st.expander("🚀 How to use Betty"):
        st.markdown("""
        **Sample Questions:**
        - "Transform 'improve customer satisfaction' into measurable outcomes"
        - "Is 'implement agile' a What or How?"
        - "Map this outcome to GPS tiers"
        
        **Features:**
        - 📤 Upload documents for context
        - 👍👎 Rate Betty's responses
        - 📊 View analytics in Admin Dashboard
        """)
    
    st.markdown("---")
    st.caption("💡 Betty AI Assistant v2.0")
    st.caption("Built for Molex Strategic Transformation")

