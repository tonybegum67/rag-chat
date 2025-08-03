# Betty AI Assistant 💁‍♀️

**Strategic Transformation Assistant powered by Outcome-Based Thinking (OBT)**

Betty is an AI-powered strategic transformation assistant designed to support organizations in implementing Outcome-Based Thinking, What/How Mapping, and cross-functional alignment for maximum business impact.

## 🚀 Features

- **Strategic Transformation Support**: Deep reasoning across strategic ideas, outcome statements, and business alignment
- **Outcome-Based Thinking (OBT) Coaching**: Built-in instructional coaching for OBT methodology
- **RAG-Powered Intelligence**: Retrieval-Augmented Generation with ChromaDB vector storage
- **Document Processing**: Support for PDF, DOCX, and TXT files up to 10MB
- **Knowledge Base Management**: Persistent knowledge storage with automatic updates
- **GPS Tier Mapping**: Strategic outcome classification and business capability alignment
- **Feedback Analytics**: Admin dashboard with user feedback analytics and improvement insights

## 🏗️ Architecture

```
Betty/
├── betty_app.py              # Main Betty application
├── config/                   # Configuration management
│   ├── settings.py           # App and chat configurations
│   └── __init__.py
├── utils/                    # Core utilities
│   ├── document_processor.py # Text extraction & processing
│   ├── vector_store.py       # ChromaDB interface
│   ├── feedback_manager.py   # User feedback analytics
│   └── __init__.py
├── streamlit/                # Additional Streamlit interfaces
│   ├── chat/
│   │   └── chat.py           # Generic chat interface
│   └── requirements.txt      # Streamlit-specific dependencies
├── pages/                    # Multi-page app components
│   └── admin_dashboard.py    # Analytics dashboard
├── data/                     # Data storage
│   ├── betty_chroma_db/      # Vector database
│   └── betty_feedback.db     # Feedback analytics database
├── docs/                     # Documentation and knowledge base
│   ├── Betty for Molex GPS.docx
│   ├── Molex Manufacturing BA Reference Architecture.docx
│   └── Betty Detailed Doc.md
├── requirements.txt          # Python dependencies
├── example.py               # Usage examples
├── fix_embedding_dimensions.py  # Database utilities
└── test_*.py                # Test files
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit with multi-page support
- **AI Models**: Claude 3.5 Sonnet (primary), OpenAI GPT-4o (secondary)
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: SentenceTransformer (all-mpnet-base-v2)
- **Document Processing**: PyPDF2, python-docx, tiktoken
- **Analytics**: SQLite with feedback tracking
- **Deployment**: Streamlit Cloud compatible

## 📋 Prerequisites

- Python 3.8+
- Anthropic API key (for Claude) or OpenAI API key
- Git (for cloning)

## 🚀 Quick Start

### 1. Navigate to Betty Directory
```bash
cd Betty
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
# Primary (Claude)
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export AI_PROVIDER="claude"

# Secondary (OpenAI)
export OPENAI_API_KEY="your_openai_api_key_here"
export AI_PROVIDER="openai"

# Optional configurations
export CHROMA_DB_PATH="./data/betty_chroma_db"
export CHUNK_SIZE="800"
export MAX_SEARCH_RESULTS="5"
```

### 4. Run Betty
```bash
streamlit run betty_app.py
```

Betty will be available at `http://localhost:8501`

## 🎯 Betty's Core Competencies

### Strategic Transformation Support
- Strategic ideas and concept development
- Outcome statements with What/How classification
- GPS tier mapping (Destination, Highway, Main Street, County Road)
- Business capabilities and value stream alignment
- KPI goals and measurements
- Stakeholder roles and accountability mapping

### Instructional Coaching for OBT
- **Foundation Building**: Explains Outcome-Based Thinking principles
- **Practical Examples**: Provides outcome statement examples (10 words or less)
- **Classification Guidance**: Distinguishes between What outcomes vs How activities
- **Reframing Support**: Transforms vague goals into measurable outcomes
- **System Education**: Explains GPS tier system and capability alignment

## 💬 Example Interactions

**Input**: "We want to improve our product development process"

**Betty's Response**:
```
Generated Outcome: "Product development cycle time reduced 30%" (9 words)
- Classification: What (end result)
- GPS Tier: Main Street (operational improvement)
- KPI Recommendation:
  - Goal: Reduce development cycle time by 30%
  - Measurement: Average days from concept to market launch
- Capability Connection: Links to Product Development and Innovation Management
```

## 🔧 Configuration

### Environment Variables
```bash
# Core Configuration
ANTHROPIC_API_KEY=your_api_key          # Required for Claude
OPENAI_API_KEY=your_api_key             # Required for OpenAI
AI_PROVIDER=claude                      # "claude" or "openai"

# Database Configuration
CHROMA_DB_PATH=./data/betty_chroma_db   # Vector database path

# Processing Configuration
CHUNK_SIZE=800                          # Text chunk size in tokens
CHUNK_OVERLAP=100                       # Token overlap between chunks
MAX_SEARCH_RESULTS=5                    # Number of RAG search results
MAX_FILE_SIZE_MB=10                     # Maximum upload file size

# Model Configuration
EMBEDDING_MODEL=all-mpnet-base-v2       # Embedding model
TOKENIZER_MODEL=cl100k_base             # Tokenizer for chunking
```

### Streamlit Secrets (for cloud deployment)
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
# OR
OPENAI_API_KEY = "your_openai_api_key_here"
```

## 📚 Usage Guide

### Adding Knowledge to Betty
1. **Upload Files**: Use the file uploader for temporary context
2. **Update Knowledge Base**: Click "Update Knowledge Base" in sidebar
3. **Supported Formats**: PDF, DOCX, TXT files up to 10MB

### Betty's Response Standards
- **Outcome Statements**: Maximum 10 words with measurable specificity
- **Clear Classification**: Always identifies as "What" (end result) or "How" (enabling activity)
- **KPI Standards**: Provides goal and measurement for every outcome
- **Strategic Alignment**: Highlights GPS tier placement and capability connections

### Advanced Features
- **RAG Toggle**: Enable/disable knowledge base integration
- **File Processing**: Automatic text extraction and chunking
- **Knowledge Refresh**: Update knowledge base with new documents
- **Admin Dashboard**: Access analytics and user feedback insights
- **Feedback System**: Thumbs up/down with detailed feedback collection

## 📊 Admin Dashboard

Access the admin dashboard to monitor Betty's performance:

### Features
- **Overview Metrics**: Total feedback, quality scores, satisfaction rates
- **Feedback Breakdown**: Positive vs negative feedback analysis
- **Trends Over Time**: Historical performance patterns
- **Quality Analysis**: OBT compliance and response quality metrics
- **Improvement Opportunities**: Areas for enhancement based on user feedback

### Access
Navigate to the Admin Dashboard page using the navigation buttons or visit directly.

## 🚀 Deployment

### Local Development
```bash
streamlit run betty_app.py --server.port 8501
```

### Streamlit Cloud
1. Connect GitHub repository to Streamlit Cloud
2. Add API keys to Streamlit secrets
3. Set main file to `betty_app.py`
4. Deploy automatically on code changes

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "betty_app.py", "--server.address", "0.0.0.0"]
```

## 🔍 API Reference

### Document Processor
```python
from utils.document_processor import document_processor

# Extract text from files
text = document_processor.extract_text_from_pdf(file_bytes)
text = document_processor.extract_text_from_docx(file_bytes)

# Process and chunk text
cleaned_text = document_processor.clean_text(raw_text)
chunks = document_processor.chunk_text(text, chunk_size=800, overlap=100)
```

### Vector Store
```python
from utils.vector_store import betty_vector_store

# Add documents to knowledge base
success = betty_vector_store.add_documents_from_files(
    collection_name="betty_knowledge", 
    file_paths=["docs/document.pdf"]
)

# Search knowledge base
results = betty_vector_store.search_collection(
    collection_name="betty_knowledge",
    query="outcome-based thinking",
    n_results=5
)
```

### Feedback Manager
```python
from utils.feedback_manager import feedback_manager

# Record user feedback
feedback_manager.record_feedback(
    session_id="session_123",
    user_message="User question",
    betty_response="Betty's response",
    feedback_type="thumbs_up"
)

# Get analytics
summary = feedback_manager.get_feedback_summary(days=30)
```

## 🐛 Troubleshooting

### Common Issues

**"Please set your API key"**
- Verify API key in environment variables or Streamlit secrets
- Check API key has sufficient credits/quota

**"Could not connect to ChromaDB"**
- Check database path permissions
- Clear corrupted database: `rm -rf data/betty_chroma_db/`

**"No text could be extracted"**
- Verify file format is supported (PDF, DOCX, TXT)
- Check file isn't password-protected or corrupted
- Ensure file size is under 10MB limit

**"SQLite error on Streamlit Cloud"**
- The pysqlite3 workaround is automatically applied
- Check if pysqlite3-binary is in requirements.txt

For detailed troubleshooting, see `docs/Betty Detailed Doc.md`

## 📄 Documentation

- **[Betty Detailed Doc.md](docs/Betty%20Detailed%20Doc.md)**: Comprehensive technical documentation
- **Knowledge Base**: Molex-specific documentation in docs/ folder
- **API Reference**: Complete method documentation above
- **Deployment Guides**: Local, cloud, and Docker deployment
- **Performance Optimization**: Tuning guidelines and best practices

## 📞 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: Comprehensive guides in `docs/Betty Detailed Doc.md`
- **Development**: Follow contribution guidelines for code changes

## 👥 Credits

**Author**: Tony Begum, Chief AI Officer  
**Company**: BoldARC  
**Project**: Betty AI Assistant - Strategic Transformation Platform

## 📜 License

Private repository - All rights reserved to BoldARC.

## 🏷️ Version

**Betty v2.2 Beta** - Strategic Transformation Assistant with OBT methodology

---

**Built with ❤️ for strategic transformation and outcome-based thinking**  
**Developed by BoldARC**

*Last updated: January 2025*