# Development Repository

This repository contains development projects and tools.

## 🏗️ Projects

### Betty AI Assistant 💁‍♀️

**Strategic Transformation Assistant powered by Outcome-Based Thinking (OBT)**

Betty is an AI-powered strategic transformation assistant designed to support organizations in implementing Outcome-Based Thinking, What/How Mapping, and cross-functional alignment for maximum business impact.

#### 🚀 Features

- **Strategic Transformation Support**: Deep reasoning across strategic ideas, outcome statements, and business alignment
- **Outcome-Based Thinking (OBT) Coaching**: Built-in instructional coaching for OBT methodology
- **RAG-Powered Intelligence**: Retrieval-Augmented Generation with ChromaDB vector storage
- **Document Processing**: Support for PDF, DOCX, and TXT files up to 10MB
- **Knowledge Base Management**: Persistent knowledge storage with automatic updates
- **GPS Tier Mapping**: Strategic outcome classification and business capability alignment
- **Feedback Analytics**: Admin dashboard with user feedback analytics and improvement insights

#### 📁 Project Structure

```
Betty/                        # Betty AI Assistant project
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
├── test_*.py                # Test files
└── README.md                # Project documentation
```

#### 🚀 Quick Start

```bash
# Navigate to Betty project
cd Betty

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
# OR
export OPENAI_API_KEY="your_openai_api_key_here"

# Run Betty
streamlit run betty_app.py
```

Betty will be available at `http://localhost:8501`

#### 📚 Documentation

For complete documentation, usage guides, and API reference, see [Betty/README.md](Betty/README.md)

## 🛠️ Technology Stack

- **Frontend**: Streamlit with multi-page support
- **AI Models**: Claude 3.5 Sonnet (primary), OpenAI GPT-4o (secondary)
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: SentenceTransformer (all-mpnet-base-v2)
- **Document Processing**: PyPDF2, python-docx, tiktoken
- **Analytics**: SQLite with feedback tracking
- **Deployment**: Streamlit Cloud compatible

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
OPENAI_API_KEY=your_api_key           # Required
OPENAI_MODEL=gpt-4o                   # Default model

# Database Configuration
CHROMA_DB_PATH=./betty_chroma_db      # Vector database path

# Processing Configuration
CHUNK_SIZE=500                        # Text chunk size in tokens
CHUNK_OVERLAP=50                      # Token overlap between chunks
MAX_SEARCH_RESULTS=3                  # Number of RAG search results
MAX_FILE_SIZE_MB=10                   # Maximum upload file size

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2      # Embedding model
TOKENIZER_MODEL=cl100k_base           # Tokenizer for chunking
```

### Streamlit Secrets (for cloud deployment)
Create `.streamlit/secrets.toml`:
```toml
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

## 🚀 Deployment

### Local Development
```bash
streamlit run betty_app.py --server.port 8501
```

### Streamlit Cloud
1. Connect GitHub repository to Streamlit Cloud
2. Add `OPENAI_API_KEY` to Streamlit secrets
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
chunks = document_processor.chunk_text(text, chunk_size=500, overlap=50)
```

### Vector Store
```python
from utils.vector_store import betty_vector_store

# Add documents to knowledge base
success = betty_vector_store.add_documents_from_files(
    collection_name="betty_knowledge", 
    file_paths=["document.pdf"]
)

# Search knowledge base
results = betty_vector_store.search_collection(
    collection_name="betty_knowledge",
    query="outcome-based thinking",
    n_results=3
)
```

## 🛠️ Development

### Project Structure
```
betty_for_molex/
├── betty_app.py              # Main Betty application
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration classes
├── utils/
│   ├── __init__.py
│   ├── document_processor.py # Document processing utilities
│   └── vector_store.py       # ChromaDB interface
├── requirements.txt          # Python dependencies
├── Betty Detailed Doc.md     # Comprehensive documentation
└── README.md                # This file
```

### Key Classes
- **AppConfig**: Betty-specific configuration management
- **DocumentProcessor**: Text extraction and processing utilities
- **VectorStore**: ChromaDB interface with embedding generation
- **Betty System Prompt**: Comprehensive OBT methodology and personality

### Contributing
1. Follow established code patterns and error handling
2. Update documentation for new features
3. Test thoroughly with different document types
4. Maintain compatibility with Streamlit Cloud deployment

## 📊 Performance

### Optimization Features
- **Token-based chunking** with configurable overlap
- **Duplicate file detection** prevents redundant processing
- **Caching strategies** for expensive operations
- **Batch processing** for multiple documents
- **SQLite compatibility** fix for Streamlit Cloud

### Resource Requirements
- **Memory**: ~100MB base + document storage
- **Storage**: Vector database scales with knowledge base size
- **API Costs**: Varies based on OpenAI usage patterns

## 🔒 Security

### Data Protection
- Environment variables for sensitive configuration
- No API keys stored in code or version control
- Local vector database with no external data sharing
- File size limits prevent resource exhaustion

### Privacy
- Documents processed locally before embedding
- No sensitive data sent to external services (except OpenAI API)
- Vector database stored locally with encryption at rest

## 📞 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: See individual project README files for detailed guides
- **Development**: Follow contribution guidelines for code changes

## 👥 Credits

**Author**: Tony Begum, Chief AI Officer  
**Company**: BoldARC  

## 📜 License

Private repository - All rights reserved to BoldARC.

---

**Built with ❤️ for development and innovation**  
**Developed by BoldARC**

*Last updated: January 2025*