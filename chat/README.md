# RAG Chat Application 💬

A sophisticated Retrieval-Augmented Generation (RAG) chat application built with Streamlit and ChromaDB for intelligent document-based conversations. Upload your documents and have natural conversations with their content using AI.

## ✨ Features

- **📄 Smart Document Processing**: Upload and process PDF, TXT, and DOCX documents with intelligent chunking
- **🔍 Vector Search**: ChromaDB integration with sentence-transformers for semantic document retrieval
- **🤖 AI-Powered Responses**: OpenAI GPT integration for contextual, document-grounded answers
- **💬 Interactive Chat Interface**: Clean, responsive Streamlit-based chat experience
- **📊 Feedback System**: Built-in response quality feedback and analytics
- **⚡ Performance Optimized**: Efficient caching and SQLite compatibility for cloud deployment
- **🎨 Visual Flow Diagrams**: Optional Mermaid integration for system architecture visualization

## Project Structure

```
chat/
├── chat.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── config/
│   └── settings.py        # Configuration settings
├── rag/
│   ├── chunker.py         # Document chunking utilities
│   ├── retriever.py       # Document retrieval logic
│   └── vector_store.py    # ChromaDB vector store management
├── utils/
│   └── document_parser.py # Document parsing utilities
├── storage/
│   ├── chroma_db/         # Vector database storage
│   └── documents/         # Processed document storage
└── data/
    └── betty_feedback.db  # Feedback database
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- OpenAI API key

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/tonybegum67/rag-chat.git
cd rag-chat
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API key:**
   - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
   - Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

### Usage

1. **Start the application:**
```bash
streamlit run chat.py
```

2. **Upload documents:** Use the sidebar to upload PDF, TXT, or DOCX files

3. **Start chatting:** Ask questions about your documents in natural language

4. **Provide feedback:** Rate responses to help improve the system

## ⚙️ Configuration

Customize the application by editing `config/settings.py`:

- **Model Settings**: Change OpenAI models, temperature, max tokens
- **Chunking Parameters**: Adjust document splitting strategies
- **Vector Store**: Configure ChromaDB collections and similarity thresholds
- **UI Customization**: Modify page title, icons, and styling

## 📦 Dependencies

**Core Libraries:**
- **streamlit** - Web interface framework
- **openai** - OpenAI API integration
- **chromadb==0.4.18** - Vector database for semantic search
- **sentence-transformers** - Text embedding generation
- **tiktoken** - Token counting for OpenAI models

**Document Processing:**
- **PyPDF2** - PDF document parsing
- **python-docx** - Word document processing

**System Compatibility:**
- **pysqlite3-binary** - SQLite compatibility for cloud deployment
- **streamlit-mermaid** - Optional system diagram visualization

## 🔧 Technical Architecture

The application follows a modular architecture:

- **Frontend**: Streamlit provides the interactive web interface
- **RAG Pipeline**: Custom retrieval system with chunking and semantic search  
- **Vector Store**: ChromaDB handles document embeddings and similarity search
- **AI Integration**: OpenAI GPT models generate contextual responses
- **Document Processing**: Multi-format parsing with intelligent text extraction
- **Feedback Loop**: SQLite database stores user feedback for continuous improvement

## 🚀 Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your `OPENAI_API_KEY` in the Streamlit Cloud secrets management
4. Deploy with one click

### Docker (Coming Soon)
Docker support will be added for containerized deployments.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Create an [issue](https://github.com/tonybegum67/rag-chat/issues) for bug reports
- Start a [discussion](https://github.com/tonybegum67/rag-chat/discussions) for questions
- Review the code and configuration files for customization options

## 🎯 Roadmap

- [ ] Multi-language document support
- [ ] Advanced chunking strategies
- [ ] Integration with more LLM providers
- [ ] Enhanced feedback analytics dashboard
- [ ] Docker containerization
- [ ] Real-time collaboration features