# RAG Chat Application

A Retrieval-Augmented Generation (RAG) chat application built with Streamlit and ChromaDB for document-based conversations.

## Features

- **Document Upload & Processing**: Upload and process PDF and text documents
- **Vector Storage**: ChromaDB integration for efficient document retrieval
- **RAG Pipeline**: Combines document retrieval with AI-generated responses
- **Interactive Chat**: Streamlit-based chat interface
- **Feedback System**: Built-in feedback collection for response quality

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tonybegum67/rag-chat.git
cd rag-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run chat.py
```

2. Upload documents through the web interface
3. Start chatting with your documents using the RAG system

## Configuration

Update `config/settings.py` to customize:
- Vector database settings
- Document processing parameters
- AI model configurations

## Dependencies

See `requirements.txt` for the complete list of dependencies including:
- Streamlit for the web interface
- ChromaDB for vector storage
- Various document processing libraries

## License

This project is licensed under the MIT License.