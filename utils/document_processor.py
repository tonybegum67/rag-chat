"""
Document processing utilities for Betty AI Assistant.

This module provides reusable functions for extracting text from various
document formats and processing them for use in the RAG system.
"""

import io
import re
from typing import List, Optional
import PyPDF2
import docx
import streamlit as st
import tiktoken
from config.settings import AppConfig


class DocumentProcessor:
    """Document processing utilities with improved error handling."""
    
    def __init__(self, tokenizer_model: str = None):
        """Initialize the document processor.
        
        Args:
            tokenizer_model: The tokenizer model to use for chunking.
        """
        self.tokenizer = tiktoken.get_encoding(
            tokenizer_model or AppConfig.TOKENIZER_MODEL
        )
    
    def extract_text_from_pdf(self, file: io.BytesIO) -> str:
        """Extract text from an in-memory PDF file.
        
        Args:
            file: BytesIO object containing PDF data.
            
        Returns:
            Extracted text as string, empty string if extraction fails.
        """
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_text = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(text)
                except Exception as e:
                    st.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
                    
            return "\n".join(pages_text)
            
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""
    
    def extract_text_from_docx(self, file: io.BytesIO) -> str:
        """Extract text from an in-memory DOCX file.
        
        Args:
            file: BytesIO object containing DOCX data.
            
        Returns:
            Extracted text as string, empty string if extraction fails.
        """
        try:
            doc = docx.Document(file)
            paragraphs = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
                    
            return "\n".join(paragraphs)
            
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            return ""
    
    def extract_text_from_txt(self, file: io.BytesIO) -> str:
        """Extract text from a plain text file.
        
        Args:
            file: BytesIO object containing text data.
            
        Returns:
            Extracted text as string, empty string if extraction fails.
        """
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                file.seek(0)
                return file.read().decode('latin-1')
            except Exception as e:
                st.error(f"Error reading text file: {e}")
                return ""
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text with normalized spacing and formatting.
        """
        if not text:
            return ""
        
        # Fix common formatting issues
        text = re.sub(r'([.,])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Clean up line breaks and spacing
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = None, 
        overlap: int = None
    ) -> List[str]:
        """Split text into overlapping chunks based on token count.
        
        Args:
            text: Text to chunk.
            chunk_size: Size of each chunk in tokens.
            overlap: Number of overlapping tokens between chunks.
            
        Returns:
            List of text chunks.
        """
        chunk_size = chunk_size or AppConfig.CHUNK_SIZE
        overlap = overlap or AppConfig.CHUNK_OVERLAP
        
        # Validate parameters
        if overlap >= chunk_size:
            st.warning(f"Overlap ({overlap}) must be less than chunk size ({chunk_size})")
            overlap = chunk_size // 4
        
        try:
            tokens = self.tokenizer.encode(text)
            chunks = []
            
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                if chunk_text.strip():
                    chunks.append(chunk_text)
            
            return chunks
            
        except Exception as e:
            st.error(f"Error chunking text: {e}")
            return [text]  # Return original text as fallback
    
    def get_file_type(self, filename: str) -> Optional[str]:
        """Determine file type from filename.
        
        Args:
            filename: Name of the file.
            
        Returns:
            File type string or None if unsupported.
        """
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            return 'pdf'
        elif filename_lower.endswith('.docx'):
            return 'docx'
        elif filename_lower.endswith('.txt'):
            return 'txt'
        else:
            return None
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """Process an uploaded file and extract text.
        
        Args:
            uploaded_file: Streamlit uploaded file object.
            
        Returns:
            Extracted and cleaned text.
        """
        if not uploaded_file:
            return ""
        
        # Check file size
        if uploaded_file.size > AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File {uploaded_file.name} is too large "
                    f"(max {AppConfig.MAX_FILE_SIZE_MB}MB)")
            return ""
        
        file_type = self.get_file_type(uploaded_file.name)
        if not file_type:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return ""
        
        try:
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            
            if file_type == 'pdf':
                text = self.extract_text_from_pdf(file_bytes)
            elif file_type == 'docx':
                text = self.extract_text_from_docx(file_bytes)
            elif file_type == 'txt':
                text = self.extract_text_from_txt(file_bytes)
            else:
                return ""
            
            return self.clean_text(text)
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            return ""


# Create a global instance for easy importing
document_processor = DocumentProcessor()