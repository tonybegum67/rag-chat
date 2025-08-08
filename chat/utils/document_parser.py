"""
Document parsing utilities for the Chat RAG system.
Handles PDF, DOCX, and text file processing with persistence.
"""
import io
import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Document processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

from config.settings import ChatConfig, DOCUMENTS_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    """Handles document parsing with persistent storage"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else DOCUMENTS_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "documents_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load existing document metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load metadata file: {e}")
        return {}
    
    def _save_metadata(self):
        """Save document metadata to persistent storage"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save metadata: {e}")
    
    def _generate_file_hash(self, content: bytes) -> str:
        """Generate SHA-256 hash of file content for deduplication"""
        return hashlib.sha256(content).hexdigest()
    
    def _store_document(self, filename: str, content: str, file_hash: str) -> str:
        """Store document content persistently and return storage path"""
        # Create safe filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        if not safe_filename:
            safe_filename = f"document_{file_hash[:8]}"
        
        # Create unique storage path
        # If the filename already has .txt extension, don't add another one
        if safe_filename.endswith('.txt'):
            storage_filename = f"{file_hash[:8]}_{safe_filename}"
        else:
            storage_filename = f"{file_hash[:8]}_{safe_filename}.txt"
        storage_path = self.storage_path / storage_filename
        
        # Store document content
        try:
            with open(storage_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update metadata
            self.metadata[file_hash] = {
                "filename": filename,
                "storage_path": str(storage_path),
                "storage_filename": storage_filename,
                "upload_date": datetime.now().isoformat(),
                "file_size": len(content),
                "content_length": len(content)
            }
            self._save_metadata()
            
            logger.info(f"Stored document: {filename} -> {storage_filename}")
            return str(storage_path)
            
        except IOError as e:
            logger.error(f"Could not store document {filename}: {e}")
            raise
    
    def parse_pdf(self, file_content: Union[bytes, io.BytesIO]) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not available. Install with: pip install PyPDF2")
        
        try:
            if isinstance(file_content, bytes):
                file_content = io.BytesIO(file_content)
            
            reader = PyPDF2.PdfReader(file_content)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            if not text_parts:
                raise ValueError("No text content found in PDF")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise ValueError(f"Could not parse PDF file: {str(e)}")
    
    def parse_docx(self, file_content: Union[bytes, io.BytesIO]) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
        
        try:
            if isinstance(file_content, bytes):
                file_content = io.BytesIO(file_content)
            
            doc = Document(file_content)
            paragraphs = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            
            if not paragraphs:
                raise ValueError("No text content found in DOCX")
            
            return "\n\n".join(paragraphs)
            
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            raise ValueError(f"Could not parse DOCX file: {str(e)}")
    
    def parse_excel(self, file_content: Union[bytes, io.BytesIO]) -> str:
        """Extract text from Excel file (XLS, XLSX)"""
        if not EXCEL_AVAILABLE:
            raise ImportError("openpyxl and pandas not available. Install with: pip install openpyxl pandas")
        
        try:
            if isinstance(file_content, bytes):
                file_content = io.BytesIO(file_content)
            
            # Read Excel file with pandas
            # Try to read all sheets
            excel_file = pd.ExcelFile(file_content)
            text_parts = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read the sheet
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                    
                    # Skip empty sheets
                    if df.empty or df.isna().all().all():
                        continue
                    
                    sheet_text = f"--- Sheet: {sheet_name} ---\n"
                    
                    # Convert DataFrame to text representation
                    # Handle different data types appropriately
                    rows = []
                    for index, row in df.iterrows():
                        # Convert row to string, handling NaN values
                        row_values = []
                        for value in row:
                            if pd.isna(value):
                                row_values.append("")
                            elif isinstance(value, (int, float)):
                                row_values.append(str(value))
                            else:
                                row_values.append(str(value).strip())
                        
                        # Only add non-empty rows
                        row_text = " | ".join(row_values).strip()
                        if row_text and row_text != " | " * (len(row_values) - 1):
                            rows.append(row_text)
                    
                    if rows:
                        sheet_text += "\n".join(rows)
                        text_parts.append(sheet_text)
                        
                except Exception as e:
                    logger.warning(f"Could not process sheet '{sheet_name}': {e}")
                    continue
            
            if not text_parts:
                raise ValueError("No readable content found in Excel file")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error parsing Excel file: {e}")
            raise ValueError(f"Could not parse Excel file: {str(e)}")
    
    def parse_text(self, file_content: Union[bytes, str]) -> str:
        """Process plain text file"""
        try:
            if isinstance(file_content, bytes):
                # Try multiple encodings
                encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        text = file_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode text file with any supported encoding")
            else:
                text = file_content
            
            if not text.strip():
                raise ValueError("Text file is empty or contains only whitespace")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error parsing text file: {e}")
            raise ValueError(f"Could not parse text file: {str(e)}")
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[str, str]:
        """
        Process uploaded file and return (content, file_hash)
        Handles Streamlit uploaded file objects
        """
        try:
            # Read file content
            file_content = uploaded_file.read()
            file_hash = self._generate_file_hash(file_content)
            
            # Check if we've already processed this file
            if file_hash in self.metadata:
                logger.info(f"File {uploaded_file.name} already exists (hash: {file_hash[:8]})")
                storage_path = self.metadata[file_hash]["storage_path"]
                with open(storage_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content, file_hash
            
            # Parse based on file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            if file_ext == '.pdf':
                content = self.parse_pdf(file_content)
            elif file_ext == '.docx':
                content = self.parse_docx(file_content)
            elif file_ext in ['.xlsx', '.xls']:
                content = self.parse_excel(file_content)
            elif file_ext in ['.txt', '.md']:
                content = self.parse_text(file_content)
            else:
                # Try as text file
                try:
                    content = self.parse_text(file_content)
                except:
                    raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Clean and validate content
            content = self.clean_text(content)
            if len(content.strip()) < 10:
                raise ValueError("Document content too short (minimum 10 characters)")
            
            # Store document persistently
            storage_path = self._store_document(uploaded_file.name, content, file_hash)
            
            return content, file_hash
            
        except Exception as e:
            logger.error(f"Error processing uploaded file {uploaded_file.name}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Normalize whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Handle hyphenated line breaks
        
        return text.strip()
    
    def get_stored_documents(self) -> List[Dict]:
        """Get list of all stored documents"""
        documents = []
        for file_hash, metadata in self.metadata.items():
            if Path(metadata["storage_path"]).exists():
                documents.append({
                    "hash": file_hash,
                    "filename": metadata["filename"],
                    "upload_date": metadata["upload_date"],
                    "file_size": metadata["file_size"],
                    "storage_path": metadata["storage_path"]
                })
        return documents
    
    def get_document_content(self, file_hash: str) -> Optional[str]:
        """Retrieve stored document content by hash"""
        if file_hash in self.metadata:
            storage_path = self.metadata[file_hash]["storage_path"]
            if Path(storage_path).exists():
                try:
                    with open(storage_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except IOError as e:
                    logger.error(f"Could not read stored document {file_hash}: {e}")
        return None
    
    def delete_document(self, file_hash: str) -> bool:
        """Delete stored document and metadata"""
        if file_hash in self.metadata:
            try:
                storage_path = Path(self.metadata[file_hash]["storage_path"])
                if storage_path.exists():
                    storage_path.unlink()
                
                del self.metadata[file_hash]
                self._save_metadata()
                logger.info(f"Deleted document: {file_hash}")
                return True
            except Exception as e:
                logger.error(f"Could not delete document {file_hash}: {e}")
        return False

# Global instance
document_parser = DocumentParser()