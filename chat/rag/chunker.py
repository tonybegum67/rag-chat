"""
Robust text chunking strategies for the Chat RAG system.
Implements multiple chunking methods with comprehensive error handling.
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config.settings import RAGConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict
    
    @property
    def length(self) -> int:
        return len(self.content)
    
    def validate(self) -> bool:
        """Validate chunk content and metadata"""
        if not self.content or not self.content.strip():
            return False
        if len(self.content.strip()) < RAGConfig.MIN_CHUNK_CHARS:
            return False
        if len(self.content) > RAGConfig.MAX_CHUNK_CHARS:
            return False
        return True

class ChunkingError(Exception):
    """Custom exception for chunking errors"""
    pass

class TextChunker:
    """Advanced text chunking with multiple strategies and error handling"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 method: str = "recursive"):
        self.chunk_size = max(RAGConfig.MIN_CHUNK_CHARS, min(chunk_size, RAGConfig.MAX_CHUNK_CHARS))
        self.overlap = max(0, min(overlap, chunk_size // 2))
        self.method = method
        
        # Sentence splitters in order of preference
        self.sentence_splitters = RAGConfig.SENTENCE_SPLITTERS
        
        # Validate configuration
        if self.chunk_size <= self.overlap:
            raise ChunkingError(f"Chunk size ({self.chunk_size}) must be greater than overlap ({self.overlap})")
    
    def chunk_text(self, text: str, document_metadata: Dict = None) -> List[Chunk]:
        """
        Main chunking method with error handling and validation
        
        Args:
            text: Text to chunk
            document_metadata: Metadata about the source document
            
        Returns:
            List of validated Chunk objects
        """
        if not text or not text.strip():
            raise ChunkingError("Cannot chunk empty or whitespace-only text")
        
        # Clean and normalize text
        text = self._preprocess_text(text)
        
        # Apply chunking strategy
        try:
            if self.method == "recursive":
                chunks = self._recursive_chunk(text)
            elif self.method == "fixed":
                chunks = self._fixed_chunk(text)
            elif self.method == "semantic":
                chunks = self._semantic_chunk(text)
            else:
                logger.warning(f"Unknown chunking method '{self.method}', falling back to recursive")
                chunks = self._recursive_chunk(text)
        except Exception as e:
            logger.error(f"Chunking failed with method '{self.method}': {e}")
            # Fallback to simple fixed chunking
            chunks = self._fixed_chunk(text)
        
        # Add metadata and validate
        validated_chunks = []
        base_metadata = document_metadata or {}
        
        for i, (content, start_char, end_char) in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "chunk_method": self.method,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size_config": self.chunk_size,
                "overlap_config": self.overlap
            }
            
            chunk = Chunk(
                content=content,
                index=i,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata
            )
            
            if RAGConfig.CHUNK_VALIDATION and not chunk.validate():
                logger.warning(f"Chunk {i} failed validation, skipping")
                continue
            
            validated_chunks.append(chunk)
        
        if not validated_chunks:
            raise ChunkingError("No valid chunks generated from text")
        
        logger.info(f"Generated {len(validated_chunks)} chunks using {self.method} method")
        return validated_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text before chunking"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive punctuation repetitions
        text = re.sub(r'([.!?])\1{2,}', r'\1', text)
        
        return text.strip()
    
    def _recursive_chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Recursive chunking that tries to split on natural boundaries
        """
        chunks = []
        
        def _split_text(text_part: str, start_offset: int = 0) -> None:
            if len(text_part) <= self.chunk_size:
                chunks.append((text_part, start_offset, start_offset + len(text_part)))
                return
            
            # Try to split on sentence boundaries first
            best_split = None
            best_score = 0
            
            for splitter in self.sentence_splitters:
                split_positions = [m.start() + len(splitter) 
                                 for m in re.finditer(re.escape(splitter), text_part)]
                
                for pos in split_positions:
                    if pos > self.chunk_size - self.overlap and pos < self.chunk_size + self.overlap:
                        # Score based on how close to ideal chunk size
                        score = 1.0 - abs(pos - self.chunk_size) / self.chunk_size
                        if score > best_score:
                            best_score = score
                            best_split = pos
            
            # If no good sentence boundary found, split at word boundary
            if best_split is None:
                words = text_part[:self.chunk_size].split()
                if len(words) > 1:
                    best_split = len(' '.join(words[:-1])) + 1
                else:
                    best_split = self.chunk_size
            
            # Create first chunk
            first_chunk = text_part[:best_split]
            chunks.append((first_chunk, start_offset, start_offset + len(first_chunk)))
            
            # Recursively process remainder with overlap
            if len(text_part) > best_split:
                overlap_start = max(0, best_split - self.overlap)
                remaining_text = text_part[overlap_start:]
                _split_text(remaining_text, start_offset + overlap_start)
        
        _split_text(text)
        return chunks
    
    def _fixed_chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Fixed-size chunking with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at word boundary
            if end < len(text):
                # Look backwards for space
                word_end = text.rfind(' ', start, end)
                if word_end > start + self.chunk_size // 2:  # Don't make chunks too small
                    end = word_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start, end))
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap)
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Semantic chunking based on paragraph and section boundaries
        Falls back to recursive chunking for large paragraphs
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_start = 0
        text_position = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                text_position += len(para) + 2  # Account for newlines
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append((current_chunk.strip(), chunk_start, text_position))
                
                # Start new chunk with overlap if needed
                if len(para) > self.chunk_size:
                    # Large paragraph needs recursive chunking
                    para_chunks = self._recursive_chunk(para)
                    for content, rel_start, rel_end in para_chunks:
                        chunks.append((content, text_position + rel_start, text_position + rel_end))
                    text_position += len(para) + 2
                    current_chunk = ""
                    chunk_start = text_position
                else:
                    current_chunk = para
                    chunk_start = text_position
                    text_position += len(para) + 2
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    chunk_start = text_position
                text_position += len(para) + 2
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), chunk_start, text_position))
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict:
        """Generate statistics about chunks"""
        if not chunks:
            return {}
        
        lengths = [len(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(lengths),
            "avg_chunk_size": sum(lengths) / len(lengths),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths),
            "chunking_method": self.method,
            "config_chunk_size": self.chunk_size,
            "config_overlap": self.overlap
        }
    
    def optimize_parameters(self, text: str, target_chunks: int = None) -> Tuple[int, int]:
        """
        Suggest optimal chunk size and overlap for given text
        
        Args:
            text: Sample text to analyze
            target_chunks: Desired number of chunks (optional)
            
        Returns:
            Tuple of (optimal_chunk_size, optimal_overlap)
        """
        text_length = len(text)
        
        if target_chunks:
            # Calculate chunk size for target number of chunks
            optimal_size = max(RAGConfig.MIN_CHUNK_CHARS, 
                             min(text_length // target_chunks, RAGConfig.MAX_CHUNK_CHARS))
        else:
            # Use default with adjustments based on text length
            if text_length < 2000:
                optimal_size = min(text_length // 2, 500)
            elif text_length > 50000:
                optimal_size = 2000
            else:
                optimal_size = self.chunk_size
        
        # Optimize overlap (typically 10-20% of chunk size)
        optimal_overlap = min(optimal_size // 5, 300)
        
        return optimal_size, optimal_overlap

# Global chunker instance
default_chunker = TextChunker()