"""
Document retrieval system with advanced filtering and ranking.
Handles the complete RAG pipeline with error handling.
"""
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from config.settings import ChatConfig, RAGConfig
from rag.vector_store import chat_vector_store, VectorStoreError
from rag.chunker import default_chunker, Chunk, ChunkingError
from utils.document_parser import document_parser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a document retrieval result"""
    content: str
    metadata: Dict
    similarity: float
    distance: float
    rank: int
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'similarity': self.similarity,
            'distance': self.distance,
            'rank': self.rank
        }

class RetrievalError(Exception):
    """Custom exception for retrieval operations"""
    pass

class DocumentRetriever:
    """Advanced document retrieval with filtering and ranking"""
    
    def __init__(self, vector_store=None):
        self.vector_store = vector_store or chat_vector_store
        self.chunker = default_chunker
        self.parser = document_parser
    
    def add_document(self, 
                    collection_name: str, 
                    uploaded_file,
                    chunk_size: int = None,
                    overlap: int = None,
                    metadata: Dict = None) -> Tuple[bool, str, Dict]:
        """
        Add a document to the collection with comprehensive error handling
        
        Returns:
            Tuple of (success, message, stats)
        """
        try:
            # Parse and store document
            logger.info(f"Processing document: {uploaded_file.name}")
            content, file_hash = self.parser.process_uploaded_file(uploaded_file)
            
            # Prepare document metadata
            doc_metadata = {
                "filename": uploaded_file.name,
                "file_hash": file_hash,
                "file_type": uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else 'unknown',
                "upload_date": datetime.now().isoformat(),
                "file_size": len(content),
                **(metadata or {})
            }
            
            # Configure chunker if parameters provided
            if chunk_size or overlap:
                chunker = default_chunker.__class__(
                    chunk_size=chunk_size or default_chunker.chunk_size,
                    overlap=overlap or default_chunker.overlap,
                    method=default_chunker.method
                )
            else:
                chunker = self.chunker
            
            # Generate chunks
            try:
                chunks = chunker.chunk_text(content, doc_metadata)
                if not chunks:
                    return False, "No valid chunks generated from document", {}
            except ChunkingError as e:
                logger.error(f"Chunking failed for {uploaded_file.name}: {e}")
                return False, f"Text chunking failed: {str(e)}", {}
            
            # Add chunks to vector store
            try:
                success = self.vector_store.add_chunks(collection_name, chunks)
                if not success:
                    return False, "Failed to add chunks to vector store", {}
            except VectorStoreError as e:
                logger.error(f"Vector store error for {uploaded_file.name}: {e}")
                return False, f"Vector store error: {str(e)}", {}
            
            # Generate statistics
            stats = {
                "filename": uploaded_file.name,
                "file_hash": file_hash,
                "content_length": len(content),
                "chunks_created": len(chunks),
                "chunk_stats": chunker.get_chunk_stats(chunks),
                "upload_date": doc_metadata["upload_date"]
            }
            
            logger.info(f"Successfully added {uploaded_file.name}: {len(chunks)} chunks")
            return True, f"Successfully processed {uploaded_file.name}", stats
            
        except Exception as e:
            logger.error(f"Unexpected error processing {uploaded_file.name}: {e}")
            return False, f"Processing failed: {str(e)}", {}
    
    def retrieve(self, 
                collection_name: str, 
                query: str,
                n_results: int = None,
                similarity_threshold: float = None,
                metadata_filter: Dict = None,
                rerank: bool = True) -> List[RetrievalResult]:
        """
        Retrieve relevant documents with advanced filtering and ranking
        
        Args:
            collection_name: Name of the collection to search
            query: Search query
            n_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            metadata_filter: Dictionary to filter results by metadata
            rerank: Whether to apply re-ranking based on multiple factors
            
        Returns:
            List of RetrievalResult objects
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for retrieval")
            return []
        
        n_results = n_results or ChatConfig.DEFAULT_N_RESULTS
        similarity_threshold = similarity_threshold or ChatConfig.SIMILARITY_THRESHOLD
        
        try:
            # Perform vector search
            raw_results = self.vector_store.search(
                collection_name=collection_name,
                query=query,
                n_results=min(n_results * 2, 50),  # Get more for re-ranking
                filter_metadata=metadata_filter
            )
            
            if not raw_results:
                logger.info(f"No results found for query in {collection_name}")
                return []
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in raw_results 
                if result['similarity'] >= similarity_threshold
            ]
            
            if not filtered_results:
                logger.info(f"No results above similarity threshold {similarity_threshold}")
                return []
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, result in enumerate(filtered_results):
                retrieval_results.append(RetrievalResult(
                    content=result['content'],
                    metadata=result['metadata'],
                    similarity=result['similarity'],
                    distance=result['distance'],
                    rank=i + 1
                ))
            
            # Apply re-ranking if enabled
            if rerank and len(retrieval_results) > 1:
                retrieval_results = self._rerank_results(query, retrieval_results)
            
            # Limit to requested number of results
            final_results = retrieval_results[:n_results]
            
            logger.info(f"Retrieved {len(final_results)} results for query")
            return final_results
            
        except VectorStoreError as e:
            logger.error(f"Vector store error during retrieval: {e}")
            raise RetrievalError(f"Retrieval failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {e}")
            raise RetrievalError(f"Retrieval failed: {str(e)}")
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Re-rank results based on multiple factors beyond semantic similarity
        """
        query_words = set(query.lower().split())
        
        # Calculate additional ranking factors
        for result in results:
            content_lower = result.content.lower()
            
            # Exact word matches
            content_words = set(content_lower.split())
            word_overlap = len(query_words.intersection(content_words))
            word_overlap_ratio = word_overlap / len(query_words) if query_words else 0
            
            # Query term frequency in content
            query_term_freq = sum(content_lower.count(word) for word in query_words)
            
            # Content quality indicators
            content_length_score = min(len(result.content) / 1000, 1.0)  # Prefer moderate length
            
            # Recency if available
            recency_score = 1.0
            if 'upload_date' in result.metadata:
                try:
                    upload_date = datetime.fromisoformat(result.metadata['upload_date'])
                    days_old = (datetime.now() - upload_date).days
                    recency_score = max(0.1, 1.0 - (days_old / 365))  # Decay over a year
                except:
                    pass
            
            # Composite scoring
            composite_score = (
                result.similarity * 0.4 +          # Base similarity
                word_overlap_ratio * 0.25 +        # Exact word matches
                min(query_term_freq / 10, 1.0) * 0.2 +  # Term frequency
                content_length_score * 0.1 +       # Content length
                recency_score * 0.05               # Recency
            )
            
            # Store composite score in metadata for debugging
            result.metadata['_composite_score'] = composite_score
            result.metadata['_word_overlap_ratio'] = word_overlap_ratio
            result.metadata['_query_term_freq'] = query_term_freq
        
        # Sort by composite score
        ranked_results = sorted(results, 
                              key=lambda x: x.metadata.get('_composite_score', x.similarity), 
                              reverse=True)
        
        # Update rank numbers
        for i, result in enumerate(ranked_results):
            result.rank = i + 1
        
        return ranked_results
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """Get detailed information about a collection"""
        try:
            stats = self.vector_store.get_collection_stats(collection_name)
            if not stats:
                return {}
            
            # Get stored document info
            stored_docs = self.parser.get_stored_documents()
            
            # Enhance with document storage info
            stats['stored_documents'] = len(stored_docs)
            stats['storage_path'] = str(self.parser.storage_path)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return {}
    
    def list_collections(self) -> List[Dict]:
        """List all collections with their information"""
        try:
            collection_names = self.vector_store.list_collections()
            collections_info = []
            
            for name in collection_names:
                info = self.get_collection_info(name)
                if info:
                    collections_info.append(info)
            
            return collections_info
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str, 
                         delete_documents: bool = False) -> bool:
        """
        Delete collection and optionally stored documents
        
        Args:
            collection_name: Name of collection to delete
            delete_documents: Whether to also delete stored documents
        """
        try:
            # Delete from vector store
            success = self.vector_store.delete_collection(collection_name)
            
            if success and delete_documents:
                # Get collection info to find associated documents
                # This would require tracking which documents belong to which collections
                # For now, we'll leave document storage intact
                logger.info(f"Collection {collection_name} deleted. Document storage preserved.")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def health_check(self) -> Dict:
        """Perform health check on the retrieval system"""
        health = {
            "status": "unknown",
            "vector_store": "unknown",
            "document_parser": "unknown",
            "chunker": "unknown",
            "errors": []
        }
        
        try:
            # Check vector store
            collections = self.vector_store.list_collections()
            health["vector_store"] = "healthy"
            health["collections_count"] = len(collections)
        except Exception as e:
            health["vector_store"] = "error"
            health["errors"].append(f"Vector store error: {str(e)}")
        
        try:
            # Check document parser
            stored_docs = self.parser.get_stored_documents()
            health["document_parser"] = "healthy"
            health["stored_documents_count"] = len(stored_docs)
        except Exception as e:
            health["document_parser"] = "error"
            health["errors"].append(f"Document parser error: {str(e)}")
        
        try:
            # Check chunker
            test_text = "This is a test document for health checking."
            chunks = self.chunker.chunk_text(test_text)
            health["chunker"] = "healthy" if chunks else "warning"
        except Exception as e:
            health["chunker"] = "error"
            health["errors"].append(f"Chunker error: {str(e)}")
        
        # Overall status
        if not health["errors"]:
            health["status"] = "healthy"
        elif any("error" in str(error) for error in health["errors"]):
            health["status"] = "error"
        else:
            health["status"] = "warning"
        
        return health

# Global retriever instance
document_retriever = DocumentRetriever()