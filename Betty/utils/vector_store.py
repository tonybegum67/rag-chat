"""
Vector store utilities for Betty AI Assistant.

This module provides a high-level interface for ChromaDB operations
with improved error handling and configuration management.
"""

import os
import io
from typing import List, Dict, Any, Optional
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from config.settings import AppConfig
from utils.document_processor import document_processor


class VectorStore:
    """High-level interface for vector database operations."""
    
    def __init__(
        self, 
        db_path: str = None, 
        embedding_model_name: str = None
    ):
        """Initialize the vector store.
        
        Args:
            db_path: Path to ChromaDB storage directory.
            embedding_model_name: Name of the embedding model to use.
        """
        self.db_path = db_path or AppConfig.CHROMA_DB_PATH
        self.embedding_model_name = embedding_model_name or AppConfig.EMBEDDING_MODEL
        
        # Initialize components
        self._client = None
        self._embedding_model = None
        self._reranker = None
        self._init_components()
    
    def _init_components(self):
        """Initialize ChromaDB client, embedding model, and reranker."""
        try:
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            if AppConfig.USE_RERANKING:
                self._reranker = CrossEncoder(AppConfig.RERANKER_MODEL)
        except Exception as e:
            st.error(f"Failed to initialize vector store: {e}")
            raise
    
    @property
    def client(self):
        """Get the ChromaDB client."""
        if self._client is None:
            self._init_components()
        return self._client
    
    @property
    def embedding_model(self):
        """Get the embedding model."""
        if self._embedding_model is None:
            self._init_components()
        return self._embedding_model
    
    @property
    def reranker(self):
        """Get the reranker model."""
        if self._reranker is None and AppConfig.USE_RERANKING:
            self._init_components()
        return self._reranker
    
    def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            ChromaDB collection object.
        """
        try:
            return self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            st.error(f"Failed to get/create collection '{collection_name}': {e}")
            raise
    
    def add_documents_from_files(
        self, 
        collection_name: str, 
        file_paths: List[str],
        show_progress: bool = True
    ) -> bool:
        """Add documents from file paths to a collection.
        
        Args:
            collection_name: Name of the target collection.
            file_paths: List of file paths to process.
            show_progress: Whether to show progress indicators.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # Check existing files
            existing_files = self._get_existing_files(collection)
            files_to_add = [
                fp for fp in file_paths 
                if os.path.basename(fp) not in existing_files
            ]
            
            if not files_to_add:
                if show_progress:
                    st.sidebar.info("Knowledge base is already up-to-date.")
                return True
            
            # Process files
            documents_data = []
            
            progress_text = f"Processing {len(files_to_add)} documents..."
            if show_progress:
                with st.spinner(progress_text):
                    documents_data = self._process_files_for_collection(files_to_add)
            else:
                documents_data = self._process_files_for_collection(files_to_add)
            
            if not documents_data:
                if show_progress:
                    st.sidebar.warning("No valid documents to add.")
                return False
            
            # Add to collection
            return self._add_documents_to_collection(
                collection, documents_data, show_progress
            )
            
        except Exception as e:
            st.error(f"Error adding documents to collection: {e}")
            return False
    
    def search_collection(
        self, 
        collection_name: str, 
        query: str, 
        n_results: int = None
    ) -> List[Dict[str, Any]]:
        """Search a collection for relevant documents.
        
        Args:
            collection_name: Name of the collection to search.
            query: Search query string.
            n_results: Number of results to return.
            
        Returns:
            List of search results with content and metadata.
        """
        n_results = n_results or AppConfig.MAX_SEARCH_RESULTS
        
        try:
            collection = self.client.get_collection(name=collection_name)
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            return [
                {
                    "content": doc,
                    "metadata": meta
                }
                for doc, meta in zip(
                    results["documents"][0], 
                    results["metadatas"][0]
                )
            ]
            
        except Exception as e:
            st.error(f"Error searching collection '{collection_name}': {e}")
            return []
    
    def search_collection_with_reranking(
        self, 
        collection_name: str, 
        query: str, 
        n_results: int = None,
        initial_results_multiplier: int = 3
    ) -> List[Dict[str, Any]]:
        """Search collection with cross-encoder reranking for better relevance.
        
        Args:
            collection_name: Name of the collection to search.
            query: Search query string.
            n_results: Number of final results to return.
            initial_results_multiplier: How many initial results to get before reranking.
            
        Returns:
            List of reranked search results with content and metadata.
        """
        n_results = n_results or AppConfig.MAX_SEARCH_RESULTS
        
        if not AppConfig.USE_RERANKING or self.reranker is None:
            return self.search_collection(collection_name, query, n_results)
        
        try:
            # Get more initial results for reranking
            initial_n = min(n_results * initial_results_multiplier, 20)
            initial_results = self.search_collection(collection_name, query, initial_n)
            
            if len(initial_results) <= n_results:
                return initial_results
            
            # Prepare query-document pairs for reranking
            pairs = [(query, doc['content']) for doc in initial_results]
            
            # Get relevance scores from cross-encoder
            scores = self.reranker.predict(pairs)
            
            # Combine results with scores and sort
            scored_results = [
                (doc, float(score)) 
                for doc, score in zip(initial_results, scores)
            ]
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top n_results with added relevance scores
            reranked_results = []
            for doc, score in scored_results[:n_results]:
                doc_with_score = doc.copy()
                doc_with_score['relevance_score'] = score
                reranked_results.append(doc_with_score)
            
            return reranked_results
            
        except Exception as e:
            st.error(f"Error in reranking for collection '{collection_name}': {e}")
            # Fallback to regular search
            return self.search_collection(collection_name, query, n_results)
    
    def _get_existing_files(self, collection) -> set:
        """Get set of existing filenames in a collection."""
        try:
            existing_data = collection.get(include=["metadatas"])
            return {
                meta.get('filename') 
                for meta in existing_data.get('metadatas', [])
                if meta and meta.get('filename')
            }
        except Exception:
            return set()
    
    def _process_files_for_collection(self, file_paths: List[str]) -> List[Dict]:
        """Process files and return document data for collection."""
        documents_data = []
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            
            try:
                # Read and process file
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                
                file_type = document_processor.get_file_type(filename)
                if not file_type:
                    st.warning(f"Unsupported file type: {filename}")
                    continue
                
                # Extract text based on file type
                file_io = io.BytesIO(file_bytes)
                if file_type == 'pdf':
                    text = document_processor.extract_text_from_pdf(file_io)
                elif file_type == 'docx':
                    text = document_processor.extract_text_from_docx(file_io)
                elif file_type == 'txt':
                    text = document_processor.extract_text_from_txt(file_io)
                else:
                    continue
                
                if not text.strip():
                    st.warning(f"No text extracted from {filename}")
                    continue
                
                cleaned_text = document_processor.clean_text(text)
                if AppConfig.USE_SEMANTIC_CHUNKING:
                    chunks = document_processor.semantic_chunk_text(cleaned_text)
                else:
                    chunks = document_processor.chunk_text(cleaned_text)
                
                documents_data.append({
                    'filename': filename,
                    'chunks': chunks
                })
                
            except Exception as e:
                st.error(f"Failed to process {filename}: {e}")
                continue
        
        return documents_data
    
    def _add_documents_to_collection(
        self, 
        collection, 
        documents_data: List[Dict], 
        show_progress: bool
    ) -> bool:
        """Add processed documents to ChromaDB collection."""
        try:
            all_chunks = []
            metadatas = []
            ids = []
            
            doc_offset = collection.count()
            
            for doc_idx, doc_data in enumerate(documents_data):
                filename = doc_data['filename']
                chunks = doc_data['chunks']
                
                for chunk_idx, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                        
                    all_chunks.append(chunk)
                    metadatas.append({
                        "filename": filename,
                        "chunk_index": chunk_idx
                    })
                    ids.append(f"doc_{doc_offset + doc_idx}_chunk_{chunk_idx}")
            
            if not all_chunks:
                return False
            
            # Generate embeddings and add to collection
            embeddings = self.embedding_model.encode(
                all_chunks, 
                show_progress_bar=show_progress
            ).tolist()
            
            collection.add(
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            if show_progress:
                st.sidebar.success(
                    f"Successfully added {len(documents_data)} documents "
                    f"({len(all_chunks)} chunks) to the knowledge base."
                )
            
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections in the vector store."""
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception as e:
            st.error(f"Error listing collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from the vector store.
        
        Args:
            collection_name: Name of the collection to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            st.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def reset_collection_for_embedding_model(self, collection_name: str) -> bool:
        """Reset a collection to match current embedding model dimensions.
        
        This method deletes and recreates a collection to resolve embedding
        dimension mismatches that can occur when switching embedding models.
        
        Args:
            collection_name: Name of the collection to reset.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Check if collection exists
            collections = [c.name for c in self.client.list_collections()]
            if collection_name in collections:
                st.info(f"Deleting existing collection '{collection_name}' to resolve embedding dimension mismatch...")
                self.client.delete_collection(collection_name)
                st.success(f"Collection '{collection_name}' deleted successfully.")
            
            # Create new collection (will be created with correct dimensions on first add)
            self.get_or_create_collection(collection_name)
            st.success(f"Collection '{collection_name}' recreated with correct embedding dimensions.")
            return True
            
        except Exception as e:
            st.error(f"Error resetting collection '{collection_name}': {e}")
            return False


# Create global instances for easy importing
betty_vector_store = VectorStore(
    db_path=AppConfig.CHROMA_DB_PATH,
    embedding_model_name=AppConfig.EMBEDDING_MODEL
)

chat_vector_store = VectorStore(
    db_path="./chroma_db",  # Chat app uses different path
    embedding_model_name=AppConfig.EMBEDDING_MODEL
)