"""
Vector store implementation using ChromaDB with persistent storage.
Independent of Betty's vector store for clean separation.
"""
import os
import uuid
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import ChatConfig, RAGConfig, CHROMA_DB_PATH
from rag.chunker import Chunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

class ChatVectorStore:
    """ChromaDB vector store with persistence and error handling"""
    
    def __init__(self, 
                 persist_directory: str = None,
                 embedding_model: str = None):
        
        self.persist_directory = persist_directory or CHROMA_DB_PATH
        self.embedding_model_name = embedding_model or ChatConfig.EMBEDDING_MODEL
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Connected to ChromaDB at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise VectorStoreError(f"Could not initialize vector store: {e}")
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.embedding_model_name}: {e}")
            raise VectorStoreError(f"Could not load embedding model: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=RAGConfig.BATCH_SIZE,
                show_progress_bar=len(texts) > 10
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise VectorStoreError(f"Embedding generation failed: {e}")
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name for ChromaDB compatibility"""
        if not name:
            raise VectorStoreError("Collection name cannot be empty")
        
        # Replace invalid characters
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
        
        # Add prefix if configured
        if ChatConfig.COLLECTION_PREFIX:
            sanitized = f"{ChatConfig.COLLECTION_PREFIX}_{sanitized}"
        
        # Ensure length constraints
        if len(sanitized) < 3:
            sanitized = f"{sanitized}_collection"
        if len(sanitized) > 63:
            sanitized = sanitized[:60] + "..."
        
        return sanitized
    
    def create_collection(self, 
                         name: str, 
                         metadata: Dict = None,
                         overwrite: bool = False) -> Any:
        """Create a new collection"""
        sanitized_name = self._sanitize_collection_name(name)
        
        try:
            # Check if collection exists
            try:
                existing_collection = self.client.get_collection(sanitized_name)
                if overwrite:
                    self.client.delete_collection(sanitized_name)
                    logger.info(f"Deleted existing collection: {sanitized_name}")
                else:
                    logger.info(f"Collection already exists: {sanitized_name}")
                    return existing_collection
            except Exception:
                # Collection doesn't exist, which is fine
                pass
            
            # Create collection with metadata
            collection_metadata = {
                "created_date": datetime.now().isoformat(),
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": ChatConfig.EMBEDDING_DIMENSION,
                **(metadata or {})
            }
            
            collection = self.client.create_collection(
                name=sanitized_name,
                metadata=collection_metadata
            )
            
            logger.info(f"Created collection: {sanitized_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection {sanitized_name}: {e}")
            raise VectorStoreError(f"Could not create collection: {e}")
    
    def get_collection(self, name: str) -> Optional[Any]:
        """Get existing collection"""
        sanitized_name = self._sanitize_collection_name(name)
        
        try:
            collection = self.client.get_collection(sanitized_name)
            logger.info(f"Retrieved collection: {sanitized_name} ({collection.count()} documents)")
            return collection
        except Exception as e:
            logger.warning(f"Collection {sanitized_name} not found: {e}")
            return None
    
    def get_or_create_collection(self, 
                                name: str, 
                                metadata: Dict = None) -> Any:
        """Get existing collection or create new one"""
        collection = self.get_collection(name)
        if collection is None:
            collection = self.create_collection(name, metadata)
        return collection
    
    def add_chunks(self, 
                   collection_name: str, 
                   chunks: List[Chunk],
                   batch_size: int = None) -> bool:
        """Add chunks to collection with batching"""
        if not chunks:
            logger.warning("No chunks provided to add")
            return False
        
        batch_size = batch_size or RAGConfig.BATCH_SIZE
        collection = self.get_or_create_collection(collection_name)
        
        try:
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Prepare batch data
                texts = [chunk.content for chunk in batch_chunks]
                ids = [f"{collection_name}_{uuid.uuid4().hex}" for _ in batch_chunks]
                metadatas = [chunk.metadata for chunk in batch_chunks]
                
                # Generate embeddings for batch
                embeddings = self._generate_embeddings(texts)
                
                # Add to collection
                collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
            
            total_docs = collection.count()
            logger.info(f"Successfully added {len(chunks)} chunks to {collection_name}. Total docs: {total_docs}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to {collection_name}: {e}")
            raise VectorStoreError(f"Could not add chunks: {e}")
    
    def search(self, 
               collection_name: str, 
               query: str, 
               n_results: int = 5,
               filter_metadata: Dict = None) -> List[Dict]:
        """Search collection for relevant documents"""
        collection = self.get_collection(collection_name)
        if not collection:
            logger.warning(f"Collection {collection_name} not found for search")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'][0] else 0.0,
                        'similarity': 1 - (results['distances'][0][i] if results['distances'][0] else 0.0)
                    })
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in formatted_results 
                if result['similarity'] >= ChatConfig.SIMILARITY_THRESHOLD
            ]
            
            logger.info(f"Search returned {len(filtered_results)} results above threshold")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search failed for {collection_name}: {e}")
            raise VectorStoreError(f"Search failed: {e}")
    
    def delete_collection(self, name: str) -> bool:
        """Delete collection"""
        sanitized_name = self._sanitize_collection_name(name)
        
        try:
            self.client.delete_collection(sanitized_name)
            logger.info(f"Deleted collection: {sanitized_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {sanitized_name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            # Remove prefix if configured
            if ChatConfig.COLLECTION_PREFIX:
                prefix_len = len(ChatConfig.COLLECTION_PREFIX) + 1
                collection_names = [
                    name[prefix_len:] if name.startswith(ChatConfig.COLLECTION_PREFIX + '_') else name
                    for name in collection_names
                ]
            
            logger.info(f"Found {len(collection_names)} collections")
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_collection_stats(self, name: str) -> Dict:
        """Get collection statistics"""
        collection = self.get_collection(name)
        if not collection:
            return {}
        
        try:
            count = collection.count()
            metadata = collection.metadata
            
            return {
                "name": name,
                "document_count": count,
                "created_date": metadata.get("created_date"),
                "embedding_model": metadata.get("embedding_model"),
                "embedding_dimension": metadata.get("embedding_dimension")
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {name}: {e}")
            return {}
    
    def update_collection_metadata(self, name: str, metadata: Dict) -> bool:
        """Update collection metadata"""
        collection = self.get_collection(name)
        if not collection:
            return False
        
        try:
            collection.modify(metadata=metadata)
            logger.info(f"Updated metadata for collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for {name}: {e}")
            return False
    
    def reset_database(self) -> bool:
        """Reset entire database - use with caution"""
        try:
            self.client.reset()
            logger.warning("Database reset completed")
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False

# Global vector store instance
chat_vector_store = ChatVectorStore()