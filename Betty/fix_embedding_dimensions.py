#!/usr/bin/env python3
"""
Fix embedding dimension mismatch in Betty AI Assistant.

This script resolves the embedding dimension error by recreating the collection
with the correct dimensions for the current embedding model.
"""

import sys
from config.settings import AppConfig
from utils.vector_store import betty_vector_store

def main():
    """Main function to fix embedding dimensions."""
    print("🔧 Betty AI Assistant - Embedding Dimension Fix")
    print("=" * 50)
    
    collection_name = AppConfig.KNOWLEDGE_COLLECTION_NAME
    print(f"Current embedding model: {AppConfig.EMBEDDING_MODEL}")
    print(f"Target collection: {collection_name}")
    
    # Reset the collection
    print("\n🔄 Resetting collection to fix embedding dimensions...")
    success = betty_vector_store.reset_collection_for_embedding_model(collection_name)
    
    if success:
        print("✅ Collection reset successfully!")
        print("\n📝 Next steps:")
        print("1. Restart your Streamlit application")
        print("2. Re-upload your documents to rebuild the knowledge base")
        print("3. The embedding dimension error should be resolved")
    else:
        print("❌ Failed to reset collection. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()