# Deployment Guide for RAG Chat Application

## Local Development

### Setup
1. Clone the repository
2. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run chat.py
   ```

### Local Storage
- Documents are stored in `./storage/documents/`
- ChromaDB data is stored in `./storage/chroma_db/`
- All data persists between sessions

## Streamlit Cloud Deployment

### Important Limitations
⚠️ **Ephemeral Storage**: Streamlit Cloud uses temporary storage that is cleared when the app restarts. This means:
- Uploaded documents are lost on restart
- Collections are not persistent
- User data is not preserved between sessions

### Setup Steps

1. **Fork/Push to GitHub**
   - Ensure your repository is on GitHub
   - The app automatically detects cloud environment

2. **Configure Secrets on Streamlit Cloud**
   - Go to your app settings on Streamlit Cloud
   - Navigate to "Secrets" section
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "your_api_key_here"
     ```

3. **Deploy**
   - Connect your GitHub repository to Streamlit Cloud
   - The app will automatically use cloud-optimized settings

### Cloud-Specific Settings
The application automatically adjusts for Streamlit Cloud:
- File size limit: 10MB (vs 50MB locally)
- Batch processing: 50 items (vs 100 locally)
- Parallel processing: Disabled (to conserve resources)
- Max workers: 2 (vs 4 locally)

## Persistent Storage Solutions

For production use with persistent storage, consider:

### Option 1: External Database
- PostgreSQL with pgvector extension
- MongoDB Atlas
- Pinecone, Weaviate, or Qdrant

### Option 2: Cloud Storage
- AWS S3 for documents
- Google Cloud Storage
- Azure Blob Storage

### Option 3: Managed Services
- Supabase Vector
- Redis Cloud with RediSearch
- Elasticsearch Cloud

## Environment Detection

The app automatically detects its environment:
- **Local**: Uses `./storage/` directory
- **Streamlit Cloud**: Uses temp directory with warnings

## Troubleshooting

### Issue: "Permission Denied" errors
- The app automatically falls back to temp directories
- Check logs for specific permission issues

### Issue: Collections disappear
- This is expected on Streamlit Cloud due to ephemeral storage
- Implement external storage for persistence

### Issue: ChromaDB/SQLite errors
- Ensure `pysqlite3-binary` is in requirements.txt
- The app includes compatibility workarounds

### Issue: Large files fail to upload
- Cloud limit is 10MB per file
- Consider chunking large documents before upload

## Performance Tips

1. **Optimize chunk sizes**: Use smaller chunks (500-1000) on cloud
2. **Limit retrieval results**: Keep n_results low (3-5) for faster responses
3. **Monitor token usage**: Cloud has limited resources
4. **Use caching**: Leverage Streamlit's caching for repeated operations

## Security Best Practices

1. Never commit API keys to the repository
2. Use Streamlit secrets for sensitive data
3. Implement rate limiting for production
4. Consider authentication for sensitive documents
5. Regularly rotate API keys

## Monitoring

Monitor your app's performance:
- Check Streamlit Cloud logs for errors
- Monitor OpenAI API usage
- Track document processing times
- Watch for memory usage warnings