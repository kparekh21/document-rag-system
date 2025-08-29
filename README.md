# Document Rag System

A production-ready Retrieval-Augmented Generation system that combines multiple search methods to provide accurate answers from your document collection.

## Overview

This system ingests PDF documents from Google Drive, processes them into searchable chunks, and provides a conversational interface for document-based Q&A. It uses hybrid retrieval combining BM25 keyword search, dense vector similarity, and ELSER sparse embeddings with Reciprocal Rank Fusion for optimal results.

## Architecture

**Backend:** FastAPI + Elasticsearch + Groq LLM  
**Frontend:** Streamlit web interface  
**Retrieval:** Hybrid search with BM25, Dense Vectors, ELSER  
**Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)  
**LLM:** Groq API with Qwen models  
**Storage:** Elasticsearch with vector support  
**Deployment:** Docker Compose

## Key Features

- **Hybrid Retrieval**: Combines multiple search methods using Reciprocal Rank Fusion
- **Google Drive Integration**: Automatic PDF ingestion from shared folders
- **Safety Guardrails**: Content filtering and response grounding verification
- **Scalable Architecture**: Containerized components with health monitoring
- **Interactive UI**: Clean Streamlit interface with real-time search
- **Production Ready**: Comprehensive error handling and logging

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Google Drive API credentials
- Groq API key

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd hybrid-rag-system
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file with your credentials:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here

# Optional customization
GROQ_MODEL=qwen/qwen-2.5-72b-instruct
CHUNK_SIZE=300
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Add Google Credentials

Place your `credentials.json` file (Google Service Account) in the project root.

### 4. Start Services

```bash
# Start Elasticsearch and Kibana
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt
```

### 5. Run the Application

```bash
# Start the API server
python main.py

# In another terminal, start the UI
streamlit run streamlit_app.py
```

Access the application at `http://localhost:8501`

## Usage

### Document Ingestion

1. Navigate to the "Ingest Documents" tab
2. Optionally specify a Google Drive folder ID
3. Set maximum files to process
4. Click "Start Ingestion"

The system will:
- Download PDFs from Google Drive
- Extract and chunk text content
- Generate embeddings (dense + sparse)
- Index documents in Elasticsearch

### Querying Documents

1. Use the "Query Documents" tab
2. Enter your question in natural language
3. Select retrieval mode (hybrid recommended)
4. Adjust number of documents to consider
5. Enable citations for source references

## API Endpoints

The FastAPI backend provides REST endpoints:

- `POST /query` - Submit questions and get answers
- `POST /ingest` - Trigger document ingestion
- `GET /healthz` - System health check
- `GET /stats` - Document statistics
- `GET /docs` - Interactive API documentation

## Configuration

### Retrieval Modes

- **Hybrid** (default): Combines all methods with RRF
- **BM25**: Traditional keyword search
- **Dense**: Semantic vector search
- **ELSER**: Elasticsearch's sparse vector search

### Customization

Key configuration options in `.env`:

```bash
# Chunk processing
CHUNK_SIZE=300                    # Target chunk size in tokens
CHUNK_OVERLAP=50                  # Overlap between chunks

# Retrieval
DEFAULT_TOP_K=5                   # Documents per query
DEFAULT_RETRIEVAL_MODE=hybrid     # Default search method

# LLM
GROQ_MODEL=qwen/qwen-2.5-72b-instruct
GROQ_API_KEY=your_key_here

# Elasticsearch
ELASTIC_HOST=localhost
ELASTIC_PORT=9200
ELASTIC_INDEX_NAME=documents
```

## Development

### Project Structure

```
├── main.py                 # FastAPI application
├── streamlit_app.py       # Streamlit UI
├── indexing/
│   ├── elastic_client.py  # Elasticsearch operations
│   └── embeddings.py      # Embedding generation
├── ingestion/
│   └── pdf_loader.py      # PDF processing pipeline
├── retrieval/
│   └── retriever.py       # Hybrid retrieval system
├── generation/
│   └── llm_client.py      # Groq LLM client
└── docker-compose.yml     # Container orchestration
```

### Running Tests

```bash
# Test individual components
python -m indexing.elastic_client
python -m generation.llm_client
python -m retrieval.retriever
```

### Health Monitoring

Check system health:
```bash
curl http://localhost:8000/healthz
```

## Troubleshooting

### Common Issues

**Elasticsearch Connection Failed**
```bash
# Check if Elasticsearch is running
curl http://localhost:9200
docker-compose ps
```

**Google Drive Access Denied**
- Verify `credentials.json` is valid service account key
- Ensure service account has access to the target folder
- Check `GOOGLE_DRIVE_FOLDER_ID` is correct

**Groq API Errors**
- Verify `GROQ_API_KEY` is valid
- Check API quota and rate limits
- Confirm model name is correct

**No Search Results**
- Ensure documents were ingested successfully
- Check document count: `curl http://localhost:8000/stats`
- Verify Elasticsearch index exists

### Performance Tuning

For large document collections:

1. **Increase chunk size** for longer context
2. **Adjust RRF weights** for your use case
3. **Scale Elasticsearch** with more nodes
4. **Optimize embeddings** with batch processing

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Open an issue on GitHub
