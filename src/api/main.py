"""
FastAPI Application for RAG System

Main API service providing endpoints for querying, ingestion, and health checks.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from indexing.elastic_client import get_elasticsearch_client, ElasticsearchClient
from indexing.embeddings import get_embedding_generator, EmbeddingGenerator  
from ingestion.pdf_loader import get_pdf_ingestion_pipeline, PDFIngestionPipeline
from retrieval.retriever import get_hybrid_retriever, HybridRetriever
from generation.llm_client import get_llm_client, GroqLLMClient

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components (initialized at startup)
es_client: Optional[ElasticsearchClient] = None
embedder: Optional[EmbeddingGenerator] = None
ingestion_pipeline: Optional[PDFIngestionPipeline] = None
retriever: Optional[HybridRetriever] = None
llm_client: Optional[GroqLLMClient] = None


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    retrieval_mode: str = Field(default="hybrid", description="Retrieval mode: bm25, dense, elser, or hybrid")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_citations: bool = Field(default=True, description="Whether to include citations")
    apply_guardrails: bool = Field(default=True, description="Whether to apply safety guardrails")


class Citation(BaseModel):
    """Citation model"""
    title: str = Field(..., description="Document title/filename")
    url: str = Field(..., description="Document URL")
    snippet: str = Field(..., description="Text snippet")
    relevance_score: Optional[float] = Field(None, description="Relevance score")
    retrieval_method: Optional[str] = Field(None, description="Retrieval method used")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    retrieval_mode: str = Field(..., description="Retrieval mode used")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")


class IngestionRequest(BaseModel):
    """Request model for ingestion endpoint"""
    folder_id: Optional[str] = Field(None, description="Google Drive folder ID (optional)")
    max_files: Optional[int] = Field(None, ge=1, le=100, description="Maximum number of files to process")
    force_recreate_index: bool = Field(default=False, description="Whether to recreate the index")


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""
    status: str = Field(..., description="Ingestion status")
    message: str = Field(..., description="Status message")
    files_processed: int = Field(..., description="Number of files processed")
    chunks_created: int = Field(..., description="Number of document chunks created")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Ingestion timestamp")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Overall health status")
    components: Dict[str, Any] = Field(..., description="Component health details")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Health check timestamp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting RAG System API...")
    
    global es_client, embedder, ingestion_pipeline, retriever, llm_client
    
    try:
        # Initialize components
        es_client = get_elasticsearch_client()
        embedder = get_embedding_generator()
        ingestion_pipeline = get_pdf_ingestion_pipeline()
        retriever = get_hybrid_retriever()
        llm_client = get_llm_client()
        
        # Create index if it doesn't exist
        es_client.create_index(force_recreate=False)
        
        logger.info("RAG System API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG System API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG System API...")


# Create FastAPI app
app = FastAPI(
    title=os.getenv("API_TITLE", "RAG System API"),
    description="Retrieval-Augmented Generation System with Elasticsearch and Groq",
    version=os.getenv("API_VERSION", "1.0.0"),
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_components():
    """Dependency to access initialized components"""
    if not all([es_client, embedder, ingestion_pipeline, retriever, llm_client]):
        raise HTTPException(status_code=503, detail="System components not initialized")
    
    return {
        "es_client": es_client,
        "embedder": embedder,
        "ingestion_pipeline": ingestion_pipeline,
        "retriever": retriever,
        "llm_client": llm_client
    }


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG System API",
        "version": os.getenv("API_VERSION", "1.0.0"),
        "status": "running",
        "docs": "/docs",
        "health": "/healthz"
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, components: Dict = Depends(get_components)):
    """
    Query documents and generate answers
    
    This endpoint:
    1. Retrieves relevant documents using the specified method
    2. Generates an answer using the LLM
    3. Returns the answer with citations
    """
    try:
        logger.info(f"Query received: {request.question[:50]}...")
        
        # Retrieve relevant documents
        retrieval_results = components["retriever"].retrieve(
            query=request.question,
            mode=request.retrieval_mode,
            top_k=request.top_k
        )
        
        if not retrieval_results:
            return QueryResponse(
                answer="I don't have enough information to answer this question. No relevant documents were found.",
                citations=[],
                metadata={
                    "retrieval_results": 0,
                    "safe": True,
                    "grounded": True,
                    "confidence": 1.0,
                    "reason": "no_results"
                },
                retrieval_mode=request.retrieval_mode
            )
        
        # Format context for LLM
        context = components["retriever"].format_results_for_llm(retrieval_results)
        
        # Generate answer
        llm_response = components["llm_client"].generate_rag_response(
            question=request.question,
            context=context,
            apply_guardrails=request.apply_guardrails
        )
        
        # Get citations if requested
        citations = []
        if request.include_citations:
            citation_data = components["retriever"].get_citations(retrieval_results)
            citations = [Citation(**cite) for cite in citation_data]
        
        # Prepare metadata
        metadata = {
            "retrieval_results": len(retrieval_results),
            "safe": llm_response.get("safe", True),
            "grounded": llm_response.get("grounded", True),
            "confidence": llm_response.get("confidence", 0.0),
            "model": llm_response.get("model"),
            "temperature": llm_response.get("temperature")
        }
        
        # Add any additional metadata from LLM response
        for key in ["safety_reason", "error", "reason"]:
            if key in llm_response:
                metadata[key] = llm_response[key]
        
        logger.info(f"Query processed: {len(citations)} citations, confidence={metadata.get('confidence', 0):.2f}")
        
        return QueryResponse(
            answer=llm_response["answer"],
            citations=citations,
            metadata=metadata,
            retrieval_mode=request.retrieval_mode
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest, 
                          background_tasks: BackgroundTasks,
                          components: Dict = Depends(get_components)):
    """
    Ingest documents from Google Drive
    
    This endpoint:
    1. Downloads PDFs from Google Drive
    2. Extracts text and creates chunks
    3. Generates embeddings
    4. Indexes documents in Elasticsearch
    """
    try:
        logger.info(f"Ingestion started: folder_id={request.folder_id}, max_files={request.max_files}")
        
        # Create/recreate index if requested
        if request.force_recreate_index:
            logger.info("Recreating Elasticsearch index...")
            components["es_client"].create_index(force_recreate=True)
        
        # Ingest documents
        document_chunks = components["ingestion_pipeline"].ingest_folder(
            folder_id=request.folder_id,
            max_files=request.max_files
        )
        
        if not document_chunks:
            return IngestionResponse(
                status="completed",
                message="No documents found or processed",
                files_processed=0,
                chunks_created=0
            )
        
        # Generate embeddings and index documents
        logger.info(f"Generating embeddings for {len(document_chunks)} chunks...")
        
        # Extract texts and chunk IDs
        texts = [chunk["text"] for chunk in document_chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in document_chunks]
        
        # Generate embeddings in batches
        embeddings_data = components["embedder"].batch_generate_embeddings(
            texts=texts,
            chunk_ids=chunk_ids,
            batch_size=32
        )
        
        # Combine document chunks with embeddings
        final_documents = []
        for i, chunk in enumerate(document_chunks):
            if i < len(embeddings_data):
                # Merge chunk data with embedding data
                final_doc = {**chunk, **embeddings_data[i]}
            else:
                # Fallback if embeddings failed
                final_doc = chunk
                final_doc["dense_vector"] = [0.0] * 384
                final_doc["text_expansion"] = {}
            
            final_documents.append(final_doc)
        
        # Index documents in Elasticsearch
        logger.info(f"Indexing {len(final_documents)} documents...")
        success = components["es_client"].bulk_index_documents(final_documents)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index documents in Elasticsearch")
        
        # Count unique files
        unique_files = len(set(doc["filename"] for doc in document_chunks))
        
        logger.info(f"Ingestion completed: {unique_files} files, {len(document_chunks)} chunks")
        
        return IngestionResponse(
            status="completed",
            message=f"Successfully processed {unique_files} files and created {len(document_chunks)} chunks",
            files_processed=unique_files,
            chunks_created=len(document_chunks)
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/healthz", response_model=HealthResponse)
async def health_check(components: Dict = Depends(get_components)):
    try:
        health_data = {}
        overall_status = "healthy"
        
        # Check each component
        component_checks = [
            ("elasticsearch", components["es_client"]),
            ("embeddings", components["embedder"]),
            ("ingestion", components["ingestion_pipeline"]),
            ("retrieval", components["retriever"]),
            ("llm_client", components["llm_client"])
        ]
        
        for name, component in component_checks:
            try:
                health = component.health_check()
                health_data[name] = health
                
                # Update overall status
                if health.get("status") not in ["healthy", "ok"]:
                    if health.get("status") == "partial":
                        overall_status = "partial" if overall_status == "healthy" else overall_status
                    else:
                        overall_status = "unhealthy"
                        
            except Exception as e:
                health_data[name] = {"status": "error", "error": str(e)}
                overall_status = "unhealthy"
        
        # Add system info
        health_data["system"] = {
            "status": "ok",
            "api_version": os.getenv("API_VERSION", "1.0.0"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return HealthResponse(
            status=overall_status,
            components=health_data
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return HealthResponse(
            status="unhealthy",
            components={"error": str(e), "traceback": traceback.format_exc()}
        )


@app.get("/stats")
async def get_stats(components: Dict = Depends(get_components)):
    """Get system statistics"""
    try:
        doc_count = components["es_client"].get_document_count()
        
        return {
            "document_count": doc_count,
            "index_name": components["es_client"].index_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting RAG System API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )