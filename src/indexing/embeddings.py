"""
Embeddings Component for RAG System

Handles dense vector embeddings and ELSER sparse embeddings generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles embedding generation for RAG system:
    - Dense embeddings using sentence-transformers
    - ELSER sparse embeddings via Elasticsearch ML
    """
    
    def __init__(self):
        """Initialize embedding models"""
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", 384))
        
        # Initialize dense embedding model
        self._load_dense_model()
        
        # Elasticsearch client for ELSER
        self.es_client = None
        self._setup_elasticsearch()
        
    def _load_dense_model(self):
        """Load sentence-transformers model"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.dense_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Dense embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dense model: {e}")
            raise
    
    def _setup_elasticsearch(self):
        """Setup Elasticsearch connection for ELSER"""
        try:
            host = os.getenv("ELASTIC_HOST", "localhost")
            port = int(os.getenv("ELASTIC_PORT", 9200))
            self.es_client = Elasticsearch([f"http://{host}:{port}"])
            
            # Test connection
            self.es_client.info()
            logger.info("Elasticsearch connection established for ELSER")
        except Exception as e:
            logger.warning(f"Elasticsearch connection failed: {e}")
            logger.warning("ELSER embeddings will not be available")
    
    def generate_dense_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate dense embeddings using sentence-transformers"""
        try:
            is_single = isinstance(texts, str)
            if is_single:
                texts = [texts]
            
            # Generate embeddings
            embeddings = self.dense_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10
            )
            
            # Convert to list format
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Generated dense embeddings for {len(texts)} texts")
            
            return embeddings_list[0] if is_single else embeddings_list
            
        except Exception as e:
            logger.error(f"Dense embedding generation failed: {e}")
            if isinstance(texts, str):
                return [0.0] * self.embedding_dimension
            else:
                return [[0.0] * self.embedding_dimension] * len(texts)
    
    def generate_elser_embeddings(self, texts: Union[str, List[str]], model_id: str = ".elser_model_2_linux-x86_64") -> Union[Dict, List[Dict]]:
        """Generate ELSER sparse embeddings using Elasticsearch ML"""
        if not self.es_client:
            logger.error("Elasticsearch client not available for ELSER")
            return {} if isinstance(texts, str) else [{}] * len(texts)
        
        try:
            is_single = isinstance(texts, str)
            if is_single:
                texts = [texts]
            
            embeddings = []
            for text in texts:
                try:
                    # Use Elasticsearch inference API for ELSER
                    response = self.es_client.ml.infer_trained_model(
                        model_id=model_id,
                        body={
                            "docs": [{"text_field": text}]
                        }
                    )
                    
                    # Extract sparse vector
                    if response['inference_results']:
                        sparse_embedding = response['inference_results'][0]['predicted_value']
                        embeddings.append(sparse_embedding)
                    else:
                        embeddings.append({})
                        
                except Exception as e:
                    logger.warning(f"ELSER embedding failed for one text: {e}")
                    embeddings.append({})
            
            logger.info(f"Generated ELSER embeddings for {len(texts)} texts")
            
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            logger.error(f"ELSER embedding generation failed: {e}")
            return {} if isinstance(texts, str) else [{}] * len(texts)
    
    def prepare_document_embeddings(self, text: str, chunk_id: str) -> Dict[str, Any]:
        """Prepare all embeddings for a document chunk"""
        try:
            # Generate dense embeddings
            dense_embedding = self.generate_dense_embeddings(text)
            
            # Generate ELSER embeddings (if available)
            elser_embedding = self.generate_elser_embeddings(text)
            
            # Prepare embedding document
            embedding_doc = {
                "text": text,
                "chunk_id": chunk_id,
                "dense_vector": dense_embedding,
                "text_expansion": elser_embedding,
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": len(dense_embedding) if dense_embedding else 0
            }
            
            logger.info(f"Prepared embeddings for chunk: {chunk_id}")
            return embedding_doc
            
        except Exception as e:
            logger.error(f"Failed to prepare embeddings for chunk {chunk_id}: {e}")
            return {
                "text": text,
                "chunk_id": chunk_id,
                "dense_vector": [0.0] * self.embedding_dimension,
                "text_expansion": {},
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": 0
            }
    
    def batch_generate_embeddings(self, texts: List[str], chunk_ids: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate embeddings for multiple texts in batches"""
        try:
            if len(texts) != len(chunk_ids):
                raise ValueError("Number of texts and chunk_ids must match")
            
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                current_batch = (i // batch_size) + 1
                
                logger.info(f"Processing batch {current_batch}/{total_batches}")
                
                # Generate dense embeddings for batch
                dense_embeddings = self.generate_dense_embeddings(batch_texts)
                
                # Generate ELSER embeddings for batch
                elser_embeddings = self.generate_elser_embeddings(batch_texts)
                
                # Combine results
                for j, (text, chunk_id) in enumerate(zip(batch_texts, batch_ids)):
                    embedding_doc = {
                        "text": text,
                        "chunk_id": chunk_id,
                        "dense_vector": dense_embeddings[j] if isinstance(dense_embeddings[j], list) else dense_embeddings,
                        "text_expansion": elser_embeddings[j] if isinstance(elser_embeddings, list) else elser_embeddings,
                        "embedding_model": self.embedding_model_name,
                        "embedding_dimension": len(dense_embeddings[j]) if isinstance(dense_embeddings[j], list) else len(dense_embeddings)
                    }
                    all_embeddings.append(embedding_doc)
            
            logger.info(f"Generated embeddings for {len(all_embeddings)} documents")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return []
    
    def get_query_embedding(self, query: str) -> Dict[str, Any]:
        """Generate embeddings for search query"""
        try:
            # Generate dense embedding
            dense_embedding = self.generate_dense_embeddings(query)
            
            return {
                "query": query,
                "dense_vector": dense_embedding,
                "embedding_model": self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            return {
                "query": query,
                "dense_vector": [0.0] * self.embedding_dimension,
                "embedding_model": self.embedding_model_name
            }
    
    def similarity_search(self, query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Compute similarity between query and candidate embeddings"""
        try:
            query_vector = np.array(query_embedding)
            candidate_matrix = np.array(candidate_embeddings)
            
            # Compute cosine similarities
            similarities = np.dot(candidate_matrix, query_vector) / (
                np.linalg.norm(candidate_matrix, axis=1) * np.linalg.norm(query_vector)
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "index": int(idx),
                    "similarity": float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for embedding components"""
        try:
            # Test dense model
            test_embedding = self.generate_dense_embeddings("test")
            dense_ok = len(test_embedding) == self.embedding_dimension
            
            # Test ELSER (if available)
            elser_ok = self.es_client is not None
            if elser_ok:
                try:
                    self.es_client.cluster.health()
                except:
                    elser_ok = False
            
            return {
                "status": "healthy" if dense_ok else "partial",
                "dense_embeddings": {
                    "status": "ok" if dense_ok else "error",
                    "model": self.embedding_model_name,
                    "dimension": self.embedding_dimension
                },
                "elser_embeddings": {
                    "status": "ok" if elser_ok else "unavailable",
                    "elasticsearch_connected": self.es_client is not None
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def get_embedding_generator() -> EmbeddingGenerator:
    """Get configured embedding generator instance"""
    return EmbeddingGenerator()


if __name__ == "__main__":
    print("Testing Embedding Generator...")
    
    embedder = get_embedding_generator()
    
    # Health check
    health = embedder.health_check()
    print(f"Health Status: {health}")
    
    # Test dense embeddings
    test_texts = [
        "This is a test document about machine learning.",
        "Artificial intelligence is transforming technology.",
        "Natural language processing enables text understanding."
    ]
    
    dense_embeddings = embedder.generate_dense_embeddings(test_texts)
    print(f"Dense Embeddings Shape: {len(dense_embeddings)} x {len(dense_embeddings[0])}")
    
    # Test single embedding
    single_embedding = embedder.generate_dense_embeddings("Single test text")
    print(f"Single Embedding Shape: {len(single_embedding)}")
    
    # Test query embedding
    query_emb = embedder.get_query_embedding("machine learning AI")
    print(f"Query Embedding: {len(query_emb['dense_vector'])}")
    
    # Test batch processing
    chunk_ids = [f"chunk_{i}" for i in range(len(test_texts))]
    batch_results = embedder.batch_generate_embeddings(test_texts, chunk_ids)
    print(f"Batch Results: {len(batch_results)} documents processed")
    
    print("Embedding Generator test completed!")