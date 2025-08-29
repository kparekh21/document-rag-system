"""
Elasticsearch Client for RAG System

Handles connections, index management, and document operations.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Elasticsearch client for RAG system with support for:
    - ELSER sparse embeddings
    - Dense vector embeddings  
    - BM25 text search
    - Hybrid retrieval
    """
    
    def __init__(self):
        """Initialize Elasticsearch client"""
        self.host = os.getenv("ELASTIC_HOST", "localhost")
        self.port = int(os.getenv("ELASTIC_PORT", 9200))
        self.index_name = os.getenv("ELASTIC_INDEX_NAME", "documents")
        
        # Initialize client with version compatibility
        self.client = Elasticsearch(
            [f"http://{self.host}:{self.port}"],
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Test connection
        self._test_connection()
        
    def _test_connection(self) -> bool:
        """Test Elasticsearch connection"""
        try:
            info = self.client.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
            return True
        except ConnectionError as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def get_index_mappings(self) -> Dict[str, Any]:
        """Define index mappings for RAG system"""
        return {
            "mappings": {
                "properties": {
                    # Document content
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    
                    # ELSER sparse embeddings
                    "text_expansion": {
                        "type": "sparse_vector"
                    },
                    
                    # Dense embeddings (sentence-transformers)
                    "dense_vector": {
                        "type": "dense_vector",
                        "dims": 384,  # all-MiniLM-L6-v2 dimension
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # Metadata fields
                    "filename": {
                        "type": "keyword"
                    },
                    "drive_url": {
                        "type": "keyword"
                    },
                    "chunk_id": {
                        "type": "keyword"
                    },
                    "chunk_index": {
                        "type": "integer"
                    },
                    "file_type": {
                        "type": "keyword"
                    },
                    "created_at": {
                        "type": "date"
                    },
                    
                    # Text snippets for citations
                    "text_snippet": {
                        "type": "text",
                        "store": True
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "custom_text_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            }
        }
    
    def create_index(self, force_recreate: bool = False) -> bool:
        """Create index with proper mappings"""
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                if force_recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index already exists: {self.index_name}")
                    return True
            
            # Create index with mappings
            mappings = self.get_index_mappings()
            self.client.indices.create(index=self.index_name, body=mappings)
            
            logger.info(f"Created index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def index_document(self, doc: Dict[str, Any], doc_id: Optional[str] = None) -> bool:
        """Index a single document"""
        try:
            response = self.client.index(
                index=self.index_name,
                body=doc,
                id=doc_id,
                refresh='wait_for'
            )
            
            logger.debug(f"Indexed document: {response['_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return False
    
    def bulk_index_documents(self, docs: List[Dict[str, Any]]) -> bool:
        """Bulk index multiple documents for better performance"""
        try:
            from elasticsearch.helpers import bulk
            
            # Prepare documents for bulk indexing
            actions = []
            for i, doc in enumerate(docs):
                action = {
                    "_index": self.index_name,
                    "_source": doc,
                    "_id": doc.get("chunk_id", f"doc_{i}")
                }
                actions.append(action)
            
            # Perform bulk indexing
            success_count, failed_items = bulk(
                self.client,
                actions,
                refresh='wait_for',
                request_timeout=60
            )
            
            logger.info(f"Bulk indexed {success_count} documents")
            if failed_items:
                logger.warning(f"{len(failed_items)} documents failed to index")
            
            return len(failed_items) == 0
            
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return False
    
    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search"""
        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "text_snippet"],
                        "type": "best_fields"
                    }
                },
                "size": top_k,
                "_source": True
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "score": hit['_score'],
                    "source": hit['_source'],
                    "id": hit['_id']
                }
                results.append(result)
            
            logger.info(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def search_dense_vector(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense vector similarity search"""
        try:
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'dense_vector') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                },
                "size": top_k,
                "_source": True
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "score": hit['_score'],
                    "source": hit['_source'],
                    "id": hit['_id']
                }
                results.append(result)
            
            logger.info(f"Dense vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Dense vector search failed: {e}")
            return []
    
    def search_elser(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """ELSER sparse vector search"""
        try:
            search_body = {
                "query": {
                    "text_expansion": {
                        "text_expansion": {
                            "model_id": ".elser_model_2_linux-x86_64",
                            "model_text": query
                        }
                    }
                },
                "size": top_k,
                "_source": True
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "score": hit['_score'],
                    "source": hit['_source'],
                    "id": hit['_id']
                }
                results.append(result)
            
            logger.info(f"ELSER search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"ELSER search failed: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in index"""
        try:
            response = self.client.count(index=self.index_name)
            count = response['count']
            logger.info(f"Index contains {count} documents")
            return count
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Cluster health
            cluster_health = self.client.cluster.health()
            
            # Index info
            index_exists = self.client.indices.exists(index=self.index_name)
            doc_count = self.get_document_count() if index_exists else 0
            
            return {
                "status": "healthy",
                "cluster_status": cluster_health['status'],
                "cluster_name": cluster_health['cluster_name'],
                "index_exists": index_exists,
                "document_count": doc_count,
                "elasticsearch_version": self.client.info()['version']['number']
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def delete_index(self) -> bool:
        """Delete the index"""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted index: {self.index_name}")
                return True
            else:
                logger.info(f"Index does not exist: {self.index_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False


def get_elasticsearch_client() -> ElasticsearchClient:
    """Get configured Elasticsearch client instance"""
    return ElasticsearchClient()


if __name__ == "__main__":
    print("Testing Elasticsearch Client...")
    
    client = get_elasticsearch_client()
    
    # Health check
    health = client.health_check()
    print(f"Health Status: {health}")
    
    # Create index
    success = client.create_index(force_recreate=True)
    print(f"Index Creation: {'Success' if success else 'Failed'}")
    
    # Test document indexing
    test_doc = {
        "text": "This is a test document about machine learning and artificial intelligence.",
        "text_snippet": "This is a test document about machine learning...",
        "filename": "test.pdf",
        "drive_url": "https://drive.google.com/test",
        "chunk_id": "test_chunk_1",
        "chunk_index": 0,
        "file_type": "pdf"
    }
    
    success = client.index_document(test_doc, "test_doc_1")
    print(f"Document Indexing: {'Success' if success else 'Failed'}")
    
    # Test BM25 search
    results = client.search_bm25("machine learning", top_k=3)
    print(f"BM25 Search Results: {len(results)} found")
    
    print("Elasticsearch Client test completed!")