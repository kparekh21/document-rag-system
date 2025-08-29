"""
Hybrid Retrieval System for RAG

Combines BM25, dense vector search, and ELSER with Reciprocal Rank Fusion.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import math
from collections import defaultdict
from dotenv import load_dotenv

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from indexing.elastic_client import ElasticsearchClient
from indexing.embeddings import EmbeddingGenerator

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Implements Reciprocal Rank Fusion (RRF) for combining multiple ranked lists"""
    
    def __init__(self, k: int = 60):
        """Initialize RRF with parameter k (typically 60)"""
        self.k = k
    
    def fuse_rankings(self, rankings: List[List[Dict[str, Any]]], weights: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Fuse multiple ranked lists using RRF"""
        if not rankings:
            return []
        
        # Default equal weights
        if weights is None:
            weights = [1.0] * len(rankings)
        
        if len(weights) != len(rankings):
            logger.warning("Weights length doesn't match rankings length, using equal weights")
            weights = [1.0] * len(rankings)
        
        # Collect all unique documents
        doc_scores = defaultdict(float)
        doc_data = {}
        
        for ranking_idx, ranking in enumerate(rankings):
            weight = weights[ranking_idx]
            
            for rank, result in enumerate(ranking):
                # Use document ID as key
                doc_id = result.get('id') or result.get('chunk_id') or f"doc_{rank}"
                
                # Calculate RRF score: weight / (k + rank)
                rrf_score = weight / (self.k + rank + 1)  # +1 because rank is 0-indexed
                
                doc_scores[doc_id] += rrf_score
                
                # Store document data (from first occurrence)
                if doc_id not in doc_data:
                    doc_data[doc_id] = result
        
        # Sort by RRF score (descending)
        fused_results = []
        for doc_id, rrf_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_data[doc_id].copy()
            result['rrf_score'] = rrf_score
            result['fusion_id'] = doc_id
            fused_results.append(result)
        
        logger.info(f"RRF fused {len(rankings)} rankings into {len(fused_results)} results")
        return fused_results


class HybridRetriever:
    """Hybrid retrieval system combining multiple search methods"""
    
    def __init__(self):
        """Initialize hybrid retriever"""
        # Initialize components
        self.es_client = ElasticsearchClient()
        self.embedder = EmbeddingGenerator()
        self.rrf = ReciprocalRankFusion()
        
        # Configuration
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", 5))
        self.retrieval_mode = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid")
        
        logger.info("Hybrid Retriever initialized")
    
    def retrieve_bm25(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """BM25 keyword-based retrieval"""
        top_k = top_k or self.default_top_k
        
        try:
            results = self.es_client.search_bm25(query, top_k)
            
            # Add retrieval method info
            for result in results:
                result['retrieval_method'] = 'bm25'
                result['query'] = query
            
            logger.info(f"BM25 retrieval: {len(results)} results for '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []
    
    def retrieve_dense_vector(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Dense vector similarity retrieval"""
        top_k = top_k or self.default_top_k
        
        try:
            # Generate query embedding
            query_emb = self.embedder.get_query_embedding(query)
            query_vector = query_emb['dense_vector']
            
            # Search with dense vectors
            results = self.es_client.search_dense_vector(query_vector, top_k)
            
            # Add retrieval method info
            for result in results:
                result['retrieval_method'] = 'dense_vector'
                result['query'] = query
                result['embedding_model'] = query_emb['embedding_model']
            
            logger.info(f"Dense vector retrieval: {len(results)} results for '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Dense vector retrieval failed: {e}")
            return []
    
    def retrieve_elser(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """ELSER sparse vector retrieval"""
        top_k = top_k or self.default_top_k
        
        try:
            results = self.es_client.search_elser(query, top_k)
            
            # Add retrieval method info
            for result in results:
                result['retrieval_method'] = 'elser'
                result['query'] = query
            
            logger.info(f"ELSER retrieval: {len(results)} results for '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"ELSER retrieval failed: {e}")
            return []
    
    def retrieve_hybrid(self, query: str, top_k: int = None, weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Hybrid retrieval using RRF to combine multiple methods"""
        top_k = top_k or self.default_top_k
        
        # Default weights for each method
        default_weights = {
            'bm25': 0.3,
            'dense_vector': 0.4,
            'elser': 0.3
        }
        weights = weights or default_weights
        
        try:
            # Get results from each method (retrieve more for better fusion)
            fusion_k = max(top_k * 2, 10)  # Retrieve more for fusion
            
            rankings = []
            method_weights = []
            
            # BM25 retrieval
            if weights.get('bm25', 0) > 0:
                bm25_results = self.retrieve_bm25(query, fusion_k)
                if bm25_results:
                    rankings.append(bm25_results)
                    method_weights.append(weights['bm25'])
            
            # Dense vector retrieval
            if weights.get('dense_vector', 0) > 0:
                dense_results = self.retrieve_dense_vector(query, fusion_k)
                if dense_results:
                    rankings.append(dense_results)
                    method_weights.append(weights['dense_vector'])
            
            # ELSER retrieval
            if weights.get('elser', 0) > 0:
                elser_results = self.retrieve_elser(query, fusion_k)
                if elser_results:
                    rankings.append(elser_results)
                    method_weights.append(weights['elser'])
            
            if not rankings:
                logger.warning("No retrieval methods returned results")
                return []
            
            # Apply RRF fusion
            fused_results = self.rrf.fuse_rankings(rankings, method_weights)
            
            # Take top-k results
            final_results = fused_results[:top_k]
            
            # Add hybrid retrieval metadata
            for result in final_results:
                result['retrieval_method'] = 'hybrid'
                result['fusion_methods'] = len(rankings)
                result['weights_used'] = weights
            
            logger.info(f"Hybrid retrieval: {len(final_results)} results from {len(rankings)} methods")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    def retrieve(self, query: str, mode: str = None, top_k: int = None, **kwargs) -> List[Dict[str, Any]]:
        """Main retrieval interface"""
        mode = mode or self.retrieval_mode
        top_k = top_k or self.default_top_k
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        query = query.strip()
        
        # Route to appropriate retrieval method
        if mode == 'bm25':
            return self.retrieve_bm25(query, top_k)
        elif mode == 'dense' or mode == 'dense_vector':
            return self.retrieve_dense_vector(query, top_k)
        elif mode == 'elser':
            return self.retrieve_elser(query, top_k)
        elif mode == 'hybrid':
            weights = kwargs.get('weights')
            return self.retrieve_hybrid(query, top_k, weights)
        else:
            logger.warning(f"Unknown retrieval mode: {mode}, falling back to hybrid")
            return self.retrieve_hybrid(query, top_k)
    
    def format_results_for_llm(self, results: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """Format retrieval results for LLM context"""
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source = result.get('source', {})
            text = source.get('text', 'No text available')
            filename = source.get('filename', 'Unknown file')
            
            # Format context chunk
            context_chunk = f"[Document {i}]"
            
            if include_metadata:
                context_chunk += f"\nSource: {filename}"
                
                if result.get('rrf_score'):
                    context_chunk += f"\nRelevance Score: {result['rrf_score']:.4f}"
                elif result.get('score'):
                    context_chunk += f"\nRelevance Score: {result['score']:.4f}"
            
            context_chunk += f"\nContent: {text}\n"
            context_parts.append(context_chunk)
        
        return "\n".join(context_parts)
    
    def get_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from results"""
        citations = []
        
        for result in results:
            source = result.get('source', {})
            
            citation = {
                'title': source.get('filename', 'Unknown Document'),
                'url': source.get('drive_url', '#'),
                'snippet': source.get('text_snippet', source.get('text', '')[:150] + '...')
            }
            
            # Add retrieval metadata
            if result.get('rrf_score'):
                citation['relevance_score'] = round(result['rrf_score'], 4)
            elif result.get('score'):
                citation['relevance_score'] = round(result['score'], 4)
                
            citation['retrieval_method'] = result.get('retrieval_method', 'unknown')
            
            citations.append(citation)
        
        return citations
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for retrieval system"""
        try:
            # Check components
            es_health = self.es_client.health_check()
            embedder_health = self.embedder.health_check()
            
            # Test basic retrieval
            test_query = "test query"
            test_results = self.retrieve_bm25(test_query, top_k=1)
            retrieval_ok = isinstance(test_results, list)
            
            return {
                "status": "healthy" if es_health.get('status') == 'healthy' and embedder_health.get('status') in ['healthy', 'partial'] else "partial",
                "elasticsearch": es_health,
                "embeddings": embedder_health,
                "retrieval_test": {
                    "status": "ok" if retrieval_ok else "error",
                    "test_results": len(test_results) if retrieval_ok else 0
                },
                "configuration": {
                    "default_top_k": self.default_top_k,
                    "default_mode": self.retrieval_mode,
                    "rrf_k": self.rrf.k
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def get_hybrid_retriever() -> HybridRetriever:
    """Get configured hybrid retriever instance"""
    return HybridRetriever()


if __name__ == "__main__":
    print("Testing Hybrid Retriever...")
    
    try:
        retriever = get_hybrid_retriever()
        
        # Health check
        health = retriever.health_check()
        print(f"Health Status: {health}")
        
        # Test RRF
        print("\nTesting Reciprocal Rank Fusion...")
        rrf = ReciprocalRankFusion()
        
        # Mock rankings for testing
        ranking1 = [
            {"id": "doc1", "score": 0.9, "text": "First ranking doc1"},
            {"id": "doc2", "score": 0.8, "text": "First ranking doc2"},
            {"id": "doc3", "score": 0.7, "text": "First ranking doc3"}
        ]
        
        ranking2 = [
            {"id": "doc2", "score": 0.95, "text": "Second ranking doc2"},
            {"id": "doc1", "score": 0.85, "text": "Second ranking doc1"},
            {"id": "doc4", "score": 0.75, "text": "Second ranking doc4"}
        ]
        
        fused = rrf.fuse_rankings([ranking1, ranking2], weights=[0.6, 0.4])
        print(f"RRF Results: {len(fused)} fused documents")
        for i, result in enumerate(fused[:3]):
            print(f"  {i+1}. {result['id']}: RRF={result['rrf_score']:.4f}")
        
        print("\nHybrid Retriever test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()