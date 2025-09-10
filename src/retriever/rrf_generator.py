"""
Reciprocal Rank Fusion (RRF) candidate generator.

Combines BM25 and vector search results using RRF to create a diverse
top-K candidate set for the RL environment.
"""

import json
import logging
import pickle
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Import existing retrieval modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.query_bm25 import load_index, load_docs, tokenize
from scripts.query_vector import load_vector_index, search_vectors

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Wrapper for BM25 retrieval."""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.bm25, self.doc_ids = load_index(self.index_path)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25."""
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                doc_id = self.doc_ids[idx]
                results.append((doc_id, float(scores[idx])))
        
        return results


class VectorRetriever:
    """Wrapper for vector retrieval."""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index, self.doc_ids, self.embeddings = load_vector_index(self.index_path)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using vector similarity."""
        return search_vectors(query, self.index, self.doc_ids, self.embeddings, k)


@dataclass
class CandidateDocument:
    """Represents a candidate document with features."""
    doc_id: str
    content: str
    bm25_score: float
    vector_score: float
    rrf_score: float
    rank_bm25: int
    rank_vector: int
    rank_rrf: int
    features: Dict[str, Any]


class RRFCandidateGenerator:
    """
    Generates candidate documents using Reciprocal Rank Fusion of BM25 and vector search.
    
    RRF combines rankings from multiple retrieval methods to create a diverse
    and high-quality candidate set for the RL environment.
    """
    
    def __init__(self, 
                 bm25_index_path: str = "index/coral_bm25",
                 vector_index_path: str = "index/coral_faiss",
                 corpus_path: str = "data/coral/docs.jsonl",
                 k: int = 100,
                 rrf_k: int = 60):
        """
        Initialize the RRF candidate generator.
        
        Args:
            bm25_index_path: Path to BM25 index directory
            vector_index_path: Path to vector index directory  
            corpus_path: Path to document corpus
            k: Number of candidates to return
            rrf_k: RRF parameter (higher = more weight to top ranks)
        """
        self.k = k
        self.rrf_k = rrf_k
        
        # Initialize retrievers
        self.bm25_retriever = BM25Retriever(bm25_index_path)
        self.vector_retriever = VectorRetriever(vector_index_path)
        
        # Load corpus for document content
        self.corpus = self._load_corpus(corpus_path)
        
        logger.info(f"Initialized RRF generator with k={k}, rrf_k={rrf_k}")
    
    def _load_corpus(self, corpus_path: str) -> Dict[str, str]:
        """Load document corpus."""
        corpus = {}
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    corpus[doc['id']] = doc['text']
            logger.info(f"Loaded {len(corpus)} documents from corpus")
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            raise
        return corpus
    
    def _compute_rrf_score(self, rank: int) -> float:
        """Compute RRF score for a given rank."""
        return 1.0 / (self.rrf_k + rank)
    
    def _fuse_rankings(self, 
                      bm25_results: List[Tuple[str, float]], 
                      vector_results: List[Tuple[str, float]]) -> List[CandidateDocument]:
        """
        Fuse BM25 and vector rankings using RRF.
        
        Args:
            bm25_results: List of (doc_id, score) tuples from BM25
            vector_results: List of (doc_id, score) tuples from vector search
            
        Returns:
            List of CandidateDocument objects ranked by RRF score
        """
        # Create score maps
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        
        # Create rank maps
        bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results)}
        vector_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(vector_results)}
        
        # Get all unique document IDs
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        # Compute RRF scores
        candidates = []
        for doc_id in all_doc_ids:
            # Get scores (0 if not found)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            
            # Get ranks (use a large number if not found)
            bm25_rank = bm25_ranks.get(doc_id, len(bm25_results))
            vector_rank = vector_ranks.get(doc_id, len(vector_results))
            
            # Compute RRF score
            rrf_score = self._compute_rrf_score(bm25_rank) + self._compute_rrf_score(vector_rank)
            
            # Get document content
            content = self.corpus.get(doc_id, "")
            
            # Create features
            features = {
                "content_length": len(content),
                "has_numbers": any(c.isdigit() for c in content),
                "has_dates": any(word in content.lower() for word in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]),
                "word_count": len(content.split()),
                "bm25_rank": bm25_rank,
                "vector_rank": vector_rank,
                "rank_diff": abs(bm25_rank - vector_rank)
            }
            
            candidate = CandidateDocument(
                doc_id=doc_id,
                content=content,
                bm25_score=bm25_score,
                vector_score=vector_score,
                rrf_score=rrf_score,
                rank_bm25=bm25_rank,
                rank_vector=vector_rank,
                rank_rrf=0,  # Will be set after sorting
                features=features
            )
            candidates.append(candidate)
        
        # Sort by RRF score (descending)
        candidates.sort(key=lambda x: x.rrf_score, reverse=True)
        
        # Update RRF ranks
        for rank, candidate in enumerate(candidates):
            candidate.rank_rrf = rank
        
        return candidates
    
    def generate_candidates(self, query: str) -> List[CandidateDocument]:
        """
        Generate top-K candidates for a query using RRF.
        
        Args:
            query: Search query
            
        Returns:
            List of top-K CandidateDocument objects
        """
        try:
            # Get BM25 results
            bm25_results = self.bm25_retriever.search(query, k=self.k * 2)  # Get more for fusion
            
            # Get vector results  
            vector_results = self.vector_retriever.search(query, k=self.k * 2)
            
            # Fuse rankings
            candidates = self._fuse_rankings(bm25_results, vector_results)
            
            # Return top-K
            top_candidates = candidates[:self.k]
            
            logger.info(f"Generated {len(top_candidates)} candidates for query: '{query[:50]}...'")
            return top_candidates
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            return []
    
    def get_candidate_features(self, candidates: List[CandidateDocument]) -> np.ndarray:
        """
        Extract feature matrix from candidates for RL environment.
        
        Args:
            candidates: List of CandidateDocument objects
            
        Returns:
            Feature matrix of shape (n_candidates, n_features)
        """
        if not candidates:
            return np.array([])
        
        features = []
        for candidate in candidates:
            feature_vector = [
                candidate.bm25_score,
                candidate.vector_score, 
                candidate.rrf_score,
                candidate.features["content_length"],
                candidate.features["word_count"],
                float(candidate.features["has_numbers"]),
                float(candidate.features["has_dates"]),
                candidate.features["rank_diff"]
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_candidate_ids(self, candidates: List[CandidateDocument]) -> List[str]:
        """Get list of candidate document IDs."""
        return [candidate.doc_id for candidate in candidates]
    
    def get_candidate_contents(self, candidates: List[CandidateDocument]) -> List[str]:
        """Get list of candidate document contents."""
        return [candidate.content for candidate in candidates]


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = RRFCandidateGenerator(k=10)
    
    # Test query
    query = "Who won the FA Cup in 2020?"
    
    # Generate candidates
    candidates = generator.generate_candidates(query)
    
    print(f"\nTop {len(candidates)} candidates for: '{query}'")
    print("=" * 80)
    
    for i, candidate in enumerate(candidates):
        print(f"\n{i+1}. Doc ID: {candidate.doc_id}")
        print(f"   RRF Score: {candidate.rrf_score:.4f}")
        print(f"   BM25 Score: {candidate.bm25_score:.4f} (rank {candidate.rank_bm25})")
        print(f"   Vector Score: {candidate.vector_score:.4f} (rank {candidate.rank_vector})")
        print(f"   Content: {candidate.content[:100]}...")
        print(f"   Features: {candidate.features}")
    
    # Test feature extraction
    features = generator.get_candidate_features(candidates)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature matrix:\n{features}")
