"""
Cross-encoder reranker for improving precision at top-K.

Uses a cross-encoder model to rerank candidates from RRF, addressing
the precision issues identified in Week 1.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankedCandidate:
    """Represents a reranked candidate with cross-encoder score."""
    doc_id: str
    content: str
    original_score: float
    cross_encoder_score: float
    final_score: float
    rank: int
    features: Dict


class CrossEncoderReranker:
    """
    Reranks candidates using a cross-encoder model.
    
    Cross-encoders provide better precision than bi-encoders by jointly
    encoding query-document pairs, but are slower for large candidate sets.
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 alpha: float = 0.7):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for inference
            alpha: Weight for combining original and cross-encoder scores
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.alpha = alpha
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        try:
            self.model = CrossEncoder(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded cross-encoder '{model_name}' on {self.device}")
        except Exception as e:
            logger.error(f"Error loading cross-encoder: {e}")
            raise
    
    def _prepare_pairs(self, query: str, candidates: List[Dict]) -> List[Tuple[str, str]]:
        """
        Prepare query-document pairs for cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate documents with 'content' field
            
        Returns:
            List of (query, document) pairs
        """
        pairs = []
        for candidate in candidates:
            content = candidate.get('content', '')
            # Truncate content if too long (cross-encoders have token limits)
            if len(content) > 512:
                content = content[:512] + "..."
            pairs.append((query, content))
        return pairs
    
    def _compute_scores_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Compute cross-encoder scores for a batch of query-document pairs.
        
        Args:
            pairs: List of (query, document) pairs
            
        Returns:
            List of scores
        """
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.error(f"Error computing cross-encoder scores: {e}")
            # Return zeros as fallback
            return [0.0] * len(pairs)
    
    def rerank_candidates(self, 
                         query: str, 
                         candidates: List[Dict],
                         top_k: Optional[int] = None) -> List[RerankedCandidate]:
        """
        Rerank candidates using cross-encoder scores.
        
        Args:
            query: Search query
            candidates: List of candidate documents
            top_k: Number of top candidates to return (None for all)
            
        Returns:
            List of reranked candidates
        """
        if not candidates:
            return []
        
        try:
            # Prepare query-document pairs
            pairs = self._prepare_pairs(query, candidates)
            
            # Compute cross-encoder scores
            cross_encoder_scores = self._compute_scores_batch(pairs)
            
            # Create reranked candidates
            reranked = []
            for i, (candidate, ce_score) in enumerate(zip(candidates, cross_encoder_scores)):
                # Get original score (assume it's in 'rrf_score' or 'score' field)
                original_score = candidate.get('rrf_score', candidate.get('score', 0.0))
                
                # Combine scores
                final_score = self.alpha * ce_score + (1 - self.alpha) * original_score
                
                reranked_candidate = RerankedCandidate(
                    doc_id=candidate.get('doc_id', f"doc_{i}"),
                    content=candidate.get('content', ''),
                    original_score=original_score,
                    cross_encoder_score=ce_score,
                    final_score=final_score,
                    rank=0,  # Will be set after sorting
                    features=candidate.get('features', {})
                )
                reranked.append(reranked_candidate)
            
            # Sort by final score (descending)
            reranked.sort(key=lambda x: x.final_score, reverse=True)
            
            # Update ranks
            for rank, candidate in enumerate(reranked):
                candidate.rank = rank
            
            # Return top-k if specified
            if top_k is not None:
                reranked = reranked[:top_k]
            
            logger.info(f"Reranked {len(candidates)} candidates, returning top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.error(f"Error reranking candidates: {e}")
            # Return original candidates as fallback
            fallback = []
            for i, candidate in enumerate(candidates):
                fallback.append(RerankedCandidate(
                    doc_id=candidate.get('doc_id', f"doc_{i}"),
                    content=candidate.get('content', ''),
                    original_score=candidate.get('rrf_score', candidate.get('score', 0.0)),
                    cross_encoder_score=0.0,
                    final_score=candidate.get('rrf_score', candidate.get('score', 0.0)),
                    rank=i,
                    features=candidate.get('features', {})
                ))
            return fallback
    
    def rerank_with_mmr(self, 
                       query: str, 
                       candidates: List[Dict],
                       lambda_mmr: float = 0.5,
                       top_k: int = 10) -> List[RerankedCandidate]:
        """
        Rerank candidates using cross-encoder + MMR for diversity.
        
        Args:
            query: Search query
            candidates: List of candidate documents
            lambda_mmr: MMR parameter (0.0 = pure relevance, 1.0 = pure diversity)
            top_k: Number of candidates to return
            
        Returns:
            List of reranked candidates with MMR
        """
        if not candidates:
            return []
        
        # First, get cross-encoder scores
        reranked = self.rerank_candidates(query, candidates, top_k=None)
        
        # Apply MMR selection
        selected = []
        remaining = reranked.copy()
        
        # Select first candidate (highest relevance)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining candidates using MMR
        while len(selected) < top_k and remaining:
            best_candidate = None
            best_mmr_score = -float('inf')
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Relevance score (cross-encoder score)
                relevance = candidate.cross_encoder_score
                
                # Diversity score (max similarity to already selected)
                max_similarity = 0.0
                for selected_candidate in selected:
                    # Simple similarity based on content overlap
                    similarity = self._compute_content_similarity(
                        candidate.content, selected_candidate.content
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.pop(best_idx)
        
        # Update ranks
        for rank, candidate in enumerate(selected):
            candidate.rank = rank
        
        logger.info(f"Selected {len(selected)} candidates using MMR (λ={lambda_mmr})")
        return selected
    
    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """
        Compute simple content similarity for MMR.
        
        Args:
            content1: First document content
            content2: Second document content
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize reranker
    reranker = CrossEncoderReranker()
    
    # Example candidates
    query = "Who won the FA Cup in 2020?"
    candidates = [
        {
            "doc_id": "doc1",
            "content": "Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
            "rrf_score": 0.8,
            "features": {"length": 100}
        },
        {
            "doc_id": "doc2", 
            "content": "The FA Cup is an annual football competition in England.",
            "rrf_score": 0.6,
            "features": {"length": 50}
        },
        {
            "doc_id": "doc3",
            "content": "Chelsea reached the FA Cup final in 2020 but lost to Arsenal.",
            "rrf_score": 0.7,
            "features": {"length": 80}
        }
    ]
    
    # Rerank candidates
    reranked = reranker.rerank_candidates(query, candidates, top_k=2)
    
    print(f"\nReranked candidates for: '{query}'")
    print("=" * 80)
    
    for candidate in reranked:
        print(f"\nRank {candidate.rank + 1}: {candidate.doc_id}")
        print(f"  Original Score: {candidate.original_score:.4f}")
        print(f"  Cross-Encoder Score: {candidate.cross_encoder_score:.4f}")
        print(f"  Final Score: {candidate.final_score:.4f}")
        print(f"  Content: {candidate.content}")
    
    # Test MMR reranking
    print("\n" + "=" * 80)
    print("MMR Reranking (λ=0.5)")
    print("=" * 80)
    
    mmr_reranked = reranker.rerank_with_mmr(query, candidates, lambda_mmr=0.5, top_k=2)
    
    for candidate in mmr_reranked:
        print(f"\nRank {candidate.rank + 1}: {candidate.doc_id}")
        print(f"  Final Score: {candidate.final_score:.4f}")
        print(f"  Content: {candidate.content}")
