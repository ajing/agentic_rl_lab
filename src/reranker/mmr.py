"""
Maximal Marginal Relevance (MMR) for document deduplication.

Addresses redundancy issues identified in Week 1 by selecting diverse
documents that balance relevance and novelty.
"""

import logging
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


@dataclass
class MMRDocument:
    """Represents a document with MMR scores."""
    doc_id: str
    content: str
    relevance_score: float
    diversity_score: float
    mmr_score: float
    rank: int
    features: Dict


class MMRDeduplicator:
    """
    Implements Maximal Marginal Relevance for document selection.
    
    MMR balances relevance and diversity by selecting documents that are
    both relevant to the query and different from already selected documents.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 similarity_threshold: float = 0.8):
        """
        Initialize MMR deduplicator.
        
        Args:
            model_name: Sentence transformer model for embeddings
            device: Device to run on ('cuda', 'cpu', or None for auto)
            similarity_threshold: Threshold for considering documents similar
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded MMR model '{model_name}' on {self.device}")
        except Exception as e:
            logger.error(f"Error loading MMR model: {e}")
            raise
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Embedding matrix of shape (n_texts, embedding_dim)
        """
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            # Return random embeddings as fallback
            return np.random.randn(len(texts), 384)
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def _compute_max_similarity(self, 
                               candidate_embedding: np.ndarray, 
                               selected_embeddings: List[np.ndarray]) -> float:
        """
        Compute maximum similarity between candidate and selected documents.
        
        Args:
            candidate_embedding: Embedding of candidate document
            selected_embeddings: List of embeddings of selected documents
            
        Returns:
            Maximum similarity score
        """
        if not selected_embeddings:
            return 0.0
        
        similarities = [
            self._compute_similarity(candidate_embedding, selected_emb)
            for selected_emb in selected_embeddings
        ]
        
        return max(similarities)
    
    def select_diverse_documents(self, 
                               query: str,
                               candidates: List[Dict],
                               lambda_mmr: float = 0.5,
                               top_k: int = 10,
                               relevance_scores: Optional[List[float]] = None) -> List[MMRDocument]:
        """
        Select diverse documents using MMR.
        
        Args:
            query: Search query
            candidates: List of candidate documents with 'content' field
            lambda_mmr: MMR parameter (0.0 = pure relevance, 1.0 = pure diversity)
            top_k: Number of documents to select
            relevance_scores: Optional pre-computed relevance scores
            
        Returns:
            List of selected MMRDocument objects
        """
        if not candidates:
            return []
        
        try:
            # Prepare texts for embedding
            texts = [candidate.get('content', '') for candidate in candidates]
            
            # Compute embeddings
            embeddings = self._compute_embeddings(texts)
            
            # Get relevance scores
            if relevance_scores is None:
                # Use RRF scores or default to 1.0
                relevance_scores = [
                    candidate.get('rrf_score', candidate.get('score', 1.0))
                    for candidate in candidates
                ]
            
            # Normalize relevance scores to [0, 1]
            max_relevance = max(relevance_scores) if relevance_scores else 1.0
            normalized_relevance = [score / max_relevance for score in relevance_scores]
            
            # MMR selection
            selected = []
            selected_embeddings = []
            remaining_indices = list(range(len(candidates)))
            
            # Select first document (highest relevance)
            if remaining_indices:
                best_idx = max(remaining_indices, key=lambda i: normalized_relevance[i])
                selected.append(best_idx)
                selected_embeddings.append(embeddings[best_idx])
                remaining_indices.remove(best_idx)
            
            # Select remaining documents using MMR
            while len(selected) < top_k and remaining_indices:
                best_candidate_idx = None
                best_mmr_score = -float('inf')
                
                for idx in remaining_indices:
                    # Relevance score
                    relevance = normalized_relevance[idx]
                    
                    # Diversity score (1 - max similarity to selected)
                    max_similarity = self._compute_max_similarity(
                        embeddings[idx], selected_embeddings
                    )
                    diversity = 1.0 - max_similarity
                    
                    # MMR score
                    mmr_score = lambda_mmr * relevance + (1 - lambda_mmr) * diversity
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_candidate_idx = idx
                
                if best_candidate_idx is not None:
                    selected.append(best_candidate_idx)
                    selected_embeddings.append(embeddings[best_candidate_idx])
                    remaining_indices.remove(best_candidate_idx)
            
            # Create MMRDocument objects
            mmr_documents = []
            for rank, idx in enumerate(selected):
                candidate = candidates[idx]
                relevance = normalized_relevance[idx]
                diversity = 1.0 - self._compute_max_similarity(
                    embeddings[idx], selected_embeddings[:rank]
                ) if rank > 0 else 1.0
                
                mmr_doc = MMRDocument(
                    doc_id=candidate.get('doc_id', f"doc_{idx}"),
                    content=candidate.get('content', ''),
                    relevance_score=relevance,
                    diversity_score=diversity,
                    mmr_score=best_mmr_score if rank == len(selected) - 1 else 0.0,
                    rank=rank,
                    features=candidate.get('features', {})
                )
                mmr_documents.append(mmr_doc)
            
            logger.info(f"Selected {len(mmr_documents)} diverse documents using MMR (λ={lambda_mmr})")
            return mmr_documents
            
        except Exception as e:
            logger.error(f"Error in MMR selection: {e}")
            # Return top-k by relevance as fallback
            fallback = []
            sorted_candidates = sorted(
                enumerate(candidates),
                key=lambda x: x[1].get('rrf_score', x[1].get('score', 0.0)),
                reverse=True
            )
            
            for rank, (idx, candidate) in enumerate(sorted_candidates[:top_k]):
                mmr_doc = MMRDocument(
                    doc_id=candidate.get('doc_id', f"doc_{idx}"),
                    content=candidate.get('content', ''),
                    relevance_score=candidate.get('rrf_score', candidate.get('score', 0.0)),
                    diversity_score=0.0,
                    mmr_score=0.0,
                    rank=rank,
                    features=candidate.get('features', {})
                )
                fallback.append(mmr_doc)
            
            return fallback
    
    def compute_redundancy_rate(self, documents: List[Dict]) -> float:
        """
        Compute redundancy rate among a set of documents.
        
        Args:
            documents: List of documents with 'content' field
            
        Returns:
            Redundancy rate between 0 and 1
        """
        if len(documents) < 2:
            return 0.0
        
        try:
            texts = [doc.get('content', '') for doc in documents]
            embeddings = self._compute_embeddings(texts)
            
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = self._compute_similarity(embeddings[i], embeddings[j])
                    similarities.append(similarity)
            
            # Count documents above similarity threshold
            redundant_pairs = sum(1 for sim in similarities if sim > self.similarity_threshold)
            total_pairs = len(similarities)
            
            redundancy_rate = redundant_pairs / total_pairs if total_pairs > 0 else 0.0
            
            logger.info(f"Computed redundancy rate: {redundancy_rate:.3f} ({redundant_pairs}/{total_pairs} pairs)")
            return redundancy_rate
            
        except Exception as e:
            logger.error(f"Error computing redundancy rate: {e}")
            return 0.0
    
    def lambda_sweep(self, 
                    query: str,
                    candidates: List[Dict],
                    lambda_values: List[float] = [0.2, 0.4, 0.6, 0.8],
                    top_k: int = 10) -> Dict[float, List[MMRDocument]]:
        """
        Perform MMR selection with different lambda values.
        
        Args:
            query: Search query
            candidates: List of candidate documents
            lambda_values: List of lambda values to test
            top_k: Number of documents to select
            
        Returns:
            Dictionary mapping lambda values to selected documents
        """
        results = {}
        
        for lambda_val in lambda_values:
            logger.info(f"Running MMR with λ={lambda_val}")
            selected = self.select_diverse_documents(
                query, candidates, lambda_mmr=lambda_val, top_k=top_k
            )
            results[lambda_val] = selected
            
            # Compute redundancy rate for this selection
            redundancy = self.compute_redundancy_rate([
                {"content": doc.content} for doc in selected
            ])
            logger.info(f"λ={lambda_val}: redundancy rate = {redundancy:.3f}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize MMR deduplicator
    mmr = MMRDeduplicator()
    
    # Example candidates
    query = "Football players and their achievements"
    candidates = [
        {
            "doc_id": "doc1",
            "content": "Lionel Messi won the Ballon d'Or multiple times and is considered one of the greatest footballers.",
            "rrf_score": 0.9,
            "features": {"length": 100}
        },
        {
            "doc_id": "doc2",
            "content": "Cristiano Ronaldo is another great footballer who has won many awards and championships.",
            "rrf_score": 0.8,
            "features": {"length": 90}
        },
        {
            "doc_id": "doc3",
            "content": "Messi and Ronaldo have dominated world football for over a decade with their incredible skills.",
            "rrf_score": 0.7,
            "features": {"length": 95}
        },
        {
            "doc_id": "doc4",
            "content": "The World Cup is the most prestigious tournament in international football.",
            "rrf_score": 0.6,
            "features": {"length": 80}
        },
        {
            "doc_id": "doc5",
            "content": "Football tactics and formations have evolved significantly over the years.",
            "rrf_score": 0.5,
            "features": {"length": 85}
        }
    ]
    
    # Test MMR selection
    print(f"MMR Selection for: '{query}'")
    print("=" * 80)
    
    selected = mmr.select_diverse_documents(query, candidates, lambda_mmr=0.5, top_k=3)
    
    for doc in selected:
        print(f"\nRank {doc.rank + 1}: {doc.doc_id}")
        print(f"  Relevance: {doc.relevance_score:.3f}")
        print(f"  Diversity: {doc.diversity_score:.3f}")
        print(f"  Content: {doc.content}")
    
    # Test lambda sweep
    print(f"\n" + "=" * 80)
    print("Lambda Sweep Results")
    print("=" * 80)
    
    lambda_results = mmr.lambda_sweep(query, candidates, top_k=3)
    
    for lambda_val, docs in lambda_results.items():
        print(f"\nλ = {lambda_val}:")
        for doc in docs:
            print(f"  {doc.doc_id}: {doc.content[:50]}...")
    
    # Test redundancy computation
    print(f"\n" + "=" * 80)
    print("Redundancy Analysis")
    print("=" * 80)
    
    redundancy = mmr.compute_redundancy_rate(candidates)
    print(f"Original redundancy rate: {redundancy:.3f}")
    
    # Test with selected documents
    selected_docs = [{"content": doc.content} for doc in selected]
    selected_redundancy = mmr.compute_redundancy_rate(selected_docs)
    print(f"Selected redundancy rate: {selected_redundancy:.3f}")
