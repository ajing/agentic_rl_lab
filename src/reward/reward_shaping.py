"""
Reward shaping for RL environment.

Implements comprehensive reward shaping that combines:
- Final answer rewards (from LLM judge or reward model)
- Step-wise novelty rewards (MMR-based diversity)
- Step-wise relevance rewards (document quality)
- Shaped rewards for better RL training
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import torch

from .reward_model import LightweightRewardModel
from src.reranker.mmr import MMRDeduplicator

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Components of the shaped reward."""
    final_reward: float = 0.0
    novelty_reward: float = 0.0
    relevance_reward: float = 0.0
    diversity_reward: float = 0.0
    coherence_reward: float = 0.0
    total_reward: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # Final reward weights
    final_weight: float = 1.0
    
    # Step-wise reward weights
    novelty_weight: float = 0.3
    relevance_weight: float = 0.2
    diversity_weight: float = 0.2
    coherence_weight: float = 0.1
    
    # Novelty parameters
    novelty_threshold: float = 0.7  # Threshold for considering documents novel
    novelty_decay: float = 0.9     # Decay factor for repeated novelty
    
    # Relevance parameters
    relevance_threshold: float = 0.5  # Minimum relevance score
    relevance_bonus: float = 0.1     # Bonus for high relevance
    
    # Diversity parameters
    diversity_penalty: float = 0.05  # Penalty for low diversity
    max_diversity_bonus: float = 0.2  # Maximum diversity bonus
    
    # Coherence parameters
    coherence_window: int = 3  # Window for coherence calculation
    coherence_bonus: float = 0.1  # Bonus for coherent selections


class RewardShaper:
    """
    Comprehensive reward shaping system.
    
    Combines multiple reward signals to provide rich feedback
    for RL training, encouraging both quality and diversity.
    """
    
    def __init__(self,
                 reward_model: Optional[LightweightRewardModel] = None,
                 mmr_deduplicator: Optional[MMRDeduplicator] = None,
                 config: Optional[RewardConfig] = None):
        """
        Initialize the reward shaper.
        
        Args:
            reward_model: Trained reward model for answer scoring
            mmr_deduplicator: MMR deduplicator for diversity calculation
            config: Reward configuration
        """
        self.reward_model = reward_model
        self.mmr_deduplicator = mmr_deduplicator
        self.config = config or RewardConfig()
        
        # Track state for step-wise rewards
        self.selected_documents: List[str] = []
        self.document_scores: Dict[str, float] = {}
        self.novelty_history: List[float] = []
        
        logger.info("Initialized reward shaper")
    
    def reset_episode(self):
        """Reset state for a new episode."""
        self.selected_documents = []
        self.document_scores = {}
        self.novelty_history = []
        logger.debug("Reset reward shaper for new episode")
    
    def compute_final_reward(self, 
                           query: str, 
                           answer: str, 
                           context: List[str],
                           use_reward_model: bool = True) -> float:
        """
        Compute final reward for the complete answer.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Selected context documents
            use_reward_model: Whether to use reward model or fallback
            
        Returns:
            Final reward score
        """
        if use_reward_model and self.reward_model is not None:
            try:
                with torch.no_grad():
                    result = self.reward_model(query, answer, context)
                    reward_score = result["reward_score"].item()
                    logger.debug(f"Final reward from model: {reward_score:.3f}")
                    return reward_score
            except Exception as e:
                logger.warning(f"Error computing final reward: {e}")
        
        # Fallback: simple heuristic based on answer length and context usage
        answer_length = len(answer.split())
        context_usage = len(context)
        
        # Simple scoring heuristic
        length_score = min(answer_length / 50, 1.0)  # Normalize to 0-1
        context_score = min(context_usage / 5, 1.0)  # Normalize to 0-1
        
        fallback_score = (length_score + context_score) / 2
        logger.debug(f"Final reward (fallback): {fallback_score:.3f}")
        return fallback_score
    
    def compute_novelty_reward(self, 
                             new_document: str, 
                             selected_documents: List[str]) -> float:
        """
        Compute novelty reward for selecting a new document.
        
        Args:
            new_document: Newly selected document
            selected_documents: Previously selected documents
            
        Returns:
            Novelty reward score
        """
        if not selected_documents:
            # First document gets maximum novelty
            return 1.0
        
        if self.mmr_deduplicator is not None:
            try:
                # Compute similarity with selected documents
                all_docs = selected_documents + [new_document]
                redundancy = self.mmr_deduplicator.compute_redundancy_rate([
                    {"content": doc} for doc in all_docs
                ])
                novelty = 1.0 - redundancy
                
                # Apply decay for repeated novelty patterns
                if self.novelty_history:
                    avg_historical_novelty = np.mean(self.novelty_history)
                    novelty = novelty * (1 - self.config.novelty_decay * (1 - avg_historical_novelty))
                
                self.novelty_history.append(novelty)
                
                # Keep only recent history
                if len(self.novelty_history) > 10:
                    self.novelty_history = self.novelty_history[-10:]
                
                logger.debug(f"Novelty reward: {novelty:.3f}")
                return novelty
                
            except Exception as e:
                logger.warning(f"Error computing novelty reward: {e}")
        
        # Fallback: simple word overlap
        new_words = set(new_document.lower().split())
        overlap_scores = []
        
        for selected_doc in selected_documents:
            selected_words = set(selected_doc.lower().split())
            if new_words and selected_words:
                overlap = len(new_words.intersection(selected_words)) / len(new_words.union(selected_words))
                overlap_scores.append(overlap)
        
        if overlap_scores:
            avg_overlap = np.mean(overlap_scores)
            novelty = 1.0 - avg_overlap
        else:
            novelty = 1.0
        
        logger.debug(f"Novelty reward (fallback): {novelty:.3f}")
        return novelty
    
    def compute_relevance_reward(self, 
                               document: str, 
                               query: str, 
                               document_score: float) -> float:
        """
        Compute relevance reward for document selection.
        
        Args:
            document: Selected document
            query: Original query
            document_score: Original retrieval score
            
        Returns:
            Relevance reward score
        """
        # Base relevance from retrieval score
        base_relevance = min(document_score, 1.0)
        
        # Bonus for high relevance
        if base_relevance > self.config.relevance_threshold:
            relevance_bonus = self.config.relevance_bonus * (base_relevance - self.config.relevance_threshold)
            total_relevance = base_relevance + relevance_bonus
        else:
            total_relevance = base_relevance
        
        # Penalty for very low relevance
        if base_relevance < 0.1:
            total_relevance *= 0.5
        
        logger.debug(f"Relevance reward: {total_relevance:.3f}")
        return min(total_relevance, 1.0)
    
    def compute_diversity_reward(self, 
                               selected_documents: List[str]) -> float:
        """
        Compute diversity reward for the current selection.
        
        Args:
            selected_documents: All selected documents so far
            
        Returns:
            Diversity reward score
        """
        if len(selected_documents) < 2:
            return 0.0  # No diversity to measure
        
        if self.mmr_deduplicator is not None:
            try:
                redundancy = self.mmr_deduplicator.compute_redundancy_rate([
                    {"content": doc} for doc in selected_documents
                ])
                diversity = 1.0 - redundancy
                
                # Scale diversity reward
                diversity_reward = diversity * self.config.max_diversity_bonus
                
                logger.debug(f"Diversity reward: {diversity_reward:.3f}")
                return diversity_reward
                
            except Exception as e:
                logger.warning(f"Error computing diversity reward: {e}")
        
        # Fallback: simple diversity measure
        all_words = set()
        doc_word_sets = []
        
        for doc in selected_documents:
            words = set(doc.lower().split())
            doc_word_sets.append(words)
            all_words.update(words)
        
        if not all_words:
            return 0.0
        
        # Compute average pairwise Jaccard similarity
        similarities = []
        for i in range(len(doc_word_sets)):
            for j in range(i + 1, len(doc_word_sets)):
                intersection = len(doc_word_sets[i].intersection(doc_word_sets[j]))
                union = len(doc_word_sets[i].union(doc_word_sets[j]))
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity
            diversity_reward = diversity * self.config.max_diversity_bonus
        else:
            diversity_reward = 0.0
        
        logger.debug(f"Diversity reward (fallback): {diversity_reward:.3f}")
        return diversity_reward
    
    def compute_coherence_reward(self, 
                               selected_documents: List[str],
                               query: str) -> float:
        """
        Compute coherence reward for document selection pattern.
        
        Args:
            selected_documents: All selected documents
            query: Original query
            
        Returns:
            Coherence reward score
        """
        if len(selected_documents) < self.config.coherence_window:
            return 0.0  # Not enough documents for coherence
        
        # Simple coherence: check if documents share common themes
        recent_docs = selected_documents[-self.config.coherence_window:]
        
        # Extract common words across recent documents
        all_words = set()
        for doc in recent_docs:
            words = set(doc.lower().split())
            all_words.update(words)
        
        if not all_words:
            return 0.0
        
        # Count how many documents contain each word
        word_counts = {}
        for word in all_words:
            count = sum(1 for doc in recent_docs if word in doc.lower())
            word_counts[word] = count
        
        # Coherence based on shared vocabulary
        shared_words = sum(1 for count in word_counts.values() if count > 1)
        total_words = len(all_words)
        
        if total_words > 0:
            coherence_ratio = shared_words / total_words
            coherence_reward = coherence_ratio * self.config.coherence_bonus
        else:
            coherence_reward = 0.0
        
        logger.debug(f"Coherence reward: {coherence_reward:.3f}")
        return coherence_reward
    
    def compute_step_reward(self, 
                          query: str,
                          selected_document: str,
                          document_score: float,
                          step: int) -> RewardComponents:
        """
        Compute step-wise reward for document selection.
        
        Args:
            query: Original query
            selected_document: Newly selected document
            document_score: Retrieval score for the document
            step: Current step number
            
        Returns:
            RewardComponents with all reward signals
        """
        # Add document to selection
        self.selected_documents.append(selected_document)
        self.document_scores[selected_document] = document_score
        
        # Compute individual reward components
        novelty_reward = self.compute_novelty_reward(selected_document, self.selected_documents[:-1])
        relevance_reward = self.compute_relevance_reward(selected_document, query, document_score)
        diversity_reward = self.compute_diversity_reward(self.selected_documents)
        coherence_reward = self.compute_coherence_reward(self.selected_documents, query)
        
        # Combine step-wise rewards
        step_reward = (
            self.config.novelty_weight * novelty_reward +
            self.config.relevance_weight * relevance_reward +
            self.config.diversity_weight * diversity_reward +
            self.config.coherence_weight * coherence_reward
        )
        
        return RewardComponents(
            final_reward=0.0,  # Will be set later
            novelty_reward=novelty_reward,
            relevance_reward=relevance_reward,
            diversity_reward=diversity_reward,
            coherence_reward=coherence_reward,
            total_reward=step_reward,
            metadata={
                "step": step,
                "document_score": document_score,
                "num_selected": len(self.selected_documents)
            }
        )
    
    def compute_episode_reward(self,
                             query: str,
                             answer: str,
                             final_reward: Optional[float] = None) -> RewardComponents:
        """
        Compute final episode reward combining all components.
        
        Args:
            query: Original query
            answer: Final generated answer
            final_reward: Pre-computed final reward (optional)
            
        Returns:
            Complete RewardComponents
        """
        # Compute final reward if not provided
        if final_reward is None:
            final_reward = self.compute_final_reward(query, answer, self.selected_documents)
        
        # Compute final diversity and coherence
        diversity_reward = self.compute_diversity_reward(self.selected_documents)
        coherence_reward = self.compute_coherence_reward(self.selected_documents, query)
        
        # Combine all rewards
        total_reward = (
            self.config.final_weight * final_reward +
            self.config.diversity_weight * diversity_reward +
            self.config.coherence_weight * coherence_reward
        )
        
        return RewardComponents(
            final_reward=final_reward,
            novelty_reward=0.0,  # Step-wise only
            relevance_reward=0.0,  # Step-wise only
            diversity_reward=diversity_reward,
            coherence_reward=coherence_reward,
            total_reward=total_reward,
            metadata={
                "episode_length": len(self.selected_documents),
                "avg_document_score": np.mean(list(self.document_scores.values())) if self.document_scores else 0.0
            }
        )
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """Get summary of reward statistics."""
        return {
            "num_documents_selected": len(self.selected_documents),
            "avg_novelty": np.mean(self.novelty_history) if self.novelty_history else 0.0,
            "document_scores": self.document_scores,
            "config": {
                "final_weight": self.config.final_weight,
                "novelty_weight": self.config.novelty_weight,
                "relevance_weight": self.config.relevance_weight,
                "diversity_weight": self.config.diversity_weight,
                "coherence_weight": self.config.coherence_weight
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize reward shaper
    config = RewardConfig(
        final_weight=1.0,
        novelty_weight=0.3,
        relevance_weight=0.2,
        diversity_weight=0.2,
        coherence_weight=0.1
    )
    
    shaper = RewardShaper(config=config)
    
    # Simulate episode
    query = "Who won the FA Cup in 2020?"
    
    # Step 1
    shaper.reset_episode()
    step1_reward = shaper.compute_step_reward(
        query, 
        "Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
        0.9,
        1
    )
    
    print(f"Step 1 reward: {step1_reward.total_reward:.3f}")
    print(f"  Novelty: {step1_reward.novelty_reward:.3f}")
    print(f"  Relevance: {step1_reward.relevance_reward:.3f}")
    
    # Step 2
    step2_reward = shaper.compute_step_reward(
        query,
        "The FA Cup is an annual football competition in England.",
        0.6,
        2
    )
    
    print(f"Step 2 reward: {step2_reward.total_reward:.3f}")
    print(f"  Novelty: {step2_reward.novelty_reward:.3f}")
    print(f"  Diversity: {step2_reward.diversity_reward:.3f}")
    
    # Final episode reward
    final_reward = shaper.compute_episode_reward(
        query,
        "Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final at Wembley Stadium."
    )
    
    print(f"Final episode reward: {final_reward.total_reward:.3f}")
    print(f"  Final answer: {final_reward.final_reward:.3f}")
    print(f"  Diversity: {final_reward.diversity_reward:.3f}")
    print(f"  Coherence: {final_reward.coherence_reward:.3f}")
    
    # Summary
    summary = shaper.get_reward_summary()
    print(f"Reward summary: {summary}")
