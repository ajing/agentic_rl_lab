"""
RAFT (Reward rAnked FineTuning) implementation.

Implements RAFT training where candidate answers are generated,
ranked by reward model, and the best ones are used for fine-tuning.
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.env.rag_environment import RAGEnvironment, RAGAction, ConversationTurn
from src.policy.episode_runner import EpisodeRunner, PolicyConfig, EpisodeResult
from src.reward.reward_model import LightweightRewardModel
from src.reward.reward_shaping import RewardShaper
from src.training.dpo_trainer import DPOTrainerWrapper, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class RAFTExample:
    """Represents a RAFT training example."""
    query: str
    context: List[str]
    answer: str
    reward_score: float
    rank: int
    metadata: Dict[str, Any] = None


@dataclass
class RAFTConfig:
    """Configuration for RAFT training."""
    # Candidate generation
    num_candidates_per_query: int = 8
    candidate_policies: List[str] = None  # Will be set to default policies
    
    # Ranking configuration
    top_k_for_training: int = 2  # Use top-k candidates for training
    reward_threshold: float = 0.5  # Minimum reward threshold
    
    # Training configuration
    training_config: Optional[TrainingConfig] = None
    
    # Output configuration
    output_dir: str = "outputs/raft_training"
    save_candidates: bool = True
    
    def __post_init__(self):
        if self.candidate_policies is None:
            self.candidate_policies = ["random", "greedy", "epsilon_greedy_0.1", "epsilon_greedy_0.3"]


class RAFTDataset(Dataset):
    """Dataset for RAFT training."""
    
    def __init__(self, examples: List[RAFTExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "query": example.query,
            "context": example.context,
            "answer": example.answer,
            "reward_score": example.reward_score,
            "rank": example.rank,
            "metadata": example.metadata
        }


class RAFTTrainer:
    """
    RAFT (Reward rAnked FineTuning) trainer.
    
    Generates multiple candidate answers, ranks them by reward model,
    and fine-tunes on the best candidates.
    """
    
    def __init__(self,
                 rag_env: RAGEnvironment,
                 episode_runner: EpisodeRunner,
                 reward_model: LightweightRewardModel,
                 reward_shaper: Optional[RewardShaper] = None,
                 config: Optional[RAFTConfig] = None):
        """
        Initialize the RAFT trainer.
        
        Args:
            rag_env: RAG environment for generating episodes
            episode_runner: Episode runner for policy execution
            reward_model: Reward model for ranking candidates
            reward_shaper: Optional reward shaper for additional scoring
            config: RAFT configuration
        """
        self.rag_env = rag_env
        self.episode_runner = episode_runner
        self.reward_model = reward_model
        self.reward_shaper = reward_shaper
        self.config = config or RAFTConfig()
        
        # Create candidate generation policies
        self.candidate_policies = self._create_candidate_policies()
        
        logger.info("Initialized RAFT trainer")
    
    def _create_candidate_policies(self) -> List[PolicyConfig]:
        """Create policies for candidate generation."""
        policies = []
        
        for policy_name in self.config.candidate_policies:
            if policy_name == "random":
                policies.append(PolicyConfig(policy_type="random", selection_strategy="random"))
            elif policy_name == "greedy":
                policies.append(PolicyConfig(policy_type="greedy", selection_strategy="top_score"))
            elif policy_name.startswith("epsilon_greedy_"):
                epsilon = float(policy_name.split("_")[-1])
                policies.append(PolicyConfig(
                    policy_type="epsilon_greedy", 
                    selection_strategy="top_score", 
                    epsilon=epsilon
                ))
        
        return policies
    
    def generate_candidates(self, 
                          query: str, 
                          conversation_history: List[ConversationTurn]) -> List[RAFTExample]:
        """
        Generate candidate answers for a query.
        
        Args:
            query: Input query
            conversation_history: Conversation history
            
        Returns:
            List of RAFTExample candidates
        """
        candidates = []
        
        for i, policy_config in enumerate(self.candidate_policies):
            try:
                # Run episode with the policy
                episode_result = self.episode_runner.run_episode(query, policy_config, conversation_history)
                
                # Generate answer from trajectory
                answer = self._generate_answer_from_episode(episode_result)
                context = [doc.content for doc in episode_result.final_state.selected_documents]
                
                # Score with reward model
                reward_score = self._score_answer(query, answer, context)
                
                # Create RAFT example
                candidate = RAFTExample(
                    query=query,
                    context=context,
                    answer=answer,
                    reward_score=reward_score,
                    rank=0,  # Will be set after ranking
                    metadata={
                        "policy": policy_config.policy_type,
                        "episode_reward": episode_result.total_reward,
                        "num_steps": len(episode_result.trajectory),
                        "termination_reason": episode_result.termination_reason
                    }
                )
                
                candidates.append(candidate)
                logger.debug(f"Generated candidate {i+1} with reward {reward_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Error generating candidate with policy {policy_config.policy_type}: {e}")
                continue
        
        return candidates
    
    def _generate_answer_from_episode(self, episode_result: EpisodeResult) -> str:
        """
        Generate answer from episode result.
        
        Args:
            episode_result: Result of the episode
            
        Returns:
            Generated answer
        """
        # Simple implementation: combine selected document contents
        selected_docs = episode_result.final_state.selected_documents
        
        if not selected_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Combine document contents with some structure
        answer_parts = []
        for i, doc in enumerate(selected_docs):
            content = doc.content[:400] + "..." if len(doc.content) > 400 else doc.content
            answer_parts.append(f"[{i+1}] {content}")
        
        answer = " ".join(answer_parts)
        
        # Truncate if too long
        if len(answer) > 1200:
            answer = answer[:1200] + "..."
        
        return answer
    
    def _score_answer(self, query: str, answer: str, context: List[str]) -> float:
        """
        Score an answer using the reward model.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Supporting context
            
        Returns:
            Reward score
        """
        try:
            with torch.no_grad():
                result = self.reward_model(query, answer, context)
                reward_score = result["reward_score"].item()
                return reward_score
        except Exception as e:
            logger.warning(f"Error scoring answer: {e}")
            # Fallback: simple heuristic
            return min(len(answer.split()) / 100, 1.0)
    
    def rank_candidates(self, candidates: List[RAFTExample]) -> List[RAFTExample]:
        """
        Rank candidates by reward score.
        
        Args:
            candidates: List of candidates to rank
            
        Returns:
            Ranked list of candidates
        """
        # Sort by reward score (descending)
        ranked_candidates = sorted(candidates, key=lambda x: x.reward_score, reverse=True)
        
        # Assign ranks
        for i, candidate in enumerate(ranked_candidates):
            candidate.rank = i + 1
        
        logger.info(f"Ranked {len(ranked_candidates)} candidates")
        return ranked_candidates
    
    def filter_candidates(self, candidates: List[RAFTExample]) -> List[RAFTExample]:
        """
        Filter candidates based on quality criteria.
        
        Args:
            candidates: List of ranked candidates
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for candidate in candidates:
            # Check reward threshold
            if candidate.reward_score < self.config.reward_threshold:
                continue
            
            # Check answer length (not too short, not too long)
            answer_length = len(candidate.answer.split())
            if answer_length < 10 or answer_length > 500:
                continue
            
            # Check context availability
            if not candidate.context:
                continue
            
            filtered.append(candidate)
        
        logger.info(f"Filtered {len(filtered)}/{len(candidates)} candidates")
        return filtered
    
    def select_training_candidates(self, candidates: List[RAFTExample]) -> List[RAFTExample]:
        """
        Select candidates for training.
        
        Args:
            candidates: List of filtered candidates
            
        Returns:
            Selected candidates for training
        """
        # Take top-k candidates
        selected = candidates[:self.config.top_k_for_training]
        
        # Ensure diversity if we have enough candidates
        if len(candidates) > self.config.top_k_for_training:
            selected = self._ensure_diversity(selected, candidates)
        
        logger.info(f"Selected {len(selected)} candidates for training")
        return selected
    
    def _ensure_diversity(self, 
                         selected: List[RAFTExample], 
                         all_candidates: List[RAFTExample]) -> List[RAFTExample]:
        """
        Ensure diversity in selected candidates.
        
        Args:
            selected: Currently selected candidates
            all_candidates: All available candidates
            
        Returns:
            Diverse set of selected candidates
        """
        if len(selected) <= 1:
            return selected
        
        # Check for diversity based on policy and answer similarity
        diverse_selected = [selected[0]]  # Always keep the best
        
        for candidate in selected[1:]:
            is_diverse = True
            
            for selected_candidate in diverse_selected:
                # Check policy diversity
                if candidate.metadata["policy"] == selected_candidate.metadata["policy"]:
                    # Check answer similarity
                    similarity = self._compute_answer_similarity(
                        candidate.answer, selected_candidate.answer
                    )
                    if similarity > 0.7:  # High similarity threshold
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_selected.append(candidate)
        
        return diverse_selected
    
    def _compute_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Compute similarity between two answers."""
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def build_raft_dataset(self, 
                          queries: List[Tuple[str, List[ConversationTurn]]],
                          output_path: str) -> Dict[str, Any]:
        """
        Build RAFT training dataset.
        
        Args:
            queries: List of (query, conversation_history) tuples
            output_path: Path to save the dataset
            
        Returns:
            Dataset statistics
        """
        logger.info(f"Building RAFT dataset with {len(queries)} queries")
        
        all_candidates = []
        total_generated = 0
        
        for i, (query, history) in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: '{query}'")
            
            # Generate candidates
            candidates = self.generate_candidates(query, history)
            total_generated += len(candidates)
            
            # Rank and filter candidates
            ranked_candidates = self.rank_candidates(candidates)
            filtered_candidates = self.filter_candidates(ranked_candidates)
            selected_candidates = self.select_training_candidates(filtered_candidates)
            
            all_candidates.extend(selected_candidates)
            logger.info(f"Query {i+1}: {len(selected_candidates)} candidates selected")
        
        # Save dataset
        self._save_raft_dataset(all_candidates, output_path)
        
        # Compute statistics
        stats = self._compute_dataset_stats(all_candidates, total_generated)
        
        return stats
    
    def _save_raft_dataset(self, candidates: List[RAFTExample], output_path: str):
        """Save RAFT dataset to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [asdict(candidate) for candidate in candidates]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(candidates)} RAFT examples to {output_path}")
    
    def _compute_dataset_stats(self, candidates: List[RAFTExample], total_generated: int) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not candidates:
            return {"error": "No candidates generated"}
        
        reward_scores = [c.reward_score for c in candidates]
        ranks = [c.rank for c in candidates]
        answer_lengths = [len(c.answer.split()) for c in candidates]
        
        # Policy distribution
        policy_counts = {}
        for candidate in candidates:
            policy = candidate.metadata["policy"]
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
        
        stats = {
            "total_candidates": len(candidates),
            "total_generated": total_generated,
            "selection_rate": len(candidates) / total_generated if total_generated > 0 else 0.0,
            "avg_reward_score": np.mean(reward_scores),
            "std_reward_score": np.std(reward_scores),
            "min_reward_score": np.min(reward_scores),
            "max_reward_score": np.max(reward_scores),
            "avg_rank": np.mean(ranks),
            "avg_answer_length": np.mean(answer_lengths),
            "unique_queries": len(set(c.query for c in candidates)),
            "policy_distribution": policy_counts
        }
        
        return stats
    
    def train_raft_model(self, 
                        raft_dataset_path: str,
                        model_output_path: str,
                        training_config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """
        Train a model using RAFT dataset.
        
        Args:
            raft_dataset_path: Path to RAFT dataset
            model_output_path: Path to save trained model
            training_config: Training configuration
            
        Returns:
            Training results
        """
        # Load RAFT dataset
        with open(raft_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to training format
        training_examples = []
        for example in data:
            training_examples.append({
                "prompt": example["query"],
                "chosen": example["answer"],
                "rejected": "I don't have enough information to answer this question."  # Dummy rejected
            })
        
        # Initialize trainer
        config = training_config or self.config.training_config or TrainingConfig()
        trainer = DPOTrainerWrapper(config)
        
        # Train using DPO (can be adapted for other methods)
        results = trainer.train_dpo(training_examples, save_path=model_output_path)
        
        logger.info(f"RAFT training completed. Results: {results}")
        return results


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = RAFTConfig(
        num_candidates_per_query=4,
        top_k_for_training=2,
        reward_threshold=0.3,
        output_dir="outputs/raft_test"
    )
    
    # Example queries
    test_queries = [
        ("Who won the FA Cup in 2020?", []),
        ("What is the capital of France?", []),
        ("Tell me about the history of the internet.", [])
    ]
    
    print("RAFT trainer ready for use")
    print(f"Configuration: {config}")
    print(f"Test queries: {len(test_queries)}")
    
    # Note: Actual training would require initialized components
    # raft_trainer = RAFTTrainer(rag_env, episode_runner, reward_model, reward_shaper, config)
    # stats = raft_trainer.build_raft_dataset(test_queries, "outputs/raft_dataset.json")
