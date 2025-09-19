"""
Expert trajectory generation for Behavioral Cloning (BC).

Generates high-quality expert trajectories using strong baseline policies
to provide training data for BC pretraining.
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from ..env.rag_environment import RAGEnvironment, RAGState, RAGAction, ConversationTurn
from ..policy.episode_runner import EpisodeRunner, PolicyConfig, EpisodeResult
from ..reward.reward_shaping import RewardShaper, RewardConfig
from ..reward.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class ExpertTrajectory:
    """Represents an expert trajectory for BC training."""
    query: str
    conversation_history: List[ConversationTurn]
    states: List[RAGState]
    actions: List[RAGAction]
    rewards: List[float]
    final_answer: str
    total_reward: float
    metadata: Dict[str, Any] = None


@dataclass
class ExpertConfig:
    """Configuration for expert trajectory generation."""
    # Expert policy configuration
    expert_policy: str = "greedy"  # "greedy", "oracle", "reward_guided"
    selection_strategy: str = "top_score"
    
    # Trajectory filtering
    min_reward_threshold: float = 0.5
    max_trajectories_per_query: int = 5
    diversity_threshold: float = 0.3
    
    # Reward configuration
    use_reward_shaping: bool = True
    reward_config: Optional[RewardConfig] = None
    
    # Quality filtering
    use_llm_judge: bool = True
    min_judge_score: float = 0.6
    
    # Output configuration
    output_dir: str = "outputs/expert_trajectories"
    save_format: str = "json"  # "json", "jsonl"


class ExpertTrajectoryGenerator:
    """
    Generates expert trajectories for Behavioral Cloning.
    
    Uses strong baseline policies to create high-quality trajectories
    that serve as training data for BC pretraining.
    """
    
    def __init__(self,
                 rag_env: RAGEnvironment,
                 episode_runner: EpisodeRunner,
                 reward_shaper: Optional[RewardShaper] = None,
                 llm_judge: Optional[LLMJudge] = None,
                 config: Optional[ExpertConfig] = None):
        """
        Initialize the expert trajectory generator.
        
        Args:
            rag_env: RAG environment for generating episodes
            episode_runner: Episode runner for policy execution
            reward_shaper: Reward shaper for quality assessment
            llm_judge: LLM judge for answer quality evaluation
            config: Expert configuration
        """
        self.rag_env = rag_env
        self.episode_runner = episode_runner
        self.reward_shaper = reward_shaper
        self.llm_judge = llm_judge
        self.config = config or ExpertConfig()
        
        # Expert policy configurations
        self.expert_policies = self._create_expert_policies()
        
        logger.info("Initialized expert trajectory generator")
    
    def _create_expert_policies(self) -> List[PolicyConfig]:
        """Create expert policy configurations."""
        policies = []
        
        if self.config.expert_policy == "greedy":
            policies.append(PolicyConfig(
                policy_type="greedy",
                selection_strategy="top_score"
            ))
        elif self.config.expert_policy == "oracle":
            # Oracle policy uses perfect information (for simulation)
            policies.append(PolicyConfig(
                policy_type="greedy",
                selection_strategy="oracle"  # Would need custom implementation
            ))
        elif self.config.expert_policy == "reward_guided":
            # Multiple policies with different strategies
            policies.extend([
                PolicyConfig(policy_type="greedy", selection_strategy="top_score"),
                PolicyConfig(policy_type="epsilon_greedy", selection_strategy="top_score", epsilon=0.1),
                PolicyConfig(policy_type="epsilon_greedy", selection_strategy="top_score", epsilon=0.2),
            ])
        
        return policies
    
    def generate_expert_trajectory(self, 
                                 query: str, 
                                 conversation_history: List[ConversationTurn],
                                 policy_config: PolicyConfig) -> Optional[ExpertTrajectory]:
        """
        Generate a single expert trajectory.
        
        Args:
            query: Input query
            conversation_history: Conversation history
            policy_config: Policy configuration to use
            
        Returns:
            ExpertTrajectory or None if generation failed
        """
        try:
            # Run episode with the policy
            episode_result = self.episode_runner.run_episode(query, policy_config, conversation_history)
            
            # Extract trajectory components
            states = [step[0] for step in episode_result.trajectory]
            actions = [step[1] for step in episode_result.trajectory]
            rewards = [step[2].step_reward for step in episode_result.trajectory]
            
            # Generate final answer from selected documents
            final_answer = self._generate_answer_from_trajectory(episode_result)
            
            # Compute shaped rewards if enabled
            if self.config.use_reward_shaping and self.reward_shaper:
                shaped_rewards = self._compute_shaped_rewards(
                    query, states, actions, final_answer
                )
                rewards = shaped_rewards
            
            # Create expert trajectory
            trajectory = ExpertTrajectory(
                query=query,
                conversation_history=conversation_history,
                states=states,
                actions=actions,
                rewards=rewards,
                final_answer=final_answer,
                total_reward=episode_result.total_reward,
                metadata={
                    "policy_config": asdict(policy_config),
                    "episode_result": asdict(episode_result),
                    "num_steps": len(states),
                    "termination_reason": episode_result.termination_reason
                }
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error generating expert trajectory: {e}")
            return None
    
    def _generate_answer_from_trajectory(self, episode_result: EpisodeResult) -> str:
        """
        Generate a final answer from the episode trajectory.
        
        Args:
            episode_result: Result of the episode
            
        Returns:
            Generated answer
        """
        # Simple implementation: concatenate selected document contents
        # In practice, this would use a generator model
        selected_docs = episode_result.final_state.selected_documents
        
        if not selected_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Combine document contents
        answer_parts = []
        for doc in selected_docs:
            # Truncate long documents
            content = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            answer_parts.append(content)
        
        # Create a coherent answer
        answer = " ".join(answer_parts)
        
        # Simple post-processing
        if len(answer) > 1000:
            answer = answer[:1000] + "..."
        
        return answer
    
    def _compute_shaped_rewards(self, 
                              query: str, 
                              states: List[RAGState], 
                              actions: List[RAGAction],
                              final_answer: str) -> List[float]:
        """
        Compute shaped rewards for the trajectory.
        
        Args:
            query: Original query
            states: Trajectory states
            actions: Trajectory actions
            final_answer: Final generated answer
            
        Returns:
            List of shaped rewards
        """
        if not self.reward_shaper:
            return [0.0] * len(actions)
        
        # Reset reward shaper
        self.reward_shaper.reset_episode()
        
        shaped_rewards = []
        
        for i, (state, action) in enumerate(zip(states, actions)):
            if action.action_type == "select_document" and action.document_id:
                # Find the selected document
                selected_doc = None
                for doc in state.candidate_pool:
                    if doc.doc_id == action.document_id:
                        selected_doc = doc
                        break
                
                if selected_doc:
                    # Compute step reward
                    step_reward_components = self.reward_shaper.compute_step_reward(
                        query, selected_doc.content, 
                        selected_doc.bm25_score + selected_doc.vector_score, i
                    )
                    shaped_rewards.append(step_reward_components.total_reward)
                else:
                    shaped_rewards.append(0.0)
            else:
                shaped_rewards.append(0.0)
        
        # Add final reward
        if shaped_rewards:
            final_reward_components = self.reward_shaper.compute_episode_reward(
                query, final_answer
            )
            # Add final reward to the last step
            shaped_rewards[-1] += final_reward_components.final_reward
        
        return shaped_rewards
    
    def filter_trajectory_quality(self, trajectory: ExpertTrajectory) -> bool:
        """
        Filter trajectory based on quality criteria.
        
        Args:
            trajectory: Trajectory to evaluate
            
        Returns:
            True if trajectory passes quality filters
        """
        # Check minimum reward threshold
        if trajectory.total_reward < self.config.min_reward_threshold:
            logger.debug(f"Trajectory rejected: low reward ({trajectory.total_reward:.3f})")
            return False
        
        # Check minimum number of steps
        if len(trajectory.actions) < 2:
            logger.debug("Trajectory rejected: too few steps")
            return False
        
        # Check answer quality with LLM judge if available
        if self.config.use_llm_judge and self.llm_judge:
            try:
                # Create a simple answer pair for evaluation
                from ..reward.llm_judge import AnswerPair
                
                # Use a dummy rejected answer for comparison
                dummy_rejected = "I don't have enough information to answer this question."
                
                answer_pair = AnswerPair(
                    query=trajectory.query,
                    answer_a=trajectory.final_answer,
                    answer_b=dummy_rejected,
                    context_a=trajectory.states[-1].selected_documents if trajectory.states else [],
                    context_b=[],
                    metadata={"trajectory_id": id(trajectory)}
                )
                
                result = self.llm_judge.compare_answers(answer_pair)
                
                if result.confidence < self.config.min_judge_score:
                    logger.debug(f"Trajectory rejected: low judge score ({result.confidence:.3f})")
                    return False
                
            except Exception as e:
                logger.warning(f"Error in LLM judge evaluation: {e}")
                # Don't reject based on judge error
        
        return True
    
    def generate_expert_dataset(self, 
                              queries: List[Tuple[str, List[ConversationTurn]]],
                              output_path: str) -> Dict[str, Any]:
        """
        Generate a complete expert dataset.
        
        Args:
            queries: List of (query, conversation_history) tuples
            output_path: Path to save the dataset
            
        Returns:
            Dataset statistics
        """
        logger.info(f"Generating expert dataset with {len(queries)} queries")
        
        all_trajectories = []
        rejected_count = 0
        
        for i, (query, history) in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: '{query}'")
            
            query_trajectories = []
            
            # Generate trajectories with different expert policies
            for policy_config in self.expert_policies:
                for attempt in range(self.config.max_trajectories_per_query):
                    trajectory = self.generate_expert_trajectory(query, history, policy_config)
                    
                    if trajectory and self.filter_trajectory_quality(trajectory):
                        query_trajectories.append(trajectory)
                        logger.debug(f"Generated quality trajectory with reward {trajectory.total_reward:.3f}")
                    else:
                        rejected_count += 1
                    
                    # Stop if we have enough good trajectories for this query
                    if len(query_trajectories) >= self.config.max_trajectories_per_query:
                        break
            
            # Apply diversity filtering within query
            if len(query_trajectories) > 1:
                query_trajectories = self._filter_diverse_trajectories(query_trajectories)
            
            all_trajectories.extend(query_trajectories)
            logger.info(f"Query {i+1}: {len(query_trajectories)} trajectories accepted")
        
        # Save dataset
        self._save_expert_dataset(all_trajectories, output_path)
        
        # Compute statistics
        stats = self._compute_dataset_stats(all_trajectories, rejected_count)
        
        return stats
    
    def _filter_diverse_trajectories(self, trajectories: List[ExpertTrajectory]) -> List[ExpertTrajectory]:
        """
        Filter trajectories to maintain diversity.
        
        Args:
            trajectories: List of trajectories for a single query
            
        Returns:
            Filtered list of diverse trajectories
        """
        if len(trajectories) <= 1:
            return trajectories
        
        # Sort by reward (descending)
        trajectories.sort(key=lambda t: t.total_reward, reverse=True)
        
        diverse_trajectories = [trajectories[0]]  # Always keep the best
        
        for trajectory in trajectories[1:]:
            # Check diversity against already selected trajectories
            is_diverse = True
            
            for selected in diverse_trajectories:
                # Simple diversity check: different number of steps or different final answers
                step_diff = abs(len(trajectory.actions) - len(selected.actions))
                answer_similarity = self._compute_answer_similarity(
                    trajectory.final_answer, selected.final_answer
                )
                
                if step_diff < 2 and answer_similarity > (1 - self.config.diversity_threshold):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_trajectories.append(trajectory)
        
        return diverse_trajectories
    
    def _compute_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Compute similarity between two answers."""
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _save_expert_dataset(self, trajectories: List[ExpertTrajectory], output_path: str):
        """Save expert dataset to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.save_format == "json":
            # Save as JSON array
            data = [asdict(trajectory) for trajectory in trajectories]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif self.config.save_format == "jsonl":
            # Save as JSONL
            with open(output_path, 'w', encoding='utf-8') as f:
                for trajectory in trajectories:
                    json.dump(asdict(trajectory), f, ensure_ascii=False)
                    f.write('\n')
        
        logger.info(f"Saved {len(trajectories)} expert trajectories to {output_path}")
    
    def _compute_dataset_stats(self, trajectories: List[ExpertTrajectory], rejected_count: int) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not trajectories:
            return {"error": "No trajectories generated"}
        
        rewards = [t.total_reward for t in trajectories]
        step_counts = [len(t.actions) for t in trajectories]
        answer_lengths = [len(t.final_answer.split()) for t in trajectories]
        
        stats = {
            "total_trajectories": len(trajectories),
            "rejected_trajectories": rejected_count,
            "acceptance_rate": len(trajectories) / (len(trajectories) + rejected_count),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "avg_steps": np.mean(step_counts),
            "std_steps": np.std(step_counts),
            "avg_answer_length": np.mean(answer_lengths),
            "unique_queries": len(set(t.query for t in trajectories)),
            "policies_used": list(set(t.metadata["policy_config"]["policy_type"] for t in trajectories))
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = ExpertConfig(
        expert_policy="reward_guided",
        min_reward_threshold=0.3,
        max_trajectories_per_query=3,
        use_llm_judge=False,  # Disable for testing
        output_dir="outputs/expert_test"
    )
    
    # Example queries
    test_queries = [
        ("Who won the FA Cup in 2020?", []),
        ("What is the capital of France?", []),
        ("Tell me about the history of the internet.", [])
    ]
    
    print("Expert trajectory generator ready for use")
    print(f"Configuration: {config}")
    print(f"Test queries: {len(test_queries)}")
    
    # Note: Actual generation would require initialized RAG environment and components
    # generator = ExpertTrajectoryGenerator(rag_env, episode_runner, reward_shaper, llm_judge, config)
    # stats = generator.generate_expert_dataset(test_queries, "outputs/expert_trajectories.json")
