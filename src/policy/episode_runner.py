"""
Episode runner for RAG RL environment.

Implements random and greedy policies for testing the RL environment
and generating trajectories for behavioral cloning.
"""

import logging
import random
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from ..env.rag_environment import RAGEnvironment, RLAction, RLEpisode, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for a policy."""
    name: str
    policy_type: str  # "random", "greedy", "epsilon_greedy"
    epsilon: float = 0.0  # For epsilon-greedy
    temperature: float = 1.0  # For random sampling


class RandomPolicy:
    """Random policy that selects actions uniformly at random."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def select_action(self, valid_actions: List[RLAction], state_features: np.ndarray) -> RLAction:
        """Select a random action from valid actions."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Prefer select actions over terminate
        select_actions = [a for a in valid_actions if a.action_type == "select"]
        if select_actions:
            return random.choice(select_actions)
        else:
            return valid_actions[0]  # Should be terminate action


class GreedyPolicy:
    """Greedy policy that selects the highest-scoring action."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def select_action(self, valid_actions: List[RLAction], state_features: np.ndarray) -> RLAction:
        """Select the action with highest expected reward."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # For greedy policy, select the action with highest score
        # We'll use the candidate scores as proxy for expected reward
        select_actions = [a for a in valid_actions if a.action_type == "select"]
        
        if select_actions:
            # In a real implementation, you'd use a learned value function
            # For now, we'll use a simple heuristic based on state features
            # (candidate scores are in the state features)
            best_action = select_actions[0]  # Default to first
            best_score = 0.0
            
            for action in select_actions:
                # Simple heuristic: prefer actions with higher candidate scores
                # This is a placeholder - in practice you'd use learned Q-values
                score = random.random()  # Placeholder
                if score > best_score:
                    best_score = score
                    best_action = action
            
            return best_action
        else:
            return valid_actions[0]  # Should be terminate action


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy that balances exploration and exploitation."""
    
    def __init__(self, epsilon: float = 0.1, seed: Optional[int] = None):
        self.epsilon = epsilon
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def select_action(self, valid_actions: List[RLAction], state_features: np.ndarray) -> RLAction:
        """Select action using epsilon-greedy strategy."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Explore with probability epsilon
        if random.random() < self.epsilon:
            # Random action
            select_actions = [a for a in valid_actions if a.action_type == "select"]
            if select_actions:
                return random.choice(select_actions)
            else:
                return valid_actions[0]
        else:
            # Greedy action
            greedy_policy = GreedyPolicy(self.seed)
            return greedy_policy.select_action(valid_actions, state_features)


class EpisodeRunner:
    """
    Runs episodes in the RAG RL environment using different policies.
    
    Generates trajectories for analysis and behavioral cloning training.
    """
    
    def __init__(self, 
                 environment: RAGEnvironment,
                 output_dir: str = "outputs/episodes"):
        """
        Initialize the episode runner.
        
        Args:
            environment: RAG RL environment
            output_dir: Directory to save episode logs
        """
        self.env = environment
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Policy registry
        self.policies = {
            "random": RandomPolicy,
            "greedy": GreedyPolicy,
            "epsilon_greedy": EpsilonGreedyPolicy
        }
        
        logger.info(f"Initialized episode runner with output directory: {self.output_dir}")
    
    def run_episode(self, 
                   query: str,
                   policy_config: PolicyConfig,
                   conversation_history: Optional[List[ConversationTurn]] = None,
                   max_steps: Optional[int] = None,
                   save_episode: bool = True) -> RLEpisode:
        """
        Run a single episode with the specified policy.
        
        Args:
            query: Query to process
            policy_config: Policy configuration
            conversation_history: Previous conversation turns
            max_steps: Override max steps for this episode
            save_episode: Whether to save episode to disk
            
        Returns:
            Completed RLEpisode
        """
        # Initialize policy
        if policy_config.policy_type == "random":
            policy = RandomPolicy()
        elif policy_config.policy_type == "greedy":
            policy = GreedyPolicy()
        elif policy_config.policy_type == "epsilon_greedy":
            policy = EpsilonGreedyPolicy(epsilon=policy_config.epsilon)
        else:
            raise ValueError(f"Unknown policy type: {policy_config.policy_type}")
        
        # Reset environment
        state = self.env.reset(query, conversation_history)
        
        # Override max steps if specified
        original_max_steps = self.env.max_steps
        if max_steps is not None:
            self.env.max_steps = max_steps
        
        try:
            # Run episode
            step_count = 0
            while True:
                # Get valid actions
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break
                
                # Get state features
                state_features = self.env.get_state_features()
                
                # Select action
                action = policy.select_action(valid_actions, state_features)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                step_count += 1
                logger.debug(f"Step {step_count}: {action.action_type} {action.doc_id}, "
                           f"reward={reward.total_reward:.3f}")
                
                if done:
                    break
            
            # Finalize episode
            episode = self.env.finalize_episode()
            
            # Save episode if requested
            if save_episode:
                filename = f"{policy_config.name}_episode_{episode.episode_id}.json"
                filepath = self.output_dir / filename
                self.env.save_episode(str(filepath))
                logger.info(f"Saved episode to {filepath}")
            
            return episode
            
        finally:
            # Restore original max steps
            self.env.max_steps = original_max_steps
    
    def run_batch_episodes(self, 
                          queries: List[str],
                          policy_config: PolicyConfig,
                          conversation_histories: Optional[List[List[ConversationTurn]]] = None,
                          max_steps: Optional[int] = None,
                          save_episodes: bool = True) -> List[RLEpisode]:
        """
        Run multiple episodes in batch.
        
        Args:
            queries: List of queries to process
            policy_config: Policy configuration
            conversation_histories: List of conversation histories (optional)
            max_steps: Override max steps for episodes
            save_episodes: Whether to save episodes to disk
            
        Returns:
            List of completed RLEpisodes
        """
        episodes = []
        
        for i, query in enumerate(queries):
            logger.info(f"Running episode {i+1}/{len(queries)}: '{query[:50]}...'")
            
            history = conversation_histories[i] if conversation_histories else None
            episode = self.run_episode(
                query=query,
                policy_config=policy_config,
                conversation_history=history,
                max_steps=max_steps,
                save_episode=save_episodes
            )
            episodes.append(episode)
        
        return episodes
    
    def compare_policies(self, 
                        query: str,
                        policy_configs: List[PolicyConfig],
                        conversation_history: Optional[List[ConversationTurn]] = None,
                        num_runs: int = 3) -> Dict[str, List[RLEpisode]]:
        """
        Compare multiple policies on the same query.
        
        Args:
            query: Query to test
            policy_configs: List of policy configurations to compare
            conversation_history: Previous conversation turns
            num_runs: Number of runs per policy
            
        Returns:
            Dictionary mapping policy names to lists of episodes
        """
        results = {}
        
        for policy_config in policy_configs:
            logger.info(f"Testing policy: {policy_config.name}")
            episodes = []
            
            for run in range(num_runs):
                episode = self.run_episode(
                    query=query,
                    policy_config=policy_config,
                    conversation_history=conversation_history,
                    save_episode=False
                )
                episodes.append(episode)
            
            results[policy_config.name] = episodes
        
        return results
    
    def analyze_episodes(self, episodes: List[RLEpisode]) -> Dict[str, Any]:
        """
        Analyze a list of episodes and compute statistics.
        
        Args:
            episodes: List of episodes to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not episodes:
            return {}
        
        # Compute statistics
        total_rewards = [ep.metrics["total_reward"] for ep in episodes]
        episode_lengths = [ep.metrics["episode_length"] for ep in episodes]
        num_docs_selected = [ep.metrics["num_documents_selected"] for ep in episodes]
        
        analysis = {
            "num_episodes": len(episodes),
            "total_reward": {
                "mean": np.mean(total_rewards),
                "std": np.std(total_rewards),
                "min": np.min(total_rewards),
                "max": np.max(total_rewards)
            },
            "episode_length": {
                "mean": np.mean(episode_lengths),
                "std": np.std(episode_lengths),
                "min": np.min(episode_lengths),
                "max": np.max(episode_lengths)
            },
            "documents_selected": {
                "mean": np.mean(num_docs_selected),
                "std": np.std(num_docs_selected),
                "min": np.min(num_docs_selected),
                "max": np.max(num_docs_selected)
            }
        }
        
        return analysis
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str):
        """Save analysis results to a file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved analysis to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize environment and runner
    env = RAGEnvironment(max_steps=3, k_candidates=20)
    runner = EpisodeRunner(env)
    
    # Test query
    query = "Who won the FA Cup in 2020?"
    
    # Define policies to test
    policies = [
        PolicyConfig("random", "random"),
        PolicyConfig("greedy", "greedy"),
        PolicyConfig("epsilon_greedy", "epsilon_greedy", epsilon=0.2)
    ]
    
    # Compare policies
    print(f"Comparing policies on query: '{query}'")
    print("=" * 80)
    
    results = runner.compare_policies(query, policies, num_runs=2)
    
    for policy_name, episodes in results.items():
        print(f"\n{policy_name.upper()} Policy:")
        analysis = runner.analyze_episodes(episodes)
        print(f"  Episodes: {analysis['num_episodes']}")
        print(f"  Avg Reward: {analysis['total_reward']['mean']:.3f} ± {analysis['total_reward']['std']:.3f}")
        print(f"  Avg Length: {analysis['episode_length']['mean']:.1f} ± {analysis['episode_length']['std']:.1f}")
        print(f"  Avg Docs Selected: {analysis['documents_selected']['mean']:.1f} ± {analysis['documents_selected']['std']:.1f}")
    
    # Save analysis
    all_episodes = [ep for episodes in results.values() for ep in episodes]
    overall_analysis = runner.analyze_episodes(all_episodes)
    runner.save_analysis(overall_analysis, "policy_comparison.json")
