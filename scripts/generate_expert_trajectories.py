"""
Generate expert trajectories using real CORAL data.

This script focuses specifically on generating high-quality expert trajectories
that can be used for Behavioral Cloning training.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.env.rag_environment import RAGEnvironment, ConversationTurn
from src.policy.episode_runner import EpisodeRunner, PolicyConfig
from src.reward.reward_model import LightweightRewardModel
from src.reward.reward_shaping import RewardShaper, RewardConfig

logger = logging.getLogger(__name__)


class ExpertTrajectoryGenerator:
    """
    Generate expert trajectories using real CORAL conversations.
    
    Uses strong policies to generate high-quality trajectories for BC training.
    """
    
    def __init__(self, 
                 data_dir: str = "data/coral",
                 output_dir: str = "outputs/expert_trajectories"):
        """
        Initialize the expert trajectory generator.
        
        Args:
            data_dir: Directory containing CORAL data
            output_dir: Directory for output trajectories
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rag_env = None
        self.episode_runner = None
        self.reward_model = None
        self.reward_shaper = None
        
        # Data storage
        self.coral_conversations = []
        
        logger.info(f"Initialized ExpertTrajectoryGenerator with data_dir: {self.data_dir}")
    
    def setup_components(self):
        """Setup components for trajectory generation."""
        logger.info("Setting up components...")
        
        # Initialize RAG environment
        self.rag_env = RAGEnvironment(
            corpus_path=str(self.data_dir / "docs.jsonl"),
            bm25_index_path="index/coral_bm25",
            vector_index_path="index/coral_faiss",
            use_query_rewriting=True,
            use_cross_encoder=True,
            use_mmr=True,
            max_steps=5,
            k_candidates=100
        )
        
        # Initialize episode runner
        self.episode_runner = EpisodeRunner(self.rag_env)
        
        # Initialize reward model
        self.reward_model = LightweightRewardModel()
        
        # Initialize reward shaper
        reward_config = RewardConfig(
            final_weight=1.0,
            novelty_weight=0.3,
            relevance_weight=0.2,
            diversity_weight=0.2,
            coherence_weight=0.1
        )
        self.reward_shaper = RewardShaper(
            reward_model=self.reward_model,
            config=reward_config
        )
        
        logger.info("Components setup complete")
    
    def load_coral_conversations(self, max_conversations: int = 1000):
        """Load CORAL conversations from training data."""
        logger.info(f"Loading up to {max_conversations} CORAL conversations...")
        
        # Load training conversations
        train_conversations_path = self.data_dir / "raw" / "train" / "new_train_conversation.json"
        if not train_conversations_path.exists():
            logger.error(f"Training conversations not found at {train_conversations_path}")
            return
        
        with open(train_conversations_path, 'r', encoding='utf-8') as f:
            conversations_data = json.load(f)
        
        # Convert to our format
        for conv in conversations_data[:max_conversations]:
            conversation_turns = []
            for i, turn in enumerate(conv.get('turns', [])):
                conversation_turns.append(ConversationTurn(
                    turn_id=i,
                    question=turn.get('question', ''),
                    answer=turn.get('answer', None)
                ))
            
            self.coral_conversations.append({
                'conversation_id': conv.get('conversation_id', ''),
                'turns': conversation_turns
            })
        
        logger.info(f"Loaded {len(self.coral_conversations)} conversations")
    
    def extract_queries_from_conversations(self) -> List[Tuple[str, List[ConversationTurn]]]:
        """Extract queries and conversation history from CORAL conversations."""
        logger.info("Extracting queries from conversations...")
        
        queries_with_history = []
        
        for conv in self.coral_conversations:
            if not conv['turns']:
                continue
            
            # Extract user queries and build conversation history
            current_history = []
            
            for i, turn in enumerate(conv['turns']):
                if turn.question:  # This is a user query
                    query = turn.question
                    
                    # Add to queries with history
                    queries_with_history.append((query, current_history.copy()))
                    
                    # Add this turn to history
                    current_history.append(turn)
                elif turn.answer and current_history:  # This is an assistant response
                    current_history.append(turn)
        
        logger.info(f"Extracted {len(queries_with_history)} queries with history")
        return queries_with_history
    
    def generate_trajectory_for_query(self, 
                                    query: str, 
                                    history: List[ConversationTurn],
                                    policy: PolicyConfig) -> Dict[str, Any]:
        """Generate a single expert trajectory for a query."""
        try:
            # Reset reward shaper for new episode
            self.reward_shaper.reset_episode()
            
            # Run episode
            episode_result = self.episode_runner.run_episode(query, policy, history)
            
            # Extract trajectory information
            trajectory = {
                'query': query,
                'conversation_history': [{'question': t.question, 'answer': t.answer, 'turn_id': t.turn_id} for t in history],
                'episode_result': {
                    'selected_doc_ids': episode_result.selected_doc_ids,
                    'total_reward': episode_result.total_reward,
                    'episode_length': episode_result.final_state.current_step,
                    'rewritten_query': episode_result.rewritten_query
                },
                'policy_used': {
                    'name': policy.name,
                    'policy_type': policy.policy_type,
                    'epsilon': getattr(policy, 'epsilon', None)
                },
                'documents_selected': []
            }
            
            # Add document information
            for doc_id in episode_result.selected_doc_ids:
                # Find document content
                doc_content = ""
                for doc in episode_result.final_state.candidate_pool:
                    if doc.doc_id == doc_id:
                        doc_content = doc.content
                        break
                
                trajectory['documents_selected'].append({
                    'doc_id': doc_id,
                    'content': doc_content[:500] + "..." if len(doc_content) > 500 else doc_content
                })
            
            return trajectory
            
        except Exception as e:
            logger.warning(f"Error generating trajectory for query '{query[:50]}...': {e}")
            return None
    
    def generate_expert_trajectories(self, 
                                   num_trajectories: int = 500,
                                   use_multiple_policies: bool = True) -> List[Dict[str, Any]]:
        """Generate expert trajectories using strong policies."""
        logger.info(f"Generating {num_trajectories} expert trajectories...")
        
        # Extract queries
        queries_with_history = self.extract_queries_from_conversations()
        
        if not queries_with_history:
            logger.error("No queries extracted from conversations")
            return []
        
        # Define expert policies
        if use_multiple_policies:
            expert_policies = [
                PolicyConfig(name="greedy_policy", policy_type="greedy"),
                PolicyConfig(name="epsilon_greedy_0.1", policy_type="epsilon_greedy", epsilon=0.1),
                PolicyConfig(name="epsilon_greedy_0.2", policy_type="epsilon_greedy", epsilon=0.2)
            ]
        else:
            expert_policies = [
                PolicyConfig(name="greedy_policy", policy_type="greedy")
            ]
        
        # Generate trajectories
        trajectories = []
        queries_used = set()
        
        with tqdm(total=num_trajectories, desc="Generating trajectories") as pbar:
            while len(trajectories) < num_trajectories and len(queries_used) < len(queries_with_history):
                # Select random query
                query, history = random.choice(queries_with_history)
                query_key = f"{query}|{len(history)}"
                
                if query_key in queries_used:
                    continue
                
                queries_used.add(query_key)
                
                # Select random policy
                policy = random.choice(expert_policies)
                
                # Generate trajectory
                trajectory = self.generate_trajectory_for_query(query, history, policy)
                
                if trajectory is not None:
                    trajectories.append(trajectory)
                    pbar.update(1)
        
        logger.info(f"Generated {len(trajectories)} expert trajectories")
        return trajectories
    
    def filter_high_quality_trajectories(self, 
                                       trajectories: List[Dict[str, Any]],
                                       min_reward: float = 0.3,
                                       min_documents: int = 1,
                                       max_documents: int = 5) -> List[Dict[str, Any]]:
        """Filter trajectories based on quality criteria."""
        logger.info(f"Filtering trajectories with min_reward={min_reward}, "
                   f"min_docs={min_documents}, max_docs={max_documents}")
        
        filtered_trajectories = []
        
        for traj in trajectories:
            episode_result = traj['episode_result']
            
            # Check reward threshold
            if episode_result['total_reward'] < min_reward:
                continue
            
            # Check document count
            num_docs = len(episode_result['selected_doc_ids'])
            if num_docs < min_documents or num_docs > max_documents:
                continue
            
            # Check episode length
            if episode_result['episode_length'] < 1:
                continue
            
            filtered_trajectories.append(traj)
        
        logger.info(f"Filtered to {len(filtered_trajectories)} high-quality trajectories "
                   f"(from {len(trajectories)} total)")
        return filtered_trajectories
    
    def save_trajectories(self, trajectories: List[Dict[str, Any]], filename: str = "expert_trajectories.json"):
        """Save trajectories to file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")
        
        # Also save a summary
        summary = {
            'total_trajectories': len(trajectories),
            'avg_reward': np.mean([t['episode_result']['total_reward'] for t in trajectories]),
            'avg_documents': np.mean([len(t['episode_result']['selected_doc_ids']) for t in trajectories]),
            'avg_episode_length': np.mean([t['episode_result']['episode_length'] for t in trajectories]),
            'policy_distribution': {}
        }
        
        # Calculate policy distribution
        for traj in trajectories:
            policy_key = traj['policy_used']['name']
            summary['policy_distribution'][policy_key] = summary['policy_distribution'].get(policy_key, 0) + 1
        
        with open(self.output_dir / "trajectory_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved trajectory summary to {self.output_dir / 'trajectory_summary.json'}")
        return output_path
    
    def run_trajectory_generation(self, 
                                max_conversations: int = 1000,
                                num_trajectories: int = 500,
                                min_reward: float = 0.3,
                                use_multiple_policies: bool = True) -> str:
        """Run the complete trajectory generation process."""
        logger.info("Starting expert trajectory generation...")
        
        # Setup components
        self.setup_components()
        
        # Load conversations
        self.load_coral_conversations(max_conversations)
        
        # Generate trajectories
        trajectories = self.generate_expert_trajectories(num_trajectories, use_multiple_policies)
        
        # Filter high-quality trajectories
        filtered_trajectories = self.filter_high_quality_trajectories(
            trajectories, min_reward=min_reward
        )
        
        # Save trajectories
        output_path = self.save_trajectories(filtered_trajectories)
        
        logger.info("Expert trajectory generation completed!")
        return str(output_path)


def main():
    """Run expert trajectory generation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting expert trajectory generation...")
    
    # Initialize generator
    generator = ExpertTrajectoryGenerator(
        data_dir="data/coral",
        output_dir="outputs/expert_trajectories"
    )
    
    # Run generation
    output_path = generator.run_trajectory_generation(
        max_conversations=500,
        num_trajectories=300,
        min_reward=0.2,
        use_multiple_policies=True
    )
    
    logger.info(f"Expert trajectories saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    output_path = main()
    print(f"\nExpert trajectories generated: {output_path}")
