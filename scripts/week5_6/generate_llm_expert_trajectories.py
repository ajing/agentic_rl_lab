"""
LLM Expert Trajectory Generator.

This script generates expert trajectories using LLM-based document selection
with robust error handling and fallback mechanisms.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import numpy as np
from tqdm import tqdm

# Add src to path (go up to project root, then to src)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.env.rag_environment import RAGEnvironment, ConversationTurn
from src.policy.episode_runner import EpisodeRunner, PolicyConfig
from src.policy.llm_expert_policy import LLMExpertPolicy, LLMExpertConfig

logger = logging.getLogger(__name__)


class LLMExpertTrajectoryGenerator:
    """
    LLM Expert Trajectory Generator with robust error handling.
    """
    
    def __init__(self, 
                 data_dir: str = "data/coral",
                 output_dir: str = "outputs/llm_expert_trajectories"):
        """Initialize the generator."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rag_env = None
        self.episode_runner = None
        
        logger.info(f"Initialized LLMExpertTrajectoryGenerator")
    
    def setup_components(self):
        """Setup components with basic configuration."""
        try:
            # Initialize RAG environment with basic configuration
            self.rag_env = RAGEnvironment(
                bm25_index_path="index/coral_bm25",
                vector_index_path="index/coral_faiss",
                corpus_path="data/coral/docs.jsonl",
                max_steps=3,  # Shorter episodes
                k_candidates=10,  # Fewer candidates
                use_query_rewriting=False,  # Disable for simplicity
                use_cross_encoder=False,
                use_mmr=False
            )
            
            # Initialize episode runner
            self.episode_runner = EpisodeRunner(
                environment=self.rag_env,
                output_dir=str(self.output_dir)
            )
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def load_sample_queries(self, num_queries: int = 20) -> List[str]:
        """Load a small sample of queries for testing."""
        logger.info(f"Loading {num_queries} sample queries...")
        
        # Load conversations
        conversations_file = self.data_dir / "raw" / "train" / "new_train_conversation.json"
        
        if not conversations_file.exists():
            logger.error(f"Conversations file not found: {conversations_file}")
            return []
        
        with open(conversations_file, 'r', encoding='utf-8') as f:
            conversations_data = json.load(f)
        
        # Extract queries
        queries = []
        for conv_data in conversations_data[:num_queries * 2]:  # Load more to account for filtering
            for turn_data in conv_data.get('turns', []):
                if turn_data.get('question'):
                    queries.append(turn_data['question'])
                    if len(queries) >= num_queries:
                        break
            if len(queries) >= num_queries:
                break
        
        logger.info(f"Loaded {len(queries)} sample queries")
        return queries
    
    def generate_single_trajectory(self, query: str) -> Dict[str, Any]:
        """Generate a single trajectory with robust error handling."""
        try:
            # Create LLM expert policy configuration
            llm_config = LLMExpertConfig(
                model="gpt-4o-mini",
                temperature=0.1,
                k_documents=2,
                max_documents_to_evaluate=5,  # Very limited for cost control
                use_ranking_context=True,
                use_content_preview=True
            )
            
            # Create policy config
            policy_config = PolicyConfig(
                name="llm_expert_simple",
                policy_type="llm_expert"
            )
            policy_config.llm_config = llm_config
            
            # Run episode with no conversation history to avoid format issues
            episode = self.episode_runner.run_episode(
                query=query,
                policy_config=policy_config,
                conversation_history=None,  # No history to avoid format issues
                max_steps=3,
                save_episode=False
            )
            
            # Extract trajectory information
            trajectory = {
                'query': query,
                'conversation_history': [],  # Empty for simplicity
                'episode_id': episode.episode_id,
                'total_reward': episode.total_reward,
                'episode_length': len(episode.states),
                'selected_doc_ids': episode.states[-1].selected_doc_ids if episode.states else [],
                'documents_selected': []
            }
            
            # Add document information with actual content
            for doc_id in trajectory['selected_doc_ids']:
                # Get actual document content from the corpus
                doc_content = ""
                if hasattr(self.rag_env.candidate_generator, 'corpus'):
                    doc_content = self.rag_env.candidate_generator.corpus.get(doc_id, "")
                    # Truncate content for readability (first 200 characters)
                    if len(doc_content) > 200:
                        doc_content = doc_content[:200] + "..."
                
                trajectory['documents_selected'].append({
                    'doc_id': doc_id,
                    'content': doc_content
                })
            
            return trajectory
            
        except Exception as e:
            logger.warning(f"Error generating trajectory for query '{query[:50]}...': {e}")
            return None
    
    def generate_trajectories(self, num_trajectories: int = 10) -> List[Dict[str, Any]]:
        """Generate trajectories with robust error handling."""
        logger.info(f"Generating {num_trajectories} LLM expert trajectories...")
        
        # Load sample queries
        queries = self.load_sample_queries(num_trajectories * 2)  # Load more to account for failures
        
        if not queries:
            logger.error("No queries loaded")
            return []
        
        # Generate trajectories
        trajectories = []
        successful_count = 0
        
        with tqdm(total=num_trajectories, desc="Generating trajectories") as pbar:
            for query in queries:
                if successful_count >= num_trajectories:
                    break
                
                trajectory = self.generate_single_trajectory(query)
                
                if trajectory and trajectory['total_reward'] > 0:
                    trajectories.append(trajectory)
                    successful_count += 1
                    pbar.update(1)
                    
                    logger.info(f"Generated trajectory {successful_count}/{num_trajectories}, "
                               f"reward: {trajectory['total_reward']:.3f}")
                else:
                    logger.debug(f"Skipped trajectory with reward: {trajectory['total_reward'] if trajectory else 'None'}")
        
        logger.info(f"Generated {len(trajectories)} trajectories successfully")
        return trajectories
    
    def save_trajectories(self, trajectories: List[Dict[str, Any]]) -> str:
        """Save trajectories to file."""
        output_path = self.output_dir / "llm_expert_trajectories.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)
        
        # Create summary
        summary = {
            "total_trajectories": len(trajectories),
            "avg_reward": np.mean([t['total_reward'] for t in trajectories]) if trajectories else 0,
            "avg_episode_length": np.mean([t['episode_length'] for t in trajectories]) if trajectories else 0,
            "avg_documents_selected": np.mean([len(t['selected_doc_ids']) for t in trajectories]) if trajectories else 0,
            "reward_distribution": {
                "min": min([t['total_reward'] for t in trajectories]) if trajectories else 0,
                "max": max([t['total_reward'] for t in trajectories]) if trajectories else 0,
                "std": np.std([t['total_reward'] for t in trajectories]) if trajectories else 0
            }
        }
        
        summary_path = self.output_dir / "llm_expert_trajectory_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")
        logger.info(f"Summary saved to {summary_path}")
        
        return str(output_path)
    
    def run_generation(self, num_trajectories: int = 10) -> str:
        """Run the complete generation process."""
        logger.info("Starting simple LLM expert trajectory generation...")
        
        # Setup components
        self.setup_components()
        
        # Generate trajectories
        trajectories = self.generate_trajectories(num_trajectories)
        
        # Save results
        output_path = self.save_trajectories(trajectories)
        
        logger.info(f"LLM expert trajectory generation completed. Output: {output_path}")
        return output_path


def main():
    """Run LLM expert trajectory generation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LLM expert trajectory generation...")
    
    # Initialize generator
    generator = LLMExpertTrajectoryGenerator(
        data_dir="data/coral",
        output_dir="outputs/llm_expert_trajectories"
    )
    
    # Run generation with larger batch for training
    output_path = generator.run_generation(num_trajectories=100)  # Good size for BC training
    
    logger.info(f"LLM expert trajectories saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    output_path = main()
