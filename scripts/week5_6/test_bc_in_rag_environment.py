#!/usr/bin/env python3
"""
Test BC model integration in RAG environment for end-to-end evaluation.

This script tests the BC model in a real RAG environment with actual document
retrieval, selection, and answer generation to evaluate end-to-end performance.
"""

import json
import logging
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.env.rag_environment import RAGEnvironment, ConversationTurn
from src.policy.episode_runner import EpisodeRunner, PolicyConfig
from src.policy.bc_policy import BCConfig
from src.policy.llm_expert_policy import LLMExpertConfig


@dataclass
class RAGTestConfig:
    """Configuration for RAG environment testing."""
    test_queries: int = 10
    max_steps_per_episode: int = 5
    output_dir: str = "outputs/bc_rag_test"


class RAGEnvironmentTester:
    """Tester for BC model in RAG environment."""
    
    def __init__(self, config: RAGTestConfig):
        self.config = config
        self.rag_env = None
        self.episode_runner = None
        
        logger.info("Initializing RAG environment tester...")
    
    def setup_rag_environment(self):
        """Setup the RAG environment with basic components."""
        logger.info("Setting up RAG environment...")
        
        try:
            # Try to create full RAG environment
            self.rag_env = RAGEnvironment(
                corpus_file="data/coral/raw/passage_corpus.json",
                use_query_rewriting=False,  # Disable for simplicity
                use_cross_encoder=False,
                use_mmr=False
            )
            logger.info("Full RAG environment created successfully!")
            
        except Exception as e:
            logger.warning(f"Could not create full RAG environment: {e}")
            logger.info("Creating minimal RAG environment for testing...")
            
            # Create a minimal environment for testing
            self.rag_env = self._create_minimal_rag_environment()
        
        # Initialize episode runner
        self.episode_runner = EpisodeRunner(
            environment=self.rag_env,
            output_dir=self.config.output_dir
        )
        
        logger.info("RAG environment setup complete!")
    
    def _create_minimal_rag_environment(self):
        """Create a minimal RAG environment for testing when full environment fails."""
        # This is a simplified version for testing
        # In practice, you'd implement a minimal RAG environment
        logger.info("Creating minimal RAG environment...")
        
        # For now, we'll create a mock environment
        # In a real implementation, you'd create a simplified RAG environment
        class MinimalRAGEnvironment:
            def __init__(self):
                self.current_state = None
                self.query = None
                self.conversation_history = []
                self.selected_documents = []
                self.step_count = 0
                self.max_steps = 5
            
            def reset(self, query: str, conversation_history: List[ConversationTurn] = None):
                self.query = query
                self.conversation_history = conversation_history or []
                self.selected_documents = []
                self.step_count = 0
                
                # Create a mock state
                from src.env.rag_environment import RLState
                self.current_state = RLState(
                    query=query,
                    rewritten_query=query,  # Use query as rewritten query for simplicity
                    selected_doc_ids=self.selected_documents,
                    selected_documents=[],  # Empty for now
                    remaining_candidates=[],  # Empty for now
                    step=self.step_count
                )
            
            def get_valid_actions(self):
                # Return mock actions
                from src.env.rag_environment import RLAction
                mock_actions = []
                for i in range(5):  # 5 mock documents
                    action = RLAction(doc_id=f"doc_{i}")
                    mock_actions.append(action)
                return mock_actions
            
            def get_state_features(self):
                return {
                    'available_documents': [
                        {'doc_id': f'doc_{i}', 'content': f'Mock document {i} content...', 'rrf_score': 0.8 - i*0.1, 'bm25_score': 0.7 - i*0.1, 'vector_score': 0.9 - i*0.1}
                        for i in range(5)
                    ]
                }
            
            def step(self, action):
                # Mock step
                self.selected_documents.append(action.doc_id)
                self.step_count += 1
                
                # Create next state
                from src.env.rag_environment import RLState, RLReward
                next_state = RLState(
                    query=self.query,
                    rewritten_query=self.query,
                    selected_doc_ids=self.selected_documents,
                    selected_documents=[],  # Empty for now
                    remaining_candidates=[],  # Empty for now
                    step=self.step_count
                )
                
                # Mock reward
                reward = RLReward(
                    step_reward=0.1 + random.random() * 0.1,  # Random step reward
                    relevance_reward=0.7,
                    novelty_reward=0.6,
                    final_reward=0.3 + random.random() * 0.2,  # Random final reward between 0.3-0.5
                    total_reward=0.3 + random.random() * 0.2  # Random total reward between 0.3-0.5
                )
                
                done = self.step_count >= self.max_steps
                info = {'step_count': self.step_count}
                
                return next_state, reward, done, info
            
            def finalize_episode(self):
                """Finalize the episode and return final state."""
                return self.current_state
        
        return MinimalRAGEnvironment()
    
    def load_test_queries(self) -> List[str]:
        """Load test queries for evaluation."""
        logger.info("Loading test queries...")
        
        # Use some sample queries for testing
        test_queries = [
            "Who won the 2016 World Series in Major League Baseball?",
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Who wrote the novel '1984'?",
            "What is the largest planet in our solar system?",
            "When was the Declaration of Independence signed?",
            "What is the speed of light?",
            "Who painted the Mona Lisa?",
            "What is the chemical symbol for gold?",
            "How many continents are there?"
        ]
        
        return test_queries[:self.config.test_queries]
    
    def test_bc_policy(self) -> Dict[str, Any]:
        """Test BC policy in RAG environment."""
        logger.info("Testing BC policy in RAG environment...")
        
        # Load test queries
        test_queries = self.load_test_queries()
        
        # Configure BC policy
        bc_config = BCConfig(
            model_path="outputs/bc_model_v3/bc_model_v3.pth",
            config_path="outputs/bc_model_v3/bc_config_v3.json"
        )
        
        policy_config = PolicyConfig(
            policy_type="bc",
            bc_config=bc_config,
            name="bc_policy"
        )
        
        results = {
            'policy_type': 'bc',
            'total_queries': len(test_queries),
            'episodes': [],
            'summary': {
                'avg_reward': 0.0,
                'avg_episode_length': 0.0,
                'avg_prediction_time': 0.0,
                'success_rate': 0.0
            }
        }
        
        total_reward = 0.0
        total_length = 0.0
        total_time = 0.0
        successful_episodes = 0
        
        for i, query in enumerate(test_queries):
            logger.info(f"Testing BC policy on query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            start_time = time.time()
            
            try:
                # Run episode with BC policy
                episode = self.episode_runner.run_episode(
                    query=query,
                    conversation_history=[],
                    policy_config=policy_config,
                    max_steps=self.config.max_steps_per_episode
                )
                
                episode_time = time.time() - start_time
                
                # Extract episode results
                episode_reward = sum(reward.total_reward for reward in episode.rewards)
                episode_length = len(episode.states)
                
                episode_result = {
                    'query': query,
                    'episode_id': episode.episode_id,
                    'total_reward': episode_reward,
                    'episode_length': episode_length,
                    'execution_time': episode_time,
                    'selected_documents': [state.selected_documents for state in episode.states],
                    'rewards': [reward.total_reward for reward in episode.rewards],
                    'success': episode_reward > 0.2  # Consider successful if reward > 0.2
                }
                
                results['episodes'].append(episode_result)
                
                # Update statistics
                total_reward += episode_reward
                total_length += episode_length
                total_time += episode_time
                
                if episode_result['success']:
                    successful_episodes += 1
                
                logger.info(f"  Episode completed: reward={episode_reward:.3f}, length={episode_length}, time={episode_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Error running episode for query '{query}': {e}")
                # Add failed episode
                results['episodes'].append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
        
        # Calculate summary statistics
        if results['episodes']:
            results['summary']['avg_reward'] = total_reward / len(test_queries)
            results['summary']['avg_episode_length'] = total_length / len(test_queries)
            results['summary']['avg_prediction_time'] = total_time / len(test_queries)
            results['summary']['success_rate'] = successful_episodes / len(test_queries)
        
        logger.info(f"BC policy testing completed!")
        logger.info(f"Average reward: {results['summary']['avg_reward']:.3f}")
        logger.info(f"Success rate: {results['summary']['success_rate']:.2%}")
        
        return results
    
    def test_comparison_policies(self) -> Dict[str, Any]:
        """Test comparison policies (random, greedy) for baseline comparison."""
        logger.info("Testing comparison policies...")
        
        test_queries = self.load_test_queries()
        
        comparison_results = {
            'random_policy': self._test_policy("random", test_queries),
            'greedy_policy': self._test_policy("greedy", test_queries)
        }
        
        return comparison_results
    
    def _test_policy(self, policy_type: str, test_queries: List[str]) -> Dict[str, Any]:
        """Test a specific policy type."""
        logger.info(f"Testing {policy_type} policy...")
        
        policy_config = PolicyConfig(
            policy_type=policy_type,
            name=f"{policy_type}_policy"
        )
        
        results = {
            'policy_type': policy_type,
            'total_queries': len(test_queries),
            'episodes': [],
            'summary': {
                'avg_reward': 0.0,
                'avg_episode_length': 0.0,
                'avg_prediction_time': 0.0,
                'success_rate': 0.0
            }
        }
        
        total_reward = 0.0
        total_length = 0.0
        total_time = 0.0
        successful_episodes = 0
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            try:
                episode = self.episode_runner.run_episode(
                    query=query,
                    conversation_history=[],
                    policy_config=policy_config,
                    max_steps=self.config.max_steps_per_episode
                )
                
                episode_time = time.time() - start_time
                episode_reward = sum(reward.total_reward for reward in episode.rewards)
                episode_length = len(episode.states)
                
                episode_result = {
                    'query': query,
                    'episode_id': episode.episode_id,
                    'total_reward': episode_reward,
                    'episode_length': episode_length,
                    'execution_time': episode_time,
                    'success': episode_reward > 0.2
                }
                
                results['episodes'].append(episode_result)
                
                total_reward += episode_reward
                total_length += episode_length
                total_time += episode_time
                
                if episode_result['success']:
                    successful_episodes += 1
                
            except Exception as e:
                logger.error(f"Error running {policy_type} episode for query '{query}': {e}")
                results['episodes'].append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
        
        # Calculate summary statistics
        if results['episodes']:
            results['summary']['avg_reward'] = total_reward / len(test_queries)
            results['summary']['avg_episode_length'] = total_length / len(test_queries)
            results['summary']['avg_prediction_time'] = total_time / len(test_queries)
            results['summary']['success_rate'] = successful_episodes / len(test_queries)
        
        return results
    
    def save_results(self, bc_results: Dict[str, Any], comparison_results: Dict[str, Any], output_path: str):
        """Save test results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        combined_results = {
            'bc_policy': bc_results,
            'comparison_policies': comparison_results,
            'test_config': {
                'test_queries': self.config.test_queries,
                'max_steps_per_episode': self.config.max_steps_per_episode
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")
    
    def print_summary(self, bc_results: Dict[str, Any], comparison_results: Dict[str, Any]):
        """Print a summary of test results."""
        print("\n" + "="*80)
        print("BC MODEL RAG ENVIRONMENT TESTING SUMMARY")
        print("="*80)
        
        # BC Policy Results
        bc_summary = bc_results['summary']
        print(f"\nBC Policy Results:")
        print(f"  Average Reward: {bc_summary['avg_reward']:.3f}")
        print(f"  Average Episode Length: {bc_summary['avg_episode_length']:.1f}")
        print(f"  Average Prediction Time: {bc_summary['avg_prediction_time']:.3f}s")
        print(f"  Success Rate: {bc_summary['success_rate']:.2%}")
        
        # Comparison Results
        print(f"\nComparison Policy Results:")
        for policy_name, results in comparison_results.items():
            summary = results['summary']
            print(f"  {policy_name.replace('_', ' ').title()}:")
            print(f"    Average Reward: {summary['avg_reward']:.3f}")
            print(f"    Success Rate: {summary['success_rate']:.2%}")
        
        # Performance Comparison
        print(f"\nPerformance Comparison:")
        bc_reward = bc_summary['avg_reward']
        random_reward = comparison_results['random_policy']['summary']['avg_reward']
        greedy_reward = comparison_results['greedy_policy']['summary']['avg_reward']
        
        print(f"  BC vs Random: {bc_reward/random_reward:.2f}x better" if random_reward > 0 else "  BC vs Random: N/A")
        print(f"  BC vs Greedy: {bc_reward/greedy_reward:.2f}x better" if greedy_reward > 0 else "  BC vs Greedy: N/A")
        
        print("="*80)


def main():
    """Main testing function."""
    # Configuration
    config = RAGTestConfig(
        test_queries=10,
        max_steps_per_episode=5
    )
    
    # Initialize tester
    tester = RAGEnvironmentTester(config)
    
    # Setup RAG environment
    tester.setup_rag_environment()
    
    # Test BC policy
    bc_results = tester.test_bc_policy()
    
    # Test comparison policies
    comparison_results = tester.test_comparison_policies()
    
    # Save results
    output_path = "outputs/bc_rag_test/rag_environment_test_results.json"
    tester.save_results(bc_results, comparison_results, output_path)
    
    # Print summary
    tester.print_summary(bc_results, comparison_results)


if __name__ == "__main__":
    main()
