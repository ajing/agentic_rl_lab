"""
Comprehensive evaluation of Week 3-4 components on CORAL dataset.

Tests the complete pipeline:
1. LLM-as-a-Judge preference collection
2. Reward model training and evaluation
3. Reward shaping effectiveness
4. Expert trajectory generation
5. RAFT training pipeline
6. End-to-end performance comparison
"""

import json
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import asdict

# Import our components
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reward.llm_judge import LLMJudge, AnswerPair, PreferenceResult
from src.reward.reward_model import LightweightRewardModel, RewardExample, RewardModelTrainer
from src.reward.reward_shaping import RewardShaper, RewardConfig
from src.data.preference_dataset import PreferenceDatasetBuilder, PreferenceExample
from src.training.expert_trajectories import ExpertTrajectoryGenerator, ExpertConfig
from src.training.raft_trainer import RAFTTrainer, RAFTConfig
from src.env.rag_environment import RAGEnvironment, ConversationTurn
from src.policy.episode_runner import EpisodeRunner, PolicyConfig

logger = logging.getLogger(__name__)


class Week3Evaluator:
    """
    Comprehensive evaluator for Week 3-4 components.
    
    Tests all new components and provides detailed performance analysis.
    """
    
    def __init__(self, 
                 corpus_path: str = "data/coral/docs.jsonl",
                 bm25_index_path: str = "index/coral_bm25",
                 vector_index_path: str = "index/coral_faiss",
                 output_dir: str = "outputs/eval_week3"):
        """
        Initialize the evaluator.
        
        Args:
            corpus_path: Path to CORAL corpus
            bm25_index_path: Path to BM25 index
            vector_index_path: Path to vector index
            output_dir: Output directory for results
        """
        self.corpus_path = Path(corpus_path)
        self.bm25_index_path = Path(bm25_index_path)
        self.vector_index_path = Path(vector_index_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rag_env = None
        self.episode_runner = None
        self.llm_judge = None
        self.reward_model = None
        self.reward_shaper = None
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized Week 3 evaluator with output dir: {self.output_dir}")
    
    def setup_components(self, use_llm_judge: bool = True):
        """Setup all components for evaluation."""
        logger.info("Setting up components...")
        
        # Initialize RAG environment
        self.rag_env = RAGEnvironment(
            corpus_path=str(self.corpus_path),
            bm25_index_path=str(self.bm25_index_path),
            vector_index_path=str(self.vector_index_path),
            use_query_rewriter=False,  # Disable for faster evaluation
            use_cross_encoder=True,
            use_mmr=True,
            max_steps=5,
            k_candidates=50  # Smaller for faster evaluation
        )
        
        # Initialize episode runner
        self.episode_runner = EpisodeRunner(self.rag_env)
        
        # Initialize LLM judge (optional)
        if use_llm_judge:
            try:
                self.llm_judge = LLMJudge()
                logger.info("LLM judge initialized")
            except Exception as e:
                logger.warning(f"Could not initialize LLM judge: {e}")
                self.llm_judge = None
        
        # Initialize reward model
        self.reward_model = LightweightRewardModel()
        logger.info("Reward model initialized")
        
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
        logger.info("Reward shaper initialized")
    
    def load_test_queries(self, num_queries: int = 20) -> List[Tuple[str, List[ConversationTurn]]]:
        """Load test queries from CORAL dataset."""
        # For now, create some test queries
        # In practice, you would load from the actual CORAL test set
        test_queries = [
            ("Who won the FA Cup in 2020?", []),
            ("What is the capital of France?", []),
            ("Tell me about the history of the internet.", []),
            ("What are the benefits of renewable energy?", []),
            ("How does machine learning work?", []),
            ("What is the population of Tokyo?", []),
            ("Explain the theory of relativity.", []),
            ("What are the main causes of climate change?", []),
            ("How do vaccines work?", []),
            ("What is the structure of DNA?", []),
            ("Who wrote Romeo and Juliet?", []),
            ("What is the speed of light?", []),
            ("How do solar panels work?", []),
            ("What is artificial intelligence?", []),
            ("Explain photosynthesis.", []),
            ("What is the largest planet in our solar system?", []),
            ("How does the human brain work?", []),
            ("What is quantum computing?", []),
            ("Explain the water cycle.", []),
            ("What is blockchain technology?", [])
        ]
        
        return test_queries[:num_queries]
    
    def evaluate_llm_judge(self, test_queries: List[Tuple[str, List[ConversationTurn]]]) -> Dict[str, Any]:
        """Evaluate LLM-as-a-Judge component."""
        logger.info("Evaluating LLM-as-a-Judge...")
        
        if not self.llm_judge:
            return {"error": "LLM judge not available"}
        
        # Generate some answer pairs for evaluation
        answer_pairs = []
        for query, history in test_queries[:5]:  # Use subset for evaluation
            # Generate two different answers using different policies
            policy1 = PolicyConfig(policy_type="greedy", selection_strategy="top_score")
            policy2 = PolicyConfig(policy_type="random", selection_strategy="random")
            
            try:
                episode1 = self.episode_runner.run_episode(query, policy1, history)
                episode2 = self.episode_runner.run_episode(query, policy2, history)
                
                # Create answer pair
                answer1 = self._extract_answer_from_episode(episode1)
                answer2 = self._extract_answer_from_episode(episode2)
                
                context1 = [doc.content for doc in episode1.final_state.selected_documents]
                context2 = [doc.content for doc in episode2.final_state.selected_documents]
                
                answer_pair = AnswerPair(
                    query=query,
                    answer_a=answer1,
                    answer_b=answer2,
                    context_a=context1,
                    context_b=context2,
                    metadata={"episode1_reward": episode1.total_reward, "episode2_reward": episode2.total_reward}
                )
                
                answer_pairs.append(answer_pair)
                
            except Exception as e:
                logger.warning(f"Error generating answer pair for query '{query}': {e}")
                continue
        
        # Evaluate with LLM judge
        results = []
        for i, answer_pair in enumerate(answer_pairs):
            try:
                result = self.llm_judge.compare_answers(answer_pair)
                results.append({
                    "query": answer_pair.query,
                    "preferred_answer": result.preferred_answer,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "criteria_scores": result.criteria_scores
                })
                logger.info(f"Judge evaluation {i+1}/{len(answer_pairs)}: {result.preferred_answer} (confidence: {result.confidence:.3f})")
            except Exception as e:
                logger.error(f"Error in judge evaluation: {e}")
                continue
        
        # Compute statistics
        if results:
            confidences = [r["confidence"] for r in results]
            stats = {
                "total_evaluations": len(results),
                "avg_confidence": np.mean(confidences),
                "std_confidence": np.std(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
                "results": results
            }
        else:
            stats = {"error": "No successful evaluations"}
        
        return stats
    
    def evaluate_reward_model(self, test_queries: List[Tuple[str, List[ConversationTurn]]]) -> Dict[str, Any]:
        """Evaluate reward model component."""
        logger.info("Evaluating reward model...")
        
        # Generate training examples
        examples = []
        for query, history in test_queries[:10]:  # Use subset for evaluation
            try:
                episode = self.episode_runner.run_episode(
                    query, 
                    PolicyConfig(policy_type="greedy", selection_strategy="top_score"), 
                    history
                )
                
                answer = self._extract_answer_from_episode(episode)
                context = [doc.content for doc in episode.final_state.selected_documents]
                
                # Score with reward model
                with torch.no_grad():
                    result = self.reward_model(query, answer, context)
                    reward_score = result["reward_score"].item()
                
                example = RewardExample(
                    query=query,
                    answer=answer,
                    context=context,
                    preference_score=reward_score,
                    criteria_scores={},  # Simplified for evaluation
                    metadata={"episode_reward": episode.total_reward}
                )
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error generating example for query '{query}': {e}")
                continue
        
        # Evaluate reward model
        if examples:
            scores = [ex.preference_score for ex in examples]
            episode_rewards = [ex.metadata["episode_reward"] for ex in examples]
            
            # Compute correlation between reward model and episode rewards
            correlation = np.corrcoef(scores, episode_rewards)[0, 1] if len(scores) > 1 else 0.0
            
            stats = {
                "total_examples": len(examples),
                "avg_reward_score": np.mean(scores),
                "std_reward_score": np.std(scores),
                "min_reward_score": np.min(scores),
                "max_reward_score": np.max(scores),
                "correlation_with_episode_rewards": correlation,
                "examples": [asdict(ex) for ex in examples]
            }
        else:
            stats = {"error": "No examples generated"}
        
        return stats
    
    def evaluate_reward_shaping(self, test_queries: List[Tuple[str, List[ConversationTurn]]]) -> Dict[str, Any]:
        """Evaluate reward shaping component."""
        logger.info("Evaluating reward shaping...")
        
        results = []
        for query, history in test_queries[:5]:  # Use subset for evaluation
            try:
                # Reset reward shaper
                self.reward_shaper.reset_episode()
                
                # Run episode with reward shaping
                episode = self.episode_runner.run_episode(
                    query,
                    PolicyConfig(policy_type="greedy", selection_strategy="top_score"),
                    history
                )
                
                # Compute shaped rewards
                answer = self._extract_answer_from_episode(episode)
                shaped_rewards = self.reward_shaper.compute_episode_reward(query, answer)
                
                result = {
                    "query": query,
                    "original_reward": episode.total_reward,
                    "shaped_reward": shaped_rewards.total_reward,
                    "final_reward": shaped_rewards.final_reward,
                    "novelty_reward": shaped_rewards.novelty_reward,
                    "relevance_reward": shaped_rewards.relevance_reward,
                    "diversity_reward": shaped_rewards.diversity_reward,
                    "coherence_reward": shaped_rewards.coherence_reward,
                    "num_documents": len(episode.final_state.selected_documents)
                }
                
                results.append(result)
                logger.info(f"Reward shaping for '{query}': {episode.total_reward:.3f} -> {shaped_rewards.total_reward:.3f}")
                
            except Exception as e:
                logger.warning(f"Error in reward shaping for query '{query}': {e}")
                continue
        
        # Compute statistics
        if results:
            original_rewards = [r["original_reward"] for r in results]
            shaped_rewards = [r["shaped_reward"] for r in results]
            
            stats = {
                "total_episodes": len(results),
                "avg_original_reward": np.mean(original_rewards),
                "avg_shaped_reward": np.mean(shaped_rewards),
                "reward_improvement": np.mean(shaped_rewards) - np.mean(original_rewards),
                "results": results
            }
        else:
            stats = {"error": "No successful evaluations"}
        
        return stats
    
    def evaluate_expert_trajectories(self, test_queries: List[Tuple[str, List[ConversationTurn]]]) -> Dict[str, Any]:
        """Evaluate expert trajectory generation."""
        logger.info("Evaluating expert trajectory generation...")
        
        # Initialize expert generator
        expert_config = ExpertConfig(
            expert_policy="greedy",
            min_reward_threshold=0.3,
            max_trajectories_per_query=2,
            use_llm_judge=False,  # Disable for faster evaluation
            output_dir=str(self.output_dir / "expert_trajectories")
        )
        
        expert_generator = ExpertTrajectoryGenerator(
            self.rag_env,
            self.episode_runner,
            self.reward_shaper,
            self.llm_judge,
            expert_config
        )
        
        # Generate expert trajectories
        try:
            stats = expert_generator.generate_expert_dataset(
                test_queries[:5],  # Use subset for evaluation
                str(self.output_dir / "expert_trajectories.json")
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in expert trajectory generation: {e}")
            return {"error": str(e)}
    
    def evaluate_raft_training(self, test_queries: List[Tuple[str, List[ConversationTurn]]]) -> Dict[str, Any]:
        """Evaluate RAFT training pipeline."""
        logger.info("Evaluating RAFT training...")
        
        # Initialize RAFT trainer
        raft_config = RAFTConfig(
            num_candidates_per_query=4,
            top_k_for_training=2,
            reward_threshold=0.3,
            output_dir=str(self.output_dir / "raft_training")
        )
        
        raft_trainer = RAFTTrainer(
            self.rag_env,
            self.episode_runner,
            self.reward_model,
            self.reward_shaper,
            raft_config
        )
        
        # Build RAFT dataset
        try:
            stats = raft_trainer.build_raft_dataset(
                test_queries[:5],  # Use subset for evaluation
                str(self.output_dir / "raft_dataset.json")
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in RAFT training: {e}")
            return {"error": str(e)}
    
    def _extract_answer_from_episode(self, episode_result) -> str:
        """Extract answer from episode result."""
        selected_docs = episode_result.final_state.selected_documents
        
        if not selected_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Combine document contents
        answer_parts = []
        for doc in selected_docs:
            content = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            answer_parts.append(content)
        
        answer = " ".join(answer_parts)
        
        if len(answer) > 1000:
            answer = answer[:1000] + "..."
        
        return answer
    
    def run_comprehensive_evaluation(self, num_queries: int = 20) -> Dict[str, Any]:
        """Run comprehensive evaluation of all Week 3-4 components."""
        logger.info("Starting comprehensive Week 3-4 evaluation...")
        
        # Setup components
        self.setup_components(use_llm_judge=True)
        
        # Load test queries
        test_queries = self.load_test_queries(num_queries)
        logger.info(f"Loaded {len(test_queries)} test queries")
        
        # Run evaluations
        evaluation_results = {}
        
        # 1. LLM Judge evaluation
        logger.info("=" * 60)
        logger.info("1. LLM-as-a-Judge Evaluation")
        logger.info("=" * 60)
        evaluation_results["llm_judge"] = self.evaluate_llm_judge(test_queries)
        
        # 2. Reward Model evaluation
        logger.info("=" * 60)
        logger.info("2. Reward Model Evaluation")
        logger.info("=" * 60)
        evaluation_results["reward_model"] = self.evaluate_reward_model(test_queries)
        
        # 3. Reward Shaping evaluation
        logger.info("=" * 60)
        logger.info("3. Reward Shaping Evaluation")
        logger.info("=" * 60)
        evaluation_results["reward_shaping"] = self.evaluate_reward_shaping(test_queries)
        
        # 4. Expert Trajectories evaluation
        logger.info("=" * 60)
        logger.info("4. Expert Trajectories Evaluation")
        logger.info("=" * 60)
        evaluation_results["expert_trajectories"] = self.evaluate_expert_trajectories(test_queries)
        
        # 5. RAFT Training evaluation
        logger.info("=" * 60)
        logger.info("5. RAFT Training Evaluation")
        logger.info("=" * 60)
        evaluation_results["raft_training"] = self.evaluate_raft_training(test_queries)
        
        # Save results
        results_file = self.output_dir / "week3_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        # Generate summary
        summary = self._generate_evaluation_summary(evaluation_results)
        
        return {
            "evaluation_results": evaluation_results,
            "summary": summary,
            "results_file": str(results_file)
        }
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation summary."""
        summary = {
            "total_components_evaluated": len(results),
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "component_summaries": {}
        }
        
        for component, result in results.items():
            if "error" in result:
                summary["failed_evaluations"] += 1
                summary["component_summaries"][component] = {"status": "failed", "error": result["error"]}
            else:
                summary["successful_evaluations"] += 1
                summary["component_summaries"][component] = {"status": "success", "key_metrics": {}}
                
                # Extract key metrics based on component
                if component == "llm_judge" and "avg_confidence" in result:
                    summary["component_summaries"][component]["key_metrics"]["avg_confidence"] = result["avg_confidence"]
                elif component == "reward_model" and "avg_reward_score" in result:
                    summary["component_summaries"][component]["key_metrics"]["avg_reward_score"] = result["avg_reward_score"]
                elif component == "reward_shaping" and "reward_improvement" in result:
                    summary["component_summaries"][component]["key_metrics"]["reward_improvement"] = result["reward_improvement"]
                elif component == "expert_trajectories" and "total_trajectories" in result:
                    summary["component_summaries"][component]["key_metrics"]["total_trajectories"] = result["total_trajectories"]
                elif component == "raft_training" and "total_candidates" in result:
                    summary["component_summaries"][component]["key_metrics"]["total_candidates"] = result["total_candidates"]
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize evaluator
    evaluator = Week3Evaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(num_queries=10)
    
    # Print summary
    print("\n" + "=" * 80)
    print("WEEK 3-4 EVALUATION SUMMARY")
    print("=" * 80)
    
    summary = results["summary"]
    print(f"Total components evaluated: {summary['total_components_evaluated']}")
    print(f"Successful evaluations: {summary['successful_evaluations']}")
    print(f"Failed evaluations: {summary['failed_evaluations']}")
    
    print("\nComponent Results:")
    for component, comp_summary in summary["component_summaries"].items():
        status = comp_summary["status"]
        print(f"  {component}: {status.upper()}")
        if status == "success" and "key_metrics" in comp_summary:
            for metric, value in comp_summary["key_metrics"].items():
                print(f"    {metric}: {value}")
        elif status == "failed":
            print(f"    Error: {comp_summary['error']}")
    
    print(f"\nDetailed results saved to: {results['results_file']}")
