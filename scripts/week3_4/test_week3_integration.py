"""
Integration test for Week 3-4 components with Week 2 infrastructure.

Tests the new components working together with the existing RAG environment.
"""

import logging
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def test_week3_with_week2():
    """Test Week 3 components integrated with Week 2 infrastructure."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import Week 2 components
        from src.env.rag_environment import RAGEnvironment, ConversationTurn
        from src.policy.episode_runner import EpisodeRunner, PolicyConfig
        
        # Import Week 3 components
        from src.reward.reward_model import LightweightRewardModel
        from src.reward.reward_shaping import RewardShaper, RewardConfig
        from src.data.preference_dataset import PreferenceDatasetBuilder
        
        logger.info("✅ All imports successful")
        
        # Initialize Week 2 components
        rag_env = RAGEnvironment(
            corpus_path="data/coral/docs.jsonl",
            bm25_index_path="index/coral_bm25", 
            vector_index_path="index/coral_faiss",
            use_query_rewriter=False,  # Disable for faster testing
            use_cross_encoder=True,
            use_mmr=True,
            max_steps=3,  # Shorter for testing
            k_candidates=20
        )
        
        episode_runner = EpisodeRunner(rag_env)
        logger.info("✅ Week 2 components initialized")
        
        # Initialize Week 3 components
        reward_model = LightweightRewardModel()
        reward_config = RewardConfig()
        reward_shaper = RewardShaper(reward_model=reward_model, config=reward_config)
        
        logger.info("✅ Week 3 components initialized")
        
        # Test integration with realistic conversational scenarios
        realistic_scenarios = [
            {
                "query": "I'm planning a trip to Japan next month. What should I know about the weather and cultural etiquette?",
                "history": [],
                "expected_topics": ["weather", "cultural etiquette", "travel tips"]
            },
            {
                "query": "What are the main differences between machine learning and deep learning?",
                "history": [],
                "expected_topics": ["machine learning", "deep learning", "neural networks"]
            },
            {
                "query": "How does climate change affect global food security?",
                "history": [],
                "expected_topics": ["climate change", "food security", "agriculture"]
            }
        ]
        
        logger.info("Testing integration with realistic conversational scenarios...")
        
        for i, scenario in enumerate(realistic_scenarios):
            test_query = scenario["query"]
            test_history = scenario["history"]
        
            # Run episode
            episode_result = episode_runner.run_episode(
                test_query,
                PolicyConfig(policy_type="greedy", selection_strategy="top_score"),
                test_history
            )
            
            logger.info(f"  Scenario {i+1}: Episode completed with {len(episode_result.selected_doc_ids)} documents selected")
            
            # Test reward shaping
            reward_shaper.reset_episode()
            
            # Simulate step-wise rewards with realistic document content
            realistic_docs = [
                f"Document about {scenario['expected_topics'][0]} for query: {test_query[:50]}...",
                f"Additional information on {scenario['expected_topics'][1] if len(scenario['expected_topics']) > 1 else 'related topics'}",
                f"Supporting details for {scenario['expected_topics'][2] if len(scenario['expected_topics']) > 2 else 'context'}"
            ]
            
            for j, doc_content in enumerate(realistic_docs[:len(episode_result.selected_doc_ids)]):
                step_reward = reward_shaper.compute_step_reward(
                    test_query, doc_content, 0.8 - j * 0.1, j+1
                )
                logger.info(f"    Step {j+1} reward: {step_reward.total_reward:.3f} "
                           f"(Novelty: {step_reward.novelty_reward:.3f}, "
                           f"Relevance: {step_reward.relevance_reward:.3f})")
            
            # Test final reward with realistic answer
            realistic_answers = [
                "For Japan in October, expect mild temperatures around 15-20°C with occasional rain. Cultural etiquette includes bowing when greeting, removing shoes indoors, and being quiet on public transport.",
                "Machine learning is a broader field that includes various algorithms for learning from data, while deep learning is a subset that uses neural networks with multiple layers for complex pattern recognition.",
                "Climate change impacts food security through extreme weather events, changing precipitation patterns, and rising temperatures that affect crop yields and agricultural productivity."
            ]
            
            final_answer = realistic_answers[i] if i < len(realistic_answers) else "Comprehensive answer covering the main aspects of the query."
            final_reward = reward_shaper.compute_episode_reward(test_query, final_answer)
            logger.info(f"    Final episode reward: {final_reward.total_reward:.3f}")
            
            # Test reward model scoring
            context = realistic_docs[:len(episode_result.selected_doc_ids)]
            with reward_model.eval():
                import torch
                with torch.no_grad():
                    result = reward_model(test_query, final_answer, context)
                    reward_score = result["reward_score"].item()
                    criteria_scores = {k: v.item() for k, v in result["criteria_scores"].items()}
                    logger.info(f"    Reward model score: {reward_score:.3f} "
                               f"(Accuracy: {criteria_scores['accuracy']:.3f}, "
                               f"Completeness: {criteria_scores['completeness']:.3f})")
        
        logger.info("✅ Integration test completed with realistic scenarios")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def test_preference_dataset_builder():
    """Test preference dataset builder with mock data."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.data.preference_dataset import PreferenceDatasetBuilder, PreferenceExample
        from src.reward.llm_judge import AnswerPair
        
        # Create realistic preference examples
        preferences = [
            PreferenceExample(
                query="I'm planning a trip to Japan next month. What should I know about the weather and cultural etiquette?",
                chosen_answer="For Japan in October, expect mild temperatures around 15-20°C with occasional rain. Pack layers and an umbrella. Cultural etiquette includes bowing when greeting, removing shoes indoors, and being quiet on public transport. Avoid pointing with your finger and don't eat while walking.",
                rejected_answer="Japan is a country in Asia. The weather changes with seasons. People there have different customs.",
                chosen_context=[
                    "Japan has four distinct seasons with October being part of autumn, featuring mild weather and beautiful fall foliage.",
                    "Japanese culture emphasizes respect, harmony, and proper etiquette in social interactions.",
                    "Weather in Japan varies by region, with Tokyo experiencing mild autumn temperatures in October."
                ],
                rejected_context=[
                    "Japan is an island nation in East Asia with a rich cultural heritage.",
                    "The country experiences seasonal weather patterns typical of temperate climates."
                ],
                preference_score=0.9,
                reasoning="The chosen answer provides specific, actionable information about weather and detailed cultural etiquette guidelines, while the rejected answer is too vague and generic."
            ),
            PreferenceExample(
                query="What are the main differences between machine learning and deep learning?",
                chosen_answer="Machine learning is a broader field that includes various algorithms for learning from data, while deep learning is a subset that uses neural networks with multiple layers. Deep learning excels at complex pattern recognition in unstructured data like images and text, while traditional ML works well with structured data and is often more interpretable.",
                rejected_answer="Machine learning and deep learning are both types of artificial intelligence that help computers learn from data.",
                chosen_context=[
                    "Machine learning encompasses algorithms that learn patterns from data without explicit programming.",
                    "Deep learning uses artificial neural networks with multiple hidden layers to model complex relationships.",
                    "Traditional ML algorithms include decision trees, SVM, and linear regression, while deep learning includes CNNs, RNNs, and transformers."
                ],
                rejected_context=[
                    "Artificial intelligence includes various approaches to making computers intelligent.",
                    "Both machine learning and deep learning are important technologies in AI."
                ],
                preference_score=0.85,
                reasoning="The chosen answer clearly distinguishes between the two concepts with specific examples and use cases, while the rejected answer is too simplistic and doesn't explain the differences."
            )
        ]
        
        logger.info("✅ Preference examples created")
        
        # Test AnswerPair creation
        answer_pair = AnswerPair(
            query="Who won the FA Cup in 2020?",
            answer_a="Arsenal won the FA Cup in 2020.",
            answer_b="Chelsea won the FA Cup in 2020.",
            context_a=["Arsenal defeated Chelsea in the final."],
            context_b=["Chelsea defeated Arsenal in the final."]
        )
        
        logger.info("✅ AnswerPair creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Preference dataset test failed: {e}")
        return False

def test_training_components():
    """Test training components initialization."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.training.dpo_trainer import DPOTrainerWrapper, TrainingConfig
        from src.training.expert_trajectories import ExpertTrajectoryGenerator, ExpertConfig
        from src.training.raft_trainer import RAFTTrainer, RAFTConfig
        
        # Test DPO trainer config
        dpo_config = TrainingConfig(
            model_name="microsoft/DialoGPT-small",  # Small model for testing
            batch_size=2,
            num_epochs=1,
            use_qlora=False  # Disable for testing
        )
        logger.info("✅ DPO training config created")
        
        # Test expert config
        expert_config = ExpertConfig(
            expert_policy="greedy",
            min_reward_threshold=0.3,
            max_trajectories_per_query=2,
            use_llm_judge=False
        )
        logger.info("✅ Expert config created")
        
        # Test RAFT config
        raft_config = RAFTConfig(
            num_candidates_per_query=4,
            top_k_for_training=2,
            reward_threshold=0.3
        )
        logger.info("✅ RAFT config created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training components test failed: {e}")
        return False

def main():
    """Run integration tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Week 3-4 integration tests...")
    
    tests = [
        ("Week 3 + Week 2 Integration", test_week3_with_week2),
        ("Preference Dataset Builder", test_preference_dataset_builder),
        ("Training Components", test_training_components)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "CRASH"
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! Week 3-4 components integrate well with Week 2.")
    else:
        logger.warning("⚠️  Some integration tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
