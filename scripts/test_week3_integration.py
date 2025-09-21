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
        
        # Test integration: Run episode with reward shaping
        test_query = "Who won the FA Cup in 2020?"
        test_history = []
        
        # Run episode
        episode_result = episode_runner.run_episode(
            test_query,
            PolicyConfig(policy_type="greedy", selection_strategy="top_score"),
            test_history
        )
        
        logger.info(f"✅ Episode completed: {len(episode_result.selected_doc_ids)} documents selected")
        
        # Test reward shaping
        reward_shaper.reset_episode()
        
        # Simulate step-wise rewards
        for i, doc_id in enumerate(episode_result.selected_doc_ids):
            # Find document content
            doc_content = ""
            for doc in episode_result.final_state.candidate_pool:
                if doc.doc_id == doc_id:
                    doc_content = doc.content
                    break
            
            if doc_content:
                step_reward = reward_shaper.compute_step_reward(
                    test_query, doc_content, 0.8, i+1
                )
                logger.info(f"✅ Step {i+1} reward: {step_reward.total_reward:.3f}")
        
        # Test final reward
        final_answer = "Arsenal won the FA Cup in 2020."  # Simplified
        final_reward = reward_shaper.compute_episode_reward(test_query, final_answer)
        logger.info(f"✅ Final episode reward: {final_reward.total_reward:.3f}")
        
        # Test reward model scoring
        context = [doc.content for doc in episode_result.final_state.selected_documents]
        with reward_model.eval():
            import torch
            with torch.no_grad():
                result = reward_model(test_query, final_answer, context)
                reward_score = result["reward_score"].item()
                logger.info(f"✅ Reward model score: {reward_score:.3f}")
        
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
        
        # Create mock preference examples
        preferences = [
            PreferenceExample(
                query="Who won the FA Cup in 2020?",
                chosen_answer="Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
                rejected_answer="The FA Cup is an annual football competition in England.",
                chosen_context=["Arsenal defeated Chelsea in the 2020 FA Cup final."],
                rejected_context=["The FA Cup is England's premier cup competition."],
                preference_score=0.9,
                reasoning="The chosen answer directly answers the question with specific details."
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
