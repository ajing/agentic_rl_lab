"""
Simple smoke test for Week 3-4 components.

Quick verification that all components can be imported and basic functionality works.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_imports():
    """Test that all Week 3-4 components can be imported."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test reward components
        from src.reward.llm_judge import LLMJudge, AnswerPair, PreferenceResult
        logger.info("✅ LLM Judge imports successful")
        
        from src.reward.reward_model import LightweightRewardModel, RewardExample, RewardModelTrainer
        logger.info("✅ Reward Model imports successful")
        
        from src.reward.reward_shaping import RewardShaper, RewardConfig, RewardComponents
        logger.info("✅ Reward Shaping imports successful")
        
        # Test data components
        from src.data.preference_dataset import PreferenceDatasetBuilder, PreferenceExample
        logger.info("✅ Preference Dataset imports successful")
        
        # Test training components
        from src.training.dpo_trainer import DPOTrainerWrapper, TrainingConfig
        logger.info("✅ DPO Trainer imports successful")
        
        from src.training.expert_trajectories import ExpertTrajectoryGenerator, ExpertConfig
        logger.info("✅ Expert Trajectories imports successful")
        
        from src.training.raft_trainer import RAFTTrainer, RAFTConfig
        logger.info("✅ RAFT Trainer imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of components."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test reward model initialization
        from src.reward.reward_model import LightweightRewardModel
        model = LightweightRewardModel()
        logger.info("✅ Reward model initialization successful")
        
        # Test reward config
        from src.reward.reward_shaping import RewardConfig
        config = RewardConfig()
        logger.info("✅ Reward config creation successful")
        
        # Test training config
        from src.training.dpo_trainer import TrainingConfig
        train_config = TrainingConfig()
        logger.info("✅ Training config creation successful")
        
        # Test expert config
        from src.training.expert_trajectories import ExpertConfig
        expert_config = ExpertConfig()
        logger.info("✅ Expert config creation successful")
        
        # Test RAFT config
        from src.training.raft_trainer import RAFTConfig
        raft_config = RAFTConfig()
        logger.info("✅ RAFT config creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

def test_reward_model_inference():
    """Test reward model inference."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.reward.reward_model import LightweightRewardModel
        import torch
        
        model = LightweightRewardModel()
        model.eval()
        
        # Test inference
        with torch.no_grad():
            result = model(
                "Who won the FA Cup in 2020?",
                "Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
                ["Arsenal defeated Chelsea in the 2020 FA Cup final."]
            )
            
            reward_score = result["reward_score"].item()
            logger.info(f"✅ Reward model inference successful: {reward_score:.3f}")
            
            # Check that we get criteria scores
            criteria_scores = result["criteria_scores"]
            logger.info(f"✅ Criteria scores generated: {list(criteria_scores.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reward model inference failed: {e}")
        return False

def test_reward_shaping():
    """Test reward shaping functionality."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.reward.reward_shaping import RewardShaper, RewardConfig
        
        config = RewardConfig()
        shaper = RewardShaper(config=config)
        
        # Test episode reset
        shaper.reset_episode()
        logger.info("✅ Reward shaper reset successful")
        
        # Test step reward computation
        step_reward = shaper.compute_step_reward(
            "Who won the FA Cup in 2020?",
            "Arsenal won the FA Cup in 2020.",
            0.8,
            1
        )
        logger.info(f"✅ Step reward computation successful: {step_reward.total_reward:.3f}")
        
        # Test novelty reward
        novelty = shaper.compute_novelty_reward(
            "Chelsea is a football club in London.",
            ["Arsenal is a football club in London."]
        )
        logger.info(f"✅ Novelty reward computation successful: {novelty:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reward shaping test failed: {e}")
        return False

def test_llm_judge_structure():
    """Test LLM judge structure (without API calls)."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.reward.llm_judge import AnswerPair, PreferenceResult
        
        # Test AnswerPair creation
        answer_pair = AnswerPair(
            query="Who won the FA Cup in 2020?",
            answer_a="Arsenal won the FA Cup in 2020.",
            answer_b="Chelsea won the FA Cup in 2020.",
            context_a=["Arsenal defeated Chelsea in the final."],
            context_b=["Chelsea defeated Arsenal in the final."]
        )
        logger.info("✅ AnswerPair creation successful")
        
        # Test PreferenceResult creation
        preference_result = PreferenceResult(
            preferred_answer="A",
            confidence=0.8,
            reasoning="Answer A is more accurate based on the context.",
            criteria_scores={"accuracy": {"A": 5, "B": 3}}
        )
        logger.info("✅ PreferenceResult creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM judge structure test failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Week 3-4 component smoke tests...")
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Reward Model Inference", test_reward_model_inference),
        ("Reward Shaping", test_reward_shaping),
        ("LLM Judge Structure", test_llm_judge_structure)
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
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All smoke tests passed! Week 3-4 components are ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
