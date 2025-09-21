"""
Core functionality test for Week 3-4 components.

Simple test focusing on the most important components.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def test_core_components():
    """Test core Week 3-4 components."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test reward model
        from src.reward.reward_model import LightweightRewardModel
        import torch
        
        model = LightweightRewardModel()
        model.eval()
        
        with torch.no_grad():
            result = model(
                "Who won the FA Cup in 2020?",
                "Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
                ["Arsenal defeated Chelsea in the 2020 FA Cup final."]
            )
            
            reward_score = result["reward_score"].item()
            logger.info(f"✅ Reward model working: {reward_score:.3f}")
        
        # Test reward shaping
        from src.reward.reward_shaping import RewardShaper, RewardConfig
        
        config = RewardConfig()
        shaper = RewardShaper(config=config)
        shaper.reset_episode()
        
        step_reward = shaper.compute_step_reward(
            "Who won the FA Cup in 2020?",
            "Arsenal won the FA Cup in 2020.",
            0.8,
            1
        )
        logger.info(f"✅ Reward shaping working: {step_reward.total_reward:.3f}")
        
        # Test LLM judge structure
        from src.reward.llm_judge import AnswerPair, PreferenceResult
        
        answer_pair = AnswerPair(
            query="Who won the FA Cup in 2020?",
            answer_a="Arsenal won the FA Cup in 2020.",
            answer_b="Chelsea won the FA Cup in 2020.",
            context_a=["Arsenal defeated Chelsea in the final."],
            context_b=["Chelsea defeated Arsenal in the final."]
        )
        logger.info("✅ LLM judge structure working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Core components test failed: {e}")
        return False

def main():
    """Run core component test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Testing core Week 3-4 components...")
    
    success = test_core_components()
    
    if success:
        logger.info("🎉 Core components are working!")
    else:
        logger.error("❌ Core components have issues.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
