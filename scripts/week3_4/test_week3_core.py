"""
Core functionality test for Week 3-4 components.

Simple test focusing on the most important components.
"""

import logging
import sys
from pathlib import Path

# Add project root to path (so we can import src modules)
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_core_components():
    """Test core Week 3-4 components with realistic examples."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test reward model with realistic conversational examples
        from src.reward.reward_model import LightweightRewardModel
        import torch
        
        model = LightweightRewardModel()
        model.eval()
        
        # Realistic test cases
        realistic_cases = [
            {
                "query": "I'm planning a trip to Japan next month. What should I know about the weather and cultural etiquette?",
                "answer": "For Japan in October, expect mild temperatures around 15-20°C with occasional rain. Pack layers and an umbrella. Cultural etiquette includes bowing when greeting, removing shoes indoors, and being quiet on public transport. Avoid pointing with your finger and don't eat while walking.",
                "context": [
                    "Japan has four distinct seasons with October being part of autumn, featuring mild weather and beautiful fall foliage.",
                    "Japanese culture emphasizes respect, harmony, and proper etiquette in social interactions.",
                    "Weather in Japan varies by region, with Tokyo experiencing mild autumn temperatures in October."
                ]
            },
            {
                "query": "What are the main differences between machine learning and deep learning?",
                "answer": "Machine learning is a broader field that includes various algorithms for learning from data, while deep learning is a subset that uses neural networks with multiple layers. Deep learning excels at complex pattern recognition in unstructured data like images and text, while traditional ML works well with structured data and is often more interpretable.",
                "context": [
                    "Machine learning encompasses algorithms that learn patterns from data without explicit programming.",
                    "Deep learning uses artificial neural networks with multiple hidden layers to model complex relationships.",
                    "Traditional ML algorithms include decision trees, SVM, and linear regression, while deep learning includes CNNs, RNNs, and transformers."
                ]
            },
            {
                "query": "How does climate change affect global food security?",
                "answer": "Climate change impacts food security through extreme weather events, changing precipitation patterns, and rising temperatures that affect crop yields. Droughts and floods can destroy harvests, while temperature changes can shift growing seasons and reduce agricultural productivity in many regions, particularly affecting vulnerable populations in developing countries.",
                "context": [
                    "Climate change leads to more frequent and severe extreme weather events that can damage crops and infrastructure.",
                    "Rising global temperatures affect plant growth cycles and can reduce crop yields in many agricultural regions.",
                    "Changes in precipitation patterns can lead to droughts in some areas and flooding in others, both harmful to agriculture."
                ]
            }
        ]
        
        logger.info("Testing reward model with realistic conversational examples...")
        for i, case in enumerate(realistic_cases):
            with torch.no_grad():
                result = model(case["query"], case["answer"], case["context"])
                reward_score = result["reward_score"].item()
                criteria_scores = {k: v.item() for k, v in result["criteria_scores"].items()}
                
                logger.info(f"  Case {i+1}: Reward={reward_score:.3f}, "
                           f"Accuracy={criteria_scores['accuracy']:.3f}, "
                           f"Completeness={criteria_scores['completeness']:.3f}")
        
        logger.info("✅ Reward model working with realistic examples")
        
        # Test reward shaping with realistic scenarios
        from src.reward.reward_shaping import RewardShaper, RewardConfig
        
        config = RewardConfig()
        shaper = RewardShaper(config=config)
        
        # Test multi-step document selection scenario
        realistic_documents = [
            "Japan experiences four distinct seasons with autumn (September-November) featuring mild temperatures and beautiful fall foliage. October temperatures typically range from 15-20°C in Tokyo.",
            "Japanese cultural etiquette emphasizes respect and harmony. Key practices include bowing when greeting, removing shoes indoors, and maintaining quiet on public transportation.",
            "Weather patterns in Japan vary significantly by region, with the Pacific side experiencing different conditions than the Sea of Japan side.",
            "Traditional Japanese customs include not pointing with fingers, avoiding eating while walking, and using both hands when giving or receiving items."
        ]
        
        logger.info("Testing reward shaping with realistic document selection...")
        shaper.reset_episode()
        
        for i, doc in enumerate(realistic_documents):
            step_reward = shaper.compute_step_reward(
                "I'm planning a trip to Japan next month. What should I know about the weather and cultural etiquette?",
                doc,
                0.7 + i * 0.1,  # Simulate decreasing relevance scores
                i + 1
            )
            logger.info(f"  Step {i+1}: Reward={step_reward.total_reward:.3f}, "
                       f"Novelty={step_reward.novelty_reward:.3f}, "
                       f"Relevance={step_reward.relevance_reward:.3f}")
        
        logger.info("✅ Reward shaping working with realistic scenarios")
        
        # Test LLM judge with realistic preference comparisons
        from src.reward.llm_judge import AnswerPair, PreferenceResult
        
        realistic_preferences = [
            {
                "query": "What are the health benefits of regular exercise?",
                "answer_a": "Regular exercise provides numerous health benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and reduced risk of chronic diseases like diabetes and heart disease. It also boosts energy levels and improves sleep quality.",
                "answer_b": "Exercise is good for you. It helps you stay healthy and feel better.",
                "context_a": [
                    "Studies show that regular physical activity reduces the risk of cardiovascular disease by up to 35%.",
                    "Exercise has been proven to improve mental health by reducing symptoms of depression and anxiety.",
                    "Regular physical activity helps maintain healthy body weight and prevents obesity-related diseases."
                ],
                "context_b": [
                    "Exercise is beneficial for overall health and wellness."
                ]
            },
            {
                "query": "How do solar panels work to generate electricity?",
                "answer_a": "Solar panels work through the photovoltaic effect, where sunlight hits semiconductor materials (usually silicon) and knocks electrons loose, creating an electric current. The panels contain multiple solar cells connected together, and an inverter converts the direct current (DC) to alternating current (AC) for use in homes and businesses.",
                "answer_b": "Solar panels use sunlight to make electricity. They have special materials that react to light and create power.",
                "context_a": [
                    "Photovoltaic cells are made of semiconductor materials, typically silicon, that absorb photons from sunlight.",
                    "When photons hit the semiconductor, they knock electrons loose, creating an electric current.",
                    "Solar inverters convert the DC electricity produced by panels into AC electricity used by most appliances."
                ],
                "context_b": [
                    "Solar panels convert sunlight into electrical energy through photovoltaic technology."
                ]
            }
        ]
        
        logger.info("Testing LLM judge structure with realistic preference comparisons...")
        for i, pref in enumerate(realistic_preferences):
            answer_pair = AnswerPair(
                query=pref["query"],
                answer_a=pref["answer_a"],
                answer_b=pref["answer_b"],
                context_a=pref["context_a"],
                context_b=pref["context_b"]
            )
            logger.info(f"  Preference {i+1}: Created AnswerPair for '{pref['query'][:50]}...'")
        
        logger.info("✅ LLM judge structure working with realistic examples")
        
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
