"""
Realistic conversational scenarios test for Week 3-4 components.

Tests the components with realistic, multi-turn conversational queries
that represent real-world RAG use cases.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_conversational_scenarios():
    """Test with realistic conversational scenarios."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.reward.reward_model import LightweightRewardModel
        from src.reward.reward_shaping import RewardShaper, RewardConfig
        from src.reward.llm_judge import AnswerPair, PreferenceResult
        import torch
        
        # Initialize components
        reward_model = LightweightRewardModel()
        reward_model.eval()
        
        config = RewardConfig()
        reward_shaper = RewardShaper(reward_model=reward_model, config=config)
        
        # Realistic conversational scenarios
        scenarios = [
            {
                "name": "Travel Planning",
                "turns": [
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
                        "query": "What about transportation? How should I get around Tokyo?",
                        "answer": "Tokyo has an excellent public transportation system. The JR Yamanote Line is perfect for tourists, connecting major districts. Get a Suica or Pasmo card for easy payment. Avoid rush hours (7-9 AM, 5-7 PM) when trains are extremely crowded. Taxis are expensive but available 24/7.",
                        "context": [
                            "Tokyo's public transportation system includes JR lines, subway systems, and buses.",
                            "Suica and Pasmo are rechargeable IC cards that work on most public transport.",
                            "Rush hour in Tokyo is among the most crowded in the world, especially on weekdays."
                        ]
                    }
                ]
            },
            {
                "name": "Learning Technology",
                "turns": [
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
                        "query": "Which one should I learn first as a beginner?",
                        "answer": "Start with traditional machine learning first. Learn the fundamentals like linear regression, decision trees, and clustering. Understanding these concepts will give you a solid foundation. Once you're comfortable with the basics, move to deep learning. Python with libraries like scikit-learn and TensorFlow/PyTorch are excellent starting points.",
                        "context": [
                            "Traditional machine learning provides fundamental concepts that apply to all AI approaches.",
                            "Scikit-learn is a popular Python library for traditional machine learning algorithms.",
                            "Deep learning frameworks like TensorFlow and PyTorch require understanding of neural network basics."
                        ]
                    }
                ]
            },
            {
                "name": "Health and Wellness",
                "turns": [
                    {
                        "query": "What are the health benefits of regular exercise?",
                        "answer": "Regular exercise provides numerous health benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and reduced risk of chronic diseases like diabetes and heart disease. It also boosts energy levels and improves sleep quality.",
                        "context": [
                            "Studies show that regular physical activity reduces the risk of cardiovascular disease by up to 35%.",
                            "Exercise has been proven to improve mental health by reducing symptoms of depression and anxiety.",
                            "Regular physical activity helps maintain healthy body weight and prevents obesity-related diseases."
                        ]
                    },
                    {
                        "query": "How much exercise do I need per week?",
                        "answer": "The World Health Organization recommends at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week, plus muscle-strengthening activities on 2 or more days. This can be broken down into 30 minutes of moderate exercise 5 days a week, or 25 minutes of vigorous exercise 3 days a week.",
                        "context": [
                            "WHO guidelines recommend 150 minutes of moderate or 75 minutes of vigorous aerobic activity weekly.",
                            "Moderate activities include brisk walking, cycling, or swimming.",
                            "Vigorous activities include running, high-intensity interval training, or competitive sports."
                        ]
                    }
                ]
            }
        ]
        
        logger.info("Testing realistic conversational scenarios...")
        
        for scenario in scenarios:
            logger.info(f"\n--- {scenario['name']} Scenario ---")
            
            for i, turn in enumerate(scenario['turns']):
                logger.info(f"Turn {i+1}: {turn['query'][:60]}...")
                
                # Test reward model scoring
                with torch.no_grad():
                    result = reward_model(turn['query'], turn['answer'], turn['context'])
                    reward_score = result["reward_score"].item()
                    criteria_scores = {k: v.item() for k, v in result["criteria_scores"].items()}
                    
                    logger.info(f"  Reward Model: {reward_score:.3f} "
                               f"(Accuracy: {criteria_scores['accuracy']:.3f}, "
                               f"Completeness: {criteria_scores['completeness']:.3f}, "
                               f"Clarity: {criteria_scores['clarity']:.3f})")
                
                # Test reward shaping
                reward_shaper.reset_episode()
                
                for j, doc in enumerate(turn['context']):
                    step_reward = reward_shaper.compute_step_reward(
                        turn['query'], doc, 0.8 - j * 0.1, j + 1
                    )
                    logger.info(f"  Step {j+1} Reward: {step_reward.total_reward:.3f} "
                               f"(Novelty: {step_reward.novelty_reward:.3f}, "
                               f"Relevance: {step_reward.relevance_reward:.3f})")
                
                # Test final episode reward
                final_reward = reward_shaper.compute_episode_reward(turn['query'], turn['answer'])
                logger.info(f"  Final Episode Reward: {final_reward.total_reward:.3f}")
        
        logger.info("✅ Conversational scenarios test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversational scenarios test failed: {e}")
        return False

def test_preference_comparisons():
    """Test preference comparisons with realistic examples."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.reward.llm_judge import AnswerPair, PreferenceResult
        
        # Realistic preference comparison scenarios
        preference_scenarios = [
            {
                "name": "Technical Explanation Quality",
                "query": "How do solar panels work to generate electricity?",
                "answer_a": "Solar panels work through the photovoltaic effect, where sunlight hits semiconductor materials (usually silicon) and knocks electrons loose, creating an electric current. The panels contain multiple solar cells connected together, and an inverter converts the direct current (DC) to alternating current (AC) for use in homes and businesses.",
                "answer_b": "Solar panels use sunlight to make electricity. They have special materials that react to light and create power that can be used in buildings.",
                "context_a": [
                    "Photovoltaic cells are made of semiconductor materials, typically silicon, that absorb photons from sunlight.",
                    "When photons hit the semiconductor, they knock electrons loose, creating an electric current.",
                    "Solar inverters convert the DC electricity produced by panels into AC electricity used by most appliances."
                ],
                "context_b": [
                    "Solar panels convert sunlight into electrical energy through photovoltaic technology.",
                    "The electricity generated can be used directly or stored in batteries for later use."
                ],
                "expected_preference": "A"  # More detailed and technically accurate
            },
            {
                "name": "Travel Advice Quality",
                "query": "What should I know before visiting Japan for the first time?",
                "answer_a": "Before visiting Japan, learn basic Japanese phrases, understand cultural etiquette like bowing and removing shoes, research transportation options like the JR Pass, and be aware of cash-based society in many places. Also, respect local customs like being quiet on public transport and not eating while walking.",
                "answer_b": "Japan is a great country to visit. The people are nice and the food is good. Make sure to see Tokyo and maybe some temples.",
                "context_a": [
                    "Japanese culture places high value on respect, harmony, and proper social etiquette.",
                    "The JR Pass offers unlimited travel on JR trains and is cost-effective for tourists.",
                    "Many Japanese businesses still prefer cash payments over credit cards."
                ],
                "context_b": [
                    "Japan is known for its unique culture, delicious cuisine, and beautiful landscapes.",
                    "Tokyo is the capital and largest city, offering modern attractions and traditional sites."
                ],
                "expected_preference": "A"  # More practical and actionable advice
            },
            {
                "name": "Health Information Quality",
                "query": "What are the benefits of regular exercise?",
                "answer_a": "Regular exercise provides numerous health benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and reduced risk of chronic diseases like diabetes and heart disease. It also boosts energy levels and improves sleep quality.",
                "answer_b": "Exercise is good for you. It helps you stay healthy and feel better. You should try to exercise regularly.",
                "context_a": [
                    "Studies show that regular physical activity reduces the risk of cardiovascular disease by up to 35%.",
                    "Exercise has been proven to improve mental health by reducing symptoms of depression and anxiety.",
                    "Regular physical activity helps maintain healthy body weight and prevents obesity-related diseases."
                ],
                "context_b": [
                    "Exercise is beneficial for overall health and wellness.",
                    "Regular physical activity is recommended by health professionals worldwide."
                ],
                "expected_preference": "A"  # More comprehensive and specific
            }
        ]
        
        logger.info("Testing preference comparisons with realistic examples...")
        
        for scenario in preference_scenarios:
            logger.info(f"\n--- {scenario['name']} ---")
            
            # Create AnswerPair
            answer_pair = AnswerPair(
                query=scenario['query'],
                answer_a=scenario['answer_a'],
                answer_b=scenario['answer_b'],
                context_a=scenario['context_a'],
                context_b=scenario['context_b']
            )
            
            logger.info(f"Query: {scenario['query'][:60]}...")
            logger.info(f"Answer A: {scenario['answer_a'][:80]}...")
            logger.info(f"Answer B: {scenario['answer_b'][:80]}...")
            logger.info(f"Expected Preference: {scenario['expected_preference']}")
            
            # Note: In a real test, you would call the LLM judge here
            # For now, we just verify the structure is correct
            logger.info("✅ AnswerPair structure validated")
        
        logger.info("✅ Preference comparisons test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Preference comparisons test failed: {e}")
        return False

def test_multi_turn_conversations():
    """Test multi-turn conversation handling."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.reward.reward_model import LightweightRewardModel
        import torch
        
        reward_model = LightweightRewardModel()
        reward_model.eval()
        
        # Multi-turn conversation example
        conversation = [
            {
                "turn": 1,
                "query": "I'm interested in learning about machine learning. Where should I start?",
                "answer": "Start with the fundamentals: linear algebra, statistics, and programming in Python. Learn basic algorithms like linear regression and decision trees. Online courses like Andrew Ng's Machine Learning course on Coursera are excellent starting points.",
                "context": [
                    "Machine learning requires strong mathematical foundations in linear algebra and statistics.",
                    "Python is the most popular programming language for machine learning.",
                    "Andrew Ng's course is widely regarded as one of the best introductions to machine learning."
                ]
            },
            {
                "turn": 2,
                "query": "What about deep learning? Should I learn that next?",
                "answer": "Yes, but only after mastering the basics. Deep learning builds on traditional ML concepts. Start with neural networks, then move to CNNs for images and RNNs for sequences. Frameworks like TensorFlow and PyTorch are essential tools.",
                "context": [
                    "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
                    "Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks.",
                    "Recurrent Neural Networks (RNNs) are designed for sequential data like text and time series."
                ]
            },
            {
                "turn": 3,
                "query": "How long does it typically take to become proficient?",
                "answer": "It varies, but expect 6-12 months of consistent study to become proficient in the basics. Focus on hands-on projects and real datasets. Join communities like Kaggle to practice and learn from others. Remember, it's a continuous learning journey.",
                "context": [
                    "Learning machine learning is a continuous process that requires consistent practice.",
                    "Hands-on projects with real datasets are crucial for developing practical skills.",
                    "Online communities like Kaggle provide opportunities for learning and collaboration."
                ]
            }
        ]
        
        logger.info("Testing multi-turn conversation handling...")
        
        for turn_data in conversation:
            logger.info(f"\nTurn {turn_data['turn']}: {turn_data['query'][:60]}...")
            
            # Test reward model scoring for each turn
            with torch.no_grad():
                result = reward_model(
                    turn_data['query'], 
                    turn_data['answer'], 
                    turn_data['context']
                )
                reward_score = result["reward_score"].item()
                criteria_scores = {k: v.item() for k, v in result["criteria_scores"].items()}
                
                logger.info(f"  Reward Score: {reward_score:.3f}")
                logger.info(f"  Accuracy: {criteria_scores['accuracy']:.3f}")
                logger.info(f"  Completeness: {criteria_scores['completeness']:.3f}")
                logger.info(f"  Relevance: {criteria_scores['relevance']:.3f}")
        
        logger.info("✅ Multi-turn conversation test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-turn conversation test failed: {e}")
        return False

def main():
    """Run realistic scenario tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Testing Week 3-4 components with realistic scenarios...")
    
    tests = [
        ("Conversational Scenarios", test_conversational_scenarios),
        ("Preference Comparisons", test_preference_comparisons),
        ("Multi-turn Conversations", test_multi_turn_conversations)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "CRASH"
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("REALISTIC SCENARIOS TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All realistic scenario tests passed!")
    else:
        logger.warning("⚠️  Some realistic scenario tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
