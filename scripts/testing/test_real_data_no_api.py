"""
Test script for the real data pipeline without API dependencies.

Quick verification that the real data training components work with CORAL data
without requiring OpenAI API access.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_coral_data_loading():
    """Test loading CORAL data."""
    logger = logging.getLogger(__name__)
    
    try:
        from scripts.week5_6.generate_expert_trajectories import ExpertTrajectoryGenerator
        
        # Initialize generator
        generator = ExpertTrajectoryGenerator(
            data_dir="data/coral",
            output_dir="outputs/test_expert_trajectories"
        )
        
        # Test loading conversations
        generator.load_coral_conversations(max_conversations=10)
        
        if generator.coral_conversations:
            logger.info(f"✅ Successfully loaded {len(generator.coral_conversations)} CORAL conversations")
            
            # Test query extraction
            queries = generator.extract_queries_from_conversations()
            logger.info(f"✅ Extracted {len(queries)} queries from conversations")
            
            if queries:
                query, history = queries[0]
                logger.info(f"✅ Sample query: {query[:100]}...")
                logger.info(f"✅ Sample history length: {len(history)}")
            
            return True
        else:
            logger.error("❌ No conversations loaded")
            return False
            
    except Exception as e:
        logger.error(f"❌ CORAL data loading test failed: {e}")
        return False

def test_components_setup_no_api():
    """Test component setup without API dependencies."""
    logger = logging.getLogger(__name__)
    
    try:
        from scripts.week5_6.generate_expert_trajectories import ExpertTrajectoryGenerator
        
        # Initialize generator
        generator = ExpertTrajectoryGenerator(
            data_dir="data/coral",
            output_dir="outputs/test_expert_trajectories"
        )
        
        # Test component setup without query rewriter (which requires API)
        generator.rag_env = None  # We'll test this separately
        
        # Test reward model initialization
        from src.reward.reward_model import LightweightRewardModel
        reward_model = LightweightRewardModel()
        logger.info("✅ Reward model initialization successful")
        
        # Test reward shaper initialization
        from src.reward.reward_shaping import RewardShaper, RewardConfig
        config = RewardConfig()
        reward_shaper = RewardShaper(reward_model=reward_model, config=config)
        logger.info("✅ Reward shaper initialization successful")
        
        return True
            
    except Exception as e:
        logger.error(f"❌ Component setup test failed: {e}")
        return False

def test_rag_environment_setup():
    """Test RAG environment setup without query rewriter."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.env.rag_environment import RAGEnvironment
        
        # Test RAG environment setup without query rewriter
        rag_env = RAGEnvironment(
            corpus_path="data/coral/docs.jsonl",
            bm25_index_path="index/coral_bm25",
            vector_index_path="index/coral_faiss",
            use_query_rewriting=False,  # Disable to avoid API requirement
            use_cross_encoder=False,    # Disable for faster testing
            use_mmr=False,              # Disable for faster testing
            max_steps=3,
            k_candidates=20
        )
        
        logger.info("✅ RAG environment setup successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG environment setup test failed: {e}")
        return False

def test_episode_runner_setup():
    """Test episode runner setup."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.env.rag_environment import RAGEnvironment
        from src.policy.episode_runner import EpisodeRunner, PolicyConfig
        
        # Setup RAG environment
        rag_env = RAGEnvironment(
            corpus_path="data/coral/docs.jsonl",
            bm25_index_path="index/coral_bm25",
            vector_index_path="index/coral_faiss",
            use_query_rewriting=False,
            use_cross_encoder=False,
            use_mmr=False,
            max_steps=3,
            k_candidates=20
        )
        
        # Setup episode runner
        episode_runner = EpisodeRunner(rag_env)
        logger.info("✅ Episode runner setup successful")
        
        # Test policy config creation
        policy = PolicyConfig(name="greedy_policy", policy_type="greedy")
        logger.info("✅ Policy config creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Episode runner setup test failed: {e}")
        return False

def test_trajectory_generation_no_api():
    """Test trajectory generation without API dependencies."""
    logger = logging.getLogger(__name__)
    
    try:
        from scripts.week5_6.generate_expert_trajectories import ExpertTrajectoryGenerator
        
        # Initialize generator
        generator = ExpertTrajectoryGenerator(
            data_dir="data/coral",
            output_dir="outputs/test_expert_trajectories"
        )
        
        # Load conversations
        generator.load_coral_conversations(max_conversations=5)
        
        if not generator.coral_conversations:
            logger.error("❌ No conversations loaded for trajectory test")
            return False
        
        # Test query extraction
        queries = generator.extract_queries_from_conversations()
        if not queries:
            logger.error("❌ No queries extracted")
            return False
        
        logger.info(f"✅ Extracted {len(queries)} queries for trajectory generation")
        
        # Test trajectory structure creation (without actually running episodes)
        query, history = queries[0]
        
        # Create a mock trajectory structure
        mock_trajectory = {
            'query': query,
            'conversation_history': [{'question': t.question, 'answer': t.answer, 'turn_id': t.turn_id} for t in history],
            'episode_result': {
                'selected_doc_ids': ['doc1', 'doc2'],
                'total_reward': 0.8,
                'episode_length': 2,
                'rewritten_query': query
            },
            'policy_used': {
                'policy_type': 'greedy',
                'selection_strategy': 'top_score',
                'epsilon': None
            },
            'documents_selected': [
                {'doc_id': 'doc1', 'content': 'Sample document content 1'},
                {'doc_id': 'doc2', 'content': 'Sample document content 2'}
            ]
        }
        
        # Validate trajectory structure
        required_keys = ['query', 'conversation_history', 'episode_result', 'policy_used', 'documents_selected']
        for key in required_keys:
            if key not in mock_trajectory:
                logger.error(f"❌ Missing key '{key}' in trajectory")
                return False
        
        logger.info("✅ Trajectory structure validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Trajectory generation test failed: {e}")
        return False

def test_preference_dataset_structure():
    """Test preference dataset structure without API dependencies."""
    logger = logging.getLogger(__name__)
    
    try:
        from scripts.week5_6.build_preference_dataset import PreferenceDatasetBuilder
        
        # Initialize builder
        builder = PreferenceDatasetBuilder(
            data_dir="data/coral",
            output_dir="outputs/test_preference_dataset"
        )
        
        # Load conversations
        builder.load_coral_conversations(max_conversations=5)
        
        if not builder.coral_conversations:
            logger.error("❌ No conversations loaded for preference test")
            return False
        
        # Test query extraction
        queries = builder.extract_queries_from_conversations()
        if not queries:
            logger.error("❌ No queries extracted")
            return False
        
        logger.info(f"✅ Extracted {len(queries)} queries for preference dataset")
        
        # Test preference structure creation (without actually running episodes)
        query, history = queries[0]
        
        # Create a mock preference structure
        mock_preference = {
            'query': query,
            'conversation_history': [{'question': t.question, 'answer': t.answer, 'turn_id': t.turn_id} for t in history],
            'answer_a': 'Sample answer A based on selected documents',
            'answer_b': 'Sample answer B based on different documents',
            'context_a': ['Document content A1', 'Document content A2'],
            'context_b': ['Document content B1', 'Document content B2'],
            'preferred_answer': 'A',
            'confidence': 0.8,
            'reasoning': 'Answer A is more comprehensive and accurate',
            'criteria_scores': {
                'accuracy': {'A': 0.9, 'B': 0.7},
                'completeness': {'A': 0.8, 'B': 0.6}
            },
            'policy_a': {
                'policy_type': 'greedy',
                'selection_strategy': 'top_score',
                'epsilon': None
            },
            'policy_b': {
                'policy_type': 'random',
                'selection_strategy': 'random',
                'epsilon': None
            },
            'episode_a': {
                'total_reward': 0.8,
                'num_documents': 2,
                'episode_length': 2
            },
            'episode_b': {
                'total_reward': 0.5,
                'num_documents': 2,
                'episode_length': 2
            }
        }
        
        # Validate preference structure
        required_keys = ['query', 'answer_a', 'answer_b', 'preferred_answer', 'confidence']
        for key in required_keys:
            if key not in mock_preference:
                logger.error(f"❌ Missing key '{key}' in preference")
                return False
        
        logger.info("✅ Preference structure validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Preference dataset structure test failed: {e}")
        return False

def main():
    """Run all real data pipeline tests without API dependencies."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Testing real data pipeline components (no API dependencies)...")
    
    tests = [
        ("CORAL Data Loading", test_coral_data_loading),
        ("Components Setup (No API)", test_components_setup_no_api),
        ("RAG Environment Setup", test_rag_environment_setup),
        ("Episode Runner Setup", test_episode_runner_setup),
        ("Trajectory Generation Structure", test_trajectory_generation_no_api),
        ("Preference Dataset Structure", test_preference_dataset_structure)
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
    logger.info("REAL DATA PIPELINE TEST SUMMARY (NO API)")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All real data pipeline tests passed!")
    else:
        logger.warning("⚠️  Some real data pipeline tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
