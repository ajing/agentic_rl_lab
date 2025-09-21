"""
Test script for the real data pipeline.

Quick verification that the real data training components work with CORAL data.
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

def test_components_setup():
    """Test component setup."""
    logger = logging.getLogger(__name__)
    
    try:
        from scripts.week5_6.generate_expert_trajectories import ExpertTrajectoryGenerator
        
        # Initialize generator
        generator = ExpertTrajectoryGenerator(
            data_dir="data/coral",
            output_dir="outputs/test_expert_trajectories"
        )
        
        # Test component setup
        generator.setup_components()
        
        if generator.rag_env and generator.episode_runner:
            logger.info("✅ Components setup successful")
            return True
        else:
            logger.error("❌ Component setup failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Component setup test failed: {e}")
        return False

def test_trajectory_generation():
    """Test trajectory generation with small sample."""
    logger = logging.getLogger(__name__)
    
    try:
        from scripts.week5_6.generate_expert_trajectories import ExpertTrajectoryGenerator
        
        # Initialize generator
        generator = ExpertTrajectoryGenerator(
            data_dir="data/coral",
            output_dir="outputs/test_expert_trajectories"
        )
        
        # Setup components
        generator.setup_components()
        
        # Load small sample of conversations
        generator.load_coral_conversations(max_conversations=5)
        
        if not generator.coral_conversations:
            logger.error("❌ No conversations loaded for trajectory test")
            return False
        
        # Generate small number of trajectories
        trajectories = generator.generate_expert_trajectories(
            num_trajectories=3,
            use_multiple_policies=False  # Use single policy for faster testing
        )
        
        if trajectories:
            logger.info(f"✅ Generated {len(trajectories)} test trajectories")
            
            # Check trajectory structure
            traj = trajectories[0]
            required_keys = ['query', 'conversation_history', 'episode_result', 'policy_used', 'documents_selected']
            
            for key in required_keys:
                if key not in traj:
                    logger.error(f"❌ Missing key '{key}' in trajectory")
                    return False
            
            logger.info("✅ Trajectory structure validation passed")
            return True
        else:
            logger.error("❌ No trajectories generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Trajectory generation test failed: {e}")
        return False

def test_preference_dataset_building():
    """Test preference dataset building with small sample."""
    logger = logging.getLogger(__name__)
    
    try:
        from scripts.week5_6.build_preference_dataset import PreferenceDatasetBuilder
        
        # Initialize builder
        builder = PreferenceDatasetBuilder(
            data_dir="data/coral",
            output_dir="outputs/test_preference_dataset"
        )
        
        # Setup components (without LLM judge for faster testing)
        builder.setup_components(use_llm_judge=False)
        
        # Load small sample of conversations
        builder.load_coral_conversations(max_conversations=5)
        
        if not builder.coral_conversations:
            logger.error("❌ No conversations loaded for preference test")
            return False
        
        # Build small preference dataset
        preferences = builder.build_preference_dataset(
            num_preferences=3,
            use_multiple_policies=False  # Use fewer policies for faster testing
        )
        
        if preferences:
            logger.info(f"✅ Generated {len(preferences)} test preferences")
            
            # Check preference structure
            pref = preferences[0]
            required_keys = ['query', 'answer_a', 'answer_b', 'preferred_answer', 'confidence']
            
            for key in required_keys:
                if key not in pref:
                    logger.error(f"❌ Missing key '{key}' in preference")
                    return False
            
            logger.info("✅ Preference structure validation passed")
            return True
        else:
            logger.error("❌ No preferences generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Preference dataset building test failed: {e}")
        return False

def main():
    """Run all real data pipeline tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Testing real data pipeline components...")
    
    tests = [
        ("CORAL Data Loading", test_coral_data_loading),
        ("Components Setup", test_components_setup),
        ("Trajectory Generation", test_trajectory_generation),
        ("Preference Dataset Building", test_preference_dataset_building)
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
    logger.info("REAL DATA PIPELINE TEST SUMMARY")
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
