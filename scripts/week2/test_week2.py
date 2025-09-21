#!/usr/bin/env python3
"""
Week 2 smoke tests for RAG RL environment.

Tests the complete pipeline: query rewriting -> RRF -> cross-encoder -> MMR -> RL environment
on a small subset of CORAL data.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add project root to path (so we can import src modules)
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.retriever.query_rewriter import QueryRewriter, ConversationTurn
from src.retriever.rrf_generator import RRFCandidateGenerator
from src.reranker.cross_encoder import CrossEncoderReranker
from src.reranker.mmr import MMRDeduplicator
from src.env.rag_environment import RAGEnvironment
from src.policy.episode_runner import EpisodeRunner, PolicyConfig

logger = logging.getLogger(__name__)


def load_coral_test_data(data_path: str, num_samples: int = 3) -> List[Dict]:
    """Load a small sample of CORAL test data."""
    test_data = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                conversation = json.loads(line.strip())
                test_data.append(conversation)
        
        logger.info(f"Loaded {len(test_data)} CORAL conversations")
        return test_data
        
    except Exception as e:
        logger.error(f"Error loading CORAL data: {e}")
        return []


def test_query_rewriter():
    """Test query rewriting module."""
    logger.info("Testing Query Rewriter...")
    
    try:
        rewriter = QueryRewriter()
        
        # Test conversation
        history = [
            ConversationTurn(1, "Who is Kariqi?", "Kariqi is a football player."),
            ConversationTurn(2, "What team does he play for?", "He plays for a European team.")
        ]
        
        current_query = "What role did he play during the 2017 UEFA European Under-21 Championship qualification?"
        
        result = rewriter.rewrite_query(current_query, history)
        
        logger.info(f"Original: {result.original_query}")
        logger.info(f"Rewritten: {result.rewritten_query}")
        logger.info(f"Confidence: {result.confidence}")
        
        return True
        
    except Exception as e:
        logger.error(f"Query rewriter test failed: {e}")
        return False


def test_rrf_generator():
    """Test RRF candidate generator."""
    logger.info("Testing RRF Generator...")
    
    try:
        generator = RRFCandidateGenerator(k=20)
        
        query = "Who won the FA Cup in 2020?"
        candidates = generator.generate_candidates(query)
        
        logger.info(f"Generated {len(candidates)} candidates")
        if candidates:
            logger.info(f"Top candidate: {candidates[0].doc_id} (score: {candidates[0].rrf_score:.3f})")
        
        return len(candidates) > 0
        
    except Exception as e:
        logger.error(f"RRF generator test failed: {e}")
        return False


def test_cross_encoder():
    """Test cross-encoder reranker."""
    logger.info("Testing Cross-Encoder Reranker...")
    
    try:
        reranker = CrossEncoderReranker()
        
        query = "Who won the FA Cup in 2020?"
        candidates = [
            {
                "doc_id": "doc1",
                "content": "Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
                "rrf_score": 0.8
            },
            {
                "doc_id": "doc2",
                "content": "The FA Cup is an annual football competition in England.",
                "rrf_score": 0.6
            }
        ]
        
        reranked = reranker.rerank_candidates(query, candidates)
        
        logger.info(f"Reranked {len(reranked)} candidates")
        if reranked:
            logger.info(f"Top reranked: {reranked[0].doc_id} (score: {reranked[0].final_score:.3f})")
        
        return len(reranked) > 0
        
    except Exception as e:
        logger.error(f"Cross-encoder test failed: {e}")
        return False


def test_mmr():
    """Test MMR deduplicator."""
    logger.info("Testing MMR Deduplicator...")
    
    try:
        mmr = MMRDeduplicator()
        
        query = "Football players and their achievements"
        candidates = [
            {
                "doc_id": "doc1",
                "content": "Lionel Messi won the Ballon d'Or multiple times.",
                "rrf_score": 0.9
            },
            {
                "doc_id": "doc2",
                "content": "Cristiano Ronaldo is another great footballer.",
                "rrf_score": 0.8
            },
            {
                "doc_id": "doc3",
                "content": "Messi and Ronaldo have dominated world football.",
                "rrf_score": 0.7
            }
        ]
        
        selected = mmr.select_diverse_documents(query, candidates, top_k=2)
        
        logger.info(f"Selected {len(selected)} diverse documents")
        if selected:
            logger.info(f"Top selected: {selected[0].doc_id} (relevance: {selected[0].relevance_score:.3f})")
        
        return len(selected) > 0
        
    except Exception as e:
        logger.error(f"MMR test failed: {e}")
        return False


def test_rl_environment():
    """Test RL environment."""
    logger.info("Testing RL Environment...")
    
    try:
        env = RAGEnvironment(max_steps=3, k_candidates=20)
        
        query = "Who won the FA Cup in 2020?"
        state = env.reset(query)
        
        logger.info(f"Environment reset: {len(state.remaining_candidates)} candidates")
        
        # Take a few steps
        for step in range(2):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action = valid_actions[0]  # Take first action
            next_state, reward, done, info = env.step(action)
            
            logger.info(f"Step {step + 1}: {action.action_type} {action.doc_id}, reward={reward.total_reward:.3f}")
            
            if done:
                break
        
        episode = env.finalize_episode("Arsenal won the FA Cup in 2020.")
        
        logger.info(f"Episode completed: reward={episode.metrics['total_reward']:.3f}, "
                   f"length={episode.metrics['episode_length']}")
        
        return True
        
    except Exception as e:
        logger.error(f"RL environment test failed: {e}")
        return False


def test_episode_runner():
    """Test episode runner with different policies."""
    logger.info("Testing Episode Runner...")
    
    try:
        env = RAGEnvironment(max_steps=3, k_candidates=20)
        runner = EpisodeRunner(env)
        
        query = "Who won the FA Cup in 2020?"
        
        # Test different policies
        policies = [
            PolicyConfig(policy_type="random", selection_strategy="random"),
            PolicyConfig(policy_type="greedy", selection_strategy="top_score")
        ]
        
        results = runner.compare_policies(query, policies, num_runs=2)
        
        for policy_name, episodes in results.items():
            analysis = runner.analyze_episodes(episodes)
            logger.info(f"{policy_name}: avg reward={analysis['total_reward']['mean']:.3f}, "
                       f"avg length={analysis['episode_length']['mean']:.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Episode runner test failed: {e}")
        return False


def test_coral_integration(coral_data_path: str):
    """Test integration with CORAL data."""
    logger.info("Testing CORAL Integration...")
    
    try:
        # Load CORAL data
        coral_data = load_coral_test_data(coral_data_path, num_samples=2)
        if not coral_data:
            logger.warning("No CORAL data loaded, skipping integration test")
            return False
        
        # Initialize environment
        env = RAGEnvironment(max_steps=3, k_candidates=20)
        runner = EpisodeRunner(env)
        
        # Test on CORAL conversations
        for i, conversation in enumerate(coral_data):
            logger.info(f"Testing CORAL conversation {i + 1}")
            
            # Extract first question
            if "turns" in conversation and conversation["turns"]:
                first_turn = conversation["turns"][0]
                query = first_turn.get("question", "")
                
                if query:
                    # Run episode
                    policy_config = PolicyConfig(policy_type="random", selection_strategy="random")
                    episode = runner.run_episode(query, policy_config, save_episode=False)
                    
                    logger.info(f"  Query: {query[:50]}...")
                    logger.info(f"  Reward: {episode.metrics['total_reward']:.3f}")
                    logger.info(f"  Length: {episode.metrics['episode_length']}")
        
        return True
        
    except Exception as e:
        logger.error(f"CORAL integration test failed: {e}")
        return False


def main():
    """Run all Week 2 smoke tests."""
    parser = argparse.ArgumentParser(description="Week 2 smoke tests")
    parser.add_argument("--coral-data", type=str, 
                       default="data/coral/raw/test/new_test_conversation.json",
                       help="Path to CORAL test data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Week 2 smoke tests...")
    
    # Run individual component tests
    tests = [
        ("Query Rewriter", test_query_rewriter),
        ("RRF Generator", test_rrf_generator),
        ("Cross-Encoder", test_cross_encoder),
        ("MMR Deduplicator", test_mmr),
        ("RL Environment", test_rl_environment),
        ("Episode Runner", test_episode_runner),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
            logger.info(f"{test_name} test: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            results[test_name] = "ERROR"
            logger.error(f"{test_name} test: ERROR - {e}")
    
    # Run CORAL integration test
    logger.info(f"\n{'='*60}")
    logger.info("Running CORAL Integration test...")
    logger.info(f"{'='*60}")
    
    try:
        coral_success = test_coral_integration(args.coral_data)
        results["CORAL Integration"] = "PASS" if coral_success else "FAIL"
    except Exception as e:
        results["CORAL Integration"] = "ERROR"
        logger.error(f"CORAL integration test: ERROR - {e}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        logger.info(f"{status} {test_name}: {result}")
        if result == "PASS":
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Week 2 implementation is working.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
