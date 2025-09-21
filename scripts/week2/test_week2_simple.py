#!/usr/bin/env python3
"""
Simplified Week 2 smoke tests.

Tests individual components without full integration to validate
the core functionality works.
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

logger = logging.getLogger(__name__)


def test_query_rewriter():
    """Test query rewriting module."""
    logger.info("Testing Query Rewriter...")
    
    try:
        from src.retriever.query_rewriter import QueryRewriter, ConversationTurn
        
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


def test_cross_encoder():
    """Test cross-encoder reranker."""
    logger.info("Testing Cross-Encoder Reranker...")
    
    try:
        from src.reranker.cross_encoder import CrossEncoderReranker
        
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
        from src.reranker.mmr import MMRDeduplicator
        
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


def test_rl_environment_basic():
    """Test basic RL environment functionality."""
    logger.info("Testing RL Environment (Basic)...")
    
    try:
        from src.env.rag_environment import RAGEnvironment, RLAction
        
        # Create environment with minimal components
        env = RAGEnvironment(
            max_steps=3, 
            k_candidates=20,
            use_query_rewriting=False,  # Disable to avoid API calls
            use_cross_encoder=False,    # Disable to avoid model loading
            use_mmr=False              # Disable to avoid model loading
        )
        
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


def test_episode_runner_basic():
    """Test episode runner with basic functionality."""
    logger.info("Testing Episode Runner (Basic)...")
    
    try:
        from src.env.rag_environment import RAGEnvironment
        from src.policy.episode_runner import EpisodeRunner, PolicyConfig
        
        env = RAGEnvironment(
            max_steps=3, 
            k_candidates=20,
            use_query_rewriting=False,
            use_cross_encoder=False,
            use_mmr=False
        )
        runner = EpisodeRunner(env)
        
        query = "Who won the FA Cup in 2020?"
        
        # Test random policy
        policy_config = PolicyConfig(policy_type="random", selection_strategy="random")
        episode = runner.run_episode(query, policy_config, save_episode=False)
        
        logger.info(f"Random policy: reward={episode.metrics['total_reward']:.3f}, "
                   f"length={episode.metrics['episode_length']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Episode runner test failed: {e}")
        return False


def main():
    """Run simplified Week 2 smoke tests."""
    parser = argparse.ArgumentParser(description="Simplified Week 2 smoke tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting simplified Week 2 smoke tests...")
    
    # Run individual component tests
    tests = [
        ("Query Rewriter", test_query_rewriter),
        ("Cross-Encoder", test_cross_encoder),
        ("MMR Deduplicator", test_mmr),
        ("RL Environment (Basic)", test_rl_environment_basic),
        ("Episode Runner (Basic)", test_episode_runner_basic),
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
        logger.info("🎉 All tests passed! Week 2 core components are working.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
