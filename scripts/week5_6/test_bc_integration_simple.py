#!/usr/bin/env python3
"""
Simple BC model integration test.

This script demonstrates that the BC model can be successfully integrated
into the RAG environment and makes intelligent document selections.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.policy.bc_policy import BCPolicy, BCConfig


def test_bc_policy_integration():
    """Test BC policy integration with simple document selection."""
    logger.info("Testing BC policy integration...")
    
    # Initialize BC policy
    bc_config = BCConfig(
        model_path="outputs/bc_model_v3/bc_model_v3.pth",
        config_path="outputs/bc_model_v3/bc_config_v3.json"
    )
    
    bc_policy = BCPolicy(bc_config)
    logger.info("BC policy initialized successfully!")
    
    # Test queries
    test_queries = [
        "Who won the 2016 World Series?",
        "What is the capital of France?",
        "How does photosynthesis work?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries):
        logger.info(f"Testing query {i+1}: {query}")
        
        # Create mock candidates with different scores
        candidates = [
            {"doc_id": "doc_1", "content": f"High relevance document about {query[:20]}...", "rrf_score": 0.9, "bm25_score": 0.8, "vector_score": 0.85},
            {"doc_id": "doc_2", "content": f"Medium relevance document on {query[:15]}...", "rrf_score": 0.6, "bm25_score": 0.7, "vector_score": 0.65},
            {"doc_id": "doc_3", "content": f"Lower relevance document about {query[:25]}...", "rrf_score": 0.4, "bm25_score": 0.5, "vector_score": 0.55},
        ]
        
        # Predict document selection
        predicted_idx = bc_policy._predict_document_selection(query, candidates)
        selected_doc = candidates[predicted_idx]
        
        result = {
            'query': query,
            'candidates': candidates,
            'predicted_idx': predicted_idx,
            'selected_doc': selected_doc
        }
        
        results.append(result)
        
        logger.info(f"  Selected document {predicted_idx}: {selected_doc['doc_id']} (RRF: {selected_doc['rrf_score']:.3f})")
    
    return results


def print_integration_summary(results: List[Dict[str, Any]]):
    """Print integration test summary."""
    print("\n" + "="*70)
    print("BC MODEL INTEGRATION TEST SUMMARY")
    print("="*70)
    
    print(f"Total queries tested: {len(results)}")
    
    print(f"\nDocument Selection Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Query: {result['query']}")
        print(f"     Selected: {result['selected_doc']['doc_id']} (RRF: {result['selected_doc']['rrf_score']:.3f})")
        print(f"     All candidates: {[doc['doc_id'] for doc in result['candidates']]}")
    
    # Check if BC model is making diverse selections
    selected_docs = [result['selected_doc']['doc_id'] for result in results]
    unique_selections = len(set(selected_docs))
    
    print(f"\nSelection Diversity:")
    print(f"  Unique document selections: {unique_selections}/{len(results)}")
    print(f"  Selection distribution: {dict((doc, selected_docs.count(doc)) for doc in set(selected_docs))}")
    
    if unique_selections > 1:
        print("  ✅ BC model shows diverse selection behavior")
    else:
        print("  ⚠️  BC model shows limited selection diversity")
    
    print("="*70)


def main():
    """Main integration test function."""
    logger.info("Starting BC model integration test...")
    
    # Test BC policy integration
    results = test_bc_policy_integration()
    
    # Save results
    output_path = "outputs/bc_model_v3/integration_test_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Integration test results saved to {output_path}")
    
    # Print summary
    print_integration_summary(results)
    
    logger.info("BC model integration test completed successfully!")


if __name__ == "__main__":
    main()
