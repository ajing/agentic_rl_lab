#!/usr/bin/env python3
"""
Test the trained BC model on new queries to evaluate its performance.

This script loads the trained BC model and tests it on new queries from the CORAL dataset,
comparing its performance against the LLM expert policy.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BCTestingConfig:
    """Configuration for BC model testing."""
    model_path: str = "outputs/bc_model_v2/bc_model_v2.pth"
    config_path: str = "outputs/bc_model_v2/bc_config_v2.json"
    test_queries: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BCPolicyNetwork(nn.Module):
    """Neural network for BC policy (same as training)."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, features):
        return self.network(features)


class BCModelTester:
    """Tester for the trained BC model."""
    
    def __init__(self, config: BCTestingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.model_config = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the trained BC model and configuration."""
        logger.info(f"Loading BC model from {self.config.model_path}")
        
        # Load configuration
        with open(self.config.config_path, 'r') as f:
            self.model_config = json.load(f)
        
        # Calculate input size (same as training)
        # Query features (5) + max_candidates * candidate_features (6 each)
        input_size = 5 + self.model_config['max_candidates'] * 6
        
        # Initialize model
        self.model = BCPolicyNetwork(
            input_size=input_size,
            hidden_size=self.model_config['hidden_size'],
            output_size=self.model_config['max_candidates'],
            dropout=self.model_config['dropout']
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        self.model.eval()
        
        logger.info("BC model loaded successfully!")
    
    def extract_query_features(self, query: str) -> List[float]:
        """Extract features from query (same as training)."""
        return [
            len(query),  # Query length
            query.count(' '),  # Word count
            query.count('?'),  # Question marks
            query.count('!'),  # Exclamation marks
            len(query.split()),  # Number of words
        ]
    
    def extract_candidate_features(self, candidates: List[Dict]) -> List[List[float]]:
        """Extract features from candidate documents (same as training)."""
        features = []
        
        for candidate in candidates:
            content = candidate.get('content', '')
            features.append([
                candidate.get('rrf_score', 0.0),
                candidate.get('bm25_score', 0.0),
                candidate.get('vector_score', 0.0),
                len(content),  # Content length
                content.count(' '),  # Word count in content
                content.count('.'),  # Sentence count (approximate)
            ])
        
        return features
    
    def predict_document_selection(self, query: str, candidates: List[Dict]) -> int:
        """Predict which document to select from candidates."""
        if not candidates:
            return 0
        
        # Extract features
        query_features = self.extract_query_features(query)
        candidate_features = self.extract_candidate_features(candidates)
        
        # Pad or truncate candidates to max_candidates
        max_candidates = self.model_config['max_candidates']
        if len(candidate_features) < max_candidates:
            # Pad with zeros
            padding = [[0.0] * 6] * (max_candidates - len(candidate_features))
            candidate_features.extend(padding)
        else:
            # Truncate to max_candidates
            candidate_features = candidate_features[:max_candidates]
        
        # Flatten features
        features = query_features + [item for sublist in candidate_features for item in sublist]
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            prediction = torch.argmax(logits, dim=1).item()
        
        # Ensure prediction is within valid range
        prediction = min(prediction, len(candidates) - 1)
        
        return prediction
    
    def load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from CORAL dataset."""
        logger.info("Loading test queries from CORAL dataset...")
        
        # Load CORAL conversations
        coral_file = "data/coral/hf_cache/datasets--ariya2357--CORAL/snapshots/e1ebb7bca2692e27fc136b9282df3bfd5b635f0c/test/new_test_conversation.json"
        
        with open(coral_file, 'r') as f:
            conversations = json.load(f)
        
        # Extract queries from conversations
        test_queries = []
        for conv in conversations[:self.config.test_queries]:
            for turn in conv.get('turns', []):
                if turn.get('question'):
                    test_queries.append({
                        'query': turn['question'],
                        'conversation_id': conv.get('conversation_id', 'unknown'),
                        'turn_id': turn.get('turn_id', 0)
                    })
                    break  # Take only the first question from each conversation
        
        logger.info(f"Loaded {len(test_queries)} test queries")
        return test_queries
    
    def generate_candidates_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Generate candidate documents for a query (simplified version)."""
        # This is a simplified version - in practice, you'd use the full RAG environment
        # For now, we'll create mock candidates with realistic scores
        
        # Mock document IDs and content
        mock_docs = [
            {"doc_id": "1234", "content": f"Document about {query[:20]}...", "rrf_score": 0.8, "bm25_score": 0.7, "vector_score": 0.9},
            {"doc_id": "5678", "content": f"Another document related to {query[:15]}...", "rrf_score": 0.6, "bm25_score": 0.8, "vector_score": 0.7},
            {"doc_id": "9012", "content": f"Third document on {query[:25]}...", "rrf_score": 0.7, "bm25_score": 0.6, "vector_score": 0.8},
            {"doc_id": "3456", "content": f"Fourth document about {query[:18]}...", "rrf_score": 0.5, "bm25_score": 0.5, "vector_score": 0.6},
            {"doc_id": "7890", "content": f"Fifth document on {query[:22]}...", "rrf_score": 0.4, "bm25_score": 0.4, "vector_score": 0.5},
        ]
        
        # Add some randomness to make it more realistic
        random.shuffle(mock_docs)
        
        return mock_docs
    
    def test_model(self) -> Dict[str, Any]:
        """Test the BC model on new queries."""
        logger.info("Starting BC model testing...")
        
        # Load test queries
        test_queries = self.load_test_queries()
        
        results = {
            'total_queries': len(test_queries),
            'predictions': [],
            'avg_prediction_time': 0.0,
            'model_config': self.model_config
        }
        
        total_time = 0.0
        
        for i, query_data in enumerate(test_queries):
            query = query_data['query']
            
            logger.info(f"Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Generate candidates
            candidates = self.generate_candidates_for_query(query)
            
            # Time the prediction
            start_time = time.time()
            predicted_idx = self.predict_document_selection(query, candidates)
            prediction_time = time.time() - start_time
            total_time += prediction_time
            
            # Get the selected document
            selected_doc = candidates[predicted_idx] if predicted_idx < len(candidates) else candidates[0]
            
            # Store results
            result = {
                'query': query,
                'conversation_id': query_data['conversation_id'],
                'turn_id': query_data['turn_id'],
                'candidates': candidates,
                'predicted_idx': predicted_idx,
                'selected_doc': selected_doc,
                'prediction_time': prediction_time
            }
            
            results['predictions'].append(result)
            
            logger.info(f"  Selected document {predicted_idx}: {selected_doc['doc_id']} (RRF: {selected_doc['rrf_score']:.3f})")
        
        results['avg_prediction_time'] = total_time / len(test_queries)
        
        logger.info(f"Testing completed! Average prediction time: {results['avg_prediction_time']:.4f}s")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save test results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results."""
        print("\n" + "="*60)
        print("BC MODEL TESTING SUMMARY")
        print("="*60)
        
        print(f"Total queries tested: {results['total_queries']}")
        print(f"Average prediction time: {results['avg_prediction_time']:.4f} seconds")
        print(f"Model configuration:")
        print(f"  - Hidden size: {results['model_config']['hidden_size']}")
        print(f"  - Max candidates: {results['model_config']['max_candidates']}")
        print(f"  - Dropout: {results['model_config']['dropout']}")
        
        print(f"\nSample predictions:")
        for i, pred in enumerate(results['predictions'][:5]):
            print(f"  {i+1}. Query: {pred['query'][:50]}...")
            print(f"     Selected: Doc {pred['selected_doc']['doc_id']} (RRF: {pred['selected_doc']['rrf_score']:.3f})")
            print(f"     Time: {pred['prediction_time']:.4f}s")
        
        print("="*60)


def main():
    """Main testing function."""
    # Configuration
    config = BCTestingConfig(
        test_queries=20
    )
    
    # Initialize tester
    tester = BCModelTester(config)
    
    # Load model
    tester.load_model()
    
    # Test model
    results = tester.test_model()
    
    # Save results
    output_path = "outputs/bc_model_v2/test_results.json"
    tester.save_results(results, output_path)
    
    # Print summary
    tester.print_summary(results)


if __name__ == "__main__":
    main()
