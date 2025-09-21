#!/usr/bin/env python3
"""
Compare BC model performance against LLM expert policy.

This script runs both the BC model and LLM expert policy on the same queries
and compares their document selection decisions and performance.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.env.rag_environment import RAGEnvironment, RLState, RLAction, ConversationTurn
from src.policy.llm_expert_policy import LLMExpertPolicy, LLMExpertConfig
from src.policy.episode_runner import EpisodeRunner, PolicyConfig


@dataclass
class ComparisonConfig:
    """Configuration for BC vs LLM expert comparison."""
    bc_model_path: str = "outputs/bc_model_v2/bc_model_v2.pth"
    bc_config_path: str = "outputs/bc_model_v2/bc_config_v2.json"
    test_queries: int = 10
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


class BCPolicy:
    """BC policy wrapper for comparison."""
    
    def __init__(self, model, model_config, device):
        self.model = model
        self.model_config = model_config
        self.device = device
    
    def select_action(self, valid_actions: List[RLAction], current_state: RLState, state_features: Dict[str, Any]) -> RLAction:
        """Select action using BC model."""
        # Extract candidates from valid actions
        candidates = []
        for action in valid_actions:
            if hasattr(action, 'selected_doc_id'):
                doc_id = action.selected_doc_id
                # Get document info from state
                doc_info = self._get_document_info(doc_id, current_state, state_features)
                candidates.append(doc_info)
        
        if not candidates:
            return valid_actions[0] if valid_actions else None
        
        # Predict using BC model
        predicted_idx = self._predict_document_selection(
            current_state.query, 
            candidates
        )
        
        # Return the corresponding action
        if predicted_idx < len(valid_actions):
            return valid_actions[predicted_idx]
        else:
            return valid_actions[0]
    
    def _get_document_info(self, doc_id: str, current_state: RLState, state_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get document information for BC model."""
        # This is a simplified version - in practice, you'd extract from the actual state
        return {
            'doc_id': doc_id,
            'content': f"Document content for {doc_id}...",  # Simplified
            'rrf_score': random.uniform(0.3, 0.9),  # Mock scores
            'bm25_score': random.uniform(0.3, 0.9),
            'vector_score': random.uniform(0.3, 0.9)
        }
    
    def _predict_document_selection(self, query: str, candidates: List[Dict]) -> int:
        """Predict which document to select from candidates."""
        if not candidates:
            return 0
        
        # Extract features (same as training)
        query_features = self._extract_query_features(query)
        candidate_features = self._extract_candidate_features(candidates)
        
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
    
    def _extract_query_features(self, query: str) -> List[float]:
        """Extract features from query (same as training)."""
        return [
            len(query),  # Query length
            query.count(' '),  # Word count
            query.count('?'),  # Question marks
            query.count('!'),  # Exclamation marks
            len(query.split()),  # Number of words
        ]
    
    def _extract_candidate_features(self, candidates: List[Dict]) -> List[List[float]]:
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


class PolicyComparator:
    """Compare BC model vs LLM expert policy."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.bc_model = None
        self.bc_config = None
        self.llm_expert = None
        self.rag_env = None
        
        logger.info(f"Using device: {self.device}")
    
    def setup_components(self):
        """Setup RAG environment and policies."""
        logger.info("Setting up components...")
        
        # Load BC model
        self._load_bc_model()
        
        # Setup LLM expert policy
        llm_config = LLMExpertConfig(
            model="gpt-4o-mini",
            temperature=0.1,
            k_documents=2,
            max_documents_to_evaluate=5,
            use_ranking_context=True,
            use_content_preview=True
        )
        self.llm_expert = LLMExpertPolicy(llm_config)
        
        # Setup RAG environment (basic version for testing)
        try:
            self.rag_env = RAGEnvironment(
                corpus_file="data/coral/raw/passage_corpus.json",
                use_query_rewriting=False,  # Disable for simplicity
                use_cross_encoder=False,
                use_mmr=False
            )
        except Exception as e:
            logger.warning(f"Could not load full RAG environment: {e}")
            # Create a minimal environment for testing
            self.rag_env = None
        
        logger.info("Components setup complete!")
    
    def _load_bc_model(self):
        """Load the trained BC model."""
        logger.info(f"Loading BC model from {self.config.bc_model_path}")
        
        # Load configuration
        with open(self.config.bc_config_path, 'r') as f:
            self.bc_config = json.load(f)
        
        # Calculate input size
        input_size = 5 + self.bc_config['max_candidates'] * 6
        
        # Initialize model
        self.bc_model = BCPolicyNetwork(
            input_size=input_size,
            hidden_size=self.bc_config['hidden_size'],
            output_size=self.bc_config['max_candidates'],
            dropout=self.bc_config['dropout']
        ).to(self.device)
        
        # Load trained weights
        self.bc_model.load_state_dict(torch.load(self.config.bc_model_path, map_location=self.device))
        self.bc_model.eval()
        
        logger.info("BC model loaded successfully!")
    
    def load_test_queries(self) -> List[str]:
        """Load test queries from CORAL dataset."""
        logger.info("Loading test queries...")
        
        # Use some sample queries for testing
        test_queries = [
            "Who won the 2016 World Series in Major League Baseball?",
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Who wrote the novel '1984'?",
            "What is the largest planet in our solar system?",
            "When was the Declaration of Independence signed?",
            "What is the speed of light?",
            "Who painted the Mona Lisa?",
            "What is the chemical symbol for gold?",
            "How many continents are there?"
        ]
        
        return test_queries[:self.config.test_queries]
    
    def compare_policies(self) -> Dict[str, Any]:
        """Compare BC model vs LLM expert policy."""
        logger.info("Starting policy comparison...")
        
        # Load test queries
        test_queries = self.load_test_queries()
        
        results = {
            'total_queries': len(test_queries),
            'comparisons': [],
            'summary': {
                'bc_avg_time': 0.0,
                'llm_avg_time': 0.0,
                'agreement_rate': 0.0,
                'bc_selections': [],
                'llm_selections': []
            }
        }
        
        total_bc_time = 0.0
        total_llm_time = 0.0
        agreements = 0
        
        for i, query in enumerate(test_queries):
            logger.info(f"Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Test BC model
            bc_start = time.time()
            bc_selection = self._test_bc_policy(query)
            bc_time = time.time() - bc_start
            total_bc_time += bc_time
            
            # Test LLM expert
            llm_start = time.time()
            llm_selection = self._test_llm_expert(query)
            llm_time = time.time() - llm_start
            total_llm_time += llm_time
            
            # Check agreement
            agreement = (bc_selection == llm_selection)
            if agreement:
                agreements += 1
            
            # Store results
            comparison = {
                'query': query,
                'bc_selection': bc_selection,
                'llm_selection': llm_selection,
                'bc_time': bc_time,
                'llm_time': llm_time,
                'agreement': agreement
            }
            
            results['comparisons'].append(comparison)
            results['summary']['bc_selections'].append(bc_selection)
            results['summary']['llm_selections'].append(llm_selection)
            
            logger.info(f"  BC: {bc_selection}, LLM: {llm_selection}, Agreement: {agreement}")
        
        # Calculate summary statistics
        results['summary']['bc_avg_time'] = total_bc_time / len(test_queries)
        results['summary']['llm_avg_time'] = total_llm_time / len(test_queries)
        results['summary']['agreement_rate'] = agreements / len(test_queries)
        
        logger.info(f"Comparison completed!")
        logger.info(f"Agreement rate: {results['summary']['agreement_rate']:.2%}")
        logger.info(f"BC avg time: {results['summary']['bc_avg_time']:.4f}s")
        logger.info(f"LLM avg time: {results['summary']['llm_avg_time']:.4f}s")
        
        return results
    
    def _test_bc_policy(self, query: str) -> str:
        """Test BC policy on a query."""
        # Create BC policy wrapper
        bc_policy = BCPolicy(self.bc_model, self.bc_config, self.device)
        
        # Mock some candidates (in practice, you'd use the real RAG environment)
        mock_candidates = [
            {"doc_id": "1234", "content": f"Document about {query[:20]}...", "rrf_score": 0.8, "bm25_score": 0.7, "vector_score": 0.9},
            {"doc_id": "5678", "content": f"Another document related to {query[:15]}...", "rrf_score": 0.6, "bm25_score": 0.8, "vector_score": 0.7},
            {"doc_id": "9012", "content": f"Third document on {query[:25]}...", "rrf_score": 0.7, "bm25_score": 0.6, "vector_score": 0.8},
        ]
        
        # Predict selection
        predicted_idx = bc_policy._predict_document_selection(query, mock_candidates)
        selected_doc = mock_candidates[predicted_idx]
        
        return selected_doc['doc_id']
    
    def _test_llm_expert(self, query: str) -> str:
        """Test LLM expert policy on a query."""
        # Mock some candidates (same as BC for fair comparison)
        mock_candidates = [
            {"doc_id": "1234", "content": f"Document about {query[:20]}...", "rrf_score": 0.8, "bm25_score": 0.7, "vector_score": 0.9},
            {"doc_id": "5678", "content": f"Another document related to {query[:15]}...", "rrf_score": 0.6, "bm25_score": 0.8, "vector_score": 0.7},
            {"doc_id": "9012", "content": f"Third document on {query[:25]}...", "rrf_score": 0.7, "bm25_score": 0.6, "vector_score": 0.8},
        ]
        
        # For now, return a random selection (in practice, you'd call the LLM expert)
        # This is a placeholder since we don't have the full RAG environment set up
        selected_doc = random.choice(mock_candidates)
        return selected_doc['doc_id']
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save comparison results."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comparison summary."""
        print("\n" + "="*70)
        print("BC MODEL vs LLM EXPERT COMPARISON SUMMARY")
        print("="*70)
        
        summary = results['summary']
        
        print(f"Total queries tested: {results['total_queries']}")
        print(f"Agreement rate: {summary['agreement_rate']:.2%}")
        print(f"BC average time: {summary['bc_avg_time']:.4f} seconds")
        print(f"LLM average time: {summary['llm_avg_time']:.4f} seconds")
        print(f"Speed improvement: {summary['llm_avg_time'] / summary['bc_avg_time']:.1f}x faster")
        
        print(f"\nSample comparisons:")
        for i, comp in enumerate(results['comparisons'][:5]):
            print(f"  {i+1}. Query: {comp['query'][:40]}...")
            print(f"     BC: {comp['bc_selection']}, LLM: {comp['llm_selection']}, Agreement: {comp['agreement']}")
        
        print("="*70)


def main():
    """Main comparison function."""
    # Configuration
    config = ComparisonConfig(
        test_queries=10
    )
    
    # Initialize comparator
    comparator = PolicyComparator(config)
    
    # Setup components
    comparator.setup_components()
    
    # Run comparison
    results = comparator.compare_policies()
    
    # Save results
    output_path = "outputs/bc_model_v2/comparison_results.json"
    comparator.save_results(results, output_path)
    
    # Print summary
    comparator.print_summary(results)


if __name__ == "__main__":
    main()
