"""
BC (Behavioral Cloning) Policy for document selection in RAG environment.

This policy uses a trained neural network to imitate expert document selection behavior.
"""

import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.env.rag_environment import RLAction, RLState

logger = logging.getLogger(__name__)


@dataclass
class BCConfig:
    """Configuration for BC policy."""
    model_path: str = "outputs/bc_model_v3/bc_model_v3.pth"
    config_path: str = "outputs/bc_model_v3/bc_config_v3.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_candidates: int = 10


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
    """
    BC Policy that uses a trained neural network for document selection.
    
    This policy loads a pre-trained BC model and uses it to select documents
    from available candidates based on query and document features.
    """
    
    def __init__(self, config: BCConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.model_config = None
        
        # Load the model
        self._load_model()
        
        logger.info(f"Initialized BCPolicy with model from {config.model_path}")
    
    def _load_model(self):
        """Load the trained BC model and configuration."""
        try:
            # Load configuration
            with open(self.config.config_path, 'r') as f:
                self.model_config = json.load(f)
            
            # Calculate input size
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
            
        except Exception as e:
            logger.error(f"Failed to load BC model: {e}")
            raise
    
    def select_action(self, valid_actions: List[RLAction], current_state: RLState, state_features: Dict[str, Any]) -> RLAction:
        """
        Select an action using the BC model.
        
        Args:
            valid_actions: List of valid actions (document selections)
            current_state: Current RAG state
            state_features: Additional state features
            
        Returns:
            Selected action
        """
        if not valid_actions:
            logger.warning("No valid actions available")
            return None
        
        # Extract candidates from valid actions
        candidates = self._extract_candidates_from_actions(valid_actions, current_state, state_features)
        
        if not candidates:
            logger.warning("No candidates extracted from actions")
            return valid_actions[0]
        
        # Predict using BC model
        predicted_idx = self._predict_document_selection(
            current_state.query, 
            candidates
        )
        
        # Return the corresponding action
        if predicted_idx < len(valid_actions):
            selected_action = valid_actions[predicted_idx]
            logger.debug(f"BC selected action: {selected_action.doc_id}")
            return selected_action
        else:
            logger.warning(f"Predicted index {predicted_idx} out of range, using first action")
            return valid_actions[0]
    
    def _extract_candidates_from_actions(self, valid_actions: List[RLAction], current_state: RLState, state_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract candidate document information from valid actions."""
        candidates = []
        
        for action in valid_actions:
            if hasattr(action, 'doc_id'):
                doc_id = action.doc_id
                
                # Get document info from state features or current state
                doc_info = self._get_document_info(doc_id, current_state, state_features)
                candidates.append(doc_info)
        
        return candidates
    
    def _get_document_info(self, doc_id: str, current_state: RLState, state_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get document information for BC model."""
        # Try to get document info from state features first
        if 'available_documents' in state_features:
            for doc in state_features['available_documents']:
                if doc.get('doc_id') == doc_id:
                    return {
                        'doc_id': doc_id,
                        'content': doc.get('content', ''),
                        'rrf_score': doc.get('rrf_score', 0.0),
                        'bm25_score': doc.get('bm25_score', 0.0),
                        'vector_score': doc.get('vector_score', 0.0)
                    }
        
        # Fallback: create basic document info
        return {
            'doc_id': doc_id,
            'content': f"Document content for {doc_id}...",
            'rrf_score': 0.5,  # Default scores
            'bm25_score': 0.5,
            'vector_score': 0.5
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
