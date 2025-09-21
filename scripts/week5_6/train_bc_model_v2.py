#!/usr/bin/env python3
"""
Train Behavioral Cloning (BC) model with LLM expert trajectories - Version 2.

This version uses a more practical approach:
- Instead of predicting exact document IDs, predict which document to select from available candidates
- Uses a smaller, more manageable action space
- Focuses on relative ranking rather than absolute document IDs
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BCTrainingConfig:
    """Configuration for BC training."""
    learning_rate: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 100
    hidden_size: int = 128
    dropout: float = 0.1
    weight_decay: float = 1e-5
    validation_split: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_candidates: int = 10  # Maximum number of candidate documents to consider


class BCTrainingExample:
    """Single training example for BC."""
    def __init__(self, query: str, candidates: List[Dict], selected_idx: int, reward: float):
        self.query = query
        self.candidates = candidates  # List of candidate documents with scores
        self.selected_idx = selected_idx  # Index of selected document in candidates
        self.reward = reward


class BCDataset(Dataset):
    """Dataset for BC training."""
    
    def __init__(self, examples: List[BCTrainingExample], max_candidates: int = 10):
        self.examples = examples
        self.max_candidates = max_candidates
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract features
        query_features = self._extract_query_features(example.query)
        candidate_features = self._extract_candidate_features(example.candidates)
        
        # Pad or truncate candidates to max_candidates
        if len(candidate_features) < self.max_candidates:
            # Pad with zeros
            padding = [[0.0] * len(candidate_features[0])] * (self.max_candidates - len(candidate_features))
            candidate_features.extend(padding)
        else:
            # Truncate to max_candidates
            candidate_features = candidate_features[:self.max_candidates]
        
        # Flatten features
        features = query_features + [item for sublist in candidate_features for item in sublist]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'action': torch.tensor(example.selected_idx, dtype=torch.long),
            'reward': torch.tensor(example.reward, dtype=torch.float32)
        }
    
    def _extract_query_features(self, query: str) -> List[float]:
        """Extract features from query."""
        return [
            len(query),  # Query length
            query.count(' '),  # Word count
            query.count('?'),  # Question marks
            query.count('!'),  # Exclamation marks
            len(query.split()),  # Number of words
        ]
    
    def _extract_candidate_features(self, candidates: List[Dict]) -> List[List[float]]:
        """Extract features from candidate documents."""
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


class BCPolicyNetwork(nn.Module):
    """Neural network for BC policy."""
    
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


class BCTrainer:
    """Trainer for Behavioral Cloning model."""
    
    def __init__(self, config: BCTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")
        
        # Will be initialized after loading data
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def load_trajectories(self, trajectory_file: str) -> List[BCTrainingExample]:
        """Load LLM expert trajectories and convert to training examples."""
        logger.info(f"Loading trajectories from {trajectory_file}")
        
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
        
        training_examples = []
        
        for traj in trajectories:
            # Extract information from trajectory
            query = traj.get('query', '')
            documents_selected = traj.get('documents_selected', [])
            total_reward = traj.get('total_reward', 0.0)
            
            # Create training examples for each document selection
            for i, doc in enumerate(documents_selected):
                # Create candidates list (all documents, with the selected one first)
                candidates = documents_selected.copy()
                
                # Move the selected document to the front
                selected_doc = candidates.pop(i)
                candidates.insert(0, selected_doc)
                
                # Create training example
                example = BCTrainingExample(
                    query=query,
                    candidates=candidates,
                    selected_idx=0,  # Selected document is now at index 0
                    reward=total_reward / len(documents_selected)  # Average reward per selection
                )
                training_examples.append(example)
        
        logger.info(f"Loaded {len(training_examples)} training examples from {len(trajectories)} trajectories")
        return training_examples
    
    def prepare_data(self, examples: List[BCTrainingExample]) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        # Split data
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - self.config.validation_split))
        
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        logger.info(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")
        
        # Create datasets
        train_dataset = BCDataset(train_examples, self.config.max_candidates)
        val_dataset = BCDataset(val_examples, self.config.max_candidates)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def initialize_model(self, input_size: int, output_size: int):
        """Initialize the BC model."""
        self.model = BCPolicyNetwork(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            output_size=output_size,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized BC model with input_size={input_size}, output_size={output_size}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # Forward pass
            logits = self.model(features)
            loss = self.criterion(logits, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == actions).sum().item()
            total_predictions += actions.size(0)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, actions)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == actions).sum().item()
                total_predictions += actions.size(0)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train the BC model."""
        logger.info(f"Starting BC training for {self.config.num_epochs} epochs...")
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            train_losses.append(train_metrics['loss'])
            train_accuracies.append(train_metrics['accuracy'])
            
            # Validation
            val_metrics = self.validate(val_loader)
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_state = self.model.state_dict().copy()
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
                )
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def save_model(self, output_dir: str):
        """Save the trained model and training history."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "bc_model_v2.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = output_path / "bc_config_v2.json"
        with open(config_path, 'w') as f:
            json.dump({
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs,
                'hidden_size': self.config.hidden_size,
                'dropout': self.config.dropout,
                'weight_decay': self.config.weight_decay,
                'device': self.config.device,
                'max_candidates': self.config.max_candidates
            }, f, indent=2)
        
        logger.info(f"Saved BC model to {model_path}")
        logger.info(f"Saved BC config to {config_path}")


def main():
    """Main training function."""
    # Configuration
    config = BCTrainingConfig(
        learning_rate=1e-3,
        batch_size=16,
        num_epochs=100,
        hidden_size=128,
        dropout=0.1,
        max_candidates=10
    )
    
    # Paths
    trajectory_file = "outputs/llm_expert_trajectories/llm_expert_trajectories.json"
    output_dir = "outputs/bc_model_v2"
    
    # Initialize trainer
    trainer = BCTrainer(config)
    
    # Load trajectories
    training_examples = trainer.load_trajectories(trajectory_file)
    
    if not training_examples:
        logger.error("No training examples found!")
        return
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(training_examples)
    
    # Get input/output sizes from first example
    sample_example = training_examples[0]
    sample_dataset = BCDataset([sample_example], config.max_candidates)
    sample_batch = next(iter(DataLoader(sample_dataset, batch_size=1)))
    
    input_size = sample_batch['features'].shape[1]
    output_size = config.max_candidates  # Predict which of the max_candidates to select
    
    logger.info(f"Input size: {input_size}, Output size: {output_size}")
    
    # Initialize model
    trainer.initialize_model(input_size, output_size)
    
    # Train model
    training_history = trainer.train(train_loader, val_loader)
    
    # Save model
    trainer.save_model(output_dir)
    
    # Save training history
    history_path = Path(output_dir) / "training_history_v2.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("BC training completed!")
    logger.info(f"Best validation accuracy: {training_history['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
