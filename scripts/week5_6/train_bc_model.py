#!/usr/bin/env python3
"""
Train Behavioral Cloning (BC) model with LLM expert trajectories.

This script trains a neural network to imitate the expert behavior demonstrated
in the LLM expert trajectories by learning to predict document selections
given the current state and available actions.
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
    batch_size: int = 32
    num_epochs: int = 50
    hidden_size: int = 256
    dropout: float = 0.1
    weight_decay: float = 1e-5
    validation_split: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BCTrainingExample:
    """Single training example for BC."""
    def __init__(self, state_features: Dict[str, Any], action: Dict[str, Any], reward: float):
        self.state_features = state_features
        self.action = action
        self.reward = reward


class BCDataset(Dataset):
    """Dataset for BC training."""
    
    def __init__(self, examples: List[BCTrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract state features
        state_features = self._extract_state_features(example.state_features)
        
        # Extract action (document ID to select)
        action = example.action.get('selected_doc_id', -1)
        
        return {
            'state_features': torch.tensor(state_features, dtype=torch.float32),
            'action': torch.tensor(action, dtype=torch.long),
            'reward': torch.tensor(example.reward, dtype=torch.float32)
        }
    
    def _extract_state_features(self, state_features: Dict[str, Any]) -> List[float]:
        """Extract numerical features from state."""
        features = []
        
        # Query features (simplified - just length and some basic stats)
        query = state_features.get('query', '')
        features.extend([
            len(query),  # Query length
            query.count(' '),  # Word count
            query.count('?'),  # Question marks
        ])
        
        # Conversation history features
        history = state_features.get('conversation_history', [])
        features.extend([
            len(history),  # Number of previous turns
            sum(len(turn.get('question', '')) for turn in history),  # Total history length
        ])
        
        # Available documents features
        available_docs = state_features.get('available_documents', [])
        features.extend([
            len(available_docs),  # Number of available documents
        ])
        
        # Document features (for top 5 documents)
        for i in range(5):
            if i < len(available_docs):
                doc = available_docs[i]
                features.extend([
                    doc.get('rrf_score', 0.0),
                    doc.get('bm25_score', 0.0),
                    doc.get('vector_score', 0.0),
                    len(doc.get('content', '')),
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # Padding
        
        # Selected documents features
        selected_docs = state_features.get('selected_documents', [])
        features.extend([
            len(selected_docs),  # Number of already selected documents
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
        
    def forward(self, state_features):
        return self.network(state_features)


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
            conversation_history = traj.get('conversation_history', [])
            selected_doc_ids = traj.get('selected_doc_ids', [])
            documents_selected = traj.get('documents_selected', [])
            total_reward = traj.get('total_reward', 0.0)
            
            # Create training examples for each document selection
            for i, doc_id in enumerate(selected_doc_ids):
                # Find the corresponding document info
                doc_info = None
                for doc in documents_selected:
                    if doc.get('doc_id') == doc_id:
                        doc_info = doc
                        break
                
                # Create state features (simplified for this format)
                state_features = {
                    'query': query,
                    'conversation_history': conversation_history,
                    'available_documents': documents_selected,  # All available docs
                    'selected_documents': selected_doc_ids[:i],  # Previously selected docs
                    'current_document': doc_info
                }
                
                # Create action (document ID to select)
                action = {
                    'selected_doc_id': int(doc_id) if doc_id.isdigit() else 0
                }
                
                # Use average reward per selection
                reward = total_reward / len(selected_doc_ids) if selected_doc_ids else 0.0
                
                # Create training example
                example = BCTrainingExample(state_features, action, reward)
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
        train_dataset = BCDataset(train_examples)
        val_dataset = BCDataset(val_examples)
        
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
            state_features = batch['state_features'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # Forward pass
            logits = self.model(state_features)
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
                state_features = batch['state_features'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                logits = self.model(state_features)
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
            if (epoch + 1) % 10 == 0:
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
        model_path = output_path / "bc_model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = output_path / "bc_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs,
                'hidden_size': self.config.hidden_size,
                'dropout': self.config.dropout,
                'weight_decay': self.config.weight_decay,
                'device': self.config.device
            }, f, indent=2)
        
        logger.info(f"Saved BC model to {model_path}")
        logger.info(f"Saved BC config to {config_path}")


def main():
    """Main training function."""
    # Configuration
    config = BCTrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=50,
        hidden_size=256,
        dropout=0.1
    )
    
    # Paths
    trajectory_file = "outputs/llm_expert_trajectories/llm_expert_trajectories.json"
    output_dir = "outputs/bc_model"
    
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
    sample_dataset = BCDataset([sample_example])
    sample_batch = next(iter(DataLoader(sample_dataset, batch_size=1)))
    
    input_size = sample_batch['state_features'].shape[1]
    
    # Determine output size (max document ID + 1)
    max_doc_id = 0
    for example in training_examples:
        action = example.action.get('selected_doc_id', -1)
        if action > max_doc_id:
            max_doc_id = action
    
    output_size = max_doc_id + 1
    
    logger.info(f"Input size: {input_size}, Output size: {output_size}")
    
    # Initialize model
    trainer.initialize_model(input_size, output_size)
    
    # Train model
    training_history = trainer.train(train_loader, val_loader)
    
    # Save model
    trainer.save_model(output_dir)
    
    # Save training history
    history_path = Path(output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("BC training completed!")
    logger.info(f"Best validation accuracy: {training_history['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
