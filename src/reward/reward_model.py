"""
Lightweight reward model for offline scoring.

Distills preferences from LLM-as-a-Judge into a fast, lightweight model
for scoring answers during training and inference.
"""

import json
import logging
import pickle
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RewardExample:
    """Represents a training example for the reward model."""
    query: str
    answer: str
    context: List[str]
    preference_score: float  # 0.0 to 1.0 (from LLM judge)
    criteria_scores: Dict[str, float]  # Individual criteria scores
    metadata: Dict[str, Any] = None


class RewardDataset(Dataset):
    """Dataset for reward model training."""
    
    def __init__(self, examples: List[RewardExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "query": example.query,
            "answer": example.answer,
            "context": example.context,
            "preference_score": example.preference_score,
            "criteria_scores": example.criteria_scores,
            "metadata": example.metadata
        }


class LightweightRewardModel(nn.Module):
    """
    Lightweight reward model for fast inference.
    
    Uses a sentence transformer backbone with a small MLP head
    to predict reward scores from query-answer-context triplets.
    """
    
    def __init__(self,
                 backbone_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 device: Optional[str] = None):
        """
        Initialize the reward model.
        
        Args:
            backbone_model: Sentence transformer model for encoding
            hidden_dim: Hidden dimension for MLP head
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = SentenceTransformer(backbone_model)
        self.backbone.to(self.device)
        
        # Get embedding dimension
        embedding_dim = self.backbone.get_sentence_embedding_dimension()
        
        # MLP head for reward prediction
        self.reward_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),  # query + answer + context
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Individual criteria heads
        self.criteria_heads = nn.ModuleDict({
            "accuracy": nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            "completeness": nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            "clarity": nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            "relevance": nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            "attribution": nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
        self.to(self.device)
        
        logger.info(f"Initialized reward model with backbone: {backbone_model}")
        logger.info(f"Embedding dimension: {embedding_dim}, Hidden dimension: {hidden_dim}")
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode list of texts using the backbone model."""
        with torch.no_grad():
            embeddings = self.backbone.encode(texts, convert_to_tensor=True, device=self.device)
        return embeddings
    
    def forward(self, query: str, answer: str, context: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the reward model.
        
        Args:
            query: Input query
            answer: Answer to score
            context: Supporting context documents
            
        Returns:
            Dictionary with reward scores
        """
        # Prepare input text
        context_str = " ".join(context) if context else ""
        
        # Encode query, answer, and context
        query_emb = self.encode_texts([query])
        answer_emb = self.encode_texts([answer])
        context_emb = self.encode_texts([context_str])
        
        # Concatenate embeddings
        combined_emb = torch.cat([query_emb, answer_emb, context_emb], dim=1)
        
        # Predict overall reward
        reward_score = self.reward_head(combined_emb)
        
        # Predict individual criteria scores
        criteria_scores = {}
        for criterion, head in self.criteria_heads.items():
            criteria_scores[criterion] = head(combined_emb)
        
        return {
            "reward_score": reward_score,
            "criteria_scores": criteria_scores
        }


class RewardModelTrainer:
    """
    Trainer for the lightweight reward model.
    
    Handles training, validation, and evaluation of the reward model
    using preference data from LLM-as-a-Judge.
    """
    
    def __init__(self,
                 model: LightweightRewardModel,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 batch_size: int = 32,
                 num_epochs: int = 10,
                 device: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Reward model to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            batch_size: Training batch size
            num_epochs: Number of training epochs
            device: Device to train on
        """
        self.model = model
        self.device = device or model.device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized trainer with LR={learning_rate}, batch_size={batch_size}")
    
    def prepare_data(self, examples: List[RewardExample]) -> DataLoader:
        """Prepare data for training."""
        dataset = RewardDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        return dataloader
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        return {
            "queries": [item["query"] for item in batch],
            "answers": [item["answer"] for item in batch],
            "contexts": [item["context"] for item in batch],
            "preference_scores": torch.tensor([item["preference_score"] for item in batch], dtype=torch.float32),
            "criteria_scores": [item["criteria_scores"] for item in batch],
            "metadata": [item["metadata"] for item in batch]
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_reward_loss = 0.0
        total_criteria_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Get batch data
            queries = batch["queries"]
            answers = batch["answers"]
            contexts = batch["contexts"]
            target_scores = batch["preference_scores"].to(self.device)
            target_criteria = batch["criteria_scores"]
            
            # Forward pass
            batch_reward_loss = 0.0
            batch_criteria_loss = 0.0
            
            for i in range(len(queries)):
                outputs = self.model(queries[i], answers[i], contexts[i])
                
                # Reward loss
                reward_loss = self.criterion(outputs["reward_score"], target_scores[i:i+1])
                batch_reward_loss += reward_loss
                
                # Criteria losses
                for criterion, score in outputs["criteria_scores"].items():
                    if criterion in target_criteria[i]:
                        target_criterion_score = torch.tensor(
                            [target_criteria[i][criterion]], dtype=torch.float32
                        ).to(self.device)
                        criterion_loss = self.criterion(score, target_criterion_score)
                        batch_criteria_loss += criterion_loss
            
            # Average losses
            batch_reward_loss /= len(queries)
            batch_criteria_loss /= len(queries)
            total_loss_batch = batch_reward_loss + 0.1 * batch_criteria_loss  # Weight criteria loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_reward_loss += batch_reward_loss.item()
            total_criteria_loss += batch_criteria_loss.item()
            num_batches += 1
        
        return {
            "total_loss": total_loss / num_batches,
            "reward_loss": total_reward_loss / num_batches,
            "criteria_loss": total_criteria_loss / num_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                queries = batch["queries"]
                answers = batch["answers"]
                contexts = batch["contexts"]
                target_scores = batch["preference_scores"].to(self.device)
                
                batch_loss = 0.0
                for i in range(len(queries)):
                    outputs = self.model(queries[i], answers[i], contexts[i])
                    loss = self.criterion(outputs["reward_score"], target_scores[i:i+1])
                    batch_loss += loss
                    
                    predictions.append(outputs["reward_score"].item())
                    targets.append(target_scores[i].item())
                
                total_loss += batch_loss.item() / len(queries)
                num_batches += 1
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # Convert to binary for accuracy (threshold at 0.5)
        binary_preds = (predictions > 0.5).astype(int)
        binary_targets = (targets > 0.5).astype(int)
        accuracy = accuracy_score(binary_targets, binary_preds)
        
        try:
            auc = roc_auc_score(binary_targets, predictions)
        except ValueError:
            auc = 0.5  # Default if only one class present
        
        return {
            "loss": total_loss / num_batches,
            "mse": mse,
            "mae": mae,
            "accuracy": accuracy,
            "auc": auc
        }
    
    def train(self, train_examples: List[RewardExample], 
              val_examples: Optional[List[RewardExample]] = None,
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the reward model.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples (optional)
            save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        logger.info(f"Starting training with {len(train_examples)} examples")
        
        # Prepare data
        train_loader = self.prepare_data(train_examples)
        val_loader = self.prepare_data(val_examples) if val_examples else None
        
        # Training history
        history = {
            "train_loss": [],
            "train_reward_loss": [],
            "train_criteria_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auc": []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["total_loss"])
            history["train_reward_loss"].append(train_metrics["reward_loss"])
            history["train_criteria_loss"].append(train_metrics["criteria_loss"])
            
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Reward Loss: {train_metrics['reward_loss']:.4f}, "
                       f"Criteria Loss: {train_metrics['criteria_loss']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_auc"].append(val_metrics["auc"])
                
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Accuracy: {val_metrics['accuracy']:.4f}, "
                           f"AUC: {val_metrics['auc']:.4f}")
                
                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    if save_path:
                        self.save_model(save_path)
                        logger.info(f"Saved best model to {save_path}")
        
        return history
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "backbone_model": self.model.backbone.get_sentence_embedding_dimension(),
                "hidden_dim": self.model.reward_head[0].out_features,
                "dropout": self.model.reward_head[2].p
            }
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(model_data["model_state_dict"])
        logger.info(f"Loaded model from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example data
    examples = [
        RewardExample(
            query="Who won the FA Cup in 2020?",
            answer="Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
            context=["Arsenal defeated Chelsea 2-1 in the 2020 FA Cup final."],
            preference_score=0.9,
            criteria_scores={"accuracy": 0.9, "completeness": 0.8, "clarity": 0.9, "relevance": 0.9, "attribution": 0.8}
        ),
        RewardExample(
            query="Who won the FA Cup in 2020?",
            answer="The FA Cup is an annual football competition in England.",
            context=["The FA Cup is England's premier cup competition."],
            preference_score=0.3,
            criteria_scores={"accuracy": 0.7, "completeness": 0.2, "clarity": 0.8, "relevance": 0.3, "attribution": 0.6}
        )
    ]
    
    # Initialize model and trainer
    model = LightweightRewardModel()
    trainer = RewardModelTrainer(model, num_epochs=2)
    
    # Train
    history = trainer.train(examples)
    
    # Test inference
    model.eval()
    with torch.no_grad():
        result = model("Who won the FA Cup in 2020?", 
                      "Arsenal won the FA Cup in 2020.", 
                      ["Arsenal defeated Chelsea in the final."])
        print(f"Reward score: {result['reward_score'].item():.3f}")
        print(f"Criteria scores: {[(k, v.item()) for k, v in result['criteria_scores'].items()]}")
