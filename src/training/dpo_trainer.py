"""
DPO/IPO/ORPO training with TRL and QLoRA.

Implements preference-based training for generator alignment
using Direct Preference Optimization and related methods.
"""

import json
import logging
import torch
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# TRL imports
try:
    from trl import DPOTrainer, DPOTrainingArguments, DPOConfig
    from trl import IPOTrainer, IPOTrainingArguments, IPOConfig
    from trl import ORPOTrainer, ORPOTrainingArguments, ORPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logger.warning("TRL not available. Install with: pip install trl")

# Transformers imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")

# PEFT imports for QLoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. Install with: pip install peft")


@dataclass
class TrainingConfig:
    """Configuration for DPO training."""
    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"  # Default to smaller model for testing
    max_length: int = 512
    temperature: float = 0.7
    
    # Training configuration
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # QLoRA configuration
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will be set based on model
    
    # DPO configuration
    beta: float = 0.1  # DPO temperature parameter
    label_smoothing: float = 0.0
    
    # Output configuration
    output_dir: str = "outputs/dpo_training"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for common models
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class PreferenceData:
    """Represents preference data for training."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any] = None


class DPOTrainerWrapper:
    """
    Wrapper for DPO/IPO/ORPO training with TRL and QLoRA.
    
    Provides a unified interface for preference-based training
    with support for different algorithms and configurations.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the DPO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        if not TRL_AVAILABLE:
            raise ImportError("TRL is required for DPO training. Install with: pip install trl")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required. Install with: pip install transformers")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Initialized DPO trainer with config: {config}")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with QLoRA if enabled."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if self.config.use_qlora and PEFT_AVAILABLE:
            # QLoRA configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("Applied QLoRA configuration")
            
        else:
            # Standard model loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Loaded standard model")
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def prepare_preference_data(self, preferences: List[PreferenceData]) -> List[Dict[str, str]]:
        """
        Prepare preference data for training.
        
        Args:
            preferences: List of preference examples
            
        Returns:
            List of formatted training examples
        """
        formatted_data = []
        
        for pref in preferences:
            # Format prompt for the model
            prompt = self._format_prompt(pref.prompt)
            
            formatted_data.append({
                "prompt": prompt,
                "chosen": pref.chosen,
                "rejected": pref.rejected
            })
        
        logger.info(f"Prepared {len(formatted_data)} preference examples")
        return formatted_data
    
    def _format_prompt(self, query: str) -> str:
        """
        Format query as a prompt for the model.
        
        Args:
            query: Input query
            
        Returns:
            Formatted prompt
        """
        # Simple prompt format - can be customized based on model
        return f"Human: {query}\n\nAssistant:"
    
    def train_dpo(self, 
                  train_data: List[Dict[str, str]], 
                  eval_data: Optional[List[Dict[str, str]]] = None,
                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train using Direct Preference Optimization (DPO).
        
        Args:
            train_data: Training preference data
            eval_data: Evaluation preference data
            save_path: Path to save the trained model
            
        Returns:
            Training results
        """
        if not self.model or not self.tokenizer:
            self.setup_model_and_tokenizer()
        
        # DPO training arguments
        training_args = DPOTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_data else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_data else False,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # DPO configuration
        dpo_config = DPOConfig(
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
        )
        
        # Initialize DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use the same model as reference
            args=training_args,
            beta=dpo_config.beta,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_length // 2,
        )
        
        # Train
        logger.info("Starting DPO training...")
        train_result = self.trainer.train()
        
        # Save model
        if save_path:
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved trained model to {save_path}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "eval_loss": train_result.metrics.get("eval_loss", None)
        }
    
    def train_ipo(self, 
                  train_data: List[Dict[str, str]], 
                  eval_data: Optional[List[Dict[str, str]]] = None,
                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train using Identity Preference Optimization (IPO).
        
        Args:
            train_data: Training preference data
            eval_data: Evaluation preference data
            save_path: Path to save the trained model
            
        Returns:
            Training results
        """
        if not self.model or not self.tokenizer:
            self.setup_model_and_tokenizer()
        
        # IPO training arguments
        training_args = IPOTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_data else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_data else False,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # IPO configuration
        ipo_config = IPOConfig(
            beta=self.config.beta,
        )
        
        # Initialize IPO trainer
        self.trainer = IPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            beta=ipo_config.beta,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_length // 2,
        )
        
        # Train
        logger.info("Starting IPO training...")
        train_result = self.trainer.train()
        
        # Save model
        if save_path:
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved trained model to {save_path}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "eval_loss": train_result.metrics.get("eval_loss", None)
        }
    
    def train_orpo(self, 
                   train_data: List[Dict[str, str]], 
                   eval_data: Optional[List[Dict[str, str]]] = None,
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train using Odds Ratio Preference Optimization (ORPO).
        
        Args:
            train_data: Training preference data
            eval_data: Evaluation preference data
            save_path: Path to save the trained model
            
        Returns:
            Training results
        """
        if not self.model or not self.tokenizer:
            self.setup_model_and_tokenizer()
        
        # ORPO training arguments
        training_args = ORPOTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_data else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_data else False,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # ORPO configuration
        orpo_config = ORPOConfig(
            beta=self.config.beta,
        )
        
        # Initialize ORPO trainer
        self.trainer = ORPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            beta=orpo_config.beta,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_length // 2,
        )
        
        # Train
        logger.info("Starting ORPO training...")
        train_result = self.trainer.train()
        
        # Save model
        if save_path:
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved trained model to {save_path}")
        
        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "eval_loss": train_result.metrics.get("eval_loss", None)
        }
    
    def generate_answer(self, 
                       query: str, 
                       context: Optional[List[str]] = None,
                       max_new_tokens: int = 256) -> str:
        """
        Generate an answer using the trained model.
        
        Args:
            query: Input query
            context: Optional context documents
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call setup_model_and_tokenizer() first.")
        
        # Format prompt
        if context:
            context_str = " ".join(context[:3])  # Use top 3 context documents
            prompt = f"Context: {context_str}\n\nHuman: {query}\n\nAssistant:"
        else:
            prompt = f"Human: {query}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (everything after "Assistant:")
        if "Assistant:" in generated_text:
            answer = generated_text.split("Assistant:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def evaluate_model(self, 
                      test_data: List[Dict[str, str]], 
                      reward_model=None) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_data: Test preference data
            reward_model: Optional reward model for evaluation
            
        Returns:
            Evaluation metrics
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded.")
        
        self.model.eval()
        
        total_rewards = []
        preference_accuracy = 0
        
        for example in test_data:
            # Generate answers
            chosen_gen = self.generate_answer(example["prompt"])
            rejected_gen = self.generate_answer(example["prompt"])
            
            # Score with reward model if available
            if reward_model:
                try:
                    chosen_score = reward_model.score(example["prompt"], chosen_gen)
                    rejected_score = reward_model.score(example["prompt"], rejected_gen)
                    total_rewards.extend([chosen_score, rejected_score])
                    
                    # Check if model preference matches ground truth
                    if chosen_score > rejected_score:
                        preference_accuracy += 1
                        
                except Exception as e:
                    logger.warning(f"Error scoring with reward model: {e}")
        
        metrics = {
            "avg_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "preference_accuracy": preference_accuracy / len(test_data) if test_data else 0.0,
            "num_evaluated": len(test_data)
        }
        
        return metrics


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example preference data
    example_preferences = [
        PreferenceData(
            prompt="Who won the FA Cup in 2020?",
            chosen="Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final at Wembley Stadium.",
            rejected="The FA Cup is an annual football competition in England.",
            metadata={"source": "example"}
        ),
        PreferenceData(
            prompt="What is the capital of France?",
            chosen="The capital of France is Paris, which is located in the north-central part of the country.",
            rejected="France is a country in Europe.",
            metadata={"source": "example"}
        )
    ]
    
    # Training configuration
    config = TrainingConfig(
        model_name="microsoft/DialoGPT-small",  # Use smaller model for testing
        batch_size=2,
        num_epochs=1,
        use_qlora=False,  # Disable QLoRA for testing
        output_dir="outputs/dpo_test"
    )
    
    # Initialize trainer
    trainer = DPOTrainerWrapper(config)
    
    # Prepare data
    train_data = trainer.prepare_preference_data(example_preferences)
    
    print(f"Prepared {len(train_data)} training examples")
    print("DPO trainer ready for use")
    
    # Note: Actual training would require more data and computational resources
    # trainer.train_dpo(train_data, save_path="outputs/dpo_model")
