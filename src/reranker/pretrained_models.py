"""
Pre-trained cross-encoder models for reranking.

This module provides a curated list of pre-trained cross-encoder models
that work well for different use cases without requiring training.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PretrainedModel:
    """Represents a pre-trained cross-encoder model."""
    name: str
    description: str
    use_case: str
    size: str
    performance: str


# Pre-trained cross-encoder models for reranking
PRETRAINED_MODELS = {
    # MS MARCO models (best for general passage ranking)
    "ms-marco-MiniLM-L-6-v2": PretrainedModel(
        name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Lightweight model trained on MS MARCO passage ranking",
        use_case="General passage reranking, good speed/quality balance",
        size="Small (22M parameters)",
        performance="Good"
    ),
    
    "ms-marco-MiniLM-L-12-v2": PretrainedModel(
        name="cross-encoder/ms-marco-MiniLM-L-12-v2", 
        description="Larger model trained on MS MARCO passage ranking",
        use_case="Higher quality passage reranking",
        size="Medium (33M parameters)",
        performance="Better"
    ),
    
    "ms-marco-MiniLM-L-6-v2-distilbert": PretrainedModel(
        name="cross-encoder/ms-marco-MiniLM-L-6-v2-distilbert",
        description="DistilBERT-based model for MS MARCO",
        use_case="Fast inference with good quality",
        size="Small (66M parameters)",
        performance="Good"
    ),
    
    # Natural Questions models (good for Q&A)
    "nq-distilbert-base-v1": PretrainedModel(
        name="cross-encoder/nq-distilbert-base-v1",
        description="DistilBERT model trained on Natural Questions",
        use_case="Question-answering and factoid queries",
        size="Medium (66M parameters)",
        performance="Good for Q&A"
    ),
    
    # Multi-lingual models
    "xlm-r-100langs-bert-base-nli-stsb": PretrainedModel(
        name="cross-encoder/xlm-r-100langs-bert-base-nli-stsb",
        description="Multilingual XLM-RoBERTa model",
        use_case="Multilingual queries and documents",
        size="Large (270M parameters)",
        performance="Good for multiple languages"
    ),
    
    # Specialized models
    "quora-distilbert-base": PretrainedModel(
        name="cross-encoder/quora-distilbert-base",
        description="DistilBERT trained on Quora duplicate questions",
        use_case="Semantic similarity and duplicate detection",
        size="Medium (66M parameters)",
        performance="Good for similarity"
    ),
    
    "sts-distilbert-base": PretrainedModel(
        name="cross-encoder/sts-distilbert-base",
        description="DistilBERT trained on semantic textual similarity",
        use_case="Semantic similarity scoring",
        size="Medium (66M parameters)",
        performance="Good for similarity"
    )
}


def get_recommended_models(use_case: str = "general") -> List[PretrainedModel]:
    """
    Get recommended models for specific use cases.
    
    Args:
        use_case: Use case type ("general", "qa", "multilingual", "similarity")
        
    Returns:
        List of recommended models
    """
    recommendations = {
        "general": [
            "ms-marco-MiniLM-L-6-v2",
            "ms-marco-MiniLM-L-12-v2", 
            "ms-marco-MiniLM-L-6-v2-distilbert"
        ],
        "qa": [
            "nq-distilbert-base-v1",
            "ms-marco-MiniLM-L-12-v2"
        ],
        "multilingual": [
            "xlm-r-100langs-bert-base-nli-stsb"
        ],
        "similarity": [
            "quora-distilbert-base",
            "sts-distilbert-base"
        ]
    }
    
    model_keys = recommendations.get(use_case, recommendations["general"])
    return [PRETRAINED_MODELS[key] for key in model_keys]


def print_model_info(model_name: str):
    """Print information about a specific model."""
    for key, model in PRETRAINED_MODELS.items():
        if model.name == model_name or key == model_name:
            print(f"Model: {model.name}")
            print(f"Description: {model.description}")
            print(f"Use Case: {model.use_case}")
            print(f"Size: {model.size}")
            print(f"Performance: {model.performance}")
            return
    
    print(f"Model '{model_name}' not found in pretrained models list")


def list_all_models():
    """List all available pre-trained models."""
    print("Available Pre-trained Cross-Encoder Models:")
    print("=" * 60)
    
    for key, model in PRETRAINED_MODELS.items():
        print(f"\n{key}:")
        print(f"  Name: {model.name}")
        print(f"  Description: {model.description}")
        print(f"  Use Case: {model.use_case}")
        print(f"  Size: {model.size}")
        print(f"  Performance: {model.performance}")


# Example usage
if __name__ == "__main__":
    print("Pre-trained Cross-Encoder Models for RAG Reranking")
    print("=" * 60)
    
    # Show recommended models for general use
    print("\nRecommended for General RAG Reranking:")
    general_models = get_recommended_models("general")
    for model in general_models:
        print(f"- {model.name}")
        print(f"  {model.description}")
    
    # Show all models
    print("\n" + "=" * 60)
    list_all_models()
    
    # Show specific model info
    print("\n" + "=" * 60)
    print("Default Model Info:")
    print_model_info("ms-marco-MiniLM-L-6-v2")
