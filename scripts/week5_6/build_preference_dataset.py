"""
Build preference datasets using LLM-as-a-Judge with real CORAL data.

This script generates preference data by comparing responses from different policies
and using an LLM judge to determine which responses are better.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.env.rag_environment import RAGEnvironment, ConversationTurn
from src.policy.episode_runner import EpisodeRunner, PolicyConfig
from src.reward.llm_judge import LLMJudge, AnswerPair, PreferenceResult
from src.reward.reward_model import LightweightRewardModel
from src.reward.reward_shaping import RewardShaper, RewardConfig

logger = logging.getLogger(__name__)


class PreferenceDatasetBuilder:
    """
    Build preference datasets using LLM-as-a-Judge with real CORAL data.
    
    Generates preference pairs by comparing responses from different policies
    and using an LLM judge to determine preferences.
    """
    
    def __init__(self, 
                 data_dir: str = "data/coral",
                 output_dir: str = "outputs/preference_dataset"):
        """
        Initialize the preference dataset builder.
        
        Args:
            data_dir: Directory containing CORAL data
            output_dir: Directory for output preference data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.rag_env = None
        self.episode_runner = None
        self.llm_judge = None
        self.reward_model = None
        self.reward_shaper = None
        
        # Data storage
        self.coral_conversations = []
        
        logger.info(f"Initialized PreferenceDatasetBuilder with data_dir: {self.data_dir}")
    
    def setup_components(self, use_llm_judge: bool = True):
        """Setup components for preference dataset building."""
        logger.info("Setting up components...")
        
        # Initialize RAG environment
        self.rag_env = RAGEnvironment(
            corpus_path=str(self.data_dir / "docs.jsonl"),
            bm25_index_path="index/coral_bm25",
            vector_index_path="index/coral_faiss",
            use_query_rewriting=True,
            use_cross_encoder=True,
            use_mmr=True,
            max_steps=5,
            k_candidates=100
        )
        
        # Initialize episode runner
        self.episode_runner = EpisodeRunner(self.rag_env)
        
        # Initialize LLM judge
        if use_llm_judge:
            try:
                self.llm_judge = LLMJudge()
                logger.info("LLM judge initialized")
            except Exception as e:
                logger.warning(f"Could not initialize LLM judge: {e}")
                self.llm_judge = None
        else:
            self.llm_judge = None
        
        # Initialize reward model
        self.reward_model = LightweightRewardModel()
        
        # Initialize reward shaper
        reward_config = RewardConfig(
            final_weight=1.0,
            novelty_weight=0.3,
            relevance_weight=0.2,
            diversity_weight=0.2,
            coherence_weight=0.1
        )
        self.reward_shaper = RewardShaper(
            reward_model=self.reward_model,
            config=reward_config
        )
        
        logger.info("Components setup complete")
    
    def load_coral_conversations(self, max_conversations: int = 1000):
        """Load CORAL conversations from training data."""
        logger.info(f"Loading up to {max_conversations} CORAL conversations...")
        
        # Load training conversations
        train_conversations_path = self.data_dir / "raw" / "train" / "new_train_conversation.json"
        if not train_conversations_path.exists():
            logger.error(f"Training conversations not found at {train_conversations_path}")
            return
        
        with open(train_conversations_path, 'r', encoding='utf-8') as f:
            conversations_data = json.load(f)
        
        # Convert to our format
        for conv in conversations_data[:max_conversations]:
            conversation_turns = []
            for i, turn in enumerate(conv.get('turns', [])):
                conversation_turns.append(ConversationTurn(
                    turn_id=i,
                    question=turn.get('question', ''),
                    answer=turn.get('answer', None)
                ))
            
            self.coral_conversations.append({
                'conversation_id': conv.get('conversation_id', ''),
                'turns': conversation_turns
            })
        
        logger.info(f"Loaded {len(self.coral_conversations)} conversations")
    
    def extract_queries_from_conversations(self) -> List[Tuple[str, List[ConversationTurn]]]:
        """Extract queries and conversation history from CORAL conversations."""
        logger.info("Extracting queries from conversations...")
        
        queries_with_history = []
        
        for conv in self.coral_conversations:
            if not conv['turns']:
                continue
            
            # Extract user queries and build conversation history
            current_history = []
            
            for i, turn in enumerate(conv['turns']):
                if turn.question:  # This is a user query
                    query = turn.question
                    
                    # Add to queries with history
                    queries_with_history.append((query, current_history.copy()))
                    
                    # Add this turn to history
                    current_history.append(turn)
                elif turn.answer and current_history:  # This is an assistant response
                    current_history.append(turn)
        
        logger.info(f"Extracted {len(queries_with_history)} queries with history")
        return queries_with_history
    
    def generate_response_from_documents(self, 
                                       query: str, 
                                       selected_docs: List[Dict[str, Any]]) -> str:
        """Generate a response from selected documents."""
        if not selected_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Combine document contents
        context_parts = []
        for i, doc in enumerate(selected_docs):
            content = doc.get('content', '')
            if content:
                # Truncate long documents
                if len(content) > 300:
                    content = content[:300] + "..."
                context_parts.append(f"Document {i+1}: {content}")
        
        context = "\n".join(context_parts)
        
        # Generate response (simplified - in practice, you'd use an LLM)
        response = f"Based on the available information: {context[:500]}..."
        
        if len(response) > 1000:
            response = response[:1000] + "..."
        
        return response
    
    def generate_preference_pair(self, 
                               query: str, 
                               history: List[ConversationTurn],
                               policy_a: PolicyConfig,
                               policy_b: PolicyConfig) -> Optional[Dict[str, Any]]:
        """Generate a preference pair by comparing two policies."""
        try:
            # Run episode with policy A
            episode_a = self.episode_runner.run_episode(query, policy_a, history)
            
            # Run episode with policy B
            episode_b = self.episode_runner.run_episode(query, policy_b, history)
            
            # Generate responses
            docs_a = [{'content': doc.content} for doc in episode_a.final_state.selected_documents]
            docs_b = [{'content': doc.content} for doc in episode_b.final_state.selected_documents]
            
            response_a = self.generate_response_from_documents(query, docs_a)
            response_b = self.generate_response_from_documents(query, docs_b)
            
            # Create answer pair
            answer_pair = AnswerPair(
                query=query,
                answer_a=response_a,
                answer_b=response_b,
                context_a=[doc['content'] for doc in docs_a],
                context_b=[doc['content'] for doc in docs_b]
            )
            
            # Get preference from LLM judge
            if self.llm_judge:
                try:
                    preference_result = self.llm_judge.compare_answers(answer_pair)
                    
                    # Create preference example
                    preference_example = {
                        'query': query,
                        'conversation_history': [{'question': t.question, 'answer': t.answer, 'turn_id': t.turn_id} for t in history],
                        'answer_a': response_a,
                        'answer_b': response_b,
                        'context_a': [doc['content'] for doc in docs_a],
                        'context_b': [doc['content'] for doc in docs_b],
                        'preferred_answer': preference_result.preferred_answer,
                        'confidence': preference_result.confidence,
                        'reasoning': preference_result.reasoning,
                        'criteria_scores': preference_result.criteria_scores,
                        'policy_a': {
                            'name': policy_a.name,
                            'policy_type': policy_a.policy_type,
                            'epsilon': getattr(policy_a, 'epsilon', None)
                        },
                        'policy_b': {
                            'name': policy_b.name,
                            'policy_type': policy_b.policy_type,
                            'epsilon': getattr(policy_b, 'epsilon', None)
                        },
                        'episode_a': {
                            'total_reward': episode_a.total_reward,
                            'num_documents': len(episode_a.selected_doc_ids),
                            'episode_length': episode_a.final_state.current_step
                        },
                        'episode_b': {
                            'total_reward': episode_b.total_reward,
                            'num_documents': len(episode_b.selected_doc_ids),
                            'episode_length': episode_b.final_state.current_step
                        }
                    }
                    
                    return preference_example
                    
                except Exception as e:
                    logger.warning(f"Error in LLM judge evaluation: {e}")
                    return None
            else:
                # No LLM judge available, create preference based on episode rewards
                preferred_answer = "A" if episode_a.total_reward > episode_b.total_reward else "B"
                
                preference_example = {
                    'query': query,
                    'conversation_history': [{'question': t.question, 'answer': t.answer, 'turn_id': t.turn_id} for t in history],
                    'answer_a': response_a,
                    'answer_b': response_b,
                    'context_a': [doc['content'] for doc in docs_a],
                    'context_b': [doc['content'] for doc in docs_b],
                    'preferred_answer': preferred_answer,
                    'confidence': 0.8,  # Default confidence
                    'reasoning': f"Preference based on episode reward: A={episode_a.total_reward:.3f}, B={episode_b.total_reward:.3f}",
                    'criteria_scores': {},
                    'policy_a': {
                        'name': policy_a.name,
                        'policy_type': policy_a.policy_type,
                        'epsilon': getattr(policy_a, 'epsilon', None)
                    },
                    'policy_b': {
                        'name': policy_b.name,
                        'policy_type': policy_b.policy_type,
                        'epsilon': getattr(policy_b, 'epsilon', None)
                    },
                    'episode_a': {
                        'total_reward': episode_a.total_reward,
                        'num_documents': len(episode_a.selected_doc_ids),
                        'episode_length': episode_a.final_state.current_step
                    },
                    'episode_b': {
                        'total_reward': episode_b.total_reward,
                        'num_documents': len(episode_b.selected_doc_ids),
                        'episode_length': episode_b.final_state.current_step
                    }
                }
                
                return preference_example
                
        except Exception as e:
            logger.warning(f"Error generating preference pair for query '{query[:50]}...': {e}")
            return None
    
    def build_preference_dataset(self, 
                               num_preferences: int = 1000,
                               use_multiple_policies: bool = True) -> List[Dict[str, Any]]:
        """Build preference dataset by comparing different policies."""
        logger.info(f"Building preference dataset with {num_preferences} examples...")
        
        # Extract queries
        queries_with_history = self.extract_queries_from_conversations()
        
        if not queries_with_history:
            logger.error("No queries extracted from conversations")
            return []
        
        # Define policies for comparison
        if use_multiple_policies:
            policies = [
                PolicyConfig(name="greedy_policy", policy_type="greedy"),
                PolicyConfig(name="random_policy", policy_type="random"),
                PolicyConfig(name="epsilon_greedy_0.3", policy_type="epsilon_greedy", epsilon=0.3),
                PolicyConfig(name="epsilon_greedy_0.5", policy_type="epsilon_greedy", epsilon=0.5)
            ]
        else:
            policies = [
                PolicyConfig(name="greedy_policy", policy_type="greedy"),
                PolicyConfig(name="random_policy", policy_type="random")
            ]
        
        # Generate preference pairs
        preference_examples = []
        queries_used = set()
        
        with tqdm(total=num_preferences, desc="Building preference dataset") as pbar:
            while len(preference_examples) < num_preferences and len(queries_used) < len(queries_with_history):
                # Select random query
                query, history = random.choice(queries_with_history)
                query_key = f"{query}|{len(history)}"
                
                if query_key in queries_used:
                    continue
                
                queries_used.add(query_key)
                
                # Select two different policies
                policy_a, policy_b = random.sample(policies, 2)
                
                # Generate preference pair
                preference_example = self.generate_preference_pair(query, history, policy_a, policy_b)
                
                if preference_example is not None:
                    preference_examples.append(preference_example)
                    pbar.update(1)
                
                # Add small delay to avoid rate limiting
                if self.llm_judge:
                    time.sleep(0.1)
        
        logger.info(f"Built preference dataset with {len(preference_examples)} examples")
        return preference_examples
    
    def filter_high_quality_preferences(self, 
                                      preferences: List[Dict[str, Any]],
                                      min_confidence: float = 0.6,
                                      min_documents: int = 1) -> List[Dict[str, Any]]:
        """Filter preferences based on quality criteria."""
        logger.info(f"Filtering preferences with min_confidence={min_confidence}, "
                   f"min_documents={min_documents}")
        
        filtered_preferences = []
        
        for pref in preferences:
            # Check confidence threshold
            if pref['confidence'] < min_confidence:
                continue
            
            # Check that both policies selected some documents
            if (pref['episode_a']['num_documents'] < min_documents or 
                pref['episode_b']['num_documents'] < min_documents):
                continue
            
            # Check that responses are not empty
            if not pref['answer_a'].strip() or not pref['answer_b'].strip():
                continue
            
            filtered_preferences.append(pref)
        
        logger.info(f"Filtered to {len(filtered_preferences)} high-quality preferences "
                   f"(from {len(preferences)} total)")
        return filtered_preferences
    
    def save_preference_dataset(self, 
                              preferences: List[Dict[str, Any]], 
                              filename: str = "preference_dataset.jsonl"):
        """Save preference dataset to file."""
        output_path = self.output_dir / filename
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for pref in preferences:
                f.write(json.dumps(pref, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(preferences)} preferences to {output_path}")
        
        # Also save a summary
        summary = {
            'total_preferences': len(preferences),
            'avg_confidence': np.mean([p['confidence'] for p in preferences]),
            'preference_distribution': {
                'A': sum(1 for p in preferences if p['preferred_answer'] == 'A'),
                'B': sum(1 for p in preferences if p['preferred_answer'] == 'B')
            },
            'policy_comparisons': {},
            'avg_documents_a': np.mean([p['episode_a']['num_documents'] for p in preferences]),
            'avg_documents_b': np.mean([p['episode_b']['num_documents'] for p in preferences]),
            'avg_reward_a': np.mean([p['episode_a']['total_reward'] for p in preferences]),
            'avg_reward_b': np.mean([p['episode_b']['total_reward'] for p in preferences])
        }
        
        # Calculate policy comparison distribution
        for pref in preferences:
            policy_key = f"{pref['policy_a']['name']}_vs_{pref['policy_b']['name']}"
            summary['policy_comparisons'][policy_key] = summary['policy_comparisons'].get(policy_key, 0) + 1
        
        with open(self.output_dir / "preference_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved preference summary to {self.output_dir / 'preference_summary.json'}")
        return output_path
    
    def run_preference_dataset_building(self, 
                                      max_conversations: int = 1000,
                                      num_preferences: int = 1000,
                                      min_confidence: float = 0.6,
                                      use_llm_judge: bool = True,
                                      use_multiple_policies: bool = True) -> str:
        """Run the complete preference dataset building process."""
        logger.info("Starting preference dataset building...")
        
        # Setup components
        self.setup_components(use_llm_judge)
        
        # Load conversations
        self.load_coral_conversations(max_conversations)
        
        # Build preference dataset
        preferences = self.build_preference_dataset(num_preferences, use_multiple_policies)
        
        # Filter high-quality preferences
        filtered_preferences = self.filter_high_quality_preferences(
            preferences, min_confidence=min_confidence
        )
        
        # Save preference dataset
        output_path = self.save_preference_dataset(filtered_preferences)
        
        logger.info("Preference dataset building completed!")
        return str(output_path)


def main():
    """Run preference dataset building."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting preference dataset building...")
    
    # Initialize builder
    builder = PreferenceDatasetBuilder(
        data_dir="data/coral",
        output_dir="outputs/preference_dataset"
    )
    
    # Run building process
    output_path = builder.run_preference_dataset_building(
        max_conversations=500,
        num_preferences=300,
        min_confidence=0.5,
        use_llm_judge=True,  # Set to False if you don't have OpenAI API access
        use_multiple_policies=True
    )
    
    logger.info(f"Preference dataset saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    output_path = main()
    print(f"\nPreference dataset built: {output_path}")
