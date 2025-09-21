"""
LLM-based Expert Policy for document selection.

This policy uses a large language model to intelligently select the most relevant
documents from a candidate set, providing high-quality expert trajectories for
behavioral cloning training.
"""

import logging
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

from src.env.rag_environment import RLAction, RLState
from src.retriever.rrf_generator import CandidateDocument

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMExpertConfig:
    """Configuration for LLM Expert Policy."""
    model: str = "gpt-4o-mini"  # Use cost-effective model
    temperature: float = 0.1  # Low temperature for consistent decisions
    max_tokens: int = 1000
    max_documents_to_evaluate: int = 20  # Limit for cost control
    selection_strategy: str = "top_k"  # "top_k", "threshold", "adaptive"
    k_documents: int = 3  # Number of documents to select
    relevance_threshold: float = 0.7  # Minimum relevance score
    use_ranking_context: bool = True  # Include BM25/vector rankings
    use_content_preview: bool = True  # Include document content preview


class LLMExpertPolicy:
    """
    Expert policy that uses LLM to intelligently select relevant documents.
    
    This policy evaluates candidate documents using an LLM and selects the most
    relevant ones based on the query and conversation context.
    """
    
    def __init__(self, config: LLMExpertConfig = None):
        """
        Initialize the LLM Expert Policy.
        
        Args:
            config: Configuration for the policy
        """
        self.config = config or LLMExpertConfig()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"Initialized LLMExpertPolicy with model: {self.config.model}")
    
    def select_action(self, 
                     valid_actions: List[RLAction], 
                     state: RLState,
                     state_features: np.ndarray = None) -> RLAction:
        """
        Select the best action using LLM-based document evaluation.
        
        Args:
            valid_actions: List of valid actions
            state: Current RL state with candidate documents
            state_features: State features (unused for LLM policy)
            
        Returns:
            Selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Filter to select actions only
        select_actions = [a for a in valid_actions if a.action_type == "select"]
        
        if not select_actions:
            # No more documents to select, terminate
            terminate_action = [a for a in valid_actions if a.action_type == "terminate"]
            return terminate_action[0] if terminate_action else valid_actions[0]
        
        # Check if we should terminate based on current selections
        if self._should_terminate(state, select_actions):
            terminate_action = [a for a in valid_actions if a.action_type == "terminate"]
            return terminate_action[0] if terminate_action else valid_actions[0]
        
        # Get candidate documents for the select actions
        candidate_docs = self._get_candidate_documents(state, select_actions)
        
        if not candidate_docs:
            # Fallback to random selection
            return random.choice(select_actions)
        
        # Use LLM to select the best document
        try:
            selected_doc_id = self._llm_select_document(
                query=state.query,
                conversation_context=state.features.get("conversation_history", []),
                candidate_docs=candidate_docs,
                already_selected=state.selected_doc_ids
            )
            
            # Find the action corresponding to the selected document
            for action in select_actions:
                if action.doc_id == selected_doc_id:
                    return action
            
            # Fallback if LLM selection doesn't match any action
            logger.warning(f"LLM selected doc_id {selected_doc_id} not found in valid actions")
            return random.choice(select_actions)
            
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, falling back to random selection")
            return random.choice(select_actions)
    
    def _should_terminate(self, state: RLState, select_actions: List[RLAction]) -> bool:
        """Check if we should terminate the episode."""
        # Terminate if we've selected enough documents
        if len(state.selected_doc_ids) >= self.config.k_documents:
            return True
        
        # Terminate if we've reached max steps
        if state.step >= 5:  # Max steps from environment
            return True
        
        # Terminate if no more candidates
        if not select_actions:
            return True
        
        return False
    
    def _get_candidate_documents(self, 
                                state: RLState, 
                                select_actions: List[RLAction]) -> List[CandidateDocument]:
        """Get candidate documents for the select actions."""
        candidate_docs = []
        
        for action in select_actions:
            # Find the corresponding candidate document
            for candidate in state.remaining_candidates:
                if candidate.doc_id == action.doc_id:
                    candidate_docs.append(candidate)
                    break
        
        # Limit the number of documents to evaluate for cost control
        if len(candidate_docs) > self.config.max_documents_to_evaluate:
            # Take top candidates by RRF score
            candidate_docs.sort(key=lambda x: x.rrf_score, reverse=True)
            candidate_docs = candidate_docs[:self.config.max_documents_to_evaluate]
        
        return candidate_docs
    
    def _llm_select_document(self,
                           query: str,
                           conversation_context: List[Dict[str, Any]],
                           candidate_docs: List[CandidateDocument],
                           already_selected: List[str]) -> str:
        """
        Use LLM to select the most relevant document.
        
        Args:
            query: Current query
            conversation_context: Previous conversation turns
            candidate_docs: Available candidate documents
            already_selected: IDs of already selected documents
            
        Returns:
            doc_id of the selected document
        """
        # Build conversation context
        context_str = self._build_conversation_context(conversation_context)
        
        # Build document descriptions
        doc_descriptions = self._build_document_descriptions(candidate_docs)
        
        # Create the prompt
        prompt = self._create_selection_prompt(
            query=query,
            context=context_str,
            documents=doc_descriptions,
            already_selected=already_selected
        )
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are an expert at selecting relevant documents for answering questions. You must respond with only the document ID of the most relevant document."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Extract document ID from response (handle cases where LLM returns explanatory text)
        selected_doc_id = None
        valid_doc_ids = [doc.doc_id for doc in candidate_docs]
        
        # Try to find a valid document ID in the response
        for doc_id in valid_doc_ids:
            if doc_id in response_text:
                selected_doc_id = doc_id
                break
        
        # If no valid doc_id found, use the first valid option
        if selected_doc_id is None:
            logger.warning(f"LLM response '{response_text}' did not contain valid doc_id, using first valid option")
            selected_doc_id = valid_doc_ids[0] if valid_doc_ids else candidate_docs[0].doc_id
        
        return selected_doc_id
    
    def _build_conversation_context(self, conversation_context: List[Any]) -> str:
        """Build conversation context string."""
        if not conversation_context:
            return "No previous conversation context."
        
        context_parts = []
        for turn in conversation_context[-3:]:  # Last 3 turns
            if isinstance(turn, dict):
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                context_parts.append(f"{speaker}: {text}")
            elif hasattr(turn, 'question') and hasattr(turn, 'answer'):
                # ConversationTurn object
                context_parts.append(f"User: {turn.question}")
                if turn.answer:
                    context_parts.append(f"Assistant: {turn.answer}")
            else:
                context_parts.append(str(turn))
        
        return "\n".join(context_parts) if context_parts else "No previous conversation context."
    
    def _build_document_descriptions(self, candidate_docs: List[CandidateDocument]) -> str:
        """Build descriptions of candidate documents."""
        descriptions = []
        
        for i, doc in enumerate(candidate_docs, 1):
            # Truncate content for cost control
            content_preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            
            description = f"Document {doc.doc_id}:\n"
            description += f"Content: {content_preview}\n"
            
            if self.config.use_ranking_context:
                description += f"BM25 Rank: {doc.rank_bm25}, Vector Rank: {doc.rank_vector}, RRF Rank: {doc.rank_rrf}\n"
                description += f"BM25 Score: {doc.bm25_score:.3f}, Vector Score: {doc.vector_score:.3f}, RRF Score: {doc.rrf_score:.3f}\n"
            
            descriptions.append(description)
        
        return "\n".join(descriptions)
    
    def _create_selection_prompt(self,
                               query: str,
                               context: str,
                               documents: str,
                               already_selected: List[str]) -> str:
        """Create the prompt for document selection."""
        prompt = f"""You are selecting the most relevant document to help answer a question.

QUESTION: {query}

CONVERSATION CONTEXT:
{context}

AVAILABLE DOCUMENTS:
{documents}

ALREADY SELECTED DOCUMENTS: {', '.join(already_selected) if already_selected else 'None'}

INSTRUCTIONS:
1. Select the document that is MOST relevant to answering the question
2. Consider the conversation context to understand what information is still needed
3. Avoid selecting documents that are too similar to already selected ones
4. Prioritize documents that directly address the question
5. Consider the ranking scores as hints, but make your own judgment

CRITICAL: You must respond with ONLY a document ID number (e.g., "12345"). Do not include any explanation, reasoning, or other text. Just the number."""

        return prompt


class LLMExpertPolicyConfig:
    """Configuration for LLM Expert Policy in the episode runner."""
    
    def __init__(self, 
                 name: str = "llm_expert",
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 k_documents: int = 3,
                 max_documents_to_evaluate: int = 20):
        self.name = name
        self.policy_type = "llm_expert"
        self.model = model
        self.temperature = temperature
        self.k_documents = k_documents
        self.max_documents_to_evaluate = max_documents_to_evaluate
        
        # Create LLM expert config
        self.llm_config = LLMExpertConfig(
            model=model,
            temperature=temperature,
            k_documents=k_documents,
            max_documents_to_evaluate=max_documents_to_evaluate
        )
    
    def create_policy(self):
        """Create an instance of the LLM Expert Policy."""
        return LLMExpertPolicy(self.llm_config)
