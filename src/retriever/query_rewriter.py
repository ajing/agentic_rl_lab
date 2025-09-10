"""
Query rewriting module for conversational RAG.

Addresses coreference resolution and topic shift issues identified in Week 1.
Uses a simple LLM-based approach to rewrite queries with context.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_id: int
    question: str
    answer: Optional[str] = None


@dataclass
class RewrittenQuery:
    """Represents a rewritten query with metadata."""
    original_query: str
    rewritten_query: str
    confidence: float
    reasoning: str


class QueryRewriter:
    """
    Rewrites conversational queries to be more self-contained.
    
    Handles coreference resolution and topic shifts by using conversation
    history to create standalone queries.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 max_history: int = 3):
        """
        Initialize the query rewriter.
        
        Args:
            model: OpenAI model to use for rewriting
            api_key: OpenAI API key (if None, uses env var)
            max_history: Maximum number of previous turns to consider
        """
        self.model = model
        self.max_history = max_history
        self.client = OpenAI(api_key=api_key)
        
        # Cache for rewritten queries to avoid redundant API calls
        self.cache: Dict[str, RewrittenQuery] = {}
    
    def _create_rewrite_prompt(self, 
                              current_query: str, 
                              history: List[ConversationTurn]) -> str:
        """Create the prompt for query rewriting."""
        
        # Build conversation context
        context_parts = []
        for turn in history[-self.max_history:]:
            context_parts.append(f"Q{turn.turn_id}: {turn.question}")
            if turn.answer:
                context_parts.append(f"A{turn.turn_id}: {turn.answer}")
        
        context = "\n".join(context_parts) if context_parts else "No previous conversation."
        
        prompt = f"""You are a query rewriting assistant for a conversational search system. Your task is to rewrite the current question to be self-contained and clear, resolving any pronouns, references, or ambiguous terms using the conversation history.

Conversation History:
{context}

Current Question: {current_query}

Instructions:
1. If the current question contains pronouns (he, she, it, they, etc.) or references (this, that, the above, etc.), resolve them using the conversation history.
2. If the question refers to entities mentioned in previous turns, make those references explicit.
3. Keep the rewritten query concise and focused.
4. If the question is already clear and self-contained, return it unchanged.
5. Provide a confidence score (0.0-1.0) for your rewrite.

Return your response as JSON with these fields:
- "rewritten_query": the rewritten question
- "confidence": confidence score (0.0-1.0)
- "reasoning": brief explanation of changes made

Example:
{{
  "rewritten_query": "What role did Kariqi play during the 2017 UEFA European Under-21 Championship qualification?",
  "confidence": 0.9,
  "reasoning": "Resolved pronoun 'he' to 'Kariqi' based on previous mention in conversation"
}}"""
        
        return prompt
    
    def rewrite_query(self, 
                     current_query: str, 
                     history: List[ConversationTurn]) -> RewrittenQuery:
        """
        Rewrite a query using conversation history.
        
        Args:
            current_query: The current question to rewrite
            history: List of previous conversation turns
            
        Returns:
            RewrittenQuery object with the rewritten query and metadata
        """
        # Check cache first
        cache_key = f"{current_query}||{hash(str(history))}"
        if cache_key in self.cache:
            logger.debug(f"Cache hit for query: {current_query[:50]}...")
            return self.cache[cache_key]
        
        try:
            prompt = self._create_rewrite_prompt(current_query, history)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(content)
                rewritten = RewrittenQuery(
                    original_query=current_query,
                    rewritten_query=result["rewritten_query"],
                    confidence=result["confidence"],
                    reasoning=result["reasoning"]
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response: {content}")
                rewritten = RewrittenQuery(
                    original_query=current_query,
                    rewritten_query=current_query,  # Use original as fallback
                    confidence=0.5,
                    reasoning="JSON parsing failed, using original query"
                )
            
            # Cache the result
            self.cache[cache_key] = rewritten
            
            logger.info(f"Rewrote query: '{current_query[:50]}...' -> '{rewritten.rewritten_query[:50]}...'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            # Return original query as fallback
            rewritten = RewrittenQuery(
                original_query=current_query,
                rewritten_query=current_query,
                confidence=0.0,
                reasoning=f"Error during rewriting: {str(e)}"
            )
            return rewritten
    
    def rewrite_batch(self, 
                     queries_with_history: List[Tuple[str, List[ConversationTurn]]]) -> List[RewrittenQuery]:
        """
        Rewrite multiple queries in batch.
        
        Args:
            queries_with_history: List of (query, history) tuples
            
        Returns:
            List of RewrittenQuery objects
        """
        results = []
        for query, history in queries_with_history:
            result = self.rewrite_query(query, history)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear the rewrite cache."""
        self.cache.clear()
        logger.info("Query rewrite cache cleared")


def load_conversation_from_coral(conversation_data: Dict) -> List[ConversationTurn]:
    """
    Load conversation turns from CORAL format.
    
    Args:
        conversation_data: CORAL conversation data
        
    Returns:
        List of ConversationTurn objects
    """
    turns = []
    
    # Handle different CORAL data formats
    if "turns" in conversation_data:
        for turn_data in conversation_data["turns"]:
            turn = ConversationTurn(
                turn_id=turn_data.get("turn_id", len(turns) + 1),
                question=turn_data.get("question", ""),
                answer=turn_data.get("answer")
            )
            turns.append(turn)
    elif "questions" in conversation_data:
        # Alternative format
        for i, question in enumerate(conversation_data["questions"]):
            turn = ConversationTurn(
                turn_id=i + 1,
                question=question,
                answer=conversation_data.get("answers", [{}])[i] if "answers" in conversation_data else None
            )
            turns.append(turn)
    
    return turns


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize rewriter
    rewriter = QueryRewriter()
    
    # Example conversation
    history = [
        ConversationTurn(1, "Who is Kariqi?", "Kariqi is a football player."),
        ConversationTurn(2, "What team does he play for?", "He plays for a European team.")
    ]
    
    current_query = "What role did he play during the 2017 UEFA European Under-21 Championship qualification?"
    
    # Rewrite the query
    result = rewriter.rewrite_query(current_query, history)
    
    print(f"Original: {result.original_query}")
    print(f"Rewritten: {result.rewritten_query}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
