"""
LLM-as-a-Judge for preference data collection.

Implements pairwise comparison and preference scoring using strong LLMs
to generate high-quality preference data for reward model training.
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import openai
from openai import OpenAI
import time

logger = logging.getLogger(__name__)


@dataclass
class AnswerPair:
    """Represents a pair of answers for comparison."""
    answer_a: str
    answer_b: str
    context_a: List[str]  # Supporting documents for answer A
    context_b: List[str]  # Supporting documents for answer B
    query: str
    metadata: Dict[str, Any] = None


@dataclass
class PreferenceResult:
    """Represents the result of a preference comparison."""
    preferred_answer: str  # "A" or "B"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    criteria_scores: Dict[str, float]  # Individual criteria scores
    metadata: Dict[str, Any] = None


class LLMJudge:
    """
    LLM-based judge for answer preference evaluation.
    
    Uses strong LLMs to compare answer pairs and generate preference data
    with detailed reasoning and bias mitigation techniques.
    """
    
    def __init__(self,
                 model: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 temperature: float = 0.1,
                 max_retries: int = 3,
                 delay_between_requests: float = 1.0):
        """
        Initialize the LLM judge.
        
        Args:
            model: OpenAI model to use for judging
            api_key: OpenAI API key (if None, uses env var)
            temperature: Temperature for generation
            max_retries: Maximum retry attempts for failed requests
            delay_between_requests: Delay between API requests (seconds)
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.delay_between_requests = delay_between_requests
        
        self.client = OpenAI(api_key=api_key)
        
        # Cache for preference results
        self.cache: Dict[str, PreferenceResult] = {}
        
        logger.info(f"Initialized LLM judge with model: {model}")
    
    def _create_judge_prompt(self, answer_pair: AnswerPair, swap_order: bool = False) -> str:
        """
        Create the judge prompt for pairwise comparison.
        
        Args:
            answer_pair: Pair of answers to compare
            swap_order: Whether to swap the order to mitigate position bias
            
        Returns:
            Formatted prompt string
        """
        if swap_order:
            answer_a, answer_b = answer_pair.answer_b, answer_pair.answer_a
            context_a, context_b = answer_pair.context_b, answer_pair.context_a
        else:
            answer_a, answer_b = answer_pair.answer_a, answer_pair.answer_b
            context_a, context_b = answer_pair.context_a, answer_pair.context_b
        
        # Build context strings
        context_a_str = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_a)])
        context_b_str = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_b)])
        
        prompt = f"""You are an expert judge evaluating the quality of answers to questions. Your task is to compare two answers and determine which one is better based on multiple criteria.

QUESTION: {answer_pair.query}

ANSWER A (with supporting context):
Context:
{context_a_str}

Answer: {answer_a}

ANSWER B (with supporting context):
Context:
{context_b_str}

Answer: {answer_b}

Please evaluate both answers based on these criteria:

1. **Accuracy & Factual Correctness**: Is the information correct and well-supported by the context?
2. **Completeness**: Does the answer fully address the question?
3. **Clarity & Coherence**: Is the answer clear, well-structured, and easy to understand?
4. **Relevance**: Does the answer stay focused on the question asked?
5. **Attribution**: Does the answer properly cite or reference the supporting context?

For each criterion, rate both answers on a scale of 1-5 (5 being best), then provide your overall preference.

Your response should be in this exact JSON format:
{{
  "preferred_answer": "A" or "B",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your decision",
  "criteria_scores": {{
    "accuracy": {{"A": 1-5, "B": 1-5}},
    "completeness": {{"A": 1-5, "B": 1-5}},
    "clarity": {{"A": 1-5, "B": 1-5}},
    "relevance": {{"A": 1-5, "B": 1-5}},
    "attribution": {{"A": 1-5, "B": 1-5}}
  }}
}}

Be objective and focus on the quality of the answers, not on minor stylistic differences."""
        
        return prompt
    
    def _parse_judge_response(self, response: str) -> PreferenceResult:
        """
        Parse the judge's response into a structured result.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Parsed PreferenceResult
        """
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            if "preferred_answer" not in result or "confidence" not in result:
                raise ValueError("Missing required fields in response")
            
            preferred_answer = result["preferred_answer"]
            if preferred_answer not in ["A", "B"]:
                raise ValueError(f"Invalid preferred_answer: {preferred_answer}")
            
            confidence = float(result["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Invalid confidence: {confidence}")
            
            # Parse criteria scores
            criteria_scores = result.get("criteria_scores", {})
            if not isinstance(criteria_scores, dict):
                criteria_scores = {}
            
            return PreferenceResult(
                preferred_answer=preferred_answer,
                confidence=confidence,
                reasoning=result.get("reasoning", ""),
                criteria_scores=criteria_scores,
                metadata={"raw_response": response}
            )
            
        except Exception as e:
            logger.error(f"Error parsing judge response: {e}")
            # Return a fallback result
            return PreferenceResult(
                preferred_answer="A",
                confidence=0.5,
                reasoning=f"Parse error: {str(e)}",
                criteria_scores={},
                metadata={"error": str(e), "raw_response": response}
            )
    
    def compare_answers(self, answer_pair: AnswerPair) -> PreferenceResult:
        """
        Compare two answers and return preference result.
        
        Args:
            answer_pair: Pair of answers to compare
            
        Returns:
            PreferenceResult with comparison outcome
        """
        # Check cache first
        cache_key = f"{hash(answer_pair.answer_a)}_{hash(answer_pair.answer_b)}_{answer_pair.query}"
        if cache_key in self.cache:
            logger.debug("Cache hit for answer comparison")
            return self.cache[cache_key]
        
        # Create prompt
        prompt = self._create_judge_prompt(answer_pair)
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                result = self._parse_judge_response(content)
                
                # Cache the result
                self.cache[cache_key] = result
                
                # Add delay between requests
                if self.delay_between_requests > 0:
                    time.sleep(self.delay_between_requests)
                
                logger.info(f"Compared answers: preferred {result.preferred_answer}, confidence {result.confidence:.2f}")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return fallback result on final failure
                    result = PreferenceResult(
                        preferred_answer="A",
                        confidence=0.5,
                        reasoning=f"API error: {str(e)}",
                        criteria_scores={},
                        metadata={"error": str(e)}
                    )
                    return result
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    def compare_with_bias_mitigation(self, answer_pair: AnswerPair) -> PreferenceResult:
        """
        Compare answers with bias mitigation by swapping order.
        
        Args:
            answer_pair: Pair of answers to compare
            
        Returns:
            PreferenceResult with bias-mitigated comparison
        """
        # First comparison (original order)
        result1 = self.compare_answers(answer_pair)
        
        # Second comparison (swapped order)
        result2 = self.compare_answers(AnswerPair(
            answer_a=answer_pair.answer_b,
            answer_b=answer_pair.answer_a,
            context_a=answer_pair.context_b,
            context_b=answer_pair.context_a,
            query=answer_pair.query,
            metadata=answer_pair.metadata
        ))
        
        # Combine results to mitigate position bias
        if result1.preferred_answer == result2.preferred_answer:
            # Consistent preference
            final_preferred = result1.preferred_answer
            final_confidence = (result1.confidence + result2.confidence) / 2
            final_reasoning = f"Consistent preference: {result1.reasoning}"
        else:
            # Inconsistent preference - use average confidence and note inconsistency
            final_preferred = "A" if result1.confidence > result2.confidence else "B"
            final_confidence = abs(result1.confidence - result2.confidence) / 2
            final_reasoning = f"Inconsistent preferences detected. Result1: {result1.preferred_answer} ({result1.confidence:.2f}), Result2: {result2.preferred_answer} ({result2.confidence:.2f})"
        
        # Combine criteria scores
        combined_criteria = {}
        for criterion in ["accuracy", "completeness", "clarity", "relevance", "attribution"]:
            if criterion in result1.criteria_scores and criterion in result2.criteria_scores:
                combined_criteria[criterion] = {
                    "A": (result1.criteria_scores[criterion].get("A", 3) + 
                          result2.criteria_scores[criterion].get("B", 3)) / 2,
                    "B": (result1.criteria_scores[criterion].get("B", 3) + 
                          result2.criteria_scores[criterion].get("A", 3)) / 2
                }
        
        return PreferenceResult(
            preferred_answer=final_preferred,
            confidence=final_confidence,
            reasoning=final_reasoning,
            criteria_scores=combined_criteria,
            metadata={
                "bias_mitigation": True,
                "result1": result1.metadata,
                "result2": result2.metadata
            }
        )
    
    def batch_compare(self, answer_pairs: List[AnswerPair], 
                     use_bias_mitigation: bool = True) -> List[PreferenceResult]:
        """
        Compare multiple answer pairs in batch.
        
        Args:
            answer_pairs: List of answer pairs to compare
            use_bias_mitigation: Whether to use bias mitigation
            
        Returns:
            List of PreferenceResult objects
        """
        results = []
        
        for i, answer_pair in enumerate(answer_pairs):
            logger.info(f"Comparing pair {i+1}/{len(answer_pairs)}")
            
            if use_bias_mitigation:
                result = self.compare_with_bias_mitigation(answer_pair)
            else:
                result = self.compare_answers(answer_pair)
            
            results.append(result)
        
        return results
    
    def save_preferences(self, results: List[PreferenceResult], 
                        answer_pairs: List[AnswerPair], 
                        filepath: str):
        """
        Save preference results to a file.
        
        Args:
            results: List of preference results
            answer_pairs: Original answer pairs
            filepath: Path to save the data
        """
        data = []
        
        for result, pair in zip(results, answer_pairs):
            data.append({
                "query": pair.query,
                "answer_a": pair.answer_a,
                "answer_b": pair.answer_b,
                "context_a": pair.context_a,
                "context_b": pair.context_b,
                "preferred_answer": result.preferred_answer,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "criteria_scores": result.criteria_scores,
                "metadata": {
                    "pair_metadata": pair.metadata,
                    "result_metadata": result.metadata
                }
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} preference results to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    import logging
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize judge
    judge = LLMJudge()
    
    # Example answer pair
    answer_pair = AnswerPair(
        query="Who won the FA Cup in 2020?",
        answer_a="Arsenal won the FA Cup in 2020, defeating Chelsea 2-1 in the final.",
        answer_b="The FA Cup is an annual football competition in England. Arsenal won it in 2020.",
        context_a=["Arsenal defeated Chelsea 2-1 in the 2020 FA Cup final at Wembley Stadium."],
        context_b=["The FA Cup is England's premier cup competition.", "Arsenal won the 2020 FA Cup final."]
    )
    
    # Compare answers
    result = judge.compare_with_bias_mitigation(answer_pair)
    
    print(f"Query: {answer_pair.query}")
    print(f"Preferred Answer: {result.preferred_answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Criteria Scores: {result.criteria_scores}")
