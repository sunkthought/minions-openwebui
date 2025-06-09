"""
Base protocol functionality for MinionS/Minions OpenWebUI functions.
This module contains shared patterns, utilities, and base classes
used by both Minion and MinionS protocols.
"""

import json
import re
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# Shared constants
CONFIDENCE_MAPPING = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
DEFAULT_NUMERIC_CONFIDENCE = 0.3
CHARS_PER_TOKEN = 3.5


@dataclass
class ParsedResponse:
    """Standardized parsed response structure."""
    success: bool
    confidence: float
    content: Any
    parse_method: str  # "json", "fallback", "text"
    error_message: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class ExecutionMetrics:
    """Shared execution metrics structure."""
    total_duration: float = 0.0
    api_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    token_savings: Optional[Dict[str, Any]] = None
    
    def add_confidence_score(self, score: float) -> None:
        """Add a confidence score to the metrics."""
        self.confidence_scores.append(score)
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence score."""
        return sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0
    
    def get_success_rate(self) -> float:
        """Calculate API call success rate."""
        total = self.successful_calls + self.failed_calls + self.timeout_calls
        return self.successful_calls / total if total > 0 else 0.0


class BaseResponseParser:
    """Base class for response parsing functionality."""
    
    @staticmethod
    def clean_json_response(response: str) -> str:
        """
        Clean and prepare response text for JSON parsing.
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
        response = response.strip()
        
        # Remove common prefixes/suffixes
        response = re.sub(r'^[^{]*(?=\{)', '', response)
        response = re.sub(r'\}[^}]*$', '}', response)
        
        return response
    
    @staticmethod
    def extract_json_with_regex(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from response using regex patterns.
        
        Args:
            response: Response text to parse
            
        Returns:
            Parsed JSON data or None if extraction fails
        """
        # Common patterns for JSON extraction
        patterns = [
            r'\{[^{}]*"[^"]*"[^{}]*\}',  # Simple object pattern
            r'\{.*\}',  # Greedy object pattern
            r'```json\s*(\{.*?\})\s*```',  # JSON code block
            r'```\s*(\{.*?\})\s*```',  # Generic code block
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    @staticmethod
    def map_confidence_to_numeric(confidence_str: str) -> float:
        """
        Map string confidence to numeric value.
        
        Args:
            confidence_str: Confidence string ("HIGH", "MEDIUM", "LOW")
            
        Returns:
            Numeric confidence value
        """
        return CONFIDENCE_MAPPING.get(confidence_str.upper(), DEFAULT_NUMERIC_CONFIDENCE)
    
    def parse_response(
        self,
        response: str,
        expected_format: str = "json",
        fallback_enabled: bool = True,
        debug_mode: bool = False
    ) -> ParsedResponse:
        """
        Parse response with multiple fallback strategies.
        
        Args:
            response: Raw response text
            expected_format: Expected response format ("json", "text")
            fallback_enabled: Whether to attempt fallback parsing
            debug_mode: Whether debug mode is enabled
            
        Returns:
            ParsedResponse object with parsing results
        """
        if expected_format.lower() == "json":
            return self._parse_json_response(response, fallback_enabled, debug_mode)
        else:
            return self._parse_text_response(response, debug_mode)
    
    def _parse_json_response(
        self,
        response: str,
        fallback_enabled: bool,
        debug_mode: bool
    ) -> ParsedResponse:
        """Parse JSON response with fallback handling."""
        # First, try direct JSON parsing
        cleaned_response = self.clean_json_response(response)
        
        try:
            data = json.loads(cleaned_response)
            confidence = self._extract_confidence_from_data(data)
            
            return ParsedResponse(
                success=True,
                confidence=confidence,
                content=data,
                parse_method="json",
                raw_response=response
            )
        except json.JSONDecodeError as e:
            if debug_mode:
                print(f"DEBUG [ResponseParser]: Direct JSON parsing failed: {str(e)}")
            
            # Try regex fallback if enabled
            if fallback_enabled:
                fallback_data = self.extract_json_with_regex(response)
                if fallback_data:
                    confidence = self._extract_confidence_from_data(fallback_data)
                    
                    return ParsedResponse(
                        success=True,
                        confidence=confidence,
                        content=fallback_data,
                        parse_method="fallback",
                        raw_response=response
                    )
            
            # Both parsing methods failed
            return ParsedResponse(
                success=False,
                confidence=DEFAULT_NUMERIC_CONFIDENCE,
                content=response,
                parse_method="text",
                error_message=f"JSON parsing failed: {str(e)}",
                raw_response=response
            )
    
    def _parse_text_response(self, response: str, debug_mode: bool) -> ParsedResponse:
        """Parse text response."""
        # For text responses, try to extract confidence if present
        confidence = self._extract_confidence_from_text(response)
        
        return ParsedResponse(
            success=True,
            confidence=confidence,
            content=response.strip(),
            parse_method="text",
            raw_response=response
        )
    
    def _extract_confidence_from_data(self, data: Dict[str, Any]) -> float:
        """Extract confidence from parsed JSON data."""
        # Common confidence field names
        confidence_fields = ['confidence', 'confidence_level', 'confidence_score']
        
        for field in confidence_fields:
            if field in data:
                confidence_value = data[field]
                if isinstance(confidence_value, str):
                    return self.map_confidence_to_numeric(confidence_value)
                elif isinstance(confidence_value, (int, float)):
                    return float(confidence_value)
        
        return DEFAULT_NUMERIC_CONFIDENCE
    
    def _extract_confidence_from_text(self, text: str) -> float:
        """Extract confidence from text response."""
        # Look for confidence keywords
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in ['high confidence', 'very confident', 'certain']):
            return CONFIDENCE_MAPPING['HIGH']
        elif any(phrase in text_lower for phrase in ['medium confidence', 'somewhat confident', 'likely']):
            return CONFIDENCE_MAPPING['MEDIUM']
        elif any(phrase in text_lower for phrase in ['low confidence', 'uncertain', 'not sure']):
            return CONFIDENCE_MAPPING['LOW']
        
        return DEFAULT_NUMERIC_CONFIDENCE


class TokenSavingsCalculator:
    """Utility class for calculating token savings."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated token count
        """
        return int(len(text) / CHARS_PER_TOKEN)
    
    @staticmethod
    def calculate_traditional_tokens(
        user_query: str,
        document_content: str,
        final_answer: str
    ) -> int:
        """
        Calculate tokens for traditional approach.
        
        Args:
            user_query: User's query
            document_content: Full document content
            final_answer: Final answer
            
        Returns:
            Estimated traditional token count
        """
        traditional_prompt = f"Query: {user_query}\n\nDocument: {document_content}\n\nPlease answer the query based on the document."
        return TokenSavingsCalculator.estimate_tokens(traditional_prompt + final_answer)
    
    @staticmethod
    def calculate_protocol_tokens(
        rounds_data: List[Dict[str, Any]],
        final_answer: str = ""
    ) -> int:
        """
        Calculate tokens used by protocol approach.
        
        Args:
            rounds_data: List of round data dictionaries
            final_answer: Final answer text
            
        Returns:
            Estimated protocol token count
        """
        total_tokens = 0
        
        for round_data in rounds_data:
            # Add tokens for questions/tasks
            if 'questions' in round_data:
                for question in round_data['questions']:
                    total_tokens += TokenSavingsCalculator.estimate_tokens(str(question))
            
            if 'tasks' in round_data:
                for task in round_data['tasks']:
                    total_tokens += TokenSavingsCalculator.estimate_tokens(str(task))
            
            # Add tokens for responses
            if 'responses' in round_data:
                for response in round_data['responses']:
                    total_tokens += TokenSavingsCalculator.estimate_tokens(str(response))
        
        # Add final answer tokens
        total_tokens += TokenSavingsCalculator.estimate_tokens(final_answer)
        
        return total_tokens
    
    @staticmethod
    def calculate_savings(
        traditional_tokens: int,
        protocol_tokens: int
    ) -> Dict[str, Any]:
        """
        Calculate token savings metrics.
        
        Args:
            traditional_tokens: Tokens for traditional approach
            protocol_tokens: Tokens for protocol approach
            
        Returns:
            Dictionary with savings metrics
        """
        savings = traditional_tokens - protocol_tokens
        savings_percentage = (savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
        
        return {
            "traditional_tokens": traditional_tokens,
            "protocol_tokens": protocol_tokens,
            "tokens_saved": savings,
            "savings_percentage": round(savings_percentage, 1)
        }


class BaseProtocolExecutor(ABC):
    """
    Abstract base class for protocol executors.
    Provides common functionality for both Minion and MinionS protocols.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.response_parser = BaseResponseParser()
        self.token_calculator = TokenSavingsCalculator()
        self.metrics = ExecutionMetrics()
        self.start_time = time.time()
    
    @abstractmethod
    async def execute_protocol(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the protocol. Must be implemented by subclasses."""
        pass
    
    def handle_api_timeout(
        self,
        error: asyncio.TimeoutError,
        context: str,
        timeout_duration: int
    ) -> str:
        """
        Handle API timeout errors consistently.
        
        Args:
            error: The timeout error
            context: Context description
            timeout_duration: Timeout duration in seconds
            
        Returns:
            Formatted error message
        """
        self.metrics.timeout_calls += 1
        error_msg = f"❌ Timeout in {context} after {timeout_duration}s"
        
        if self.debug_mode:
            print(f"DEBUG [Protocol]: {error_msg}")
        
        return error_msg
    
    def handle_api_error(
        self,
        error: Exception,
        context: str,
        response_text: Optional[str] = None
    ) -> str:
        """
        Handle general API errors consistently.
        
        Args:
            error: The exception that occurred
            context: Context description
            response_text: Optional response text
            
        Returns:
            Formatted error message
        """
        self.metrics.failed_calls += 1
        
        # Extract useful information from the error
        if hasattr(error, 'status'):
            error_detail = f"HTTP {error.status}"
            if response_text:
                preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                error_detail += f" - {preview}"
        else:
            error_detail = str(error)
        
        error_msg = f"❌ Error in {context}: {error_detail}"
        
        if self.debug_mode:
            print(f"DEBUG [Protocol]: {error_msg}")
            if response_text:
                print(f"DEBUG [Protocol]: Full response: {response_text[:500]}...")
        
        return error_msg
    
    def record_successful_call(self, duration: Optional[float] = None) -> None:
        """Record a successful API call."""
        self.metrics.successful_calls += 1
        if duration:
            if self.debug_mode:
                print(f"DEBUG [Protocol]: API call completed in {duration:.2f}s")
    
    def calculate_final_metrics(self, final_answer: str = "") -> Dict[str, Any]:
        """
        Calculate final execution metrics.
        
        Args:
            final_answer: The final answer text
            
        Returns:
            Dictionary with comprehensive metrics
        """
        self.metrics.total_duration = time.time() - self.start_time
        
        return {
            "execution_time": round(self.metrics.total_duration, 2),
            "api_calls": {
                "total": self.metrics.api_calls,
                "successful": self.metrics.successful_calls,
                "failed": self.metrics.failed_calls,
                "timeouts": self.metrics.timeout_calls,
                "success_rate": round(self.metrics.get_success_rate() * 100, 1)
            },
            "confidence_analysis": {
                "average": round(self.metrics.get_average_confidence(), 3),
                "scores": self.metrics.confidence_scores,
                "count": len(self.metrics.confidence_scores)
            },
            "token_savings": self.metrics.token_savings
        }
    
    def debug_log(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log debug message if debug mode is enabled.
        
        Args:
            message: Debug message
            data: Optional additional data
        """
        if self.debug_mode:
            print(f"DEBUG [Protocol]: {message}")
            if data:
                print(f"DEBUG [Protocol]: Data: {json.dumps(data, indent=2)}")


class ProtocolUtils:
    """Utility functions shared across protocols."""
    
    @staticmethod
    def truncate_for_display(text: str, max_length: int = 100) -> str:
        """
        Truncate text for display purposes.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
    
    @staticmethod
    def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """
        Safely get nested dictionary value.
        
        Args:
            data: Dictionary to search
            keys: List of nested keys
            default: Default value if key not found
            
        Returns:
            Nested value or default
        """
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    @staticmethod
    def merge_confidence_scores(scores: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Merge multiple confidence scores with optional weights.
        
        Args:
            scores: List of confidence scores
            weights: Optional weights for each score
            
        Returns:
            Merged confidence score
        """
        if not scores:
            return DEFAULT_NUMERIC_CONFIDENCE
        
        if weights and len(weights) == len(scores):
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else DEFAULT_NUMERIC_CONFIDENCE
        else:
            return sum(scores) / len(scores)


# Convenience functions for common operations
def parse_json_response(response: str, debug_mode: bool = False) -> ParsedResponse:
    """Parse JSON response using base parser."""
    parser = BaseResponseParser()
    return parser.parse_response(response, "json", True, debug_mode)


def calculate_token_savings(
    user_query: str,
    document_content: str,
    rounds_data: List[Dict[str, Any]],
    final_answer: str = ""
) -> Dict[str, Any]:
    """Calculate token savings using base calculator."""
    calculator = TokenSavingsCalculator()
    
    traditional_tokens = calculator.calculate_traditional_tokens(
        user_query, document_content, final_answer
    )
    
    protocol_tokens = calculator.calculate_protocol_tokens(rounds_data, final_answer)
    
    return calculator.calculate_savings(traditional_tokens, protocol_tokens)