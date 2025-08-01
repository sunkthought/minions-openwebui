"""
title: Minion Protocol Integration for Open WebUI v0.3.8
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.3.8
description: Enhanced Minion protocol with OpenAI API support, scaling strategies, and adaptive round control
required_open_webui_version: 0.5.0
license: MIT License
"""


# Centralized imports for v0.3.8 modular architecture

# Standard library imports
import asyncio
import json
import re
import hashlib
import traceback
import inspect
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Typing imports
from typing import (
    List, Dict, Any, Optional, Tuple, Callable, Awaitable, 
    Union, Set, TypedDict, Protocol
)

# Third-party imports
import aiohttp
from pydantic import BaseModel, Field, ValidationError
from fastapi import Request

"""
Centralized constants and configuration values for MinionS/Minions OpenWebUI.
This module contains all hardcoded values, magic numbers, and configuration
constants used across the codebase.
"""

from typing import Set, Dict, List
from enum import Enum


# ============================================================================
# Model & API Configuration
# ============================================================================

class ModelDefaults:
    """Default model names for different providers."""
    CLAUDE_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    OLLAMA_DEFAULT = "llama3.2"
    ANTHROPIC_API_VERSION = "2023-06-01"


class APIEndpoints:
    """API endpoints and base URLs."""
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_GENERATE_ENDPOINT = "http://localhost:11434/api/generate"
    ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"


# Models that support structured output (JSON mode)
STRUCTURED_OUTPUT_CAPABLE_MODELS: Set[str] = {
    "llama3.2", "llama3.1", "llama3", "llama2",
    "mistral", "mixtral", "mistral-nemo",
    "qwen2", "qwen2.5", 
    "gemma2", "gemma",
    "phi3", "phi",
    "command-r", "command-r-plus",
    "deepseek-coder", "deepseek-coder-v2",
    "codellama",
    "dolphin-llama3", "dolphin-mixtral",
    "solar", "starling-lm",
    "yi", "zephyr",
    "neural-chat", "openchat"
}


# ============================================================================
# Timeouts & Performance
# ============================================================================

class Timeouts:
    """Timeout values in seconds unless otherwise specified."""
    CLAUDE_DEFAULT = 60
    LOCAL_DEFAULT_MINION = 60
    LOCAL_DEFAULT_MINIONS = 30
    MAX_COMMAND_MS = 120000  # 2 minutes in milliseconds
    ABSOLUTE_MAX_MS = 600000  # 10 minutes in milliseconds


class TokenLimits:
    """Token and context length limits."""
    MAX_TOKENS_CLAUDE_MINION = 4000
    MAX_TOKENS_CLAUDE_MINIONS = 2000
    OLLAMA_NUM_PREDICT_DEFAULT = 1000
    LOCAL_MODEL_CONTEXT_LENGTH = 4096
    CHARS_PER_TOKEN = 3.5  # Conversion ratio


class ChunkSettings:
    """Document chunking configuration."""
    DEFAULT_SIZE = 5000
    DEFAULT_MAX_CHUNKS = 2
    MAX_LINES_PER_READ = 2000
    MAX_OUTPUT_CHARS = 30000


# ============================================================================
# Processing & Round Limits
# ============================================================================

class RoundLimits:
    """Limits for rounds and tasks."""
    MAX_ROUNDS_DEFAULT = 2
    MAX_TASKS_PER_ROUND = 3
    MIN_ROUNDS_BEFORE_STOPPING = 1
    CONVERGENCE_ROUNDS_MIN_NOVELTY = 2
    MAX_ROUND_TIMEOUT_FAILURE_THRESHOLD_PERCENT = 50


class QuestionLimits:
    """Limits for different types of questions."""
    MAX_EXPLORATION_QUESTIONS = 3
    MAX_DEEP_DIVE_QUESTIONS = 4
    MAX_GAP_FILLING_QUESTIONS = 2
    MAX_CLARIFICATION_ATTEMPTS = 1


# ============================================================================
# Confidence & Thresholds
# ============================================================================

class ConfidenceMapping:
    """Confidence level mappings."""
    LEVELS = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
    DEFAULT_NUMERIC = 0.3  # Corresponds to LOW
    

class ConfidenceThresholds:
    """Various confidence and threshold values."""
    DEFAULT = 0.7
    SIMPLE_QUERY = 0.85
    MEDIUM_QUERY = 0.75
    CONVERGENCE_NOVELTY = 0.10
    CONVERGENCE_SUFFICIENCY = 0.7
    DEDUPLICATION = 0.8
    FIRST_ROUND_HIGH_NOVELTY = 0.75
    KEYWORD_OVERLAP_MIN = 0.2
    HIGH_NOVELTY_PREDICTION = 0.5
    LOW_NOVELTY_PREDICTION = 0.1
    FAILURE_RATE_TREND_TOLERANCE = 0.05


# ============================================================================
# Document Size Configuration
# ============================================================================

class DocumentSizeLimits:
    """Character limits for document size classification."""
    SMALL_CHAR_LIMIT = 5000
    LARGE_CHAR_START = 50000


# ============================================================================
# Adaptive Modifiers
# ============================================================================

class AdaptiveModifiers:
    """Adaptive modifiers for confidence and sufficiency."""
    CONFIDENCE_SMALL_DOC = 0.0
    CONFIDENCE_LARGE_DOC = 0.0
    SUFFICIENCY_SIMPLE_QUERY = 0.0
    SUFFICIENCY_COMPLEX_QUERY = 0.0
    NOVELTY_SIMPLE_QUERY = 0.0
    NOVELTY_COMPLEX_QUERY = 0.0
    SUFFICIENCY_HIGH_FIRST_ROUND_NOVELTY = -0.05


# ============================================================================
# Generation Parameters
# ============================================================================

class GenerationParams:
    """Parameters for text generation."""
    LOCAL_MODEL_TEMPERATURE = 0.7
    CLAUDE_API_TEMPERATURE = 0.1
    LOCAL_MODEL_TOP_K = 40


# ============================================================================
# Performance Profiles
# ============================================================================

class PerformanceProfiles:
    """Available performance profiles."""
    HIGH_QUALITY = "high_quality"
    BALANCED = "balanced"
    FASTEST_RESULTS = "fastest_results"
    
    ALL_PROFILES = [HIGH_QUALITY, BALANCED, FASTEST_RESULTS]
    DEFAULT = BALANCED


# ============================================================================
# Text Constants
# ============================================================================

# Phrases that indicate completion
COMPLETION_PHRASES = [
    "i now have sufficient information",
    "i can now answer",
    "based on the information gathered",
    "i have enough information",
    "with this information, i can provide",
    "i can now provide a comprehensive answer",
    "based on what the local assistant has told me"
]

# Phrases indicating lack of information
NON_ANSWER_PHRASES = [
    "i don't know",
    "not sure", 
    "unclear",
    "cannot determine",
    "no information",
    "not available",
    "not found",
    "not mentioned",
    "not specified"
]

# Stop words for deduplication
STOP_WORDS: Set[str] = {
    "what", "is", "the", "are", "how", "does", "can", "you", "tell", "me", "about",
    "please", "could", "would", "explain", "describe", "provide", "give", "any",
    "there", "which", "when", "where", "who", "why", "do", "have", "has", "been",
    "was", "were", "will", "be", "being", "a", "an", "and", "or", "but", "in",
    "on", "at", "to", "for", "of", "with", "by", "from", "as", "this", "that"
}


# ============================================================================
# Enumerations
# ============================================================================

class ConversationPhase(str, Enum):
    """Phases of conversation in Minion protocol."""
    EXPLORATION = "exploration"
    DEEP_DIVE = "deep_dive"
    GAP_FILLING = "gap_filling"
    SYNTHESIS = "synthesis"


class TaskStatus(str, Enum):
    """Status values for task execution."""
    SUCCESS = "success"
    TIMEOUT_ALL_CHUNKS = "timeout_all_chunks"
    NOT_FOUND = "not_found"


class FailureTrend(str, Enum):
    """Trends in failure rates."""
    INCREASING = "increasing_failures"
    DECREASING = "decreasing_failures"
    STABLE = "stable_failures"


class PredictedValue(str, Enum):
    """Predicted values for analysis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Severity(str, Enum):
    """Severity levels."""
    HIGH = "high"
    LOW = "low"
    NONE = "none"


# ============================================================================
# Numeric Limits
# ============================================================================

class NumericLimits:
    """Various numeric limits and thresholds."""
    MIN_TASK_LENGTH = 10
    MAX_QUERY_WORDS_FOR_KEYWORDS = 5
    MAX_KEYWORDS_PER_TOPIC = 3
    MIN_WORD_LENGTH_FOR_STOP = 2
    MAX_DEBUG_OUTPUT_CHARS = 500
    MAX_CONVERSATION_LOG_CHARS = 100
    MAX_ANSWER_PREVIEW_CHARS = 70
    MAX_CLAUDE_RESPONSE_PREVIEW = 200


# ============================================================================
# Format & Display
# ============================================================================

class FormatOptions:
    """Available output format options."""
    TEXT = "text"
    JSON = "JSON"
    BULLET_POINTS = "bullet points"
    DEFAULT = TEXT


class ConfidenceLevels:
    """Text representations of confidence levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ============================================================================
# System Requirements & Markers
# ============================================================================

class SystemRequirements:
    """System version requirements."""
    MIN_OPENWEBUI_VERSION = "0.5.0"
    CURRENT_VERSION = "v0.3.7"


class SpecialMarkers:
    """Special markers and triggers in responses."""
    FINAL_ANSWER_READY = "FINAL ANSWER READY."
    FINAL_ANSWER_READY_ALT = "FINAL ANSWER READY:"
    NO_INFO_MARKER = "NONE"
    ERROR_PREFIX = "CLAUDE_ERROR:"


# ============================================================================
# Helper Functions
# ============================================================================

def get_profile_settings(profile: str) -> Dict[str, Any]:
    """
    Get settings for a specific performance profile.
    
    Args:
        profile: Name of the performance profile
        
    Returns:
        Dictionary of settings for the profile
    """
    profiles = {
        PerformanceProfiles.HIGH_QUALITY: {
            'max_rounds': 3,
            'confidence_threshold': 0.65,
            'max_tokens_claude': 4000,
        },
        PerformanceProfiles.BALANCED: {
            'max_rounds': 2,
            'confidence_threshold': 0.7,
            'max_tokens_claude': 2000,
        },
        PerformanceProfiles.FASTEST_RESULTS: {
            'max_rounds': 1,
            'confidence_threshold': 0.75,
            'max_tokens_claude': 1000,
        }
    }
    return profiles.get(profile, profiles[PerformanceProfiles.BALANCED])


def is_structured_output_model(model_name: str) -> bool:
    """
    Check if a model supports structured output.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model supports structured output, False otherwise
    """
    # Check if any supported model name is in the provided model name
    return any(supported in model_name.lower() for supported in STRUCTURED_OUTPUT_CAPABLE_MODELS)

"""
Centralized error handling for MinionS/Minions OpenWebUI functions.
This module provides consistent error handling, formatting, and logging
across all protocols and components.
"""

import json
import re
import traceback
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from pydantic import BaseModel, ValidationError


class ErrorContext:
    """Context information for error handling."""
    
    def __init__(
        self,
        round_num: Optional[int] = None,
        chunk_idx: Optional[int] = None,
        task_idx: Optional[int] = None,
        operation: Optional[str] = None,
        service_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.round_num = round_num
        self.chunk_idx = chunk_idx
        self.task_idx = task_idx
        self.operation = operation
        self.service_name = service_name
        self.model_name = model_name
    
    def __str__(self) -> str:
        """Generate context string for error messages."""
        parts = []
        
        if self.operation:
            parts.append(self.operation)
        
        if self.service_name:
            parts.append(f"({self.service_name})")
        
        if self.round_num is not None:
            parts.append(f"round {self.round_num + 1}")
        
        if self.task_idx is not None:
            parts.append(f"task {self.task_idx + 1}")
        
        if self.chunk_idx is not None:
            parts.append(f"chunk {self.chunk_idx + 1}")
        
        return " ".join(parts) if parts else "operation"


class ErrorHandler:
    """Centralized error handling with consistent formatting and logging."""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
    
    def format_error_message(
        self,
        error_type: str,
        context: Union[ErrorContext, str],
        details: str,
        include_emoji: bool = True
    ) -> str:
        """
        Format error message with consistent structure.
        
        Args:
            error_type: Type of error (e.g., "API Error", "Timeout")
            context: Error context information
            details: Specific error details
            include_emoji: Whether to include emoji prefix
            
        Returns:
            Formatted error message
        """
        emoji = "❌ " if include_emoji else ""
        context_str = str(context) if context else "operation"
        return f"{emoji}{error_type} in {context_str}: {details}"
    
    def handle_api_error(
        self,
        error: Exception,
        context: ErrorContext,
        response_text: Optional[str] = None
    ) -> str:
        """
        Handle API-related errors with consistent formatting.
        
        Args:
            error: The exception that occurred
            context: Context information for the error
            response_text: Optional API response text
            
        Returns:
            Formatted error message
        """
        error_details = str(error)
        
        # Extract useful information from common API errors
        if hasattr(error, 'status'):
            error_details = f"HTTP {error.status}"
            if response_text:
                # Truncate response for readability
                truncated_response = response_text[:200] + "..." if len(response_text) > 200 else response_text
                error_details += f" - {truncated_response}"
        elif "timeout" in str(error).lower():
            error_details = "Request timed out"
        elif "connection" in str(error).lower():
            error_details = "Connection failed"
        
        return self.format_error_message("API Error", context, error_details)
    
    def handle_timeout_error(
        self,
        error: asyncio.TimeoutError,
        context: ErrorContext,
        timeout_duration: Optional[int] = None
    ) -> str:
        """
        Handle timeout errors with duration information.
        
        Args:
            error: The timeout exception
            context: Context information for the error
            timeout_duration: Timeout duration in seconds
            
        Returns:
            Formatted error message
        """
        if timeout_duration:
            details = f"Operation timed out after {timeout_duration}s"
        else:
            details = "Operation timed out"
        
        return self.format_error_message("Timeout", context, details)
    
    def handle_json_parse_error(
        self,
        error: json.JSONDecodeError,
        response_text: str,
        context: ErrorContext,
        attempt_fallback: bool = True
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Handle JSON parsing errors with fallback mechanism.
        
        Args:
            error: The JSON decode error
            response_text: The text that failed to parse
            context: Context information for the error
            attempt_fallback: Whether to attempt regex fallback
            
        Returns:
            Tuple of (error_message, parsed_data_if_successful)
        """
        error_msg = self.format_error_message(
            "JSON Parse Error",
            context,
            f"Invalid JSON response: {str(error)}"
        )
        
        parsed_data = None
        
        if attempt_fallback and response_text:
            # Attempt regex fallback for common JSON patterns
            parsed_data = self._json_regex_fallback(response_text)
            if parsed_data:
                error_msg += " (recovered using fallback parsing)"
        
        return error_msg, parsed_data
    
    def handle_validation_error(
        self,
        error: ValidationError,
        context: ErrorContext,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Handle Pydantic validation errors.
        
        Args:
            error: The validation error
            context: Context information for the error
            data: The data that failed validation
            
        Returns:
            Formatted error message
        """
        # Extract the first error for simplicity
        first_error = error.errors()[0] if error.errors() else {}
        field = ".".join(str(x) for x in first_error.get('loc', []))
        msg = first_error.get('msg', 'Validation failed')
        
        details = f"Field '{field}': {msg}" if field else msg
        
        return self.format_error_message("Validation Error", context, details)
    
    def handle_connection_error(
        self,
        error: Exception,
        context: ErrorContext,
        base_url: Optional[str] = None
    ) -> str:
        """
        Handle connection-related errors.
        
        Args:
            error: The connection exception
            context: Context information for the error
            base_url: The base URL that failed to connect
            
        Returns:
            Formatted error message
        """
        details = "Connection failed"
        
        if base_url:
            details += f" to {base_url}"
        
        if "refused" in str(error).lower():
            details += " (connection refused)"
        elif "timeout" in str(error).lower():
            details += " (connection timeout)"
        
        return self.format_error_message("Connection Error", context, details)
    
    def log_error_to_conversation(
        self,
        message: str,
        conversation_log: List[str]
    ) -> None:
        """
        Log error message to conversation log.
        
        Args:
            message: Error message to log
            conversation_log: Conversation log list to append to
        """
        conversation_log.append(message)
    
    def log_error_to_debug(
        self,
        message: str,
        debug_log: List[str],
        error: Optional[Exception] = None,
        include_traceback: bool = None
    ) -> None:
        """
        Log error message to debug log with optional traceback.
        
        Args:
            message: Error message to log
            debug_log: Debug log list to append to
            error: Optional exception for traceback
            include_traceback: Whether to include traceback (defaults to debug_mode)
        """
        debug_log.append(f"DEBUG [ErrorHandler]: {message}")
        
        if include_traceback is None:
            include_traceback = self.debug_mode
        
        if include_traceback and error:
            tb_str = traceback.format_exc()
            debug_log.append(f"DEBUG [ErrorHandler]: Full traceback:\n{tb_str}")
    
    def log_error(
        self,
        message: str,
        conversation_log: Optional[List[str]] = None,
        debug_log: Optional[List[str]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """
        Log error to both conversation and debug logs.
        
        Args:
            message: Error message to log
            conversation_log: Optional conversation log
            debug_log: Optional debug log
            error: Optional exception for debug traceback
        """
        if conversation_log is not None:
            self.log_error_to_conversation(message, conversation_log)
        
        if debug_log is not None:
            self.log_error_to_debug(message, debug_log, error)
    
    def _json_regex_fallback(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract JSON from malformed response using regex patterns.
        
        Args:
            response_text: The malformed response text
            
        Returns:
            Parsed JSON data if successful, None otherwise
        """
        # Common patterns for extracting JSON from responses
        patterns = [
            r'\{[^{}]*"[^"]*"[^{}]*\}',  # Simple object pattern
            r'\{.*\}',  # Greedy object pattern
            r'```json\s*(\{.*?\})\s*```',  # JSON code block
            r'```\s*(\{.*?\})\s*```',  # Generic code block
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    async def api_retry_with_backoff(
        self,
        api_call_func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        context: Optional[ErrorContext] = None
    ) -> Any:
        """
        Retry API calls with exponential backoff.
        
        Args:
            api_call_func: Async function to retry
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            context: Error context for logging
            
        Returns:
            Result of successful API call
            
        Raises:
            Last exception if all retries fail
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await api_call_func()
            except Exception as e:
                last_error = e
                
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    if self.debug_mode and context:
                        error_msg = f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}"
                        print(f"DEBUG [ErrorHandler]: {error_msg}")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    break
        
        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise Exception("API retry failed with unknown error")
    
    def add_troubleshooting_hints(
        self,
        error_type: str,
        context: ErrorContext,
        base_url: Optional[str] = None
    ) -> List[str]:
        """
        Generate troubleshooting hints based on error type.
        
        Args:
            error_type: Type of error that occurred
            context: Error context information
            base_url: Optional base URL for connection issues
            
        Returns:
            List of troubleshooting hint strings
        """
        hints = []
        
        if "timeout" in error_type.lower():
            hints.extend([
                "• Try increasing the timeout value in valves",
                "• Check if the model is responding normally",
                "• Consider using a faster model or smaller input"
            ])
        
        elif "connection" in error_type.lower():
            if base_url and "localhost" in base_url:
                hints.extend([
                    f"• Ensure Ollama is running on {base_url}",
                    "• Check if the Ollama service is accessible",
                    "• Verify the base URL in valves configuration"
                ])
            else:
                hints.extend([
                    "• Check your internet connection",
                    "• Verify API endpoint URLs",
                    "• Check if service is experiencing downtime"
                ])
        
        elif "api" in error_type.lower():
            if context.service_name == "Anthropic":
                hints.extend([
                    "• Verify your Anthropic API key is valid",
                    "• Check if you have sufficient API credits",
                    "• Ensure the model name is correct"
                ])
            elif context.service_name == "Ollama":
                hints.extend([
                    f"• Verify the model '{context.model_name}' is installed in Ollama",
                    "• Check if Ollama has sufficient resources",
                    "• Try pulling the model: ollama pull <model_name>"
                ])
        
        elif "json" in error_type.lower():
            hints.extend([
                "• The model may not support structured output",
                "• Try a different model or disable JSON mode",
                "• Check if the model is properly configured"
            ])
        
        return hints
    
    def create_comprehensive_error_report(
        self,
        error: Exception,
        context: ErrorContext,
        response_text: Optional[str] = None,
        include_hints: bool = True
    ) -> str:
        """
        Create a comprehensive error report with context and hints.
        
        Args:
            error: The exception that occurred
            context: Error context information
            response_text: Optional response text
            include_hints: Whether to include troubleshooting hints
            
        Returns:
            Comprehensive error report string
        """
        # Determine error type and handle accordingly
        if isinstance(error, asyncio.TimeoutError):
            error_msg = self.handle_timeout_error(error, context)
        elif isinstance(error, json.JSONDecodeError):
            error_msg, _ = self.handle_json_parse_error(error, response_text or "", context)
        elif isinstance(error, ValidationError):
            error_msg = self.handle_validation_error(error, context)
        elif "connection" in str(error).lower():
            error_msg = self.handle_connection_error(error, context)
        else:
            error_msg = self.handle_api_error(error, context, response_text)
        
        report_parts = [error_msg]
        
        if include_hints:
            hints = self.add_troubleshooting_hints(type(error).__name__, context)
            if hints:
                report_parts.append("\nTroubleshooting suggestions:")
                report_parts.extend(hints)
        
        return "\n".join(report_parts)


# Utility functions for common error scenarios
def safe_json_parse(
    text: str,
    fallback_value: Any = None,
    error_handler: Optional[ErrorHandler] = None,
    context: Optional[ErrorContext] = None
) -> tuple[Any, Optional[str]]:
    """
    Safely parse JSON with fallback handling.
    
    Args:
        text: JSON text to parse
        fallback_value: Value to return if parsing fails
        error_handler: Optional error handler for logging
        context: Optional error context
        
    Returns:
        Tuple of (parsed_data, error_message)
    """
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        error_msg = None
        if error_handler and context:
            error_msg, fallback_data = error_handler.handle_json_parse_error(e, text, context)
            if fallback_data is not None:
                return fallback_data, error_msg
        
        return fallback_value, error_msg or f"JSON parse error: {str(e)}"


def create_error_context(
    operation: str,
    service_name: Optional[str] = None,
    round_num: Optional[int] = None,
    chunk_idx: Optional[int] = None,
    task_idx: Optional[int] = None,
    model_name: Optional[str] = None
) -> ErrorContext:
    """
    Convenience function to create error context.
    
    Args:
        operation: Description of the operation
        service_name: Name of the service (e.g., "Anthropic", "Ollama")
        round_num: Round number (0-indexed)
        chunk_idx: Chunk index (0-indexed)
        task_idx: Task index (0-indexed)
        model_name: Name of the model being used
        
    Returns:
        ErrorContext instance
    """
    return ErrorContext(
        operation=operation,
        service_name=service_name,
        round_num=round_num,
        chunk_idx=chunk_idx,
        task_idx=task_idx,
        model_name=model_name
    )

"""
Structured debug logging utilities for MinionS/Minions OpenWebUI functions.
This module provides centralized, consistent debug logging with context awareness
and multiple output channels.
"""

import time
import json
import traceback
from typing import List, Dict, Any, Optional, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum


class DebugLevel(str, Enum):
    """Debug logging levels."""
    ERROR = "ERROR"
    WARNING = "WARNING" 
    INFO = "INFO"
    TIMING = "TIMING"
    API = "API"
    STATE = "STATE"
    METRICS = "METRICS"
    TRACE = "TRACE"


@dataclass
class DebugContext:
    """Context information for debug logging."""
    component: str
    round_num: Optional[int] = None
    task_idx: Optional[int] = None
    chunk_idx: Optional[int] = None
    phase: Optional[str] = None
    operation: Optional[str] = None
    indent_level: int = 0
    
    def __str__(self) -> str:
        """Generate context string for debug messages."""
        parts = [self.component]
        
        if self.round_num is not None:
            parts.append(f"R{self.round_num + 1}")
        
        if self.task_idx is not None:
            parts.append(f"T{self.task_idx + 1}")
        
        if self.chunk_idx is not None:
            parts.append(f"C{self.chunk_idx + 1}")
        
        if self.phase:
            parts.append(f"({self.phase})")
        
        return ":".join(parts)


@dataclass
class TimingInfo:
    """Information about operation timing."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    context: Optional[DebugContext] = None
    
    def finish(self) -> float:
        """Mark timing as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration


class DebugLogger:
    """
    Centralized debug logger with context awareness and multiple output channels.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        console_output: bool = True,
        max_response_preview: int = 500,
        max_entries: int = 1000
    ):
        self.enabled = enabled
        self.console_output = console_output
        self.max_response_preview = max_response_preview
        self.max_entries = max_entries
        
        # Storage for debug entries
        self.entries: List[Dict[str, Any]] = []
        self.timings: List[TimingInfo] = []
        self.active_timers: Dict[str, TimingInfo] = {}
        
        # Current context stack for hierarchical logging
        self.context_stack: List[DebugContext] = []
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable debug logging."""
        self.enabled = enabled
    
    def push_context(self, context: DebugContext) -> None:
        """Push a new context onto the stack."""
        if self.context_stack:
            # Inherit indent level from parent
            context.indent_level = self.context_stack[-1].indent_level + 1
        self.context_stack.append(context)
    
    def pop_context(self) -> Optional[DebugContext]:
        """Pop the current context from the stack."""
        return self.context_stack.pop() if self.context_stack else None
    
    def get_current_context(self) -> Optional[DebugContext]:
        """Get the current context."""
        return self.context_stack[-1] if self.context_stack else None
    
    def log(
        self,
        level: DebugLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[DebugContext] = None,
        debug_log: Optional[List[str]] = None
    ) -> None:
        """
        Log a debug message with level and context.
        
        Args:
            level: Debug level
            message: Debug message
            data: Optional additional data
            context: Optional context (uses current if not provided)
            debug_log: Optional debug log list to append to
        """
        if not self.enabled:
            return
        
        # Use provided context or current context
        ctx = context or self.get_current_context()
        
        # Create debug entry
        entry = {
            "timestamp": time.time(),
            "level": level.value,
            "message": message,
            "context": str(ctx) if ctx else "GLOBAL",
            "indent_level": ctx.indent_level if ctx else 0,
            "data": data
        }
        
        # Store entry
        self.entries.append(entry)
        
        # Limit entries to prevent memory issues
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Format for output
        formatted_msg = self._format_message(entry)
        
        # Output to console if enabled
        if self.console_output:
            print(formatted_msg)
        
        # Append to debug log if provided
        if debug_log is not None:
            debug_log.append(formatted_msg)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self.log(DebugLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self.log(DebugLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self.log(DebugLevel.ERROR, message, **kwargs)
    
    def api(self, message: str, **kwargs) -> None:
        """Log an API-related message."""
        self.log(DebugLevel.API, message, **kwargs)
    
    def state(self, message: str, **kwargs) -> None:
        """Log a state change message."""
        self.log(DebugLevel.STATE, message, **kwargs)
    
    def metrics(self, message: str, **kwargs) -> None:
        """Log a metrics message."""
        self.log(DebugLevel.METRICS, message, **kwargs)
    
    def timing(self, message: str, duration: Optional[float] = None, **kwargs) -> None:
        """Log a timing message."""
        if duration is not None:
            message += f" ({duration:.2f}s)"
        self.log(DebugLevel.TIMING, message, **kwargs)
    
    def trace(self, message: str, **kwargs) -> None:
        """Log a detailed trace message."""
        self.log(DebugLevel.TRACE, message, **kwargs)
    
    def api_call(
        self,
        service: str,
        operation: str,
        duration: Optional[float] = None,
        response_preview: Optional[str] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log an API call with standardized format.
        
        Args:
            service: Service name (e.g., "Claude", "Ollama")
            operation: Operation description
            duration: Optional call duration
            response_preview: Optional response preview
            error: Optional error message
        """
        if error:
            message = f"{service} {operation} failed: {error}"
            level = DebugLevel.ERROR
        else:
            message = f"{service} {operation}"
            if duration is not None:
                message += f" ({duration:.2f}s)"
            level = DebugLevel.API
        
        data = {}
        if response_preview:
            # Truncate long responses
            if len(response_preview) > self.max_response_preview:
                preview = response_preview[:self.max_response_preview] + "..."
            else:
                preview = response_preview
            data["response_preview"] = preview
        
        self.log(level, message, data=data if data else None, **kwargs)
    
    def json_response(
        self,
        service: str,
        raw_response: str,
        parsed_data: Optional[Dict[str, Any]] = None,
        parse_error: Optional[str] = None,
        used_fallback: bool = False,
        **kwargs
    ) -> None:
        """
        Log JSON response parsing details.
        
        Args:
            service: Service name
            raw_response: Raw response text
            parsed_data: Successfully parsed data
            parse_error: Parsing error if any
            used_fallback: Whether fallback parsing was used
        """
        if parse_error:
            message = f"{service} JSON parsing failed: {parse_error}"
            if used_fallback:
                message += " (fallback used)"
            level = DebugLevel.WARNING
        else:
            message = f"{service} JSON parsed successfully"
            level = DebugLevel.API
        
        data = {
            "raw_response_length": len(raw_response),
            "used_fallback": used_fallback
        }
        
        if parsed_data:
            data["parsed_keys"] = list(parsed_data.keys()) if isinstance(parsed_data, dict) else None
        
        self.log(level, message, data=data, **kwargs)
    
    def confidence_analysis(
        self,
        confidence_score: float,
        threshold: float,
        component_scores: Optional[Dict[str, float]] = None,
        modifiers: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Log confidence analysis details.
        
        Args:
            confidence_score: Final confidence score
            threshold: Threshold used
            component_scores: Component-wise scores
            modifiers: Applied modifiers
        """
        status = "PASS" if confidence_score >= threshold else "FAIL"
        message = f"Confidence analysis: {confidence_score:.3f} vs {threshold:.3f} [{status}]"
        
        data = {
            "score": confidence_score,
            "threshold": threshold,
            "passed": confidence_score >= threshold
        }
        
        if component_scores:
            data["components"] = component_scores
        
        if modifiers:
            data["modifiers"] = modifiers
        
        self.log(DebugLevel.METRICS, message, data=data, **kwargs)
    
    def round_summary(
        self,
        round_num: int,
        tasks_completed: int,
        total_duration: float,
        confidence_scores: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        """
        Log round completion summary.
        
        Args:
            round_num: Round number (0-indexed)
            tasks_completed: Number of tasks completed
            total_duration: Total round duration
            confidence_scores: Confidence scores for tasks
        """
        message = f"Round {round_num + 1} completed: {tasks_completed} tasks in {total_duration:.2f}s"
        
        data = {
            "round_num": round_num,
            "tasks_completed": tasks_completed,
            "duration": total_duration
        }
        
        if confidence_scores:
            data["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
            data["confidence_range"] = [min(confidence_scores), max(confidence_scores)]
        
        self.log(DebugLevel.STATE, message, data=data, **kwargs)
    
    @contextmanager
    def timer(self, operation: str, context: Optional[DebugContext] = None):
        """
        Context manager for timing operations.
        
        Args:
            operation: Description of the operation being timed
            context: Optional debug context
            
        Yields:
            TimingInfo object that can be used to access timing data
        """
        timer_key = f"{operation}_{time.time()}"
        timing_info = TimingInfo(
            operation=operation,
            start_time=time.time(),
            context=context or self.get_current_context()
        )
        
        self.active_timers[timer_key] = timing_info
        
        try:
            yield timing_info
        finally:
            duration = timing_info.finish()
            self.timings.append(timing_info)
            del self.active_timers[timer_key]
            
            if self.enabled:
                self.timing(f"{operation} completed", duration=duration, context=context)
    
    @contextmanager
    def context(
        self,
        component: str,
        round_num: Optional[int] = None,
        task_idx: Optional[int] = None,
        chunk_idx: Optional[int] = None,
        phase: Optional[str] = None,
        operation: Optional[str] = None
    ):
        """
        Context manager for debug context.
        
        Args:
            component: Component name
            round_num: Optional round number
            task_idx: Optional task index
            chunk_idx: Optional chunk index
            phase: Optional phase name
            operation: Optional operation name
        """
        ctx = DebugContext(
            component=component,
            round_num=round_num,
            task_idx=task_idx,
            chunk_idx=chunk_idx,
            phase=phase,
            operation=operation
        )
        
        self.push_context(ctx)
        try:
            yield ctx
        finally:
            self.pop_context()
    
    def _format_message(self, entry: Dict[str, Any]) -> str:
        """Format a debug entry for output."""
        level = entry["level"]
        context = entry["context"]
        message = entry["message"]
        indent = "  " * entry["indent_level"]
        
        # Level-specific formatting
        if level == DebugLevel.ERROR.value:
            prefix = "❌"
        elif level == DebugLevel.WARNING.value:
            prefix = "⚠️"
        elif level == DebugLevel.TIMING.value:
            prefix = "⏱️"
        elif level == DebugLevel.API.value:
            prefix = "🔗"
        elif level == DebugLevel.STATE.value:
            prefix = "🔄"
        elif level == DebugLevel.METRICS.value:
            prefix = "📊"
        else:
            prefix = "🔍"
        
        formatted = f"{indent}DEBUG [{context}] {prefix} {message}"
        
        # Add data if present
        if entry.get("data"):
            data_str = json.dumps(entry["data"], indent=2)
            formatted += f"\n{indent}  Data: {data_str}"
        
        return formatted
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of debug activity."""
        if not self.entries:
            return {"message": "No debug entries recorded"}
        
        # Count by level
        level_counts = {}
        for entry in self.entries:
            level = entry["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Timing summary
        timing_summary = {
            "total_operations": len(self.timings),
            "avg_duration": sum(t.duration for t in self.timings if t.duration) / len(self.timings) if self.timings else 0,
            "longest_operation": max(self.timings, key=lambda t: t.duration or 0).operation if self.timings else None
        }
        
        return {
            "total_entries": len(self.entries),
            "level_breakdown": level_counts,
            "timing_summary": timing_summary,
            "active_timers": len(self.active_timers),
            "context_depth": len(self.context_stack)
        }
    
    def export_logs(self, include_data: bool = True) -> List[Dict[str, Any]]:
        """Export all debug logs as structured data."""
        if include_data:
            return self.entries.copy()
        else:
            return [
                {k: v for k, v in entry.items() if k != "data"}
                for entry in self.entries
            ]
    
    def clear(self) -> None:
        """Clear all debug logs and reset state."""
        self.entries.clear()
        self.timings.clear()
        self.active_timers.clear()
        self.context_stack.clear()


# Global debug logger instance
_global_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get the global debug logger instance."""
    global _global_debug_logger
    if _global_debug_logger is None:
        _global_debug_logger = DebugLogger()
    return _global_debug_logger


def init_debug_logger(
    enabled: bool = False,
    console_output: bool = True,
    **kwargs
) -> DebugLogger:
    """
    Initialize the global debug logger.
    
    Args:
        enabled: Whether debug logging is enabled
        console_output: Whether to output to console
        **kwargs: Additional DebugLogger parameters
        
    Returns:
        Configured DebugLogger instance
    """
    global _global_debug_logger
    _global_debug_logger = DebugLogger(
        enabled=enabled,
        console_output=console_output,
        **kwargs
    )
    return _global_debug_logger


# Convenience functions that use the global logger
def debug_info(message: str, **kwargs) -> None:
    """Log info message using global logger."""
    get_debug_logger().info(message, **kwargs)


def debug_timing(message: str, duration: Optional[float] = None, **kwargs) -> None:
    """Log timing message using global logger."""
    get_debug_logger().timing(message, duration=duration, **kwargs)


def debug_api_call(service: str, operation: str, **kwargs) -> None:
    """Log API call using global logger."""
    get_debug_logger().api_call(service, operation, **kwargs)


def debug_context(**kwargs):
    """Create debug context using global logger."""
    return get_debug_logger().context(**kwargs)


def debug_timer(operation: str, **kwargs):
    """Create debug timer using global logger."""
    return get_debug_logger().timer(operation, **kwargs)

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

"""
State management utilities for MinionS/Minions OpenWebUI protocols.
This module provides centralized state tracking, round management,
and execution context for both Minion and MinionS protocols.
"""

import time
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ProtocolType(str, Enum):
    """Types of protocols supported."""
    MINION = "minion"
    MINIONS = "minions"


class RoundStatus(str, Enum):
    """Status of protocol rounds."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskStatus(str, Enum):
    """Status of individual tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    TIMEOUT_ALL_CHUNKS = "timeout_all_chunks"
    NOT_FOUND = "not_found"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    status: TaskStatus
    content: Any
    confidence: float
    execution_time: float
    chunk_indices: List[int] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "content": self.content,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "chunk_indices": self.chunk_indices,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class RoundResult:
    """Result of a complete round execution."""
    round_num: int
    status: RoundStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tasks: List[TaskResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def finish(self) -> None:
        """Mark round as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def add_task_result(self, task_result: TaskResult) -> None:
        """Add a task result to this round."""
        self.tasks.append(task_result)
    
    def get_successful_tasks(self) -> List[TaskResult]:
        """Get list of successful tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.SUCCESS]
    
    def get_failed_tasks(self) -> List[TaskResult]:
        """Get list of failed tasks."""
        return [task for task in self.tasks if task.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT_ALL_CHUNKS]]
    
    def get_success_rate(self) -> float:
        """Calculate success rate for this round."""
        if not self.tasks:
            return 0.0
        successful = len(self.get_successful_tasks())
        return successful / len(self.tasks)
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence for successful tasks."""
        successful_tasks = self.get_successful_tasks()
        if not successful_tasks:
            return 0.0
        return sum(task.confidence for task in successful_tasks) / len(successful_tasks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "round_num": self.round_num,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tasks": [task.to_dict() for task in self.tasks],
            "metrics": self.metrics,
            "error_message": self.error_message,
            "success_rate": self.get_success_rate(),
            "average_confidence": self.get_average_confidence()
        }


@dataclass
class ConversationState:
    """State management for conversational protocols (Minion)."""
    questions_asked: List[str] = field(default_factory=list)
    answers_received: List[str] = field(default_factory=list)
    conversation_log: List[str] = field(default_factory=list)
    deduplication_fingerprints: set = field(default_factory=set)
    phase: str = "exploration"
    clarification_attempts: int = 0
    final_answer_detected: bool = False
    
    def add_question(self, question: str) -> None:
        """Add a question to the conversation."""
        self.questions_asked.append(question)
        self.conversation_log.append(f"Q: {question}")
    
    def add_answer(self, answer: str) -> None:
        """Add an answer to the conversation."""
        self.answers_received.append(answer)
        self.conversation_log.append(f"A: {answer}")
    
    def add_fingerprint(self, fingerprint: str) -> bool:
        """
        Add deduplication fingerprint.
        
        Args:
            fingerprint: Question/task fingerprint
            
        Returns:
            True if fingerprint was new, False if duplicate
        """
        if fingerprint in self.deduplication_fingerprints:
            return False
        self.deduplication_fingerprints.add(fingerprint)
        return True
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation state."""
        return {
            "total_questions": len(self.questions_asked),
            "total_answers": len(self.answers_received),
            "current_phase": self.phase,
            "clarification_attempts": self.clarification_attempts,
            "final_answer_detected": self.final_answer_detected,
            "unique_fingerprints": len(self.deduplication_fingerprints)
        }


@dataclass
class ProtocolMetrics:
    """Comprehensive metrics tracking for protocol execution."""
    protocol_type: ProtocolType
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    
    # API call metrics
    total_api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    timeout_api_calls: int = 0
    
    # Confidence metrics
    confidence_scores: List[float] = field(default_factory=list)
    confidence_distribution: Dict[str, int] = field(default_factory=lambda: {"HIGH": 0, "MEDIUM": 0, "LOW": 0})
    
    # Token metrics
    token_savings: Optional[Dict[str, Any]] = None
    
    # Protocol-specific metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self) -> None:
        """Mark metrics collection as finished."""
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
    
    def record_api_call(self, success: bool, timeout: bool = False) -> None:
        """Record an API call result."""
        self.total_api_calls += 1
        if timeout:
            self.timeout_api_calls += 1
        elif success:
            self.successful_api_calls += 1
        else:
            self.failed_api_calls += 1
    
    def add_confidence_score(self, score: float, level: Optional[str] = None) -> None:
        """Add a confidence score to metrics."""
        self.confidence_scores.append(score)
        
        if level:
            self.confidence_distribution[level] = self.confidence_distribution.get(level, 0) + 1
        else:
            # Categorize based on score
            if score >= 0.8:
                self.confidence_distribution["HIGH"] += 1
            elif score >= 0.5:
                self.confidence_distribution["MEDIUM"] += 1
            else:
                self.confidence_distribution["LOW"] += 1
    
    def get_api_success_rate(self) -> float:
        """Calculate API call success rate."""
        total_calls = self.successful_api_calls + self.failed_api_calls + self.timeout_api_calls
        return self.successful_api_calls / total_calls if total_calls > 0 else 0.0
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence score."""
        return sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "protocol_type": self.protocol_type.value,
            "execution_time": self.total_duration,
            "api_calls": {
                "total": self.total_api_calls,
                "successful": self.successful_api_calls,
                "failed": self.failed_api_calls,
                "timeouts": self.timeout_api_calls,
                "success_rate": round(self.get_api_success_rate() * 100, 1)
            },
            "confidence_analysis": {
                "average": round(self.get_average_confidence(), 3),
                "distribution": self.confidence_distribution,
                "scores": self.confidence_scores
            },
            "token_savings": self.token_savings,
            "custom_metrics": self.custom_metrics
        }


class ProtocolState:
    """
    Centralized state management for protocol execution.
    Handles round tracking, metrics collection, and execution context.
    """
    
    def __init__(
        self,
        protocol_type: ProtocolType,
        max_rounds: int = 2,
        debug_mode: bool = False
    ):
        self.protocol_type = protocol_type
        self.max_rounds = max_rounds
        self.debug_mode = debug_mode
        
        # State tracking
        self.current_round: int = 0
        self.rounds: List[RoundResult] = []
        self.metrics = ProtocolMetrics(protocol_type)
        self.conversation_state = ConversationState() if protocol_type == ProtocolType.MINION else None
        
        # Execution context
        self.user_query: str = ""
        self.document_content: str = ""
        self.document_chunks: List[str] = []
        self.final_answer: str = ""
        self.execution_complete: bool = False
        
        # Debug and logging
        self.debug_log: List[str] = []
        self.conversation_log: List[str] = []
    
    def start_round(self, round_num: Optional[int] = None) -> RoundResult:
        """
        Start a new round.
        
        Args:
            round_num: Optional round number (defaults to current_round)
            
        Returns:
            New RoundResult object
        """
        if round_num is None:
            round_num = self.current_round
        
        round_result = RoundResult(
            round_num=round_num,
            status=RoundStatus.IN_PROGRESS,
            start_time=time.time()
        )
        
        # Ensure we have enough space in rounds list
        while len(self.rounds) <= round_num:
            self.rounds.append(None)
        
        self.rounds[round_num] = round_result
        self.current_round = round_num
        
        if self.debug_mode:
            self.debug_log.append(f"DEBUG [ProtocolState]: Started round {round_num + 1}")
        
        return round_result
    
    def finish_round(
        self,
        round_num: Optional[int] = None,
        status: RoundStatus = RoundStatus.COMPLETED,
        error_message: Optional[str] = None
    ) -> Optional[RoundResult]:
        """
        Finish the specified round.
        
        Args:
            round_num: Round number to finish (defaults to current_round)
            status: Final status for the round
            error_message: Optional error message
            
        Returns:
            Finished RoundResult or None if round doesn't exist
        """
        if round_num is None:
            round_num = self.current_round
        
        if round_num < len(self.rounds) and self.rounds[round_num]:
            round_result = self.rounds[round_num]
            round_result.status = status
            round_result.error_message = error_message
            round_result.finish()
            
            if self.debug_mode:
                self.debug_log.append(
                    f"DEBUG [ProtocolState]: Finished round {round_num + 1} "
                    f"({status.value}) in {round_result.duration:.2f}s"
                )
            
            return round_result
        
        return None
    
    def add_task_result(
        self,
        task_result: TaskResult,
        round_num: Optional[int] = None
    ) -> None:
        """
        Add a task result to the specified round.
        
        Args:
            task_result: TaskResult to add
            round_num: Round number (defaults to current_round)
        """
        if round_num is None:
            round_num = self.current_round
        
        if round_num < len(self.rounds) and self.rounds[round_num]:
            self.rounds[round_num].add_task_result(task_result)
            
            # Update metrics
            if task_result.status == TaskStatus.SUCCESS:
                self.metrics.add_confidence_score(task_result.confidence)
        
        # Record API call for metrics
        success = task_result.status == TaskStatus.SUCCESS
        timeout = task_result.status == TaskStatus.TIMEOUT_ALL_CHUNKS
        self.metrics.record_api_call(success, timeout)
    
    def get_current_round(self) -> Optional[RoundResult]:
        """Get the current round result."""
        if self.current_round < len(self.rounds):
            return self.rounds[self.current_round]
        return None
    
    def get_round(self, round_num: int) -> Optional[RoundResult]:
        """Get a specific round result."""
        if round_num < len(self.rounds):
            return self.rounds[round_num]
        return None
    
    def get_all_successful_tasks(self) -> List[TaskResult]:
        """Get all successful tasks across all rounds."""
        successful_tasks = []
        for round_result in self.rounds:
            if round_result:
                successful_tasks.extend(round_result.get_successful_tasks())
        return successful_tasks
    
    def should_continue_rounds(self) -> bool:
        """
        Determine if more rounds should be executed.
        
        Returns:
            True if more rounds should be executed
        """
        if self.execution_complete:
            return False
        
        if self.current_round >= self.max_rounds:
            return False
        
        # Protocol-specific continuation logic can be added here
        return True
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence across all successful tasks."""
        all_successful = self.get_all_successful_tasks()
        if not all_successful:
            return 0.0
        
        return sum(task.confidence for task in all_successful) / len(all_successful)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        self.metrics.finish()
        
        summary = {
            "protocol_type": self.protocol_type.value,
            "total_rounds": len([r for r in self.rounds if r is not None]),
            "max_rounds": self.max_rounds,
            "execution_complete": self.execution_complete,
            "overall_confidence": round(self.calculate_overall_confidence(), 3),
            "metrics": self.metrics.to_dict(),
            "rounds": [round_result.to_dict() for round_result in self.rounds if round_result]
        }
        
        # Add conversation state for Minion protocol
        if self.conversation_state:
            summary["conversation"] = self.conversation_state.get_conversation_summary()
        
        return summary
    
    def export_state(self, include_debug: bool = False) -> Dict[str, Any]:
        """
        Export complete state for debugging or persistence.
        
        Args:
            include_debug: Whether to include debug logs
            
        Returns:
            Complete state dictionary
        """
        state = {
            "protocol_type": self.protocol_type.value,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "user_query": self.user_query,
            "final_answer": self.final_answer,
            "execution_complete": self.execution_complete,
            "summary": self.get_execution_summary()
        }
        
        if include_debug:
            state["debug_log"] = self.debug_log
            state["conversation_log"] = self.conversation_log
        
        return state
    
    def reset(self) -> None:
        """Reset state for new execution."""
        self.current_round = 0
        self.rounds.clear()
        self.metrics = ProtocolMetrics(self.protocol_type)
        self.conversation_state = ConversationState() if self.protocol_type == ProtocolType.MINION else None
        self.final_answer = ""
        self.execution_complete = False
        self.debug_log.clear()
        self.conversation_log.clear()


# Utility functions for creating state objects
def create_task_result(
    task_id: str,
    content: Any,
    confidence: float,
    execution_time: float,
    status: TaskStatus = TaskStatus.SUCCESS,
    chunk_indices: Optional[List[int]] = None,
    error_message: Optional[str] = None,
    **metadata
) -> TaskResult:
    """
    Convenience function to create a TaskResult.
    
    Args:
        task_id: Unique identifier for the task
        content: Task result content
        confidence: Confidence score (0.0 to 1.0)
        execution_time: Execution time in seconds
        status: Task status
        chunk_indices: List of chunk indices processed
        error_message: Optional error message
        **metadata: Additional metadata
        
    Returns:
        TaskResult object
    """
    return TaskResult(
        task_id=task_id,
        status=status,
        content=content,
        confidence=confidence,
        execution_time=execution_time,
        chunk_indices=chunk_indices or [],
        error_message=error_message,
        metadata=metadata
    )


def create_protocol_state(
    protocol_type: str,
    max_rounds: int = 2,
    debug_mode: bool = False
) -> ProtocolState:
    """
    Convenience function to create a ProtocolState.
    
    Args:
        protocol_type: Type of protocol ("minion" or "minions")
        max_rounds: Maximum number of rounds
        debug_mode: Whether debug mode is enabled
        
    Returns:
        ProtocolState object
    """
    ptype = ProtocolType.MINION if protocol_type.lower() == "minion" else ProtocolType.MINIONS
    return ProtocolState(ptype, max_rounds, debug_mode)

# Partials File: partials/model_capabilities.py

MODEL_CAPABILITIES = {
    # OpenAI Models
    "gpt-4o": {
        "max_tokens": 128000,
        "supports_json": True,
        "supports_functions": True,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "provider": "openai"
    },
    "gpt-4-turbo": {
        "max_tokens": 128000,
        "supports_json": True,
        "supports_functions": True,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "provider": "openai"
    },
    "gpt-4": {
        "max_tokens": 8192,
        "supports_json": True,
        "supports_functions": True,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
        "provider": "openai"
    },
    
    # Anthropic Models
    "claude-3-5-sonnet-20241022": {
        "max_tokens": 200000,
        "supports_json": True,
        "supports_functions": False,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "provider": "anthropic"
    },
    "claude-3-5-haiku-20241022": {
        "max_tokens": 200000,
        "supports_json": True,
        "supports_functions": False,
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.005,
        "provider": "anthropic"
    },
    "claude-3-opus-20240229": {
        "max_tokens": 200000,
        "supports_json": True,
        "supports_functions": False,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "provider": "anthropic"
    },
}

def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """Get capabilities for a specific model"""
    if model_name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model_name].copy()
    
    # Fallback defaults for unknown models
    return {
        "max_tokens": 4096,
        "supports_json": False,
        "supports_functions": False,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "provider": "unknown"
    }

def detect_ollama_capabilities(model_name: str) -> Dict[str, Any]:
    """Detect Ollama model capabilities based on model family"""
    model_lower = model_name.lower()
    
    # Base capabilities for Ollama models
    capabilities = {
        "max_tokens": 4096,
        "supports_json": False,
        "supports_functions": False,
        "cost_per_1k_input": 0.0,  # Local models are free
        "cost_per_1k_output": 0.0,
        "provider": "ollama"
    }
    
    # Detect context length based on model family
    if "llama3.2" in model_lower:
        capabilities["max_tokens"] = 128000 if ":1b" in model_lower or ":3b" in model_lower else 4096
        capabilities["supports_json"] = True
    elif "llama3.1" in model_lower:
        capabilities["max_tokens"] = 128000
        capabilities["supports_json"] = True
    elif "llama3" in model_lower:
        capabilities["max_tokens"] = 8192
        capabilities["supports_json"] = True
    elif "qwen2.5" in model_lower:
        capabilities["max_tokens"] = 32768
        capabilities["supports_json"] = True
    elif "qwen2" in model_lower:
        capabilities["max_tokens"] = 32768
        capabilities["supports_json"] = True
    elif "gemma2" in model_lower:
        capabilities["max_tokens"] = 8192
        capabilities["supports_json"] = True
    elif "mistral" in model_lower or "mixtral" in model_lower:
        capabilities["max_tokens"] = 32768
        capabilities["supports_json"] = True
    elif "phi3" in model_lower:
        capabilities["max_tokens"] = 4096
        capabilities["supports_json"] = True
    
    return capabilities

def get_effective_model_capabilities(valves: Any) -> Dict[str, Any]:
    """Get effective capabilities for the configured models"""
    supervisor_provider = getattr(valves, 'supervisor_provider', 'anthropic')
    
    if supervisor_provider == 'openai':
        supervisor_model = getattr(valves, 'openai_model', 'gpt-4o')
    else:
        supervisor_model = getattr(valves, 'remote_model', 'claude-3-5-haiku-20241022')
    
    local_model = getattr(valves, 'local_model', 'llama3.2')
    
    supervisor_caps = get_model_capabilities(supervisor_model)
    local_caps = detect_ollama_capabilities(local_model)
    
    return {
        "supervisor": supervisor_caps,
        "local": local_caps
    }

def adjust_parameters_for_capabilities(valves: Any, capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust parameters based on model capabilities"""
    adjustments = {}
    
    supervisor_caps = capabilities["supervisor"]
    local_caps = capabilities["local"]
    
    # Adjust max tokens for supervisor based on capabilities
    current_max_tokens = getattr(valves, 'max_tokens_claude', 4096)
    if current_max_tokens > supervisor_caps["max_tokens"]:
        adjustments["max_tokens_claude"] = supervisor_caps["max_tokens"]
    
    # Adjust local model context if available
    current_local_context = getattr(valves, 'local_model_context_length', 4096)
    if current_local_context > local_caps["max_tokens"]:
        adjustments["local_model_context_length"] = local_caps["max_tokens"]
    
    # Adjust structured output usage based on support
    if hasattr(valves, 'use_structured_output') and valves.use_structured_output:
        if not local_caps["supports_json"]:
            adjustments["use_structured_output"] = False
    
    return adjustments

# Partials File: partials/minion_models.py
from enum import Enum

class LocalAssistantResponse(BaseModel):
    """
    Structured response format for the local assistant in the Minion (conversational) protocol.
    This model defines the expected output structure when the local model processes
    a request from the remote model.
    """
    answer: str = Field(description="The main response to the question posed by the remote model.")
    confidence: str = Field(description="Confidence level of the answer (e.g., HIGH, MEDIUM, LOW).")
    key_points: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of key points extracted from the context related to the answer."
    )
    citations: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of direct quotes or citations from the context supporting the answer."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields during parsing
        # Consider adding an example for documentation if this model is complex:
        # schema_extra = {
        #     "example": {
        #         "answer": "The document states that the project was completed in Q4.",
        #         "confidence": "HIGH",
        #         "key_points": ["Project completion Q4"],
        #         "citations": ["The final report confirms project completion in Q4."]
        #     }
        # }

class ConversationMetrics(BaseModel):
    """
    Metrics tracking for Minion protocol conversations.
    Captures performance data for analysis and optimization.
    """
    rounds_used: int = Field(description="Number of Q&A rounds completed in the conversation")
    questions_asked: int = Field(description="Total number of questions asked by the remote model")
    avg_answer_confidence: float = Field(
        description="Average confidence score across all local model responses (0.0-1.0)"
    )
    total_tokens_used: int = Field(
        default=0,
        description="Estimated total tokens used across all API calls"
    )
    conversation_duration_ms: float = Field(
        description="Total conversation duration in milliseconds"
    )
    completion_detected: bool = Field(
        description="Whether the conversation ended via completion detection vs max rounds"
    )
    unique_topics_explored: int = Field(
        default=0,
        description="Count of distinct topics/themes in questions (optional)"
    )
    confidence_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of confidence levels (HIGH/MEDIUM/LOW counts)"
    )
    chunks_processed: int = Field(
        default=1,
        description="Number of document chunks processed in this conversation"
    )
    chunk_size_used: int = Field(
        default=0,
        description="The chunk size setting used for this conversation"
    )
    
    class Config:
        extra = "ignore"

class ConversationState(BaseModel):
    """Tracks the evolving state of a Minion conversation"""
    qa_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    topics_covered: Set[str] = Field(default_factory=set)
    key_findings: Dict[str, str] = Field(default_factory=dict)
    information_gaps: List[str] = Field(default_factory=list)
    current_phase: str = Field(default="exploration")
    knowledge_graph: Dict[str, List[str]] = Field(default_factory=dict)  # topic -> related facts
    phase_transitions: List[Dict[str, str]] = Field(default_factory=list)  # Track phase transitions
    
    def add_qa_pair(self, question: str, answer: str, confidence: str, key_points: List[str] = None):
        """Add a Q&A pair and update derived state"""
        self.qa_pairs.append({
            "round": len(self.qa_pairs) + 1,
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "key_points": key_points or [],
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        })
        
    def get_state_summary(self) -> str:
        """Generate a summary of current conversation state for the remote model"""
        summary = f"Conversation Phase: {self.current_phase}\n"
        summary += f"Questions Asked: {len(self.qa_pairs)}\n"
        
        if self.key_findings:
            summary += "\nKey Findings:\n"
            for topic, finding in self.key_findings.items():
                summary += f"- {topic}: {finding}\n"
                
        if self.information_gaps:
            summary += "\nInformation Gaps:\n"
            for gap in self.information_gaps:
                summary += f"- {gap}\n"
                
        return summary

class QuestionDeduplicator:
    """Handles detection and prevention of duplicate questions in conversations"""
    def __init__(self, similarity_threshold: float = 0.8):
        self.asked_questions: List[str] = []
        self.question_embeddings: List[str] = []  # Simplified: store normalized forms
        self.similarity_threshold = similarity_threshold
        
    def is_duplicate(self, question: str) -> Tuple[bool, Optional[str]]:
        """Check if question is semantically similar to a previous question"""
        # Normalize the question
        normalized = self._normalize_question(question)
        
        # Check for semantic similarity (simplified approach)
        for idx, prev_normalized in enumerate(self.question_embeddings):
            # Safety check to prevent IndexError
            if idx < len(self.asked_questions):
                if self._calculate_similarity(normalized, prev_normalized) > self.similarity_threshold:
                    return True, self.asked_questions[idx]
                
        return False, None
        
    def add_question(self, question: str):
        """Add question to the deduplication store"""
        self.asked_questions.append(question)
        self.question_embeddings.append(self._normalize_question(question))
        
    def _normalize_question(self, question: str) -> str:
        """Simple normalization - in production, use embeddings"""
        # Remove common words, lowercase, extract key terms
        stop_words = {"what", "is", "the", "are", "how", "does", "can", "you", "tell", "me", "about", 
                      "please", "could", "would", "explain", "describe", "provide", "give", "any",
                      "there", "which", "when", "where", "who", "why", "do", "have", "has", "been",
                      "was", "were", "will", "be", "being", "a", "an", "and", "or", "but", "in",
                      "on", "at", "to", "for", "of", "with", "by", "from", "as", "this", "that"}
        
        # Extract words and filter
        words = question.lower().replace("?", "").replace(".", "").replace(",", "").split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(sorted(key_words))
        
    def _calculate_similarity(self, q1: str, q2: str) -> float:
        """Calculate similarity between normalized questions using Jaccard similarity"""
        words1 = set(q1.split())
        words2 = set(q2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
        
    def get_all_questions(self) -> List[str]:
        """Return all previously asked questions"""
        return self.asked_questions.copy()

class ConversationPhase(str, Enum):
    """Phases of a structured conversation"""
    EXPLORATION = "exploration"
    DEEP_DIVE = "deep_dive"
    GAP_FILLING = "gap_filling"
    SYNTHESIS = "synthesis"

class ConversationFlowController:
    """Controls the flow of conversation through different phases"""
    def __init__(self):
        self.current_phase = ConversationPhase.EXPLORATION
        self.phase_question_count = {phase: 0 for phase in ConversationPhase}
        self.phase_transitions = {
            ConversationPhase.EXPLORATION: ConversationPhase.DEEP_DIVE,
            ConversationPhase.DEEP_DIVE: ConversationPhase.GAP_FILLING,
            ConversationPhase.GAP_FILLING: ConversationPhase.SYNTHESIS,
            ConversationPhase.SYNTHESIS: ConversationPhase.SYNTHESIS
        }
        
    def should_transition(self, state: ConversationState, valves: Any = None) -> bool:
        """Determine if conversation should move to next phase"""
        current_count = self.phase_question_count[self.current_phase]
        
        if self.current_phase == ConversationPhase.EXPLORATION:
            # Move on after configured questions or when main topics identified
            max_questions = valves.max_exploration_questions if valves else 3
            return current_count >= max_questions or (current_count >= 2 and len(state.topics_covered) >= 3)
            
        elif self.current_phase == ConversationPhase.DEEP_DIVE:
            # Move on after exploring key topics in detail
            max_questions = valves.max_deep_dive_questions if valves else 4
            return current_count >= max_questions or len(state.key_findings) >= 5
            
        elif self.current_phase == ConversationPhase.GAP_FILLING:
            # Move to synthesis when gaps are addressed
            max_questions = valves.max_gap_filling_questions if valves else 2
            return current_count >= max_questions or len(state.information_gaps) == 0
            
        return False
        
    def get_phase_guidance(self) -> str:
        """Get prompting guidance for current phase"""
        guidance = {
            ConversationPhase.EXPLORATION: 
                "You are in the EXPLORATION phase. Ask broad questions to understand the document's main topics and structure.",
            ConversationPhase.DEEP_DIVE:
                "You are in the DEEP DIVE phase. Focus on specific topics that are most relevant to the user's query.",
            ConversationPhase.GAP_FILLING:
                "You are in the GAP FILLING phase. Address specific information gaps identified in previous rounds.",
            ConversationPhase.SYNTHESIS:
                "You are in the SYNTHESIS phase. You should now have enough information. Prepare your final answer."
        }
        return guidance.get(self.current_phase, "")
        
    def transition_to_next_phase(self):
        """Move to the next conversation phase"""
        self.current_phase = self.phase_transitions[self.current_phase]
        self.phase_question_count[self.current_phase] = 0
        
    def increment_question_count(self):
        """Increment the question count for the current phase"""
        self.phase_question_count[self.current_phase] += 1
        
    def get_phase_status(self) -> Dict[str, Any]:
        """Get current phase status for debugging"""
        return {
            "current_phase": self.current_phase.value,
            "questions_in_phase": self.phase_question_count[self.current_phase],
            "total_questions_by_phase": {
                phase.value: count for phase, count in self.phase_question_count.items()
            }
        }

class AnswerValidator:
    """Validates answer quality and generates clarification requests"""
    
    @staticmethod
    def validate_answer(answer: str, confidence: str, question: str) -> Dict[str, Any]:
        """Validate answer quality and completeness"""
        issues = []
        
        # Check for non-answers
        non_answer_phrases = [
            "i don't know",
            "not sure",
            "unclear",
            "cannot determine",
            "no information",
            "not available",
            "not found",
            "not mentioned",
            "not specified"
        ]
        
        answer_lower = answer.lower()
        
        # Check if answer is too vague for low confidence
        if len(answer.split()) < 10 and confidence == "LOW":
            issues.append("Answer seems too brief for a low-confidence response")
            
        # Check for non-answers without clear indication
        non_answer_found = any(phrase in answer_lower for phrase in non_answer_phrases)
        if non_answer_found and "not found" not in answer_lower and "not available" not in answer_lower:
            issues.append("Answer indicates uncertainty without clearly stating if information is missing")
            
        # Check if answer addresses the question keywords
        question_keywords = set(question.lower().split()) - {
            "what", "how", "when", "where", "who", "why", "is", "are", "the", "does", "can", "could", "would", "should"
        }
        answer_keywords = set(answer_lower.split())
        
        if question_keywords:
            keyword_overlap = len(question_keywords.intersection(answer_keywords)) / len(question_keywords)
            if keyword_overlap < 0.2:
                issues.append("Answer may not directly address the question asked")
        
        # Check for extremely short answers to complex questions
        if len(question.split()) > 10 and len(answer.split()) < 5:
            issues.append("Answer seems too brief for a complex question")
            
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "needs_clarification": len(issues) > 0 and confidence != "HIGH",
            "severity": "high" if len(issues) >= 2 else "low" if len(issues) == 1 else "none"
        }
        
    @staticmethod
    def generate_clarification_request(validation_result: Dict[str, Any], original_question: str, answer: str) -> str:
        """Generate a clarification request based on validation issues"""
        if not validation_result["issues"]:
            return ""
            
        clarification = "I need clarification on your previous answer. "
        
        for issue in validation_result["issues"]:
            if "too brief" in issue:
                clarification += "Could you provide more detail or explanation? "
            elif "uncertainty" in issue:
                clarification += "Is this information not available in the document, or is it unclear? Please be explicit about what information is missing. "
            elif "not directly address" in issue:
                clarification += f"Let me rephrase the question: {original_question} "
            elif "complex question" in issue:
                clarification += "This seems like a complex topic that might need a more comprehensive answer. "
                
        # Add specific guidance based on the answer content
        if len(answer.split()) < 5:
            clarification += "If the information isn't in the document, please say so explicitly. If it is available, please provide the specific details."
        else:
            clarification += "Please provide more specific information from the document to fully address my question."
                
        return clarification.strip()


# Partials File: partials/minion_valves.py

class MinionValves(BaseModel):
    """
    Configuration settings (valves) specifically for the Minion (conversational) pipe.
    These settings control the behavior of the Minion protocol, including API keys,
    model selections, timeouts, operational parameters, extraction instructions,
    expected output format, and confidence threshold.
    """
    # Essential configuration only
    supervisor_provider: str = Field(
        default="anthropic", 
        description="Provider for supervisor model: 'anthropic' or 'openai'"
    )
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for the remote model (e.g., Claude)"
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key"
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model identifier (e.g., for Anthropic: claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality; for OpenAI: gpt-4o, gpt-4-turbo, gpt-4)",
    )
    openai_model: str = Field(
        default="gpt-4o", 
        description="OpenAI model to use when supervisor_provider is 'openai'"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    local_model: str = Field(
        default="llama3.2", description="Local Ollama model name"
    )
    max_rounds: int = Field(
        default=2, 
        description="Maximum conversation rounds between remote and local models."
    )
    show_conversation: bool = Field(
        default=True,
        description="Show full conversation between local and remote models in the output.",
    )
    timeout_local: int = Field(
        default=60, 
        description="Timeout for local model calls in seconds. Local model processes full context."
    )
    timeout_claude: int = Field(
        default=60, description="Timeout for remote model API calls in seconds."
    )
    max_tokens_claude: int = Field(
        default=4000, description="Maximum tokens for remote model's responses."
    )
    ollama_num_predict: int = Field(
        default=1000, 
        description="num_predict for Ollama generation (max output tokens for local model)."
    )
    chunk_size: int = Field(
        default=5000, 
        description="Maximum chunk size in characters for context fed to local models during conversation."
    )
    max_chunks: int = Field(
        default=2, 
        description="Maximum number of document chunks to process. Helps manage processing load for large documents."
    )
    
    # Custom local model parameters
    local_model_context_length: int = Field(
        default=4096,
        description="Context window size for the local model. Set this based on your local model's capabilities."
    )
    local_model_temperature: float = Field(
        default=0.7,
        description="Temperature for local model generation (0.0-2.0). Lower values make output more focused and deterministic.",
        ge=0.0,
        le=2.0
    )
    local_model_top_k: int = Field(
        default=40,
        description="Top-k sampling for local model. Limits vocabulary to top k tokens. Set to 0 to disable.",
        ge=0
    )
    chunk_size: int = Field(
        default=5000, 
        description="Maximum chunk size in characters for context fed to local models during conversation."
    )
    max_chunks: int = Field(
        default=2, 
        description="Maximum number of document chunks to process. Helps manage processing load for large documents."
    )
    use_structured_output: bool = Field(
        default=True, 
        description="Enable JSON structured output for local model responses (requires local model support)."
    )
    enable_completion_detection: bool = Field(
        default=True,
        description="Enable detection of when the remote model has gathered sufficient information without explicit 'FINAL ANSWER READY' marker."
    )
    enable_completion_detection: bool = Field(
        default=True,
        description="Enable detection of when the remote model has gathered sufficient information without explicit 'FINAL ANSWER READY' marker."
    )
    debug_mode: bool = Field(
        default=False, description="Show additional technical details and verbose logs."
    )
    extraction_instructions: str = Field(
        default="", title="Extraction Instructions", description="Specific instructions for the LLM on what to extract or how to process the information."
    )
    expected_format: str = Field(
        default="text", title="Expected Output Format", description="Desired format for the LLM's output (e.g., 'text', 'JSON', 'bullet points')."
    )
    confidence_threshold: float = Field(
        default=0.7, title="Confidence Threshold", description="Minimum confidence level for the LLM's response (0.0-1.0). Primarily a suggestion to the LLM.", ge=0, le=1
    )
    
    # Conversation State Tracking (v0.3.6b)
    track_conversation_state: bool = Field(
        default=True,
        description="Enable comprehensive conversation state tracking for better context awareness"
    )
    
    # Question Deduplication (v0.3.6b)
    enable_deduplication: bool = Field(
        default=True,
        description="Prevent duplicate questions by detecting semantic similarity"
    )
    deduplication_threshold: float = Field(
        default=0.8,
        description="Similarity threshold for question deduplication (0-1). Higher = stricter matching",
        ge=0.0,
        le=1.0
    )
    
    # Conversation Flow Control (v0.3.6b)
    enable_flow_control: bool = Field(
        default=True,
        description="Enable phased conversation flow (exploration → deep dive → gap filling → synthesis)"
    )
    max_exploration_questions: int = Field(
        default=3,
        description="Maximum questions in exploration phase (broad understanding)",
        ge=1,
        le=10
    )
    max_deep_dive_questions: int = Field(
        default=4,
        description="Maximum questions in deep dive phase (specific topics)",
        ge=1,
        le=10
    )
    max_gap_filling_questions: int = Field(
        default=2,
        description="Maximum questions in gap filling phase (missing information)",
        ge=1,
        le=10
    )
    
    # Answer Validation (v0.3.6b)
    enable_answer_validation: bool = Field(
        default=True,
        description="Enable answer quality validation and clarification requests"
    )
    max_clarification_attempts: int = Field(
        default=1,
        description="Maximum clarification requests per question",
        ge=0,
        le=3
    )

    # The following class is part of the Pydantic configuration and is standard.
    # It ensures that extra fields passed to the model are ignored rather than causing an error.
    class Config:
        extra = "ignore"


# Partials File: partials/common_api_calls.py

# Known models that support JSON/structured output
STRUCTURED_OUTPUT_CAPABLE_MODELS: Set[str] = {
    "llama3.2", "llama3.1", "llama3", "llama2",
    "mistral", "mixtral", "mistral-nemo",
    "qwen2", "qwen2.5", 
    "gemma2", "gemma",
    "phi3", "phi",
    "command-r", "command-r-plus",
    "deepseek-coder", "deepseek-coder-v2",
    "codellama",
    "dolphin-llama3", "dolphin-mixtral",
    "solar", "starling-lm",
    "yi", "zephyr",
    "neural-chat", "openchat"
}

def model_supports_structured_output(model_name: str) -> bool:
    """Check if a model is known to support structured output"""
    if not model_name:
        return False
    
    # Normalize model name for comparison
    model_lower = model_name.lower()
    
    # Check exact matches first
    if model_lower in STRUCTURED_OUTPUT_CAPABLE_MODELS:
        return True
    
    # Check partial matches (for versioned models like llama3.2:1b)
    for known_model in STRUCTURED_OUTPUT_CAPABLE_MODELS:
        if model_lower.startswith(known_model):
            return True
    
    return False

async def call_openai_api(
    prompt: str, 
    api_key: str, 
    model: str = "gpt-4o", 
    temperature: float = 0.1, 
    max_tokens: int = 4096,
    timeout: int = 60
) -> str:
    """Call OpenAI's API with error handling and retry logic"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"OpenAI API error: {response.status} - {error_text}"
                )
            result = await response.json()
            
            if result.get("choices") and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("Unexpected response format from OpenAI API or empty content.")

async def call_claude(
    valves: BaseModel,  # Or a more specific type if Valves is shareable
    prompt: str
) -> str:
    """Call Anthropic Claude API"""
    headers = {
        "x-api-key": valves.anthropic_api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": valves.remote_model,
        "max_tokens": valves.max_tokens_claude,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=valves.timeout_claude
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Anthropic API error: {response.status} - {error_text}"
                )
            result = await response.json()
            if result.get("content") and isinstance(result["content"], list) and len(result["content"]) > 0 and result["content"][0].get("text"):
                return result["content"][0]["text"]
            else:
                # Consider logging instead of print for shared code
                if hasattr(valves, 'debug_mode') and valves.debug_mode:
                    print(f"Unexpected Claude API response format: {result}") 
                raise Exception("Unexpected response format from Anthropic API or empty content.")

async def call_ollama(
    valves: BaseModel,  # Or a more specific type
    prompt: str,
    use_json: bool = False,
    schema: Optional[BaseModel] = None
) -> str:
    """Call Ollama API"""
    options = {
        "temperature": getattr(valves, 'local_model_temperature', 0.7),
        "num_predict": valves.ollama_num_predict,
        "num_ctx": getattr(valves, 'local_model_context_length', 4096),
    }
    
    # Add top_k only if it's greater than 0
    top_k = getattr(valves, 'local_model_top_k', 40)
    if top_k > 0:
        options["top_k"] = top_k
    
    payload = {
        "model": valves.local_model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    # Check if we should use structured output
    should_use_structured = (
        use_json and 
        hasattr(valves, 'use_structured_output') and 
        valves.use_structured_output and 
        schema and
        model_supports_structured_output(valves.local_model)
    )
    
    if should_use_structured:
        payload["format"] = "json"
        # Pydantic v1 used schema.schema_json(), v2 uses schema_json = model_json_schema(MyModel) then json.dumps(schema_json)
        # Assuming schema object has a .schema_json() method for simplicity here, may need adjustment
        try:
            schema_for_prompt = schema.schema_json() # For Pydantic v1
        except AttributeError: # Basic fallback for Pydantic v2 or other schema objects
             # This part might need refinement based on actual schema object type if not Pydantic v1 BaseModel
            if hasattr(schema, 'model_json_schema'): # Pydantic v2
                 schema_for_prompt = json.dumps(schema.model_json_schema())
            elif inspect.isclass(schema) and issubclass(schema, BaseModel): # Pydantic v1/v2 class
                 schema_for_prompt = json.dumps(schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema.schema())
            else: # Fallback, might not be perfect
                 schema_for_prompt = "{}" 
                 if hasattr(valves, 'debug_mode') and valves.debug_mode:
                      print("Warning: Could not automatically generate schema_json for prompt.")

        schema_prompt_addition = f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_for_prompt}"
        payload["prompt"] = prompt + schema_prompt_addition
    elif use_json and hasattr(valves, 'use_structured_output') and valves.use_structured_output:
        # Model doesn't support structured output but it was requested
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            print(f"DEBUG: Model '{valves.local_model}' does not support structured output. Using text-based parsing fallback.")
    
    if "format" in payload and not should_use_structured:
        del payload["format"]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{valves.ollama_base_url}/api/generate", json=payload, timeout=valves.timeout_local
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Ollama API error: {response.status} - {error_text}"
                )
            result = await response.json()
            if "response" in result:
                return result["response"].strip()
            else:
                if hasattr(valves, 'debug_mode') and valves.debug_mode:
                    print(f"Unexpected Ollama API response format: {result}")
                raise Exception("Unexpected response format from Ollama API or no response field.")

async def call_supervisor_model(valves: BaseModel, prompt: str) -> str:
    """Call the configured supervisor model (Claude or OpenAI)"""
    provider = getattr(valves, 'supervisor_provider', 'anthropic')
    
    if provider == 'openai':
        api_key = valves.openai_api_key
        model = getattr(valves, 'openai_model', 'gpt-4o')
        max_tokens = getattr(valves, 'max_tokens_claude', 4096)
        timeout = getattr(valves, 'timeout_claude', 60)
        
        if not api_key:
            raise Exception("OpenAI API key is required when using OpenAI as supervisor provider")
        
        return await call_openai_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    elif provider == 'anthropic':
        if not valves.anthropic_api_key:
            raise Exception("Anthropic API key is required when using Anthropic as supervisor provider")
        
        return await call_claude(valves, prompt)
    
    else:
        raise Exception(f"Unsupported supervisor provider: {provider}. Use 'anthropic' or 'openai'")


# Partials File: partials/common_context_utils.py

def extract_context_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract context from conversation history"""
    context_parts = []

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Assume messages longer than 200 chars contain context/documents
            if len(content) > 200:
                context_parts.append(content)

    return "\n\n".join(context_parts)

async def extract_file_content(valves, file_info: Dict[str, Any]) -> str:
    """Extract text content from a single file using Open WebUI's file API"""
    try:
        file_id = file_info.get("id")
        file_name = file_info.get("name", "unknown")

        if not file_id:
            return f"[Could not get file ID for {file_name}]"

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[DEBUG] File ID: {file_id}, Name: {file_name}, Info: {str(file_info)}]"

        # If the file info contains content directly, use it
        if "content" in file_info:
            return file_info["content"]
        
        file_type = file_info.get("type", "unknown")
        file_size = file_info.get("size", "unknown")
        
        return f"[File detected: {file_name} (Type: {file_type}, Size: {file_size})\nNote: File content extraction needs to be configured or content is not directly available in provided file_info]"

    except Exception as e:
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[Error extracting file content: {str(e)}]"
        return f"[Error extracting file content]"

async def extract_context_from_files(valves, files: List[Dict[str, Any]]) -> str:
    """Extract text content from uploaded files using Open WebUI's file system"""
    try:
        if not files:
            return ""

        files_content = []

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            files_content.append(f"[DEBUG] Found {len(files)} uploaded files")

        for file_info in files:
            if isinstance(file_info, dict):
                content = await extract_file_content(valves, file_info)
                if content:
                    if content.startswith("[") and content.endswith("]"):
                        if hasattr(valves, 'debug_mode') and valves.debug_mode:
                            files_content.append(content)
                    else:
                        file_name = file_info.get("name", "unknown_file")
                        files_content.append(f"=== FILE: {file_name} ===\n{content}")
                        
        return "\n\n".join(files_content) if files_content else ""

    except Exception as e:
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[File extraction error: {str(e)}]"
        return ""

# Partials File: partials/common_file_processing.py

def create_chunks(context: str, chunk_size: int, max_chunks: int) -> List[str]:
    """Create chunks from context"""
    if not context:
        return []
    actual_chunk_size = max(1, min(chunk_size, len(context)))
    chunks = [
        context[i : i + actual_chunk_size]
        for i in range(0, len(context), actual_chunk_size)
    ]
    return chunks[:max_chunks] if max_chunks > 0 else chunks


# Partials File: partials/minion_prompts.py

# This file will store prompt generation functions for the Minion (single-turn) protocol.

def get_minion_initial_claude_prompt(query: str, context_len: int, valves: Any) -> str:
    """
    Returns the initial prompt for Claude in the Minion protocol.
    Enhanced with better question generation guidance.
    """
    # Escape any quotes in the query to prevent f-string issues
    escaped_query = query.replace('"', '\\"').replace("'", "\\'")
    
    return f'''You are a research coordinator working with a knowledgeable local assistant who has access to specific documents.

Your task: Gather information to answer the user's query by asking strategic questions.

USER'S QUERY: "{escaped_query}"

The local assistant has FULL ACCESS to the relevant document ({context_len} characters long) and will provide factual information extracted from it.

Guidelines for effective questions:
1. Ask ONE specific, focused question at a time
2. Build upon previous answers to go deeper
3. Avoid broad questions like "What does the document say?" 
4. Good: "What are the specific budget allocations for Q2?"
   Poor: "Tell me about the budget"
5. Track what you've learned to avoid redundancy

When to conclude:
- Start your response with "I now have sufficient information" when ready to provide the final answer
- You have {valves.max_rounds} rounds maximum to gather information

QUESTION STRATEGY TIPS:
- For factual queries: Ask for specific data points, dates, numbers, or names
- For analytical queries: Ask about relationships, comparisons, or patterns
- For summary queries: Ask about key themes, main points, or conclusions
- For procedural queries: Ask about steps, sequences, or requirements

Remember: The assistant can only see the document, not your conversation history.

If you have gathered enough information to answer "{escaped_query}", respond with "FINAL ANSWER READY." followed by your comprehensive answer.

Otherwise, ask your first strategic question to the local assistant.'''

def get_minion_conversation_claude_prompt(history: List[Tuple[str, str]], original_query: str, valves: Any) -> str:
    """
    Returns the prompt for Claude during subsequent conversation rounds in the Minion protocol.
    Enhanced with better guidance for follow-up questions.
    """
    # Escape the original query
    escaped_query = original_query.replace('"', '\\"').replace("'", "\\'")
    
    current_round = len(history) // 2 + 1
    rounds_remaining = valves.max_rounds - current_round
    
    context_parts = [
        f'You are continuing to gather information to answer: "{escaped_query}"',
        f"Round {current_round} of {valves.max_rounds}",
        "",
        "INFORMATION GATHERED SO FAR:",
    ]

    for i, (role, message) in enumerate(history):
        if role == "assistant":  # Claude's previous message
            context_parts.append(f'\nQ{i//2 + 1}: {message}')
        else:  # Local model's response
            # Extract key information if structured
            if isinstance(message, str) and message.startswith('{'):
                context_parts.append(f'A{i//2 + 1}: {message}')
            else:
                context_parts.append(f'A{i//2 + 1}: {message}')

    context_parts.extend(
        [
            "",
            "DECISION POINT:",
            "Evaluate if you have sufficient information to answer the original question comprehensively.",
            "",
            "✅ If YES: Start with 'FINAL ANSWER READY.' then provide your complete answer",
            f"❓ If NO: Ask ONE more strategic question (you have {rounds_remaining} rounds left)",
            "",
            "TIPS FOR YOUR NEXT QUESTION:",
            "- What specific gaps remain in your understanding?",
            "- Can you drill deeper into any mentioned topics?",
            "- Are there related aspects you haven't explored?",
            "- Would examples or specific details strengthen your answer?",
            "",
            "Remember: Each question should build on what you've learned, not repeat previous inquiries.",
        ]
    )
    return "\n".join(context_parts)

def get_minion_local_prompt(context: str, query: str, claude_request: str, valves: Any) -> str:
    """
    Returns the prompt for the local Ollama model in the Minion protocol.
    Enhanced with better guidance for structured, useful responses.
    """
    # query is the original user query.
    # context is the document chunk.
    # claude_request (the parameter) is the specific question from the remote model to the local model.

    base_prompt = f"""You are a document analysis assistant with exclusive access to the following document:

<document>
{context}
</document>

A research coordinator needs specific information from this document to answer: "{query}"

Their current question is:
<question>
{claude_request}
</question>

RESPONSE GUIDELINES:

1. ACCURACY: Base your answer ONLY on information found in the document above
   
2. CITATIONS: When possible, include direct quotes or specific references:
   - Good: "According to section 3.2, 'the budget increased by 15%'"
   - Good: "The document states on page 4 that..."
   - Poor: "The document mentions something about budgets"

3. ORGANIZATION: For complex answers, structure your response:
   - Use bullet points or numbered lists for multiple items
   - Separate distinct pieces of information clearly
   - Highlight key findings at the beginning

4. CONFIDENCE LEVELS:
   - HIGH: Information directly answers the question with explicit statements
   - MEDIUM: Information partially addresses the question or requires some inference
   - LOW: Information is tangentially related or requires significant interpretation

5. HANDLING MISSING INFORMATION:
   - If not found: "This specific information is not available in the document"
   - If partially found: "The document provides partial information: [explain what's available]"
   - Suggest related info: "While X is not mentioned, the document does discuss Y which may be relevant"

Remember: The coordinator cannot see the document and relies entirely on your accurate extraction."""

    if valves.use_structured_output:
        structured_output_instructions = """

RESPONSE FORMAT:
Respond ONLY with a JSON object in this exact format:
{
    "answer": "Your detailed answer addressing the specific question",
    "confidence": "HIGH/MEDIUM/LOW",
    "key_points": ["Main finding 1", "Main finding 2", "..."] or null,
    "citations": ["Exact quote from document", "Another relevant quote", "..."] or null
}

JSON Guidelines:
- answer: Comprehensive response to the question (required)
- confidence: Your assessment based on criteria above (required)
- key_points: List main findings if multiple important points exist (optional)
- citations: Direct quotes that support your answer (optional but recommended)

IMPORTANT: Output ONLY the JSON object. No additional text, no markdown formatting."""
        return base_prompt + structured_output_instructions
    else:
        non_structured_instructions = """

Format your response clearly with:
- Main answer first
- Supporting details or quotes
- Confidence level (HIGH/MEDIUM/LOW) at the end
- Note if any information is not found in the document"""
        return base_prompt + non_structured_instructions

def get_minion_initial_claude_prompt_with_state(query: str, context_len: int, valves: Any, conversation_state: Optional[Any] = None, phase_guidance: Optional[str] = None) -> str:
    """
    Enhanced version of initial prompt that includes conversation state if available.
    """
    base_prompt = get_minion_initial_claude_prompt(query, context_len, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        if state_summary:
            base_prompt = base_prompt.replace(
                "Otherwise, ask your first strategic question to the local assistant.",
                f"""
CONVERSATION STATE CONTEXT:
{state_summary}

Based on this context, ask your first strategic question to the local assistant."""
            )
    
    # Add phase guidance if provided
    if phase_guidance and valves.enable_flow_control:
        base_prompt = base_prompt.replace(
            "Guidelines for effective questions:",
            f"""CURRENT PHASE:
{phase_guidance}

Guidelines for effective questions:"""
        )
    
    return base_prompt

def get_minion_conversation_claude_prompt_with_state(history: List[Tuple[str, str]], original_query: str, valves: Any, conversation_state: Optional[Any] = None, previous_questions: Optional[List[str]] = None, phase_guidance: Optional[str] = None) -> str:
    """
    Enhanced version of conversation prompt that includes conversation state if available.
    """
    base_prompt = get_minion_conversation_claude_prompt(history, original_query, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        
        # Insert state summary before decision point
        state_section = f"""
CURRENT CONVERSATION STATE:
{state_summary}

TOPICS COVERED: {', '.join(conversation_state.topics_covered) if conversation_state.topics_covered else 'None yet'}
KEY FINDINGS COUNT: {len(conversation_state.key_findings)}
INFORMATION GAPS: {len(conversation_state.information_gaps)}
"""
        
        base_prompt = base_prompt.replace(
            "DECISION POINT:",
            state_section + "\nDECISION POINT:"
        )
    
    # Add deduplication guidance if previous questions provided
    if previous_questions and valves.enable_deduplication:
        questions_section = "\nPREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT):\n"
        for i, q in enumerate(previous_questions, 1):
            questions_section += f"{i}. {q}\n"
        
        questions_section += "\n⚠️ IMPORTANT: Avoid asking questions that are semantically similar to the above. Each new question should explore genuinely new information.\n"
        
        base_prompt = base_prompt.replace(
            "Remember: Each question should build on what you've learned, not repeat previous inquiries.",
            questions_section + "Remember: Each question should build on what you've learned, not repeat previous inquiries."
        )
    
    # Add phase guidance if provided
    if phase_guidance and valves.enable_flow_control:
        base_prompt = base_prompt.replace(
            "DECISION POINT:",
            f"""CURRENT CONVERSATION PHASE:
{phase_guidance}

DECISION POINT:"""
        )
    
    return base_prompt

# Partials File: partials/minion_protocol_logic.py

def _calculate_token_savings(conversation_history: List[Tuple[str, str]], context: str, query: str) -> dict:
    """Calculate token savings for the Minion protocol"""
    chars_per_token = 3.5
    
    # Traditional approach: entire context + query sent to remote model
    traditional_tokens = int((len(context) + len(query)) / chars_per_token)
    
    # Minion approach: only conversation messages sent to remote model
    conversation_content = " ".join(
        [msg[1] for msg in conversation_history if msg[0] == "assistant"]
    )
    minion_tokens = int(len(conversation_content) / chars_per_token)
    
    # Calculate savings
    token_savings = traditional_tokens - minion_tokens
    percentage_savings = (
        (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    )
    
    return {
        'traditional_tokens': traditional_tokens,
        'minion_tokens': minion_tokens,
        'token_savings': token_savings,
        'percentage_savings': percentage_savings
    }

def _is_final_answer(response: str) -> bool:
    """Check if response contains the specific final answer marker."""
    return "FINAL ANSWER READY." in response or "FINAL ANSWER READY:" in response

def detect_completion(response: str) -> bool:
    """Check if remote model indicates it has sufficient information"""
    completion_phrases = [
        "i now have sufficient information",
        "i can now answer",
        "based on the information gathered",
        "i have enough information",
        "with this information, i can provide",
        "i can now provide a comprehensive answer",
        "based on what the local assistant has told me"
    ]
    response_lower = response.lower()
    
    # Check for explicit final answer marker first
    if "FINAL ANSWER READY." in response:
        return True
    
    # Check for completion phrases
    return any(phrase in response_lower for phrase in completion_phrases)

def _parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, LocalAssistantResponseModel: Any) -> Dict:
    """Parse local model response, supporting both text and structured formats."""
    confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
    default_numeric_confidence = 0.3  # Corresponds to LOW
    
    if is_structured and use_structured_output:
        # Clean up common formatting issues
        cleaned_response = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[7:-3].strip()
        elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
        
        # Try to extract JSON from response with explanatory text
        if not cleaned_response.startswith("{"):
            # Look for JSON object in the response
            json_start = cleaned_response.find("{")
            json_end = cleaned_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_response = cleaned_response[json_start:json_end+1]
        
        try:
            parsed_json = json.loads(cleaned_response)
            
            # Handle missing confidence field with default
            if 'confidence' not in parsed_json:
                parsed_json['confidence'] = 'LOW'
            
            # Ensure required fields have defaults if missing
            if 'answer' not in parsed_json:
                parsed_json['answer'] = None
            if 'key_points' not in parsed_json:
                parsed_json['key_points'] = None
            if 'citations' not in parsed_json:
                parsed_json['citations'] = None
            
            validated_model = LocalAssistantResponseModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            
            # Add numeric confidence for consistency
            text_confidence = model_dict.get('confidence', 'LOW').upper()
            model_dict['numeric_confidence'] = confidence_map.get(text_confidence, default_numeric_confidence)
            
            return model_dict
            
        except json.JSONDecodeError as e:
            if debug_mode:
                print(f"DEBUG: JSON decode error in Minion: {e}. Cleaned response was: {cleaned_response[:500]}")
            
            # Try regex fallback to extract key information
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
            confidence_match = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', response, re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1)
                confidence = confidence_match.group(1).upper() if confidence_match else "LOW"
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "numeric_confidence": confidence_map.get(confidence, default_numeric_confidence),
                    "key_points": None,
                    "citations": None,
                    "parse_error": f"JSON parse error (recovered): {str(e)}"
                }
            
            # Complete fallback
            return {
                "answer": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence,
                "key_points": None,
                "citations": None,
                "parse_error": str(e)
            }
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in Minion: {e}. Response was: {response[:500]}")
            return {
                "answer": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence,
                "key_points": None, 
                "citations": None, 
                "parse_error": str(e)
            }
    
    # Fallback for non-structured processing
    return {
        "answer": response, 
        "confidence": "MEDIUM", 
        "numeric_confidence": confidence_map.get("MEDIUM", 0.6),
        "key_points": None, 
        "citations": None, 
        "parse_error": None
    }

async def _execute_minion_protocol(
    valves: Any,
    query: str,
    context: str,
    call_ollama_func: Callable,
    LocalAssistantResponseModel: Any,
    ConversationStateModel: Any = None,
    QuestionDeduplicatorModel: Any = None,
    ConversationFlowControllerModel: Any = None,
    AnswerValidatorModel: Any = None
) -> str:
    """Execute the Minion protocol"""
    conversation_log = []
    debug_log = []
    conversation_history = []
    actual_final_answer = "No final answer was explicitly provided by the remote model."
    claude_declared_final = False
    
    # Initialize conversation state if enabled
    conversation_state = None
    if valves.track_conversation_state and ConversationStateModel:
        conversation_state = ConversationStateModel()
    
    # Initialize question deduplicator if enabled
    deduplicator = None
    if valves.enable_deduplication and QuestionDeduplicatorModel:
        deduplicator = QuestionDeduplicatorModel(
            similarity_threshold=valves.deduplication_threshold
        )
    
    # Initialize flow controller if enabled
    flow_controller = None
    if valves.enable_flow_control and ConversationFlowControllerModel:
        flow_controller = ConversationFlowControllerModel()
    
    # Initialize answer validator if enabled
    validator = None
    if valves.enable_answer_validation and AnswerValidatorModel:
        validator = AnswerValidatorModel()
    
    # Track clarification attempts per question
    clarification_attempts = {}
    
    # Initialize metrics tracking
    overall_start_time = asyncio.get_event_loop().time()
    metrics = {
        'confidence_scores': [],
        'confidence_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
        'rounds_completed': 0,
        'completion_via_detection': False,
        'estimated_tokens': 0,
        'chunk_size_used': valves.chunk_size,
        'context_size': len(context)
    }

    if valves.debug_mode:
        debug_log.append(f"🔍 **Debug Info (Minion v0.3.6b):**")
        debug_log.append(f"  - Query: {query[:100]}...")
        debug_log.append(f"  - Context length: {len(context)} chars")
        debug_log.append(f"  - Max rounds: {valves.max_rounds}")
        debug_log.append(f"  - Remote model: {valves.remote_model}")
        debug_log.append(f"  - Local model: {valves.local_model}")
        debug_log.append(f"  - Timeouts: Remote={valves.timeout_claude}s, Local={valves.timeout_local}s")
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**\n")

    for round_num in range(valves.max_rounds):
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {round_num + 1}/{valves.max_rounds}... (Debug Mode)**")
        
        if valves.show_conversation:
            conversation_log.append(f"### 🔄 Round {round_num + 1}")

        # Get phase guidance if flow control is enabled
        phase_guidance = None
        if flow_controller and valves.enable_flow_control:
            phase_guidance = flow_controller.get_phase_guidance()
            if valves.debug_mode:
                phase_status = flow_controller.get_phase_status()
                debug_log.append(f"  📍 Phase: {phase_status['current_phase']} (Question {phase_status['questions_in_phase'] + 1} in phase)")
        
        claude_prompt_for_this_round = ""
        if round_num == 0:
            # Use state-aware prompt if state tracking is enabled
            if conversation_state and valves.track_conversation_state:
                claude_prompt_for_this_round = get_minion_initial_claude_prompt_with_state(
                    query, len(context), valves, conversation_state, phase_guidance
                )
            else:
                claude_prompt_for_this_round = get_minion_initial_claude_prompt(query, len(context), valves)
        else:
            # Check if this is the last round and force a final answer
            is_last_round = (round_num == valves.max_rounds - 1)
            if is_last_round:
                # Override with a prompt that forces a final answer
                claude_prompt_for_this_round = f"""You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: "{query}"

The local assistant has full access to the source document and has been providing factual information extracted from it.

CONVERSATION SO FAR:
"""
                for role, message in conversation_history:
                    if role == "assistant":
                        claude_prompt_for_this_round += f"\nYou previously asked: \"{message}\""
                    else:
                        claude_prompt_for_this_round += f"\nLocal assistant responded: \"{message}\""
                
                claude_prompt_for_this_round += f"""

THIS IS YOUR FINAL OPPORTUNITY TO ANSWER. You have gathered sufficient information through {round_num} rounds of questions.

Based on ALL the information provided by the local assistant, you MUST now provide a comprehensive answer to the user's original question: "{query}"

Respond with "FINAL ANSWER READY." followed by your synthesized answer. Do NOT ask any more questions."""
            else:
                # Use state-aware prompt if state tracking is enabled
                if conversation_state and valves.track_conversation_state:
                    previous_questions = deduplicator.get_all_questions() if deduplicator else None
                    claude_prompt_for_this_round = get_minion_conversation_claude_prompt_with_state(
                        conversation_history, query, valves, conversation_state, previous_questions, phase_guidance
                    )
                else:
                    claude_prompt_for_this_round = get_minion_conversation_claude_prompt(
                        conversation_history, query, valves
                    )
        
        claude_response = ""
        try:
            if valves.debug_mode: 
                start_time_claude = asyncio.get_event_loop().time()
            claude_response = await call_supervisor_model(valves, claude_prompt_for_this_round)
            if valves.debug_mode:
                end_time_claude = asyncio.get_event_loop().time()
                time_taken_claude = end_time_claude - start_time_claude
                debug_log.append(f"  ⏱️ Remote model call in round {round_num + 1} took {time_taken_claude:.2f}s. (Debug Mode)")
        except Exception as e:
            error_message = f"❌ Error calling the remote model in round {round_num + 1}: {e}"
            conversation_log.append(error_message)
            if valves.debug_mode: 
                debug_log.append(f"  {error_message} (Debug Mode)")
            actual_final_answer = "Minion protocol failed due to remote API error."
            break

        conversation_history.append(("assistant", claude_response))
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Remote Model ({valves.remote_model}):**")
            conversation_log.append(f"{claude_response}\n")

        # Check for explicit final answer or completion indicators
        if _is_final_answer(claude_response):
            actual_final_answer = claude_response.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_declared_final = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **The remote model indicates FINAL ANSWER READY.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 The remote model declared FINAL ANSWER READY in round {round_num + 1}. (Debug Mode)")
            break
        elif valves.enable_completion_detection and detect_completion(claude_response) and round_num > 0:
            # Remote model indicates it has sufficient information
            actual_final_answer = claude_response
            claude_declared_final = True
            metrics['completion_via_detection'] = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **The remote model indicates it has sufficient information to answer.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 Completion detected: Remote model has sufficient information in round {round_num + 1}. (Debug Mode)")
            break

        # Skip local model call if this was the last round and the remote model provided final answer
        if round_num == valves.max_rounds - 1:
            continue

        # Extract question from Claude's response for deduplication check
        question_to_check = claude_response.strip()
        
        # Check for duplicate questions if deduplication is enabled
        if deduplicator and valves.enable_deduplication:
            is_dup, original_question = deduplicator.is_duplicate(question_to_check)
            
            if is_dup:
                # Log the duplicate detection
                if valves.show_conversation:
                    conversation_log.append(f"⚠️ **Duplicate question detected! Similar to: '{original_question[:100]}...'**")
                    conversation_log.append(f"**Requesting a different question...**\n")
                
                if valves.debug_mode:
                    debug_log.append(f"  ⚠️ Duplicate question detected in round {round_num + 1}. (Debug Mode)")
                
                # Create a prompt asking for a different question
                dedup_prompt = f"""The question you just asked is too similar to a previous question: "{original_question}"

Please ask a DIFFERENT question that explores new aspects of the information needed to answer: "{query}"

Focus on areas not yet covered in our conversation."""
                
                # Request a new question
                try:
                    new_claude_response = await call_claude_func(valves, dedup_prompt)
                    claude_response = new_claude_response
                    question_to_check = claude_response.strip()
                    
                    # Update conversation history with the new question
                    conversation_history[-1] = ("assistant", claude_response)
                    
                    if valves.show_conversation:
                        conversation_log.append(f"**🤖 Remote Model (New Question):**")
                        conversation_log.append(f"{claude_response}\n")
                except Exception as e:
                    # If we can't get a new question, continue with the duplicate
                    if valves.debug_mode:
                        debug_log.append(f"  ❌ Failed to get alternative question: {e} (Debug Mode)")
        
        # Add the question to deduplicator after checks
        if deduplicator:
            deduplicator.add_question(question_to_check)

        local_prompt = get_minion_local_prompt(context, query, claude_response, valves)
        
        local_response_str = ""
        try:
            if valves.debug_mode: 
                start_time_ollama = asyncio.get_event_loop().time()
                debug_log.append(f"  🔄 Calling local model {valves.local_model} at {valves.ollama_base_url} (timeout: {valves.timeout_local}s) (Debug Mode)")
            
            local_response_str = await call_ollama_func(
                valves,
                local_prompt,
                use_json=True,
                schema=LocalAssistantResponseModel
            )
            local_response_data = _parse_local_response(
                local_response_str,
                is_structured=True,
                use_structured_output=valves.use_structured_output,
                debug_mode=valves.debug_mode,
                LocalAssistantResponseModel=LocalAssistantResponseModel
            )
            
            # Track metrics from local response
            if 'numeric_confidence' in local_response_data:
                metrics['confidence_scores'].append(local_response_data['numeric_confidence'])
            
            confidence_level = local_response_data.get('confidence', 'MEDIUM').upper()
            if confidence_level in metrics['confidence_distribution']:
                metrics['confidence_distribution'][confidence_level] += 1
            
            if valves.debug_mode:
                end_time_ollama = asyncio.get_event_loop().time()
                time_taken_ollama = end_time_ollama - start_time_ollama
                debug_log.append(f"  ⏱️ Local LLM call in round {round_num + 1} took {time_taken_ollama:.2f}s. (Debug Mode)")
        except Exception as e:
            # Enhanced error reporting
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else "Unknown error (empty exception message)"
            
            if valves.debug_mode and hasattr(e, '__traceback__'):
                error_details = traceback.format_exc()
                debug_log.append(f"  ❌ Local LLM call failed with full traceback: {error_details} (Debug Mode)")
            
            if "timeout" in error_msg.lower() or error_type == "TimeoutError":
                error_message = f"❌ Error calling Local LLM in round {round_num + 1}: Timeout after {valves.timeout_local}s - Local model may be overloaded or unavailable"
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
                error_message = f"❌ Error calling Local LLM in round {round_num + 1}: Connection failed - Check if Ollama is running at {valves.ollama_base_url}"
            elif error_msg == "Unknown error (empty exception message)":
                error_message = f"❌ Error calling Local LLM in round {round_num + 1}: {error_type} with empty message - likely timeout or connection issue"
            else:
                error_message = f"❌ Error calling Local LLM in round {round_num + 1}: {error_type}: {error_msg}"
            
            conversation_log.append(error_message)
            if valves.debug_mode: 
                debug_log.append(f"  {error_message} (Debug Mode)")
                debug_log.append(f"  🔧 Troubleshooting: Check Ollama status, model availability ({valves.local_model}), and network connectivity (Debug Mode)")
            
            actual_final_answer = f"Minion protocol failed due to Local LLM API error: {error_type}"
            break

        response_for_claude = local_response_data.get("answer", "Error: Could not extract answer from local LLM.")
        if valves.use_structured_output and local_response_data.get("parse_error") and valves.debug_mode:
            response_for_claude += f" (Local LLM response parse error: {local_response_data['parse_error']})"
        elif not local_response_data.get("answer") and not local_response_data.get("parse_error"):
            response_for_claude = "Local LLM provided no answer."

        # Validate answer quality if validation is enabled
        original_response = response_for_claude
        validation_passed = True
        
        if validator and valves.enable_answer_validation:
            question_for_validation = claude_response.strip()
            answer_confidence = local_response_data.get('confidence', 'MEDIUM')
            
            # Check if we've already tried clarification for this question
            question_key = f"round_{round_num}_{hash(question_for_validation)}"
            attempts = clarification_attempts.get(question_key, 0)
            
            if attempts < valves.max_clarification_attempts:
                validation_result = validator.validate_answer(
                    response_for_claude, 
                    answer_confidence, 
                    question_for_validation
                )
                
                if validation_result["needs_clarification"]:
                    validation_passed = False
                    clarification_attempts[question_key] = attempts + 1
                    
                    if valves.show_conversation:
                        conversation_log.append(f"🔍 **Answer validation detected issues:** {', '.join(validation_result['issues'])}")
                        conversation_log.append(f"**Requesting clarification (attempt {attempts + 1}/{valves.max_clarification_attempts})...**\n")
                    
                    if valves.debug_mode:
                        debug_log.append(f"  🔍 Answer validation failed: {validation_result['issues']} (Debug Mode)")
                    
                    # Generate clarification request
                    clarification_request = validator.generate_clarification_request(
                        validation_result, 
                        question_for_validation,
                        response_for_claude
                    )
                    
                    # Get clarified response
                    try:
                        clarified_prompt = get_minion_local_prompt(context, query, clarification_request, valves)
                        
                        if valves.debug_mode:
                            debug_log.append(f"  🔄 Requesting clarification for round {round_num + 1} (Debug Mode)")
                        
                        clarified_response = await call_ollama_func(
                            valves,
                            clarified_prompt,
                            use_json=True,
                            schema=LocalAssistantResponseModel
                        )
                        
                        clarified_data = _parse_local_response(
                            clarified_response,
                            is_structured=True,
                            use_structured_output=valves.use_structured_output,
                            debug_mode=valves.debug_mode,
                            LocalAssistantResponseModel=LocalAssistantResponseModel
                        )
                        
                        # Use clarified response
                        response_for_claude = clarified_data.get("answer", response_for_claude)
                        local_response_data = clarified_data  # Update for metrics
                        
                        if valves.show_conversation:
                            conversation_log.append(f"**💻 Local Model (Clarified):**")
                            if valves.use_structured_output and clarified_data.get("parse_error") is None:
                                conversation_log.append(f"```json\n{json.dumps(clarified_data, indent=2)}\n```")
                            else:
                                conversation_log.append(f"{response_for_claude}")
                            conversation_log.append("\n")
                            
                    except Exception as e:
                        if valves.debug_mode:
                            debug_log.append(f"  ❌ Clarification request failed: {e} (Debug Mode)")
                        # Continue with original response
                        response_for_claude = original_response

        conversation_history.append(("user", response_for_claude))
        
        # Update conversation state if enabled
        if conversation_state and valves.track_conversation_state:
            # Extract the question from Claude's response
            question = claude_response.strip()
            
            # Add Q&A pair to state
            conversation_state.add_qa_pair(
                question=question,
                answer=response_for_claude,
                confidence=local_response_data.get('confidence', 'MEDIUM'),
                key_points=local_response_data.get('key_points')
            )
            
            # Extract topics from the question (simple keyword extraction)
            keywords = re.findall(r'\b[A-Z][a-z]+\b|\b\w{5,}\b', question)
            for keyword in keywords[:3]:  # Add up to 3 keywords as topics
                conversation_state.topics_covered.add(keyword.lower())
            
            # Update key findings if high confidence answer
            if local_response_data.get('confidence') == 'HIGH' and local_response_data.get('key_points'):
                for idx, point in enumerate(local_response_data['key_points'][:2]):
                    conversation_state.key_findings[f"round_{round_num+1}_finding_{idx+1}"] = point
        
        # Update flow controller if enabled
        if flow_controller and valves.enable_flow_control:
            # Increment question count for current phase
            flow_controller.increment_question_count()
            
            # Check if we should transition to next phase
            if conversation_state and flow_controller.should_transition(conversation_state, valves):
                old_phase = flow_controller.current_phase.value
                flow_controller.transition_to_next_phase()
                new_phase = flow_controller.current_phase.value
                
                # Update conversation state phase
                conversation_state.current_phase = new_phase
                conversation_state.phase_transitions.append({
                    "round": round_num + 1,
                    "from": old_phase,
                    "to": new_phase
                })
                
                if valves.show_conversation:
                    conversation_log.append(f"📊 **Phase Transition: {old_phase} → {new_phase}**\n")
                
                if valves.debug_mode:
                    debug_log.append(f"  📊 Phase transition: {old_phase} → {new_phase} (Round {round_num + 1})")
            
            # Check if we're in synthesis phase and should force completion
            if flow_controller.current_phase.value == "synthesis" and round_num > 2:
                if valves.debug_mode:
                    debug_log.append(f"  🎯 Synthesis phase reached - encouraging final answer")
        
        if valves.show_conversation:
            conversation_log.append(f"**💻 Local Model ({valves.local_model}):**")
            if valves.use_structured_output and local_response_data.get("parse_error") is None:
                conversation_log.append(f"```json\n{json.dumps(local_response_data, indent=2)}\n```")
            elif valves.use_structured_output and local_response_data.get("parse_error"):
                conversation_log.append(f"Attempted structured output, but failed. Raw response:\n{local_response_data.get('answer', 'Error displaying local response.')}")
                conversation_log.append(f"(Parse Error: {local_response_data['parse_error']})")
            else:
                conversation_log.append(f"{local_response_data.get('answer', 'Error displaying local response.')}")
            conversation_log.append("\n")

        # Update round count
        metrics['rounds_completed'] = round_num + 1
        
        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {round_num + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**\n")
    
    if not claude_declared_final and conversation_history:
        # This shouldn't happen with the fix above, but keep as fallback
        last_remote_msg = conversation_history[-1][1] if conversation_history[-1][0] == "assistant" else (conversation_history[-2][1] if len(conversation_history) > 1 and conversation_history[-2][0] == "assistant" else "No suitable final message from the remote model found.")
        actual_final_answer = f"Protocol ended without explicit final answer. The remote model's last response was: \"{last_remote_msg}\""
        if valves.show_conversation:
            conversation_log.append(f"⚠️ Protocol ended without the remote model providing a final answer.\n")

    # Calculate final metrics
    total_execution_time = asyncio.get_event_loop().time() - overall_start_time
    avg_confidence = sum(metrics['confidence_scores']) / len(metrics['confidence_scores']) if metrics['confidence_scores'] else 0.0
    
    # Estimate tokens (rough approximation)
    for role, msg in conversation_history:
        metrics['estimated_tokens'] += len(msg) // 4  # Rough token estimate
    
    if valves.debug_mode:
        debug_log.append(f"**⏱️ Total Minion protocol execution time: {total_execution_time:.2f}s. (Debug Mode)**")

    output_parts = []
    if valves.show_conversation:
        output_parts.append("## 🗣️ Collaboration Conversation")
        output_parts.extend(conversation_log)
        output_parts.append("---")
    if valves.debug_mode:
        output_parts.append("### 🔍 Debug Log")
        output_parts.extend(debug_log)
        output_parts.append("---")

    output_parts.append(f"## 🎯 Final Answer")
    output_parts.append(actual_final_answer)

    stats = _calculate_token_savings(conversation_history, context, query)
    output_parts.append(f"\n## 📊 Efficiency Stats")
    output_parts.append(f"- **Protocol:** Minion (conversational)")
    output_parts.append(f"- **Remote model:** {valves.remote_model}")
    output_parts.append(f"- **Local model:** {valves.local_model}")
    output_parts.append(f"- **Conversation rounds:** {len(conversation_history) // 2}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    output_parts.append(f"")
    output_parts.append(f"## 💰 Token Savings Analysis ({valves.remote_model})")
    output_parts.append(f"- **Traditional approach:** ~{stats['traditional_tokens']:,} tokens")
    output_parts.append(f"- **Minion approach:** ~{stats['minion_tokens']:,} tokens")
    output_parts.append(f"- **💰 Token Savings:** ~{stats['percentage_savings']:.1f}%")
    
    # Add conversation metrics
    output_parts.append(f"\n## 📈 Conversation Metrics")
    output_parts.append(f"- **Rounds used:** {metrics['rounds_completed']} of {valves.max_rounds}")
    output_parts.append(f"- **Questions asked:** {metrics['rounds_completed']}")
    output_parts.append(f"- **Average confidence:** {avg_confidence:.2f} ({['LOW', 'MEDIUM', 'HIGH'][int(avg_confidence * 2.99)]})")
    output_parts.append(f"- **Confidence distribution:**")
    for level, count in metrics['confidence_distribution'].items():
        if count > 0:
            output_parts.append(f"  - {level}: {count} response(s)")
    output_parts.append(f"- **Completion method:** {'Early completion detected' if metrics['completion_via_detection'] else 'Reached max rounds or explicit completion'}")
    output_parts.append(f"- **Total duration:** {total_execution_time*1000:.0f}ms")
    output_parts.append(f"- **Estimated tokens:** ~{metrics['estimated_tokens']:,}")
    output_parts.append(f"- **Chunk processing:** {metrics['context_size']:,} chars (max chunk size: {metrics['chunk_size_used']:,})")
    
    return "\n".join(output_parts)

# Partials File: partials/minion_pipe_method.py


async def _call_supervisor_directly(valves: Any, query: str) -> str:
    """Fallback to direct supervisor call when no context is available"""
    return await call_supervisor_model(valves, f"Please answer this question: {query}")

async def _execute_simplified_chunk_protocol(
    valves: Any,
    query: str,
    context: str,
    chunk_num: int,
    total_chunks: int,
    call_ollama_func: Callable,
    LocalAssistantResponseModel: Any,
    shared_state: Any = None,
    shared_deduplicator: Any = None
) -> str:
    """Execute a simplified 1-2 round protocol for individual chunks"""
    conversation_log = []
    debug_log = []
    
    # Create a focused prompt for this chunk
    claude_prompt = f"""You are analyzing chunk {chunk_num} of {total_chunks} from a larger document to answer: "{query}"

Here is the chunk content:

<document_chunk>
{context}
</document_chunk>

This chunk contains a portion of the document. Your task is to:
1. Ask ONE focused question to extract the most relevant information from this chunk
2. Based on the response, either ask ONE follow-up question OR provide a final answer for this chunk

Be concise and focused since this is one chunk of a larger analysis.

Ask your first question about this chunk:"""

    try:
        # Round 1: Get supervisor's question
        supervisor_response = await call_supervisor_model(valves, claude_prompt)
        
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Remote Model (Chunk {chunk_num}):** {supervisor_response}")
        
        # Get local response
        local_prompt = get_minion_local_prompt(context, query, supervisor_response, valves)
        
        local_response_str = await call_ollama_func(
            valves,
            local_prompt,
            use_json=True,
            schema=LocalAssistantResponseModel
        )
        
        # Parse local response (simplified)
        local_response_data = _parse_local_response(
            local_response_str,
            is_structured=True,
            use_structured_output=valves.use_structured_output,
            debug_mode=valves.debug_mode,
            LocalAssistantResponseModel=LocalAssistantResponseModel
        )
        
        local_answer = local_response_data.get("answer", "No answer provided")
        
        if valves.show_conversation:
            conversation_log.append(f"**💻 Local Model (Chunk {chunk_num}):** {local_answer}")
        
        # Round 2: Quick synthesis
        synthesis_prompt = f"""Based on the local assistant's response: "{local_answer}"

Provide a brief summary of what this chunk (#{chunk_num} of {total_chunks}) contributes to answering: "{query}"

Keep it concise since this is just one part of a larger document analysis."""

        final_response = await call_supervisor_model(valves, synthesis_prompt)
        
        if valves.show_conversation:
            conversation_log.append(f"**🎯 Chunk Summary:** {final_response}")
        
        # Build output
        output_parts = []
        if valves.show_conversation:
            output_parts.extend(conversation_log)
        output_parts.append(f"**Result:** {final_response}")
        
        return "\n\n".join(output_parts)
        
    except Exception as e:
        return f"❌ Error processing chunk {chunk_num}: {str(e)}"

async def minion_pipe(
    pipe_self: Any,
    body: Dict[str, Any], # Typed body
    __user__: Dict[str, Any], # Typed __user__
    __request__: Request,
    __files__: List[Dict[str, Any]] = [], # Typed __files__
    __pipe_id__: str = "minion-claude",
) -> str:
    """Execute the Minion protocol with Claude"""
    try:
        # Validate configuration
        provider = getattr(pipe_self.valves, 'supervisor_provider', 'anthropic')
        if provider == 'anthropic' and not pipe_self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."
        elif provider == 'openai' and not pipe_self.valves.openai_api_key:
            return "❌ **Error:** Please configure your OpenAI API key in the function settings."

        # Extract user message and context
        messages: List[Dict[str, Any]] = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."

        user_query: str = messages[-1]["content"]

        # Extract context from messages AND uploaded files
        context_from_messages: str = extract_context_from_messages(messages[:-1])
        context_from_files: str = await extract_context_from_files(pipe_self.valves, __files__)

        # Combine all context sources
        all_context_parts: List[str] = []
        if context_from_messages:
            all_context_parts.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files:
            all_context_parts.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")

        context: str = "\n\n".join(all_context_parts) if all_context_parts else ""

        if not context:
            direct_response = await _call_supervisor_directly(pipe_self.valves, user_query)
            provider_name = getattr(pipe_self.valves, 'supervisor_provider', 'anthropic').title()
            return (
                f"ℹ️ **Note:** No significant context detected. Using standard {provider_name} response.\n\n"
                + direct_response
            )

        # Handle chunking for large documents
        chunks = create_chunks(context, pipe_self.valves.chunk_size, pipe_self.valves.max_chunks)
        if not chunks and context:
            return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size setting."
        
        if len(chunks) <= 2:
            # For small number of chunks, combine them (performance optimization)
            combined_context = "\n\n".join([f"=== CHUNK {i+1} OF {len(chunks)} ===\n{chunk}" for i, chunk in enumerate(chunks)])
            
            result: str = await _execute_minion_protocol(
                valves=pipe_self.valves, 
                query=user_query, 
                context=combined_context, 
                call_ollama_func=call_ollama,
                LocalAssistantResponseModel=LocalAssistantResponse,
                ConversationStateModel=ConversationState,
                QuestionDeduplicatorModel=QuestionDeduplicator,
                ConversationFlowControllerModel=ConversationFlowController,
                AnswerValidatorModel=AnswerValidator
            )
            
            if len(chunks) > 1:
                chunk_info = f"\n\n---\n\n## 📄 Multi-Chunk Processing Info\n**Document processed as {len(chunks)} chunks** (max {pipe_self.valves.chunk_size:,} characters each) in a single conversation session."
                result += chunk_info
            
            return result
        else:
            # For many chunks, use lightweight chunk-by-chunk processing
            # This avoids overwhelming the local model while still being efficient
            chunk_results = []
            conversation_state = ConversationState() if pipe_self.valves.track_conversation_state else None
            deduplicator = QuestionDeduplicator(pipe_self.valves.deduplication_threshold) if pipe_self.valves.enable_deduplication else None
            
            for i, chunk in enumerate(chunks):
                chunk_header = f"## 📄 Chunk {i+1} of {len(chunks)}\n"
                
                try:
                    # Use a simplified single-round protocol for efficiency
                    chunk_result = await _execute_simplified_chunk_protocol(
                        valves=pipe_self.valves,
                        query=user_query,
                        context=chunk,
                        chunk_num=i+1,
                        total_chunks=len(chunks),
                        call_ollama_func=call_ollama,
                        LocalAssistantResponseModel=LocalAssistantResponse,
                        shared_state=conversation_state,
                        shared_deduplicator=deduplicator
                    )
                    chunk_results.append(chunk_header + chunk_result)
                except Exception as e:
                    chunk_results.append(f"{chunk_header}❌ **Error processing chunk {i+1}:** {str(e)}")
            
            # Generate final synthesis from all chunk results
            synthesis_prompt = f"""You have analyzed a document in {len(chunks)} chunks to answer: "{user_query}"

Here are the results from each chunk:

{chr(10).join(chunk_results)}

Based on all the information gathered from these chunks, provide a comprehensive final answer to the user's original question: "{user_query}"

Your response should:
1. Synthesize the key information from all chunks
2. Address the specific question asked
3. Be well-organized and coherent
4. Include the most important findings and insights

Provide your final comprehensive answer:"""

            # Try final synthesis with retry logic
            final_answer = ""
            max_retries = 2
            
            for attempt in range(max_retries + 1):
                try:
                    if pipe_self.valves.debug_mode:
                        print(f"DEBUG [Minion]: Attempting final synthesis (attempt {attempt + 1}/{max_retries + 1})")
                    
                    final_answer = await call_supervisor_model(valves=pipe_self.valves, prompt=synthesis_prompt)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries:
                        if pipe_self.valves.debug_mode:
                            print(f"DEBUG [Minion]: Synthesis attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        # All retries failed, provide fallback
                        if pipe_self.valves.debug_mode:
                            print(f"DEBUG [Minion]: All synthesis attempts failed: {str(e)}")
                        
                        # Create a basic synthesis from chunk summaries as fallback
                        chunk_summaries = []
                        for i, result in enumerate(chunk_results):
                            # Extract the "Result:" section from each chunk
                            if "**Result:**" in result:
                                summary = result.split("**Result:**")[1].strip()
                                chunk_summaries.append(f"Chunk {i+1}: {summary}")
                        
                        if chunk_summaries:
                            final_answer = f"""Based on the analysis of {len(chunks)} document chunks, here are the key findings for: "{user_query}"

{chr(10).join(chunk_summaries)}

Note: This synthesis was generated locally due to a temporary API connectivity issue. The individual chunk analyses above contain the detailed information extracted from the document."""
                        else:
                            final_answer = f"❌ Unable to generate final synthesis due to API connectivity issues: {str(e)}. Please refer to the individual chunk analyses above for the extracted information."
            
            # Combine all chunk results with final synthesis
            combined_result = "\n\n---\n\n".join(chunk_results)
            
            # Add summary header with final answer
            summary_header = f"""# 🔗 Multi-Chunk Analysis Results
            
**Document processed in {len(chunks)} chunks** (max {pipe_self.valves.chunk_size:,} characters each)

{combined_result}

---

## 🎯 Final Answer

{final_answer}

---

## 📋 Summary
The document was automatically divided into {len(chunks)} chunks for efficient processing. Each chunk was analyzed using an optimized protocol to balance performance with thoroughness."""
            
            return summary_header

    except Exception as e:
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"


class Pipe:
    class Valves(MinionValves):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Minion v0.3.8 (Conversational)"

    def pipes(self):
        """Define the available models"""
        return [
            {
                "id": "minion-claude",
                "name": f" ({self.valves.local_model} + {self.valves.remote_model})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "minion-claude",
    ) -> str:
        """Execute the Minion protocol with Claude"""
        return await minion_pipe(self, body, __user__, __request__, __files__, __pipe_id__)