"""
title: MinionS Protocol Integration for Open WebUI v0.3.7
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.3.7
description: Enhanced MinionS protocol with modular architecture, improved error handling, and structured debugging
required_open_webui_version: 0.5.0
license: MIT License
"""


# Centralized imports for v0.3.7 modular architecture

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

# Partials File: partials/minions_models.py

class TaskResult(BaseModel):
    """
    Structured response format for individual task execution by a local model
    within the MinionS (multi-task, multi-round) protocol.
    This model defines the expected output structure when a local model completes
    a sub-task defined by the remote model.
    """
    explanation: str = Field(
        description="Brief explanation of the findings or the process taken to answer the task."
    )
    citation: Optional[str] = Field(
        default=None, 
        description="Direct quote from the analyzed text (chunk) that supports the answer. Should be None if no specific citation is applicable."
    )
    answer: Optional[str] = Field(
        default=None, 
        description="The specific information extracted or the answer to the sub-task. Should be None if the information is not found in the provided text chunk."
    )
    confidence: str = Field(
        default="LOW", 
        description="Confidence level in the provided answer/explanation (e.g., HIGH, MEDIUM, LOW)."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields during parsing
        # schema_extra = {
        #     "example": {
        #         "explanation": "The document mentions a budget of $500,000 for Phase 1.",
        #         "citation": "Page 5, Paragraph 2: 'Phase 1 of the project has been allocated a budget of $500,000.'",
        #         "answer": "$500,000",
        #         "confidence": "HIGH"
        #     }
        # }

class RoundMetrics(BaseModel):
    """
    Model to store various metrics collected during a single round of the MinionS protocol.
    """
    round_number: int
    tasks_executed: int
    task_success_count: int
    task_failure_count: int
    avg_chunk_processing_time_ms: float
    total_unique_findings_count: int = 0 # Defaulting as per plan for Iteration 1
    execution_time_ms: float
    success_rate: float # Calculated as task_success_count / tasks_executed

    # New fields for Iteration 2
    avg_confidence_score: float = 0.0 # Default to 0.0
    confidence_distribution: Dict[str, int] = Field(default_factory=lambda: {"HIGH": 0, "MEDIUM": 0, "LOW": 0})
    confidence_trend: str = "N/A" # Default to N/A

    # New fields for Iteration 3
    new_findings_count_this_round: int = 0
    duplicate_findings_count_this_round: int = 0
    redundancy_percentage_this_round: float = 0.0
    # cross_round_similarity_score: float = 0.0 # Deferred

    # New fields for Iteration 4
    sufficiency_score: float = Field(default=0.0, description="Overall information sufficiency score (0-1).")
    information_components: Dict[str, bool] = Field(default_factory=dict, description="Status of identified information components from the query.")
    component_coverage_percentage: float = Field(default=0.0, description="Percentage of information components addressed (0-1).")

    # New fields for Iteration 5 (Convergence Detection)
    information_gain_rate: float = Field(default=0.0, description="Rate of new information gained in this round, typically based on the count of new findings.")
    novel_findings_percentage_this_round: float = Field(default=0.0, description="Percentage of findings in this round that are new compared to all findings from this round (new + duplicate).")
    task_failure_rate_trend: str = Field(default="N/A", description="Trend of task failures (e.g., 'increasing', 'decreasing', 'stable') compared to the previous round.")
    convergence_detected_this_round: bool = Field(default=False, description="Flag indicating if convergence criteria were met based on this round's analysis.")
    predicted_value_of_next_round: str = Field(default="N/A", description="Qualitative prediction of the potential value of executing another round (e.g., 'low', 'medium', 'high').")

    class Config:
        extra = "ignore"


# Partials File: partials/minions_valves.py

class MinionsValves(BaseModel):
    """
    Configuration settings (valves) specifically for the MinionS (multi-task, multi-round) pipe.
    These settings control the behavior of the MinionS protocol, including API keys,
    model selections, timeouts, task decomposition parameters, operational parameters,
    extraction instructions, expected output format, and confidence threshold.
    """
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for the remote model (e.g., Claude)."
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model (e.g., Claude) for task decomposition and synthesis. claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL for local model execution."
    )
    local_model: str = Field(
        default="llama3.2", description="Local Ollama model name for task execution."
    )
    max_rounds: int = Field(
        default=2, description="Maximum task decomposition rounds. Each round involves remote model decomposing tasks and local models executing them."
    )
    max_tasks_per_round: int = Field(
        default=3, description="Maximum number of new sub-tasks the remote model can create in a single decomposition round."
    )
    chunk_size: int = Field(
        default=5000, description="Maximum chunk size in characters for context fed to local models during task execution."
    )
    max_chunks: int = Field( # This was present in minions-fn-claude.py but not in minion-fn-claude.py
        default=2, description="Maximum number of document chunks to process per task by the local model. Helps manage processing load."
    )
    show_conversation: bool = Field(
        default=True,
        description="Show full task decomposition, local model execution details, and synthesis steps in the output.",
    )
    timeout_local: int = Field(
        default=30,
        description="Timeout in seconds for each local model call (per chunk, per task).",
    )
    debug_mode: bool = Field(
        default=False, description="Enable additional technical details and verbose logs for debugging."
    )
    max_round_timeout_failure_threshold_percent: int = Field(
        default=50, 
        description="If this percentage of local model calls (chunk executions) in a round time out, a warning is issued. This suggests results for that round might be incomplete."
    )
    max_tokens_claude: int = Field(
        default=2000, description="Maximum tokens for remote model API calls during decomposition and synthesis."
    )
    timeout_claude: int = Field(
        default=60, description="Timeout in seconds for remote model API calls."
    )
    ollama_num_predict: int = Field(
        default=1000, description="Maximum tokens (num_predict) for local Ollama model responses during task execution."
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
    use_structured_output: bool = Field(
        default=True, 
        description="Enable JSON structured output for local model responses (requires local model to support JSON mode and the TaskResult schema)."
    )
    structured_output_fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback parsing when structured output fails. If False, parsing errors will be propagated."
    )
    extraction_instructions: str = Field(
        default="", title="Extraction Instructions", description="Specific instructions for the LLM on what to extract or how to process the information for each task."
    )
    expected_format: str = Field(
        default="text", title="Expected Output Format", description="Desired format for the LLM's output for each task (e.g., 'text', 'JSON', 'bullet points'). Note: 'JSON' enables specific structured output fields like 'explanation', 'citation', 'answer'."
    )
    confidence_threshold: float = Field(
        default=0.7, title="Confidence Threshold", description="Minimum confidence level for the LLM's response for each task (0.0-1.0). Primarily a suggestion to the LLM.", ge=0, le=1
    )

    # --- Performance Profile Valve ---
    performance_profile: str = Field(
        default="balanced",
        title="Performance Profile",
        description="Overall performance profile: 'high_quality', 'balanced', 'fastest_results'. Affects base thresholds and max_rounds before other adaptive modifiers.",
        json_schema_extra={"enum": ["high_quality", "balanced", "fastest_results"]}
    )

    # New fields for Iteration 5: Static Early Stopping Rules
    enable_early_stopping: bool = Field(
        default=False,
        title="Enable Early Stopping",
        description="Enable early stopping of rounds based on confidence and query complexity."
    )
    simple_query_confidence_threshold: float = Field(
        default=0.85,
        title="Simple Query Confidence Threshold",
        description="Confidence threshold (0.0-1.0) to stop early for SIMPLE queries.",
        ge=0, le=1
    )
    medium_query_confidence_threshold: float = Field(
        default=0.75,
        title="Medium Query Confidence Threshold",
        description="Confidence threshold (0.0-1.0) to stop early for MEDIUM queries.",
        ge=0, le=1
    )
    min_rounds_before_stopping: int = Field(
        default=1,
        title="Minimum Rounds Before Stopping",
        description="Minimum number of rounds to execute before early stopping can be triggered.",
        ge=1
    )
    # max_rounds (already exists) will be used for COMPLEX queries.

    # New fields for Iteration 5: Convergence Detection Based Early Stopping
    convergence_novelty_threshold: float = Field(
        default=0.10,
        title="Convergence Novelty Threshold",
        description="Minimum percentage of novel findings required per round to consider it non-convergent. E.g., 0.10 means less than 10% new findings might indicate convergence if other criteria met.",
        ge=0, le=1
    )
    convergence_rounds_min_novelty: int = Field(
        default=2,
        title="Convergence Rounds for Minimum Novelty",
        description="Number of consecutive rounds novelty must be below 'convergence_novelty_threshold' to trigger convergence.",
        ge=1
    )
    convergence_sufficiency_threshold: float = Field(
        default=0.7,
        title="Convergence Sufficiency Threshold",
        description="Minimum sufficiency score required for convergence-based early stopping. E.g., 0.7 means 70% sufficiency needed.",
        ge=0, le=1
    )
    # min_rounds_before_convergence_check could be added if distinct from min_rounds_before_stopping
    # For now, ConvergenceDetector uses its own default or relies on min_rounds_before_stopping implicitly
    # if min_rounds_before_convergence_check is not explicitly set in valves.

    # --- Adaptive Threshold Valves ---
    enable_adaptive_thresholds: bool = Field(
        default=True,
        title="Enable Adaptive Thresholds",
        description="Allow the system to dynamically adjust confidence, sufficiency, and novelty thresholds based on document size, query complexity, and first-round performance."
    )
    doc_size_small_char_limit: int = Field(
        default=5000,
        title="Small Document Character Limit",
        description="Documents with character count below this are considered 'small' for threshold adjustments."
    )
    doc_size_large_char_start: int = Field(
        default=50000,
        title="Large Document Character Start",
        description="Documents with character count above this are considered 'large' for threshold adjustments."
    )
    confidence_modifier_small_doc: float = Field(
        default=0.0,
        title="Confidence Modifier for Small Docs",
        description="Value added to base confidence thresholds if document is small (e.g., -0.05 to be less strict). Applied to general confidence checks if any."
    )
    confidence_modifier_large_doc: float = Field(
        default=0.0,
        title="Confidence Modifier for Large Docs",
        description="Value added to base confidence thresholds if document is large (e.g., +0.05 to be more strict)."
    )
    sufficiency_modifier_simple_query: float = Field(
        default=0.0,
        title="Sufficiency Modifier for Simple Queries",
        description="Value added to base sufficiency thresholds for simple queries (e.g., -0.1 to require less sufficiency)."
    )
    sufficiency_modifier_complex_query: float = Field(
        default=0.0,
        title="Sufficiency Modifier for Complex Queries",
        description="Value added to base sufficiency thresholds for complex queries (e.g., +0.1 to require more)."
    )
    novelty_modifier_simple_query: float = Field(
        default=0.0,
        title="Novelty Threshold Modifier for Simple Queries",
        description="Value added to the base novelty threshold (making it potentially easier to achieve 'low novelty') for simple queries."
    )
    novelty_modifier_complex_query: float = Field(
        default=0.0,
        title="Novelty Threshold Modifier for Complex Queries",
        description="Value added to the base novelty threshold (making it potentially harder to achieve 'low novelty') for complex queries."
    )
    first_round_high_novelty_threshold: float = Field(
        default=0.75,
        title="First Round High Novelty Threshold (%)",
        description="If first round's novel_findings_percentage_this_round is above this (e.g., 0.75 for 75%), it's considered a high novelty first round.",
        ge=0, le=1
    )
    sufficiency_modifier_high_first_round_novelty: float = Field(
        default=-0.05,
        title="Sufficiency Modifier for High First Round Novelty",
        description="Value added to sufficiency thresholds if first round novelty is high (e.g., -0.05 to relax sufficiency requirement)."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields passed to the model
        # an_example = MinionsValves().dict() # For schema generation


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


# Partials File: partials/minions_prompts.py

# This file will store prompt generation functions for the MinionS (multi-turn, multi-task) protocol.

def get_minions_synthesis_claude_prompt(query: str, synthesis_input_summary: str, valves: Any) -> str:
    """
    Returns the synthesis prompt for Claude in the MinionS protocol.
    Logic moved from _execute_minions_protocol in minions_pipe_method.py.
    'synthesis_input_summary' is the aggregation of successful task results.
    """
    prompt_lines = [
        f'''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary if synthesis_input_summary else "No specific information was extracted by local models."}
'''
    ]

    # Add instructions from valves for synthesis guidance
    synthesis_instructions = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        synthesis_instructions.append(f"When synthesizing the final answer, consider these overall instructions: {valves.extraction_instructions}")
    if hasattr(valves, 'expected_format') and valves.expected_format:
        synthesis_instructions.append(f"Format the final synthesized answer as {valves.expected_format}.")
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0: # Assuming 0 is default/off
        synthesis_instructions.append(f"Aim for an overall confidence level of at least {valves.confidence_threshold} in your synthesized answer.")

    if synthesis_instructions:
        prompt_lines.append("\nSYNTHESIS GUIDELINES:")
        prompt_lines.extend(synthesis_instructions)
        prompt_lines.append("") # Add a newline for separation

    prompt_lines.append("If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.")
    prompt_lines.append("Final Answer:")
    return "\n".join(prompt_lines)

def get_minions_local_task_prompt(
    chunk: str, 
    task: str, 
    chunk_idx: int, 
    total_chunks: int, 
    valves: Any, 
) -> str:
    """
    Returns the prompt for the local Ollama model for a specific task on a chunk 
    in the MinionS protocol.
    Logic moved from execute_tasks_on_chunks in minions_protocol_logic.py.
    """
    prompt_lines = [
        f'''Text to analyze (Chunk {chunk_idx + 1}/{total_chunks} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}'''
    ]

    task_specific_instructions = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        task_specific_instructions.append(f"- Follow these specific extraction instructions: {valves.extraction_instructions}")

    # Confidence threshold is a general guideline for the task.
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0:
        task_specific_instructions.append(f"- Aim for a confidence level of at least {valves.confidence_threshold} in your findings for this task.")

    if task_specific_instructions:
        prompt_lines.append("\nTASK-SPECIFIC INSTRUCTIONS:")
        prompt_lines.extend(task_specific_instructions)

    if valves.use_structured_output:
        prompt_lines.append(f'''\n
IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any text before or after the JSON.

Required JSON structure:
{{
    "explanation": "Brief explanation of your findings for this task",
    "citation": "Direct quote from the text if applicable to this task, or null",
    "answer": "Your complete answer to the task as a SINGLE STRING",
    "confidence": "HIGH, MEDIUM, or LOW"
}}''')

        structured_output_rules = [
            "\nCRITICAL RULES FOR JSON OUTPUT:",
            "1. Output ONLY the JSON object - no markdown formatting, no explanatory text, no code blocks",
            "2. The \"answer\" field MUST be a plain text string, NOT an object or array",
            "3. If listing multiple items, format as a single string (e.g., \"Item 1: Description. Item 2: Description.\")",
            "4. Use proper JSON escaping for quotes within strings (\\\" for quotes inside string values)",
            "5. If information is not found, set \"answer\" to null and \"confidence\" to \"LOW\"",
            "6. The \"confidence\" field must be exactly one of: \"HIGH\", \"MEDIUM\", or \"LOW\"",
            "7. All string values must be properly quoted and escaped"
        ]
        prompt_lines.extend(structured_output_rules)

        # Rule 5 regarding expected_format needs to be renumbered if it was part of the list,
        # but it's added conditionally after extending structured_output_rules.
        # So, its conditional addition logic remains correct without renumbering.

        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() != "json":
            prompt_lines.append(f"5. Format the content WITHIN the \"answer\" field as {valves.expected_format.upper()}. For example, if \"bullet points\", the \"answer\" string should look like \"- Point 1\\n- Point 2\".")
        elif hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() == "json":
             prompt_lines.append("5. The overall response is already JSON. Ensure the content of the 'answer' field is a simple string, not further JSON encoded, unless the task specifically asks for a JSON string as the answer.")


        prompt_lines.append(f'''
\nEXAMPLES OF CORRECT JSON OUTPUT:

Example 1 - Information found:
{{
    "explanation": "Found budget information in the financial section",
    "citation": "The total project budget is set at $2.5 million for fiscal year 2024",
    "answer": "$2.5 million",
    "confidence": "HIGH"
}}

Example 2 - Information NOT found:
{{
    "explanation": "Searched for revenue projections but this chunk only contains expense data",
    "citation": null,
    "answer": null,
    "confidence": "LOW"
}}

Example 3 - Multiple items found:
{{
    "explanation": "Identified three risk factors mentioned in the document",
    "citation": "Key risks include: market volatility, regulatory changes, and supply chain disruptions",
    "answer": "1. Market volatility 2. Regulatory changes 3. Supply chain disruptions",
    "confidence": "HIGH"
}}''')
        
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() == "bullet points":
            prompt_lines.append(f'''
\nExample with bullet points in answer field:
{{
    "explanation": "Found multiple implementation steps",
    "citation": "The implementation plan consists of three phases...",
    "answer": "- Phase 1: Initial setup and configuration\\n- Phase 2: Testing and validation\\n- Phase 3: Full deployment",
    "confidence": "MEDIUM"
}}''')

        prompt_lines.append(f'''
\nEXAMPLES OF INCORRECT OUTPUT (DO NOT DO THIS):

Wrong - Wrapped in markdown:
```json
{{"answer": "some value"}}
```

Wrong - Answer is not a string:
{{
    "answer": {{"key": "value"}},
    "confidence": "HIGH"
}}

Wrong - Missing required fields:
{{
    "answer": "some value"
}}

Wrong - Text outside JSON:
Here is my response:
{{"answer": "some value"}}''')

    else: # Not using structured output
        prompt_lines.append("\n\nProvide a brief, specific answer based ONLY on the text provided above.")
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() != "text":
            prompt_lines.append(f"Format your entire response as {valves.expected_format.upper()}.")
        prompt_lines.append("If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\".")
    
    return "\n".join(prompt_lines)

# Partials File: partials/minions_decomposition_logic.py

# This helper will be based on the current parse_tasks from minions_protocol_logic.py
def _parse_tasks_helper(claude_response: str, max_tasks: int, debug_log: List[str], valves: Any) -> List[str]:
    """
    Parse tasks from Claude's response.
    Enhanced to match the original parse_tasks more closely.
    """
    lines = claude_response.split("\n")
    tasks = []
    for line in lines:
        line = line.strip()
        # More robust parsing for numbered or bulleted lists from original parse_tasks
        if line.startswith(tuple(f"{i}." for i in range(1, 10))) or \
           line.startswith(tuple(f"{i})" for i in range(1, 10))) or \
           line.startswith(("- ", "* ", "+ ")):
            # Attempt to split by the first space after the list marker
            parts = line.split(None, 1)
            task = parts[1].strip() if len(parts) > 1 else ""
            if len(task) > 10:  # Keep simple task filter
                tasks.append(task)
            elif valves.debug_mode and task:
                 debug_log.append(f"   [Debug] Task too short, skipped: '{task}' (Length: {len(task)})")

    if not tasks and valves.debug_mode:
        debug_log.append(f"   [Debug] No tasks parsed from Claude response: {claude_response[:200]}...")
        # Fallback could be added here if necessary, but original parse_tasks didn't have one.

    return tasks[:max_tasks]

async def decompose_task(
    valves: Any,
    call_claude_func: Callable[..., Awaitable[str]],
    query: str,
    scratchpad_content: str,
    num_chunks: int,
    max_tasks_per_round: int,
    current_round: int,
    conversation_log: List[str],
    debug_log: List[str]
) -> Tuple[List[str], str, str]:  # Added third return value for the prompt
    """
    Constructs the decomposition prompt, calls Claude, and parses tasks.
    Returns: (tasks, claude_response, decomposition_prompt)
    """
    # Base decomposition prompt
    base_decomposition_prompt = f'''You are a supervisor LLM in a multi-round process. Your goal is to answer: "{query}"
Context has been split into {num_chunks} chunks. A local LLM will process these chunks for each task you define.
Scratchpad (previous findings): {scratchpad_content if scratchpad_content else "Nothing yet."}

Based on the scratchpad and the original query, identify up to {max_tasks_per_round} specific, simple tasks for the local assistant.
If the information in the scratchpad is sufficient to answer the query, respond ONLY with the exact phrase "FINAL ANSWER READY." followed by the comprehensive answer.
Otherwise, list the new tasks clearly. Ensure tasks are actionable. Avoid redundant tasks.
Format tasks as a simple list (e.g., 1. Task A, 2. Task B).'''

    # Enhance the prompt with task formulation guidance
    decomposition_prompt = _enhance_decomposition_prompt(base_decomposition_prompt, valves)

    if valves.show_conversation:
        conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round}):** Sending prompt:\n```\n{decomposition_prompt}\n```")

    start_time_claude_decomp = 0
    if valves.debug_mode:
        start_time_claude_decomp = asyncio.get_event_loop().time()
        debug_log.append(f"   [Debug] Sending decomposition prompt to Claude (Round {current_round}):\n{decomposition_prompt}")

    try:
        claude_response = await call_claude_func(valves, decomposition_prompt)
        
        if valves.debug_mode:
            end_time_claude_decomp = asyncio.get_event_loop().time()
            time_taken_claude_decomp = end_time_claude_decomp - start_time_claude_decomp
            debug_log.append(f"   ⏱️ Claude call (Decomposition Round {current_round}) took {time_taken_claude_decomp:.2f}s.")
            debug_log.append(f"   [Debug] Claude response (Decomposition Round {current_round}):\n{claude_response}")

        tasks = _parse_tasks_helper(claude_response, max_tasks_per_round, debug_log, valves)
        
        if valves.debug_mode:
            debug_log.append(f"   Identified {len(tasks)} tasks for Round {current_round} from decomposition response.")
            for task_idx, task_item in enumerate(tasks):
                debug_log.append(f"    Task {task_idx+1} (Round {current_round}): {task_item[:100]}...")
        
        return tasks, claude_response, decomposition_prompt  # Return the prompt too

    except Exception as e:
        error_msg = f"❌ Error calling Claude for decomposition in round {current_round}: {e}"
        if valves.show_conversation:
            conversation_log.append(error_msg)
        if valves.debug_mode:
            debug_log.append(f"   {error_msg}")
        return [], f"CLAUDE_ERROR: {error_msg}", ""  # Return empty prompt on error
    
def _enhance_decomposition_prompt(base_prompt: str, valves: Any) -> str:
    """
    Enhances the decomposition prompt with additional guidance to ensure
    tasks are formulated to receive string responses.
    """
    task_formulation_guidance = '''

IMPORTANT TASK FORMULATION RULES:
1. Each task should request information that can be expressed as plain text
2. Avoid tasks that implicitly request structured data (like "Create a table of..." or "List with categories...")
3. Instead of "Extract and categorize X by Y", use "Describe X including information about Y"
4. Tasks should be answerable with narrative text, not data structures

GOOD TASK EXAMPLES:
- "Summarize the key advancements described in each Part, presenting them as a narrative"
- "Describe the AI winters mentioned, including their timeframes and characteristics"
- "Explain the progression of AI development across different periods"

BAD TASK EXAMPLES (avoid these):
- "Create a structured list of Parts with their key points"
- "Extract and categorize advancements by Part number"
- "Build a timeline table of AI winters"

Remember: The local assistant will return text strings, not structured data.'''
    
    return base_prompt + task_formulation_guidance


# Partials File: partials/minions_protocol_logic.py

# parse_tasks function removed, will be part of minions_decomposition_logic.py

# Removed create_chunks function from here

def parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, TaskResultModel: Any, structured_output_fallback_enabled: bool = True) -> Dict:
    """Parse local model response, supporting both text and structured formats"""
    confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
    default_numeric_confidence = 0.3 # Corresponds to LOW

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
            
            # Safety net: if answer is a dict/list, stringify it
            if 'answer' in parsed_json and not isinstance(parsed_json['answer'], (str, type(None))):
                if debug_mode:
                    print(f"DEBUG: Converting non-string answer to string: {type(parsed_json['answer'])}")
                parsed_json['answer'] = json.dumps(parsed_json['answer']) if parsed_json['answer'] else None
            
            # Ensure required fields have defaults if missing
            if 'explanation' not in parsed_json:
                parsed_json['explanation'] = parsed_json.get('answer', '') or "No explanation provided"
            if 'citation' not in parsed_json:
                parsed_json['citation'] = None
            
            validated_model = TaskResultModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            
            text_confidence = model_dict.get('confidence', 'LOW').upper()
            model_dict['numeric_confidence'] = confidence_map.get(text_confidence, default_numeric_confidence)

            # Check if the structured response indicates "not found" via its 'answer' field
            if model_dict.get('answer') is None:
                model_dict['_is_none_equivalent'] = True
            else:
                model_dict['_is_none_equivalent'] = False
            return model_dict
            
        except json.JSONDecodeError as e:
            if debug_mode:
                print(f"DEBUG: JSON decode error in MinionS: {e}. Cleaned response was: {cleaned_response[:500]}")
            
            if not structured_output_fallback_enabled:
                # Re-raise the error if fallback is disabled
                raise e
            
            # Try regex fallback to extract key information
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
            confidence_match = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', response, re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1)
                confidence = confidence_match.group(1).upper() if confidence_match else "LOW"
                return {
                    "answer": answer,
                    "explanation": f"Extracted from malformed JSON: {answer}",
                    "confidence": confidence,
                    "numeric_confidence": confidence_map.get(confidence, default_numeric_confidence),
                    "citation": None,
                    "parse_error": f"JSON parse error (recovered): {str(e)}",
                    "_is_none_equivalent": answer.strip().upper() == "NONE"
                }
            
            # Complete fallback
            is_none_equivalent_fallback = response.strip().upper() == "NONE"
            return {
                "answer": response, 
                "explanation": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence, 
                "parse_error": f"JSON parse error: {str(e)}", 
                "_is_none_equivalent": is_none_equivalent_fallback
            }
            
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in MinionS: {e}. Response was: {response[:500]}")
            # Fallback for parsing failure
            is_none_equivalent_fallback = response.strip().upper() == "NONE"
            return {
                "answer": response, 
                "explanation": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence, 
                "parse_error": str(e), 
                "_is_none_equivalent": is_none_equivalent_fallback
            }
    
    # Fallback for non-structured processing
    is_none_equivalent_text = response.strip().upper() == "NONE"
    # Confidence is MEDIUM by default in this path
    return {
        "answer": response, 
        "explanation": response, 
        "confidence": "MEDIUM", 
        "numeric_confidence": confidence_map['MEDIUM'], 
        "citation": None, 
        "parse_error": None, 
        "_is_none_equivalent": is_none_equivalent_text
    }

async def execute_tasks_on_chunks(
    tasks: List[str], 
    chunks: List[str], 
    conversation_log: List[str], 
    current_round: int,
    valves: Any,
    call_ollama: Callable,
    TaskResult: Any
) -> Dict:
    """Execute tasks on chunks using local model"""
    overall_task_results = []
    total_attempts_this_call = 0
    total_timeouts_this_call = 0

    # Initialize Metrics
    round_start_time = asyncio.get_event_loop().time()
    tasks_executed_count = 0
    task_success_count = 0
    task_failure_count = 0
    chunk_processing_times = []
    # Initialize Confidence Accumulators
    round_confidence_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    aggregated_task_confidences = []
    
    # Structured output metrics
    structured_output_attempts = 0
    structured_output_successes = 0

    for task_idx, task in enumerate(tasks):
        tasks_executed_count += 1 # Track Tasks Executed
        conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
        results_for_this_task_from_chunks = []
        current_task_chunk_confidences = [] # Initialize for current task
        chunk_timeout_count_for_task = 0
        num_relevant_chunks_found = 0

        for chunk_idx, chunk in enumerate(chunks):
            total_attempts_this_call += 1
            
            # Call the new function for local task prompt
            local_prompt = get_minions_local_task_prompt( # Ensure this function is defined or imported
                chunk=chunk,
                task=task,
                chunk_idx=chunk_idx,
                total_chunks=len(chunks),
                valves=valves
            )
            
            # Track Chunk Processing Time
            chunk_start_time = asyncio.get_event_loop().time()
            # start_time_ollama variable was previously used for debug,
            # let's ensure we use chunk_start_time for metrics consistently.
            # If start_time_ollama is still needed for debug, it can be kept separate.
            # For metrics, we'll use chunk_start_time and chunk_end_time.

            if valves.debug_mode:
                conversation_log.append(
                    f"   🔄 Task {task_idx + 1} - Trying chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} chars)... (Debug Mode)"
                )
                # start_time_ollama = asyncio.get_event_loop().time() # This was for debug, let's use chunk_start_time

            try:
                response_str = await asyncio.wait_for(
                    call_ollama(
                        valves,
                        local_prompt,
                        use_json=True,
                        schema=TaskResult
                    ),
                    timeout=valves.timeout_local,
                )
                chunk_end_time = asyncio.get_event_loop().time()
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)

                response_data = parse_local_response(
                    response_str,
                    is_structured=True,
                    use_structured_output=valves.use_structured_output,
                    debug_mode=valves.debug_mode,
                    TaskResultModel=TaskResult, # Pass TaskResult to parse_local_response
                    structured_output_fallback_enabled=getattr(valves, 'structured_output_fallback_enabled', True)
                )
                
                # Track structured output success
                if valves.use_structured_output:
                    structured_output_attempts += 1
                    if not response_data.get('parse_error'):
                        structured_output_successes += 1

                # Collect Confidence per Chunk
                numeric_confidence = response_data.get('numeric_confidence', 0.3) # Default to LOW numeric
                text_confidence = response_data.get('confidence', 'LOW').upper()
                response_data['fingerprint'] = None # Initialize fingerprint

                if not response_data.get('_is_none_equivalent') and not response_data.get('parse_error'):
                    if text_confidence in round_confidence_distribution:
                        round_confidence_distribution[text_confidence] += 1
                    current_task_chunk_confidences.append(numeric_confidence)

                    # Fingerprint Generation Logic
                    answer_text = response_data.get('answer')
                    if answer_text: # Ensure answer_text is not None and not empty
                        normalized_text = answer_text.lower().strip()
                        if normalized_text: # Ensure normalized_text is not empty
                            response_data['fingerprint'] = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
                
                if valves.debug_mode:
                    # end_time_ollama = asyncio.get_event_loop().time() # Already have chunk_end_time
                    time_taken_ollama = (chunk_end_time - chunk_start_time) # Use metric times for debug consistency
                    status_msg = ""
                    details_msg = ""
                    if response_data.get("parse_error"):
                        status_msg = "Parse Error"
                        details_msg = f"Error: {response_data['parse_error']}, Raw: {response_data.get('answer', '')[:70]}..."
                    elif response_data['_is_none_equivalent']:
                        status_msg = "No relevant info"
                        details_msg = f"Response indicates no info found. Confidence: {response_data.get('confidence', 'N/A')}"
                    else:
                        status_msg = "Relevant info found"
                        details_msg = f"Answer: {response_data.get('answer', '')[:70]}..., Confidence: {response_data.get('confidence', 'N/A')}"
        
                    conversation_log.append(
                         f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} processed by local LLM in {time_taken_ollama:.2f}s. Status: {status_msg}. Details: {details_msg} (Debug Mode)"
                    )

                if not response_data.get('_is_none_equivalent'): # Check with .get for safety
                    extracted_info = response_data.get('answer') or response_data.get('explanation', 'Could not extract details.')
                    # Store as dict with fingerprint
                    results_for_this_task_from_chunks.append({
                        "text": f"[Chunk {chunk_idx+1}]: {extracted_info}",
                        "fingerprint": response_data.get('fingerprint')
                    })
                    num_relevant_chunks_found += 1
                    # Note: current_task_chunk_confidences is already appended if valid (based on earlier logic)
                    
            except asyncio.TimeoutError:
                chunk_end_time = asyncio.get_event_loop().time() # Capture time even on timeout
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)
                total_timeouts_this_call += 1
                chunk_timeout_count_for_task += 1
                conversation_log.append(
                    f"   ⏰ Task {task_idx + 1} - Chunk {chunk_idx + 1} timed out after {valves.timeout_local}s"
                )
                if valves.debug_mode:
                    # end_time_ollama = asyncio.get_event_loop().time() # Already have chunk_end_time
                    time_taken_ollama = (chunk_end_time - chunk_start_time) # Use metric times
                    conversation_log.append(
                         f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} TIMEOUT after {time_taken_ollama:.2f}s. (Debug Mode)"
                    )
            except Exception as e:
                # It's good practice to also record chunk processing time if an unexpected exception occurs
                chunk_end_time = asyncio.get_event_loop().time()
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)
                conversation_log.append(
                    f"   ❌ Task {task_idx + 1} - Chunk {chunk_idx + 1} error: {str(e)}"
                )
        
        # Track Task Success/Failure and Aggregate Confidence per Task
        if results_for_this_task_from_chunks:
            task_success_count += 1
            avg_task_confidence = sum(current_task_chunk_confidences) / len(current_task_chunk_confidences) if current_task_chunk_confidences else 0.0
            aggregated_task_confidences.append({
                "task": task,
                "avg_numeric_confidence": avg_task_confidence,
                "contributing_successful_chunks": len(current_task_chunk_confidences)
            })
            # Modify overall_task_results for successful tasks
            detailed_results = [{"text": res["text"], "fingerprint": res["fingerprint"]} for res in results_for_this_task_from_chunks if isinstance(res, dict)]
            aggregated_text_result = "\n".join([res["text"] for res in detailed_results])
            overall_task_results.append({
                "task": task,
                "result": aggregated_text_result,
                "status": "success",
                "detailed_findings": detailed_results
            })
            conversation_log.append(
                f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). Avg Confidence: {avg_task_confidence:.2f}. First result snippet: {detailed_results[0]['text'][:100] if detailed_results else 'N/A'}..."
            )
        elif chunk_timeout_count_for_task > 0 and chunk_timeout_count_for_task == len(chunks):
            task_failure_count += 1 # All chunks timed out
            aggregated_task_confidences.append({"task": task, "avg_numeric_confidence": 0.0, "contributing_successful_chunks": 0})
            overall_task_results.append({
                "task": task,
                "result": f"Timeout on all {len(chunks)} chunks",
                "status": "timeout_all_chunks",
                "detailed_findings": [] # Add empty list for consistency
            })
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out."
            )
        else: # No relevant info found or other errors
            task_failure_count += 1
            aggregated_task_confidences.append({"task": task, "avg_numeric_confidence": 0.0, "contributing_successful_chunks": 0})
            overall_task_results.append({
                "task": task,
                "result": "Information not found in any relevant chunk",
                "status": "not_found",
                "detailed_findings": [] # Add empty list for consistency
            })
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found in any chunk."
            )
        # current_task_chunk_confidences is implicitly reset at the start of the task loop

    # Calculate Final Metrics
    round_end_time = asyncio.get_event_loop().time()
    execution_time_ms = (round_end_time - round_start_time) * 1000
    avg_chunk_processing_time_ms = sum(chunk_processing_times) / len(chunk_processing_times) if chunk_processing_times else 0
    success_rate = task_success_count / tasks_executed_count if tasks_executed_count > 0 else 0

    # Prepare Metrics Object (as a dictionary for now, as per instructions)
    round_metrics_data = {
        "round_number": current_round,
        "tasks_executed": tasks_executed_count,
        "task_success_count": task_success_count,
        "task_failure_count": task_failure_count,
        "avg_chunk_processing_time_ms": avg_chunk_processing_time_ms,
        "execution_time_ms": execution_time_ms,
        "success_rate": success_rate,
        # total_unique_findings_count will be handled later, defaulting in the model
    }

    # Calculate structured output success rate if applicable
    structured_output_success_rate = None
    if structured_output_attempts > 0:
        structured_output_success_rate = structured_output_successes / structured_output_attempts
    
    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_attempts_this_call,
        "total_chunk_processing_timeouts": total_timeouts_this_call,
        "round_metrics_data": round_metrics_data,
        "confidence_metrics_data": { # New confidence data
            "task_confidences": aggregated_task_confidences,
            "round_confidence_distribution": round_confidence_distribution
        },
        "structured_output_metrics": {
            "attempts": structured_output_attempts,
            "successes": structured_output_successes,
            "success_rate": structured_output_success_rate
        }
    }

def calculate_token_savings(
    decomposition_prompts: List[str], 
    synthesis_prompts: List[str],
    all_results_summary: str, 
    final_response: str, 
    context_length: int, 
    query_length: int, 
    total_chunks: int,
    total_tasks: int
) -> dict:
    """Calculate token savings for MinionS protocol"""
    chars_per_token = 3.5
    
    # Traditional approach: entire context + query sent to Claude
    traditional_tokens = int((context_length + query_length) / chars_per_token)
    
    # MinionS approach: only prompts and summaries sent to Claude
    minions_tokens = 0
    for p in decomposition_prompts:
        minions_tokens += int(len(p) / chars_per_token)
    for p in synthesis_prompts:
        minions_tokens += int(len(p) / chars_per_token)
    minions_tokens += int(len(all_results_summary) / chars_per_token)
    minions_tokens += int(len(final_response) / chars_per_token)
    
    token_savings = traditional_tokens - minions_tokens
    percentage_savings = (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    
    return {
        'traditional_tokens_claude': traditional_tokens,
        'minions_tokens_claude': minions_tokens,
        'token_savings_claude': token_savings,
        'percentage_savings_claude': percentage_savings,
        'total_rounds': len(decomposition_prompts),
        'total_chunks_processed_local': total_chunks,
        'total_tasks_executed_local': total_tasks,
    }

# Partials File: partials/minion_sufficiency_analyzer.py

class InformationSufficiencyAnalyzer:
    """
    Analyzes text to determine if it sufficiently addresses components of an initial query.
    """
    def __init__(self, query: str, debug_mode: bool = False):
        """
        Initializes the analyzer with the user's query and debug mode.
        """
        self.query = query
        self.debug_mode = debug_mode
        self.components: Dict[str, Dict[str, Any]] = {} # Stores {component_name: {"keywords": [...], "is_addressed": False, "confidence": 0.0}}
        self._identify_components()

    def _identify_components(self):
        """
        Identifies key components or topics from the user's query.
        Uses heuristics like quoted phrases, capitalized words, and generic fallbacks.
        """
        # Basic keyword extraction. This is a simple heuristic and can be expanded.
        # It looks for Nouns, Proper Nouns, and Adjectives, trying to form simple topics.
        # Example: "Compare the budget and timeline for Project Alpha and Project Beta"
        # Might identify: "budget", "timeline", "Project Alpha", "Project Beta"
        # Then forms components like "budget Project Alpha", "timeline Project Alpha", etc.

        # For simplicity in this iteration, we'll use a more direct approach:
        # Look for quoted phrases or capitalized words as potential components.
        # Or, define a few generic components if query is too simple.

        # Let's try to find quoted phrases first
        quoted_phrases = re.findall(r'"([^"]+)"', self.query)
        for phrase in quoted_phrases:
            self.components[phrase] = {"keywords": [kw.lower() for kw in phrase.split()], "is_addressed": False, "confidence": 0.0}

        # If no quoted phrases, look for capitalized words/phrases (potential proper nouns or topics)
        if not self.components:
            common_fillers = [
                "Here", "The", "This", "That", "There", "It", "Who", "What", "When", "Where", "Why", "How",
                "Is", "Are", "Was", "Were", "My", "Your", "His", "Her", "Its", "Our", "Their", "An",
                "As", "At", "But", "By", "For", "From", "In", "Into", "Of", "On", "Or", "Over",
                "So", "Then", "To", "Under", "Up", "With", "I"
            ]
            common_fillers_lower = [f.lower() for f in common_fillers]

            # Regex to find sequences of capitalized words, possibly including 'and', 'or', 'for', 'the'
            potential_topics = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+(?:and|or|for|the|[A-Z][a-zA-Z]*))*\b', self.query)

            topics = []
            for pt in potential_topics:
                is_multi_word = ' ' in pt
                # A single word is a common filler if its lowercased version is in common_fillers_lower
                is_common_filler_single_word = not is_multi_word and pt.lower() in common_fillers_lower
                # A single word is significant if it's an acronym (all upper) or longer than 3 chars
                is_significant_single_word = not is_multi_word and (pt.isupper() or len(pt) > 3)

                if is_multi_word: # Always include multi-word capitalized phrases
                    topics.append(pt)
                elif is_significant_single_word and not is_common_filler_single_word:
                    # Include significant single words only if they are NOT common fillers
                    topics.append(pt)

            if self.debug_mode and potential_topics:
                 print(f"DEBUG [SufficiencyAnalyzer]: Potential capitalized topics found: {potential_topics}")
                 print(f"DEBUG [SufficiencyAnalyzer]: Filtered topics after common word/length check: {topics}")

            for topic in topics:
                # Avoid adding overlapping sub-phrases if a larger phrase is already a component
                is_sub_phrase = False
                for existing_comp in self.components.keys():
                    if topic in existing_comp and topic != existing_comp:
                        is_sub_phrase = True
                        break
                if not is_sub_phrase:
                    self.components[topic] = {"keywords": [kw.lower() for kw in topic.split()], "is_addressed": False, "confidence": 0.0}

        # If still no components (e.g., simple query like "summarize this"), create generic ones.
        if not self.components:
            if "compare" in self.query.lower() or "contrast" in self.query.lower():
                self.components["comparison points"] = {"keywords": ["compare", "contrast", "similarit", "difference"], "is_addressed": False, "confidence": 0.0} # Added common keywords
                self.components["subject 1 details"] = {"keywords": [], "is_addressed": False, "confidence": 0.0} # Placeholder, keyword matching might be hard
                self.components["subject 2 details"] = {"keywords": [], "is_addressed": False, "confidence": 0.0} # Placeholder
            elif "summarize" in self.query.lower() or "overview" in self.query.lower():
                self.components["main points"] = {"keywords": ["summary", "summarize", "overview", "main point", "key aspect"], "is_addressed": False, "confidence": 0.0}
                self.components["details"] = {"keywords": ["detail", "specific", "elaborate"], "is_addressed": False, "confidence": 0.0}
            else: # Default fallback component
                self.components["overall query"] = {"keywords": [kw.lower() for kw in self.query.split()[:5]], "is_addressed": False, "confidence": 0.0} # Use first few words of query

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Identified components: {list(self.components.keys())}")

    def update_components(self, text_to_analyze: str, round_avg_confidence: float):
        """
        Updates the status of query components based on the provided text and confidence.
        Marks components as addressed if their keywords are found in the text.
        """
        # In this version, we'll use round_avg_confidence as a proxy for the confidence
        # of the information that might address a component.
        # A more advanced version could try to link specific task confidences.
        text_lower = text_to_analyze.lower()
        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Updating components based on text (first 100 chars): {text_lower[:100]}...")

        for comp_name, comp_data in self.components.items():
            if not comp_data["is_addressed"]:
                # If keywords are defined, require all keywords for the component to be present.
                # This is a strict rule and might need adjustment (e.g., any keyword, or a percentage).
                if comp_data["keywords"]:
                    all_keywords_present = all(kw in text_lower for kw in comp_data["keywords"])
                    if all_keywords_present:
                        comp_data["is_addressed"] = True
                        comp_data["confidence"] = round_avg_confidence # Use round's average confidence
                        if self.debug_mode:
                            print(f"DEBUG [SufficiencyAnalyzer]: Component '{comp_name}' ADDRESSED by keyword match. Confidence set to {round_avg_confidence:.2f}")
                # If no keywords (e.g. generic components like "subject 1 details"), this logic won't address them.
                # This is a limitation of the current basic keyword approach for generic components.
                # For this iteration, such components might remain unaddressed unless their names/generic keywords appear.

    def calculate_sufficiency_score(self) -> Tuple[float, float, Dict[str, bool]]:
        """
        Calculates the overall sufficiency score based on component coverage and confidence.
        Returns score, coverage percentage, and status of each component.
        """
        if not self.components:
            return 0.0, 0.0, {}

        addressed_components_count = 0
        total_confidence_of_addressed = 0.0
        component_status_for_metrics: Dict[str, bool] = {}

        for comp_name, comp_data in self.components.items():
            component_status_for_metrics[comp_name] = comp_data["is_addressed"]
            if comp_data["is_addressed"]:
                addressed_components_count += 1
                total_confidence_of_addressed += comp_data["confidence"]

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Addressed components: {addressed_components_count}/{len(self.components)}")

        component_coverage_percentage = (addressed_components_count / len(self.components)) if len(self.components) > 0 else 0.0

        avg_confidence_of_addressed = (total_confidence_of_addressed / addressed_components_count) if addressed_components_count > 0 else 0.0

        # Score is a product of coverage and average confidence of what's covered.
        sufficiency_score = component_coverage_percentage * avg_confidence_of_addressed

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Coverage: {component_coverage_percentage:.2f}, Avg Confidence of Addressed: {avg_confidence_of_addressed:.2f}, Sufficiency Score: {sufficiency_score:.2f}")

        return sufficiency_score, component_coverage_percentage, component_status_for_metrics

    def get_analysis_details(self) -> Dict[str, Any]:
        """
        Returns a dictionary with the sufficiency score, coverage, and component status.
        """
        sufficiency_score, component_coverage_percentage, component_status = self.calculate_sufficiency_score()
        return {
            "sufficiency_score": sufficiency_score,
            "component_coverage_percentage": component_coverage_percentage,
            "information_components_status": component_status # Changed key name slightly to avoid conflict if used directly in RoundMetrics
        }


# Partials File: partials/minion_convergence_detector.py

# Attempt to import RoundMetrics and Valves type hints for clarity,
# but handle potential circular dependency or generation-time issues
# by using 'Any' if direct import is problematic during generation.
try:
    # Assuming valves structure will be available, or use Any
    # from .minions_valves import MinionSValves # This might not exist as a direct importable type
    ValvesType = Any # Placeholder for valve types from pipe_self.valves
except ImportError:
    RoundMetrics = Any
    ValvesType = Any

class ConvergenceDetector:
    def __init__(self, debug_mode: bool = False):
        """Initializes the ConvergenceDetector."""
        self.debug_mode = debug_mode
        if self.debug_mode:
            print("DEBUG [ConvergenceDetector]: Initialized.")

    def calculate_round_convergence_metrics(
        self,
        current_round_metric: RoundMetrics,
        previous_round_metric: Optional[RoundMetrics]
    ) -> Dict[str, Any]:
        """
        Calculates specific convergence-related metrics for the current round.
        These will be used to update the current_round_metric object.
        """
        calculated_metrics = {}

        # 1. Information Gain Rate (using new_findings_count_this_round)
        # Ensure current_round_metric has the attribute, otherwise default to 0. Useful if RoundMetrics is Any.
        calculated_metrics["information_gain_rate"] = float(getattr(current_round_metric, "new_findings_count_this_round", 0))

        # 2. Novel Findings Percentage for this round
        new_findings = getattr(current_round_metric, "new_findings_count_this_round", 0)
        duplicate_findings = getattr(current_round_metric, "duplicate_findings_count_this_round", 0)
        total_findings_this_round = new_findings + duplicate_findings

        if total_findings_this_round > 0:
            calculated_metrics["novel_findings_percentage_this_round"] = new_findings / total_findings_this_round
        else:
            calculated_metrics["novel_findings_percentage_this_round"] = 0.0

        # 3. Task Failure Rate Trend
        trend = "N/A"
        if previous_round_metric:
            current_success_rate = getattr(current_round_metric, "success_rate", 0.0)
            previous_success_rate = getattr(previous_round_metric, "success_rate", 0.0)
            tolerance = 0.05
            if current_success_rate < previous_success_rate - tolerance:
                trend = "increasing_failures"
            elif current_success_rate > previous_success_rate + tolerance:
                trend = "decreasing_failures"
            else:
                trend = "stable_failures"
        calculated_metrics["task_failure_rate_trend"] = trend

        # 4. Predicted Value of Next Round (simple heuristic)
        predicted_value = "medium"
        novelty_current_round = calculated_metrics["novel_findings_percentage_this_round"]

        if novelty_current_round > 0.5:
            predicted_value = "high"
        elif novelty_current_round < 0.1:
            predicted_value = "low"

        if trend == "increasing_failures" and predicted_value == "medium":
            predicted_value = "low"
        elif trend == "decreasing_failures" and predicted_value == "medium":
            predicted_value = "high"

        calculated_metrics["predicted_value_of_next_round"] = predicted_value

        if self.debug_mode:
            round_num_debug = getattr(current_round_metric, "round_number", "Unknown")
            print(f"DEBUG [ConvergenceDetector]: Calculated metrics for round {round_num_debug}: {calculated_metrics}")

        return calculated_metrics

    def check_for_convergence(
        self,
        current_round_metric: RoundMetrics,
        sufficiency_score: float,
        total_rounds_executed: int,
        effective_novelty_to_use: float, # New parameter for dynamic threshold
        effective_sufficiency_to_use: float, # New parameter for dynamic threshold
        valves: ValvesType, # Still needed for min_rounds_before_convergence_check, convergence_rounds_min_novelty
        all_round_metrics: List[RoundMetrics]
    ) -> Tuple[bool, str]:
        """
        Checks if convergence criteria are met using potentially dynamic thresholds.
        """
        min_rounds_for_conv_check = getattr(valves, "min_rounds_before_convergence_check", 2)
        if total_rounds_executed < min_rounds_for_conv_check:
             if self.debug_mode:
                print(f"DEBUG [ConvergenceDetector]: Skipping convergence check for round {getattr(current_round_metric, 'round_number', 'N/A')}, min rounds for convergence check not met ({total_rounds_executed}/{min_rounds_for_conv_check}).")
             return False, ""

        # Convergence criteria:
        # 1. Low novelty for a certain number of consecutive rounds (using effective_novelty_to_use).
        # 2. Sufficiency score is above a threshold (using effective_sufficiency_to_use).

        low_novelty_streak = 0
        required_streak_length = getattr(valves, "convergence_rounds_min_novelty", 2)
        # Use the passed effective_novelty_to_use instead of getattr(valves, "convergence_novelty_threshold", 0.10)

        if len(all_round_metrics) >= required_streak_length:
            is_streak = True
            for i in range(required_streak_length):
                metric_to_check = all_round_metrics[-(i+1)]

                if not hasattr(metric_to_check, 'novel_findings_percentage_this_round') or \
                   getattr(metric_to_check, 'novel_findings_percentage_this_round') >= effective_novelty_to_use:
                    is_streak = False
                    break
            if is_streak:
                low_novelty_streak = required_streak_length

        current_round_num_debug = getattr(current_round_metric, "round_number", "N/A")
        # Use the passed effective_sufficiency_to_use instead of getattr(valves, "convergence_sufficiency_threshold", 0.7)

        if self.debug_mode:
            print(f"DEBUG [ConvergenceDetector]: Checking convergence for round {current_round_num_debug}:")
            print(f"  Using Effective Novelty Threshold: {effective_novelty_to_use:.2f}, Required streak: {required_streak_length}")
            print(f"  Actual low novelty streak achieved for check: {low_novelty_streak} (needs to be >= {required_streak_length})")
            print(f"  Sufficiency score: {sufficiency_score:.2f}, Using Effective Sufficiency Threshold: {effective_sufficiency_to_use:.2f}")

        if low_novelty_streak >= required_streak_length and sufficiency_score >= effective_sufficiency_to_use:
            reason = (
                f"Convergence detected: Novelty < {effective_novelty_to_use*100:.0f}% "
                f"for {low_novelty_streak} round(s) AND "
                f"Sufficiency ({sufficiency_score:.2f}) >= {effective_sufficiency_to_use:.2f}."
            )
            if self.debug_mode:
                print(f"DEBUG [ConvergenceDetector]: {reason}")
            return True, reason

        return False, ""


# Partials File: partials/minions_pipe_method.py
from enum import Enum # Ensured Enum is present

# Removed: from .common_query_utils import QueryComplexityClassifier, QueryComplexity

# --- Content from common_query_utils.py START ---
class QueryComplexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"

class QueryComplexityClassifier:
    """Classifies query complexity based on keywords and length."""
    def __init__(self, debug_mode: bool = False):
        """Initializes the classifier with debug mode."""
        self.debug_mode = debug_mode
        # Keywords indicating complexity
        self.complex_keywords = [
            "analyze", "compare", "contrast", "summarize", "explain in detail",
            "discuss", "critique", "evaluate", "recommend", "predict", "what if",
            "how does", "why does", "implications"
        ]
        self.medium_keywords = [
            "list", "describe", "details of", "tell me about", "what are the"
        ]
        # Question words (simple ones often start fact-based questions)
        self.simple_question_starters = ["what is", "who is", "when was", "where is", "define"]

    def classify_query(self, query: str) -> QueryComplexity:
        """
        Classifies the given query into SIMPLE, MEDIUM, or COMPLEX.
        Uses rules based on keywords and word count.
        """
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Query='{query_lower}', WordCount={word_count}")

        # Rule 1: Complex Keywords
        for keyword in self.complex_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched complex keyword '{keyword}'")
                return QueryComplexity.COMPLEX

        # Rule 2: Word Count for Complex
        if word_count > 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched complex by word count (>25)")
            return QueryComplexity.COMPLEX

        # Rule 3: Word Count for Simple (and simple question starters)
        if word_count < 10:
            is_simple_starter = any(query_lower.startswith(starter) for starter in self.simple_question_starters)
            if is_simple_starter:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched simple by word count (<10) and starter.")
                return QueryComplexity.SIMPLE

        # Rule 4: Medium Keywords
        for keyword in self.medium_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched medium keyword '{keyword}'")
                return QueryComplexity.MEDIUM

        if word_count >= 10 and word_count <= 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched medium by word count (10-25)")
            return QueryComplexity.MEDIUM

        if word_count < 10: # Default for short queries not caught by simple_question_starters
             if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Defaulting short query to MEDIUM (no simple starter)")
             return QueryComplexity.MEDIUM

        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Defaulting to MEDIUM (no other rules matched clearly)")
        return QueryComplexity.MEDIUM
# --- Content from common_query_utils.py END ---


async def _call_claude_directly(valves: Any, query: str, call_claude_func: Callable) -> str:
    """Fallback to direct Claude call when no context is available"""
    return await call_claude_func(valves, f"Please answer this question: {query}")

async def _execute_minions_protocol(
    valves: Any,
    query: str,
    context: str,
    call_claude: Callable,
    call_ollama_func: Callable,
    TaskResultModel: Any
) -> str:
    """Execute the MinionS protocol"""
    conversation_log = []
    debug_log = []
    scratchpad_content = ""
    all_round_results_aggregated = []
    all_round_metrics: List[RoundMetrics] = [] # Initialize Metrics List
    global_unique_fingerprints_seen = set() # Initialize Global Fingerprint Set
    decomposition_prompts_history = []
    synthesis_prompts_history = []
    final_response = "No answer could be synthesized."
    claude_provided_final_answer = False
    total_tasks_executed_local = 0
    total_chunks_processed_for_stats = 0
    total_chunk_processing_timeouts_accumulated = 0
    synthesis_input_summary = ""
    early_stopping_reason_for_output = None # Initialize for storing stopping reason

    overall_start_time = asyncio.get_event_loop().time()

    # User query is passed directly to _execute_minions_protocol
    user_query = query # Use the passed 'query' as user_query for clarity if needed elsewhere

    # --- Performance Profile Logic ---
    current_run_max_rounds = valves.max_rounds
    current_run_base_sufficiency_thresh = valves.convergence_sufficiency_threshold
    current_run_base_novelty_thresh = valves.convergence_novelty_threshold
    current_run_simple_query_confidence_thresh = valves.simple_query_confidence_threshold
    current_run_medium_query_confidence_thresh = valves.medium_query_confidence_threshold

    profile_applied_details = [f"🧠 Performance Profile selected: {valves.performance_profile}"]

    if valves.performance_profile == "high_quality":
        current_run_max_rounds = min(valves.max_rounds + 1, 10)
        current_run_base_sufficiency_thresh = min(0.95, valves.convergence_sufficiency_threshold + 0.1)
        current_run_base_novelty_thresh = max(0.03, valves.convergence_novelty_threshold - 0.03)
        current_run_simple_query_confidence_thresh = min(0.95, valves.simple_query_confidence_threshold + 0.1)
        current_run_medium_query_confidence_thresh = min(0.95, valves.medium_query_confidence_threshold + 0.1)
        profile_applied_details.append(f"   - Applied 'high_quality' adjustments: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")
    elif valves.performance_profile == "fastest_results":
        current_run_max_rounds = max(1, valves.max_rounds - 1)
        current_run_base_sufficiency_thresh = max(0.05, valves.convergence_sufficiency_threshold - 0.1)
        current_run_base_novelty_thresh = min(0.95, valves.convergence_novelty_threshold + 0.05)
        current_run_simple_query_confidence_thresh = max(0.05, valves.simple_query_confidence_threshold - 0.1)
        current_run_medium_query_confidence_thresh = max(0.05, valves.medium_query_confidence_threshold - 0.1)
        profile_applied_details.append(f"   - Applied 'fastest_results' adjustments: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")
    else: # balanced
        profile_applied_details.append(f"   - Using 'balanced' profile: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")

    if valves.debug_mode:
        debug_log.extend(profile_applied_details)
        debug_log.append(f"🔍 **Debug Info (MinionS v0.2.0):**\n- Query: {user_query[:100]}...\n- Context length: {len(context)} chars") # Original debug line moved after profile logic
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**")


    # Instantiate Sufficiency Analyzer
    analyzer = InformationSufficiencyAnalyzer(query=user_query, debug_mode=valves.debug_mode)
    if valves.debug_mode:
        debug_log.append(f"🧠 Sufficiency Analyzer initialized for query: {user_query[:100]}...")
        debug_log.append(f"   Identified components: {list(analyzer.components.keys())}")

    # Instantiate Convergence Detector
    convergence_detector = ConvergenceDetector(debug_mode=valves.debug_mode)
    if valves.debug_mode:
        debug_log.append(f"🧠 Convergence Detector initialized.")

    # Initialize Query Complexity Classifier and Classify Query
    query_classifier = QueryComplexityClassifier(debug_mode=valves.debug_mode)
    query_complexity_level = query_classifier.classify_query(query)

    if valves.debug_mode:
        debug_log.append(f"🧠 Query classified as: {query_complexity_level.value} (Debug Mode)")
    # Optional: Add to conversation_log if you want user to see it always
    # if valves.show_conversation:
    #     conversation_log.append(f"🧠 Initial query classified as complexity: {query_complexity_level.value}")

    # --- Dynamic Threshold Initialization ---
    doc_size_category = "medium" # Default
    context_len = len(context)
    if context_len < valves.doc_size_small_char_limit:
        doc_size_category = "small"
    elif context_len > valves.doc_size_large_char_start:
        doc_size_category = "large"

    if valves.debug_mode:
        debug_log.append(f"🧠 Document size category: {doc_size_category} (Length: {context_len} chars)")

    # Initialize effective thresholds with base values (now from current_run_... variables)
    effective_sufficiency_threshold = current_run_base_sufficiency_thresh
    effective_novelty_threshold = current_run_base_novelty_thresh
    # Base confidence thresholds for simple/medium queries will use current_run_... variables where they are applied.

    if valves.debug_mode:
        debug_log.append(f"🧠 Initial effective thresholds (after profile adjustments): Sufficiency={effective_sufficiency_threshold:.2f}, Novelty={effective_novelty_threshold:.2f}")
        debug_log.append(f"   Effective Simple Confidence Thresh (base for adaptation)={current_run_simple_query_confidence_thresh:.2f}, Medium Confidence Thresh (base for adaptation)={current_run_medium_query_confidence_thresh:.2f}")

    if valves.enable_adaptive_thresholds:
        if valves.debug_mode:
            debug_log.append(f"🧠 Adaptive thresholds ENABLED. Applying query/doc modifiers...")

        # Apply query complexity modifiers to sufficiency and novelty
        if query_complexity_level == QueryComplexity.SIMPLE:
            effective_sufficiency_threshold += valves.sufficiency_modifier_simple_query
            effective_novelty_threshold += valves.novelty_modifier_simple_query
        elif query_complexity_level == QueryComplexity.COMPLEX:
            effective_sufficiency_threshold += valves.sufficiency_modifier_complex_query
            effective_novelty_threshold += valves.novelty_modifier_complex_query

        # Clamp all thresholds to sensible ranges (e.g., 0.05 to 0.95)
        effective_sufficiency_threshold = max(0.05, min(0.95, effective_sufficiency_threshold))
        effective_novelty_threshold = max(0.05, min(0.95, effective_novelty_threshold))

        if valves.debug_mode:
            debug_log.append(f"   After query complexity mods: Eff.Sufficiency={effective_sufficiency_threshold:.2f}, Eff.Novelty={effective_novelty_threshold:.2f}")

    chunks = create_chunks(context, valves.chunk_size, valves.max_chunks)
    if not chunks and context:
        return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size."
    if not chunks and not context:
        conversation_log.append("ℹ️ No context or chunks to process with MinionS. Attempting direct call.")
        start_time_claude = 0
        if valves.debug_mode: 
            start_time_claude = asyncio.get_event_loop().time()
        try:
            final_response = await _call_claude_directly(valves, query, call_claude_func=call_claude)
            if valves.debug_mode:
                end_time_claude = asyncio.get_event_loop().time()
                time_taken_claude = end_time_claude - start_time_claude
                debug_log.append(f"⏱️ Claude direct call took {time_taken_claude:.2f}s. (Debug Mode)")
            output_parts = []
            if valves.show_conversation:
                output_parts.append("## 🗣️ MinionS Collaboration (Direct Call)")
                output_parts.extend(conversation_log)
                output_parts.append("---")
            if valves.debug_mode:
                output_parts.append("### 🔍 Debug Log")
                output_parts.extend(debug_log)
                output_parts.append("---")
            output_parts.append(f"## 🎯 Final Answer (Direct)\n{final_response}")
            return "\n".join(output_parts)
        except Exception as e:
            return f"❌ **Error in direct Claude call:** {str(e)}"

    total_chunks_processed_for_stats = len(chunks)

    # Initialize effective confidence threshold variables to store them for the performance report
    final_effective_simple_conf_thresh = current_run_simple_query_confidence_thresh
    final_effective_medium_conf_thresh = current_run_medium_query_confidence_thresh

    for current_round in range(current_run_max_rounds): # Use current_run_max_rounds
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {current_round + 1}/{current_run_max_rounds}... (Debug Mode)**") # Use current_run_max_rounds
        
        if valves.show_conversation:
            conversation_log.append(f"### 🎯 Round {current_round + 1}/{current_run_max_rounds} - Task Decomposition Phase") # Use current_run_max_rounds

        # Call the new decompose_task function
        # Note: now returns three values instead of two
        tasks, claude_response_for_decomposition, decomposition_prompt = await decompose_task(
            valves=valves,
            call_claude_func=call_claude,
            query=query,
            scratchpad_content=scratchpad_content,
            num_chunks=len(chunks),
            max_tasks_per_round=valves.max_tasks_per_round,
            current_round=current_round + 1,
            conversation_log=conversation_log,
            debug_log=debug_log
        )
        
        # Store the decomposition prompt in history
        if decomposition_prompt:  # Only add if not empty (error case)
            decomposition_prompts_history.append(decomposition_prompt)
        
        # Handle Claude communication errors from decompose_task
        if claude_response_for_decomposition.startswith("CLAUDE_ERROR:"):
            error_message = claude_response_for_decomposition.replace("CLAUDE_ERROR: ", "")
            final_response = f"MinionS protocol failed during task decomposition: {error_message}"
            break

        # Log the raw Claude response if conversation is shown
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round + 1}):**\n{claude_response_for_decomposition}\n")

        # Check for "FINAL ANSWER READY."
        if "FINAL ANSWER READY." in claude_response_for_decomposition:
            final_response = claude_response_for_decomposition.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_provided_final_answer = True
            early_stopping_reason_for_output = "Claude provided FINAL ANSWER READY." # Explicitly set reason
            if valves.show_conversation: # This log already exists
                conversation_log.append(f"**🤖 Claude indicates final answer is ready in round {current_round + 1}.**")
            scratchpad_content += f"\n\n**Round {current_round + 1}:** Claude provided final answer. Stopping." # Added "Stopping."
            break

        if not tasks:
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Claude provided no new tasks in round {current_round + 1}. Proceeding to final synthesis.**")
            break
        
        total_tasks_executed_local += len(tasks)
        
        if valves.show_conversation:
            conversation_log.append(f"### ⚡ Round {current_round + 1} - Parallel Execution Phase (Processing {len(chunks)} chunks for {len(tasks)} tasks)")
        
        execution_details = await execute_tasks_on_chunks(
            tasks, chunks, conversation_log if valves.show_conversation else debug_log, 
            current_round + 1, valves, call_ollama_func, TaskResultModel
        )
        current_round_task_results = execution_details["results"]
        round_chunk_attempts = execution_details["total_chunk_processing_attempts"]
        round_chunk_timeouts = execution_details["total_chunk_processing_timeouts"]

        # Process Metrics After execute_tasks_on_chunks
        raw_metrics_data = execution_details.get("round_metrics_data")

        # Extract and Calculate Confidence Metrics
        confidence_data = execution_details.get("confidence_metrics_data")
        task_confidences = confidence_data.get("task_confidences", []) if confidence_data else []

        round_avg_numeric_confidence = 0.0
        if task_confidences:
            total_confidence_sum = sum(tc['avg_numeric_confidence'] for tc in task_confidences if tc.get('contributing_successful_chunks', 0) > 0)
            num_successful_tasks_with_confidence = sum(1 for tc in task_confidences if tc.get('contributing_successful_chunks', 0) > 0)
            if num_successful_tasks_with_confidence > 0:
                round_avg_numeric_confidence = total_confidence_sum / num_successful_tasks_with_confidence

        round_confidence_distribution = confidence_data.get("round_confidence_distribution", {"HIGH": 0, "MEDIUM": 0, "LOW": 0}) if confidence_data else {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        # Determine Confidence Trend (before creating current RoundMetrics object)
        confidence_trend = "N/A"
        if all_round_metrics: # Check if there are previous rounds' metrics
            previous_round_metric = all_round_metrics[-1]
            previous_avg_confidence = previous_round_metric.avg_confidence_score
            diff = round_avg_numeric_confidence - previous_avg_confidence
            if diff > 0.05:
                confidence_trend = "improving"
            elif diff < -0.05:
                confidence_trend = "declining"
            else:
                confidence_trend = "stable"

        if raw_metrics_data:
            # Process Findings for Redundancy Metrics
            current_round_new_findings = 0
            current_round_duplicate_findings = 0
            # current_round_fingerprints_seen_this_round = set() # Not strictly needed for plan's definition

            task_execution_results = execution_details.get("results", [])
            for task_result in task_execution_results:
                if task_result.get("status") == "success" and task_result.get("detailed_findings"):
                    for finding in task_result["detailed_findings"]:
                        fingerprint = finding.get("fingerprint")
                        if fingerprint:
                            # Check against global set first for duplicates from previous rounds or earlier in this round
                            if fingerprint in global_unique_fingerprints_seen:
                                current_round_duplicate_findings += 1
                            else:
                                current_round_new_findings += 1
                                global_unique_fingerprints_seen.add(fingerprint)
                            # current_round_fingerprints_seen_this_round.add(fingerprint)


            total_findings_this_round = current_round_new_findings + current_round_duplicate_findings
            redundancy_percentage_this_round = (current_round_duplicate_findings / total_findings_this_round) * 100 if total_findings_this_round > 0 else 0.0

            try: # Add a try-except block for robustness when creating RoundMetrics
                round_metric = RoundMetrics(
                    round_number=raw_metrics_data["round_number"],
                    tasks_executed=raw_metrics_data["tasks_executed"],
                    task_success_count=raw_metrics_data["task_success_count"],
                    task_failure_count=raw_metrics_data["task_failure_count"],
                    avg_chunk_processing_time_ms=raw_metrics_data["avg_chunk_processing_time_ms"],
                    # total_unique_findings_count=0,  # Placeholder for Iteration 1 REMOVED
                    execution_time_ms=raw_metrics_data["execution_time_ms"],
                    success_rate=raw_metrics_data["success_rate"],
                    # Add new confidence fields
                    avg_confidence_score=round_avg_numeric_confidence,
                    confidence_distribution=round_confidence_distribution,
                    confidence_trend=confidence_trend,
                    # Add new redundancy fields
                    new_findings_count_this_round=current_round_new_findings,
                    duplicate_findings_count_this_round=current_round_duplicate_findings,
                    redundancy_percentage_this_round=redundancy_percentage_this_round,
                    total_unique_findings_count=len(global_unique_fingerprints_seen)
                    # Sufficiency fields will be added below
                )
                all_round_metrics.append(round_metric) # Add before sufficiency update

                # --> Sufficiency Analysis <--
                metric_to_update = round_metric # This is the one we just added
                if valves.debug_mode:
                    debug_log.append(f"   [Debug] Updating Sufficiency Analyzer with scratchpad content for round {current_round + 1}...")

                analyzer.update_components(text_to_analyze=scratchpad_content, round_avg_confidence=metric_to_update.avg_confidence_score)
                sufficiency_details = analyzer.get_analysis_details()

                metric_to_update.sufficiency_score = sufficiency_details["sufficiency_score"]
                metric_to_update.component_coverage_percentage = sufficiency_details["component_coverage_percentage"]
                metric_to_update.information_components = sufficiency_details["information_components_status"]

                if valves.debug_mode:
                    debug_log.append(f"   [Debug] Sufficiency for round {current_round + 1}: Score={metric_to_update.sufficiency_score:.2f}, Coverage={metric_to_update.component_coverage_percentage:.2f}")
                    debug_log.append(f"   [Debug] Component Status: {metric_to_update.information_components}")

                # --> Convergence Detection Calculations (after sufficiency is updated) <--
                if metric_to_update: # Ensure we have the current round's metric object
                    previous_round_metric_obj = all_round_metrics[-2] if len(all_round_metrics) > 1 else None

                    convergence_calcs = convergence_detector.calculate_round_convergence_metrics(
                        current_round_metric=metric_to_update,
                        previous_round_metric=previous_round_metric_obj
                    )

                    metric_to_update.information_gain_rate = convergence_calcs.get("information_gain_rate", 0.0)
                    metric_to_update.novel_findings_percentage_this_round = convergence_calcs.get("novel_findings_percentage_this_round", 0.0)
                    metric_to_update.task_failure_rate_trend = convergence_calcs.get("task_failure_rate_trend", "N/A")
                    metric_to_update.predicted_value_of_next_round = convergence_calcs.get("predicted_value_of_next_round", "N/A")
                    # convergence_detected_this_round is set by check_for_convergence below

                    if valves.debug_mode:
                        debug_log.append(f"   [Debug] Convergence Detector calculated for round {metric_to_update.round_number}: InfoGain={metric_to_update.information_gain_rate:.0f}, Novelty={metric_to_update.novel_findings_percentage_this_round:.2%}, FailTrend={metric_to_update.task_failure_rate_trend}, NextRoundValue={metric_to_update.predicted_value_of_next_round}")

                # Format and append metrics summary (now includes redundancy, sufficiency, AND convergence calcs)
                component_status_summary = {k: ('Met' if v else 'Not Met') for k,v in metric_to_update.information_components.items()}
                metrics_summary = (
                    f"**📊 Round {metric_to_update.round_number} Metrics:**\n"
                    f"  - Tasks Executed: {metric_to_update.tasks_executed}, Success Rate: {metric_to_update.success_rate:.2%}\n"
                    f"  - Task Counts (S/F): {metric_to_update.task_success_count}/{metric_to_update.task_failure_count}\n"
                    f"  - Findings (New/Dup): {metric_to_update.new_findings_count_this_round}/{metric_to_update.duplicate_findings_count_this_round}, Total Unique: {metric_to_update.total_unique_findings_count}\n"
                    f"  - Redundancy This Round: {metric_to_update.redundancy_percentage_this_round:.1f}%\n"
                    f"  - Avg Confidence: {metric_to_update.avg_confidence_score:.2f} ({metric_to_update.confidence_trend})\n"
                    f"  - Confidence Dist (H/M/L): {metric_to_update.confidence_distribution.get('HIGH',0)}/{metric_to_update.confidence_distribution.get('MEDIUM',0)}/{metric_to_update.confidence_distribution.get('LOW',0)}\n"
                    f"  - Sufficiency Score: {metric_to_update.sufficiency_score:.2f}, Info Coverage: {metric_to_update.component_coverage_percentage:.2%}\n"
                    f"  - Components Status: {component_status_summary}\n"
                    f"  - Info Gain Rate: {metric_to_update.information_gain_rate:.0f}, Novelty This Round: {metric_to_update.novel_findings_percentage_this_round:.1%}\n"
                    f"  - Task Fail Trend: {metric_to_update.task_failure_rate_trend}, Predicted Next Round Value: {metric_to_update.predicted_value_of_next_round}\n"
                    f"  - Converged This Round: {'Yes' if metric_to_update.convergence_detected_this_round else 'No'}\n" # Will be updated by convergence check later
                    f"  - Round Time: {metric_to_update.execution_time_ms:.0f} ms, Avg Chunk Time: {metric_to_update.avg_chunk_processing_time_ms:.0f} ms"
                )
                scratchpad_content += f"\n\n{metrics_summary}"
                if valves.show_conversation:
                    conversation_log.append(metrics_summary)

            except KeyError as e:
                if valves.debug_mode:
                    debug_log.append(f"⚠️ **Metrics Error:** Missing key {e} in round_metrics_data for round {current_round + 1}. Skipping metrics for this round.")
            except Exception as e: # Catch any other validation error from Pydantic
                 if valves.debug_mode:
                    debug_log.append(f"⚠️ **Metrics Error:** Could not create RoundMetrics object for round {current_round + 1} due to {type(e).__name__}: {e}. Skipping metrics for this round.")


        if round_chunk_attempts > 0:
            timeout_percentage_this_round = (round_chunk_timeouts / round_chunk_attempts) * 100
            log_msg_timeout_stat = f"**📈 Round {current_round + 1} Local LLM Timeout Stats:** {round_chunk_timeouts}/{round_chunk_attempts} chunk calls timed out ({timeout_percentage_this_round:.1f}%)."
            if valves.show_conversation: 
                conversation_log.append(log_msg_timeout_stat)
            if valves.debug_mode: 
                debug_log.append(log_msg_timeout_stat)

            if timeout_percentage_this_round >= valves.max_round_timeout_failure_threshold_percent:
                warning_msg = f"⚠️ **Warning:** Round {current_round + 1} exceeded local LLM timeout threshold of {valves.max_round_timeout_failure_threshold_percent}%. Results from this round may be incomplete or unreliable."
                if valves.show_conversation: 
                    conversation_log.append(warning_msg)
                if valves.debug_mode: 
                    debug_log.append(warning_msg)
                scratchpad_content += f"\n\n**Note from Round {current_round + 1}:** High percentage of local model timeouts ({timeout_percentage_this_round:.1f}%) occurred, results for this round may be partial."
        
        round_summary_for_scratchpad_parts = []
        for r_val in current_round_task_results:
            status_icon = "✅" if r_val['status'] == 'success' else ("⏰" if 'timeout' in r_val['status'] else "❓")
            summary_text = f"- {status_icon} Task: {r_val['task']}, Result: {r_val['result'][:200]}..." if r_val['status'] == 'success' else f"- {status_icon} Task: {r_val['task']}, Status: {r_val['result']}"
            round_summary_for_scratchpad_parts.append(summary_text)
        
        if round_summary_for_scratchpad_parts:
            scratchpad_content += f"\n\n**Results from Round {current_round + 1}:**\n" + "\n".join(round_summary_for_scratchpad_parts)
        
        all_round_results_aggregated.extend(current_round_task_results)
        total_chunk_processing_timeouts_accumulated += round_chunk_timeouts

        # Placeholder for Sufficiency-Based Stopping Logic (Debug)
        # This is checked *before* other early stopping conditions like confidence thresholds.
        # The 'metric_to_update' variable should be the most up-to-date version of the current round's metrics.
        # It now includes sufficiency and initial convergence calculation fields.

        # --- First Round Novelty Adjustment (occurs only after round 0 processing) ---
        if current_round == 0 and valves.enable_adaptive_thresholds and 'metric_to_update' in locals() and metric_to_update:
            first_round_novelty_perc = metric_to_update.novel_findings_percentage_this_round
            if first_round_novelty_perc > valves.first_round_high_novelty_threshold:
                original_eff_sufficiency_before_1st_round_adj = effective_sufficiency_threshold
                effective_sufficiency_threshold += valves.sufficiency_modifier_high_first_round_novelty
                effective_sufficiency_threshold = max(0.05, min(0.95, effective_sufficiency_threshold)) # Clamp again
                if valves.debug_mode:
                    debug_log.append(
                        f"🧠 High first round novelty ({first_round_novelty_perc:.2%}) detected. "
                        f"Adjusting effective sufficiency threshold from {original_eff_sufficiency_before_1st_round_adj:.2f} to {effective_sufficiency_threshold:.2f}."
                    )

        if valves.debug_mode and 'metric_to_update' in locals() and metric_to_update:
            # Placeholder Debug for Sufficiency (already exists, using the dynamically adjusted threshold now)
            # This hypothetical threshold is just for this debug log, actual check uses effective_sufficiency_threshold
            debug_hypothetical_sufficiency_thresh_for_log = 0.75
            if metric_to_update.sufficiency_score >= debug_hypothetical_sufficiency_thresh_for_log:
                debug_log.append(
                    f"   [Debug Placeholder] Sufficiency score {metric_to_update.sufficiency_score:.2f} >= "
                    f"{debug_hypothetical_sufficiency_thresh_for_log} (hypothetical debug value). "
                    f"Effective sufficiency for convergence check is {effective_sufficiency_threshold:.2f}."
                )
            else:
                debug_log.append(
                    f"   [Debug Placeholder] Sufficiency score {metric_to_update.sufficiency_score:.2f} < "
                    f"{debug_hypothetical_sufficiency_thresh_for_log} (hypothetical debug value). "
                    f"Effective sufficiency for convergence check is {effective_sufficiency_threshold:.2f}."
                )

        # --> Convergence Check (for early stopping) <--
        # This comes before the original early stopping logic. If convergence is met, we stop.
        if 'metric_to_update' in locals() and metric_to_update and valves.enable_early_stopping:
            converged, convergence_reason = convergence_detector.check_for_convergence(
                current_round_metric=metric_to_update,
                sufficiency_score=metric_to_update.sufficiency_score, # Base sufficiency from analyzer
                total_rounds_executed=current_round + 1,
                effective_novelty_to_use=effective_novelty_threshold, # Pass calculated value
                effective_sufficiency_to_use=effective_sufficiency_threshold, # Pass calculated value
                valves=valves,
                all_round_metrics=all_round_metrics
            )
            if converged:
                metric_to_update.convergence_detected_this_round = True
                # Update the metrics_summary in scratchpad and conversation_log one last time with Converged=Yes
                # This is a bit repetitive but ensures the log reflects the final state that caused the stop.
                component_status_summary = {k: ('Met' if v else 'Not Met') for k,v in metric_to_update.information_components.items()}
                updated_metrics_summary_for_convergence_stop = (
                    f"**📊 Round {metric_to_update.round_number} Metrics (Final Update Before Convergence Stop):**\n"
                    f"  - Tasks Executed: {metric_to_update.tasks_executed}, Success Rate: {metric_to_update.success_rate:.2%}\n"
                    f"  - Task Counts (S/F): {metric_to_update.task_success_count}/{metric_to_update.task_failure_count}\n"
                    f"  - Findings (New/Dup): {metric_to_update.new_findings_count_this_round}/{metric_to_update.duplicate_findings_count_this_round}, Total Unique: {metric_to_update.total_unique_findings_count}\n"
                    f"  - Redundancy This Round: {metric_to_update.redundancy_percentage_this_round:.1f}%\n"
                    f"  - Avg Confidence: {metric_to_update.avg_confidence_score:.2f} ({metric_to_update.confidence_trend})\n"
                    f"  - Confidence Dist (H/M/L): {metric_to_update.confidence_distribution.get('HIGH',0)}/{metric_to_update.confidence_distribution.get('MEDIUM',0)}/{metric_to_update.confidence_distribution.get('LOW',0)}\n"
                    f"  - Sufficiency Score: {metric_to_update.sufficiency_score:.2f}, Info Coverage: {metric_to_update.component_coverage_percentage:.2%}\n"
                    f"  - Components Status: {component_status_summary}\n"
                    f"  - Info Gain Rate: {metric_to_update.information_gain_rate:.0f}, Novelty This Round: {metric_to_update.novel_findings_percentage_this_round:.1%}\n"
                    f"  - Task Fail Trend: {metric_to_update.task_failure_rate_trend}, Predicted Next Round Value: {metric_to_update.predicted_value_of_next_round}\n"
                    f"  - Converged This Round: {'Yes' if metric_to_update.convergence_detected_this_round else 'No'}\n"
                    f"  - Round Time: {metric_to_update.execution_time_ms:.0f} ms, Avg Chunk Time: {metric_to_update.avg_chunk_processing_time_ms:.0f} ms"
                )
                scratchpad_content += f"\n\n{updated_metrics_summary_for_convergence_stop}" # Append the final metrics to scratchpad
                if valves.show_conversation: # Replace the last metrics log with the fully updated one
                    if conversation_log and conversation_log[-1].startswith("**📊 Round"): conversation_log[-1] = updated_metrics_summary_for_convergence_stop
                    else: conversation_log.append(updated_metrics_summary_for_convergence_stop)

                early_stopping_reason_for_output = convergence_reason
                if valves.show_conversation:
                    conversation_log.append(f"**⚠️ Early Stopping Triggered (Convergence):** {convergence_reason}")
                if valves.debug_mode:
                    debug_log.append(f"**⚠️ Early Stopping Triggered (Convergence):** {convergence_reason} (Debug Mode)")
                scratchpad_content += f"\n\n**EARLY STOPPING (Convergence Round {current_round + 1}):** {convergence_reason}"
                if valves.debug_mode:
                     debug_log.append(f"**🏁 Breaking loop due to convergence in Round {current_round + 1}. (Debug Mode)**")
                break # Exit the round loop

        # Original Early Stopping Logic (Confidence-based)
        # This will only be reached if convergence was NOT met and we didn't break above.
        # Note: 'round_metric' is the original metric object from raw_metrics_data,
        # 'metric_to_update' is the same object, but after being updated with sufficiency.
        # So, using 'metric_to_update' here for consistency if we were to integrate sufficiency into this logic.
        # However, the current early stopping is based on avg_confidence_score which is set before sufficiency.
        # For now, we keep `round_metric` for the existing logic as it was originally.
        # If sufficiency were to be a primary driver for early stopping, this would need refactoring.
        if valves.enable_early_stopping and round_metric: # Ensure round_metric exists
            stop_early = False
            stopping_reason = ""

            # Ensure we've met the minimum number of rounds
            if (current_round + 1) >= valves.min_rounds_before_stopping:
                current_avg_confidence = round_metric.avg_confidence_score # This is from the current round's raw metrics

                # Determine effective confidence threshold for current query type
                current_query_type_base_confidence_threshold = 0.0 # This will be set to the current_run_... value
                threshold_name_for_log = "N/A"
                # Store the calculated effective confidence threshold for the performance report
                # Initialize with base, then adapt if needed
                effective_confidence_threshold_for_query_this_check = 0.0


                if query_complexity_level == QueryComplexity.SIMPLE:
                    current_query_type_base_confidence_threshold = current_run_simple_query_confidence_thresh # Use current_run_
                    threshold_name_for_log = "Simple Query Confidence"
                    effective_confidence_threshold_for_query_this_check = current_query_type_base_confidence_threshold
                    if valves.enable_adaptive_thresholds:
                        if doc_size_category == "small":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_small_doc
                        elif doc_size_category == "large":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_large_doc
                        effective_confidence_threshold_for_query_this_check = max(0.05, min(0.95, effective_confidence_threshold_for_query_this_check))
                    final_effective_simple_conf_thresh = effective_confidence_threshold_for_query_this_check # Store for report

                elif query_complexity_level == QueryComplexity.MEDIUM:
                    current_query_type_base_confidence_threshold = current_run_medium_query_confidence_thresh # Use current_run_
                    threshold_name_for_log = "Medium Query Confidence"
                    effective_confidence_threshold_for_query_this_check = current_query_type_base_confidence_threshold
                    if valves.enable_adaptive_thresholds:
                        if doc_size_category == "small":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_small_doc
                        elif doc_size_category == "large":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_large_doc
                        effective_confidence_threshold_for_query_this_check = max(0.05, min(0.95, effective_confidence_threshold_for_query_this_check))
                    final_effective_medium_conf_thresh = effective_confidence_threshold_for_query_this_check # Store for report

                if valves.debug_mode and query_complexity_level != QueryComplexity.COMPLEX:
                     debug_log.append(f"   [Debug] Adaptive Confidence Check: QueryType={query_complexity_level.value}, BaseThreshForProfile={current_query_type_base_confidence_threshold:.2f}, EffectiveThreshForStopCheck={effective_confidence_threshold_for_query_this_check:.2f} (DocSize: {doc_size_category})")

                if query_complexity_level != QueryComplexity.COMPLEX and current_avg_confidence >= effective_confidence_threshold_for_query_this_check:
                    stop_early = True
                    stopping_reason = (
                        f"{query_complexity_level.value} query confidence ({current_avg_confidence:.2f}) "
                        f"met/exceeded effective threshold ({effective_confidence_threshold_for_query_this_check:.2f}) "
                        f"after round {current_round + 1}."
                    )
                # No specific confidence-based early stopping rule for COMPLEX queries; they run max_rounds or until convergence.

            if stop_early:
                if valves.show_conversation:
                    conversation_log.append(f"**⚠️ Early Stopping Triggered:** {stopping_reason}")
                if valves.debug_mode:
                    debug_log.append(f"**⚠️ Early Stopping Triggered:** {stopping_reason} (Debug Mode)")

                early_stopping_reason_for_output = stopping_reason # Store it for final output
                scratchpad_content += f"\n\n**EARLY STOPPING TRIGGERED (Round {current_round + 1}):** {stopping_reason}"
                # Add a final log message before breaking, as the "Completed Round" message will be skipped
                if valves.debug_mode:
                     debug_log.append(f"**🏁 Breaking loop due to early stopping in Round {current_round + 1}. (Debug Mode)**")
                break # Exit the round loop

        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {current_round + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**")

        if current_round == current_run_max_rounds - 1: # Use current_run_max_rounds
            if valves.show_conversation:
                conversation_log.append(f"**🏁 Reached max rounds ({current_run_max_rounds}). Proceeding to final synthesis.**") # Use current_run_max_rounds

    if not claude_provided_final_answer:
        if valves.show_conversation:
            conversation_log.append("\n### 🔄 Final Synthesis Phase")
        if not all_round_results_aggregated:
            final_response = "No information was gathered from the document by local models across the rounds."
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Claude (Synthesis):** {final_response}")
        else:
            synthesis_input_summary = "\n".join([f"- Task: {r['task']}\n  Result: {r['result']}" for r in all_round_results_aggregated if r['status'] == 'success'])
            if not synthesis_input_summary:
                synthesis_input_summary = "No definitive information was found by local models. The original query was: " + query
            
            synthesis_prompt = get_minions_synthesis_claude_prompt(query, synthesis_input_summary, valves)
            synthesis_prompts_history.append(synthesis_prompt)
            
            start_time_claude_synth = 0
            if valves.debug_mode:
                start_time_claude_synth = asyncio.get_event_loop().time()
            try:
                final_response = await call_claude(valves, synthesis_prompt)
                if valves.debug_mode:
                    end_time_claude_synth = asyncio.get_event_loop().time()
                    time_taken_claude_synth = end_time_claude_synth - start_time_claude_synth
                    debug_log.append(f"⏱️ Claude call (Final Synthesis) took {time_taken_claude_synth:.2f}s. (Debug Mode)")
                if valves.show_conversation:
                    conversation_log.append(f"**🤖 Claude (Final Synthesis):**\n{final_response}")
            except Exception as e:
                if valves.show_conversation:
                    conversation_log.append(f"❌ Error during final synthesis: {e}")
                final_response = "Error during final synthesis. Raw findings might be available in conversation log."
    
    output_parts = []
    if valves.show_conversation:
        output_parts.append("## 🗣️ MinionS Collaboration (Multi-Round)")
        output_parts.extend(conversation_log)
        output_parts.append("---")
    if valves.debug_mode:
        output_parts.append("### 🔍 Debug Log")
        output_parts.extend(debug_log)
        output_parts.append("---")
    output_parts.append(f"## 🎯 Final Answer")
    output_parts.append(final_response)

    summary_for_stats = synthesis_input_summary if not claude_provided_final_answer else scratchpad_content

    stats = calculate_token_savings(
        decomposition_prompts_history, synthesis_prompts_history,
        summary_for_stats, final_response,
        len(context), len(query), total_chunks_processed_for_stats, total_tasks_executed_local
    )
    
    # Override the total_rounds if needed to show actual rounds executed
    actual_rounds_executed = len(decomposition_prompts_history) # Number of rounds for which decomposition prompts were made
    # If loop broke early, current_round might be less than actual_rounds_executed -1
    # If no decomposition prompts (e.g. direct call, or error before first decomp), then 0.
    # If loop completed, actual_rounds_executed should be current_run_max_rounds (if prompts were made each round)
    # Or, if loop broke, it's the number of rounds that *started* decomposition.
    # A simple way: if all_round_metrics exists, it's len(all_round_metrics)
    if all_round_metrics: # This is a more reliable count of rounds that completed metric generation
        actual_rounds_executed = len(all_round_metrics)
    elif actual_rounds_executed == 0 and current_round >=0: # Fallback if decomp history is empty but loop ran
         actual_rounds_executed = min(current_round + 1, current_run_max_rounds)


    # --- Performance Report Section ---
    performance_report_parts = ["\n## 📝 Performance Report"]
    performance_report_parts.append(f"- **Total rounds executed:** {actual_rounds_executed} / {current_run_max_rounds} (Profile Max)")
    performance_report_parts.append(f"- **Stopping reason:** {early_stopping_reason_for_output if early_stopping_reason_for_output else 'Max rounds reached or no further tasks.'}")

    last_metric = all_round_metrics[-1] if all_round_metrics else None
    if last_metric:
        performance_report_parts.append(f"- **Final Sufficiency Score:** {getattr(last_metric, 'sufficiency_score', 'N/A'):.2f}")
        performance_report_parts.append(f"- **Final Component Coverage:** {getattr(last_metric, 'component_coverage_percentage', 'N/A'):.2%}")
        performance_report_parts.append(f"- **Final Information Components Status:** {str(getattr(last_metric, 'information_components', {}))}")
        performance_report_parts.append(f"- **Final Convergence Detected:** {'Yes' if getattr(last_metric, 'convergence_detected_this_round', False) else 'No'}")
    else:
        performance_report_parts.append("- *No round metrics available for final values.*")

    performance_report_parts.append("- **Effective Thresholds Used (Final Values):**")
    performance_report_parts.append(f"  - Sufficiency (for convergence): {effective_sufficiency_threshold:.2f}")
    performance_report_parts.append(f"  - Novelty (for convergence): {effective_novelty_threshold:.2f}")
    if query_complexity_level == QueryComplexity.SIMPLE:
        performance_report_parts.append(f"  - Confidence (for simple query early stop): {final_effective_simple_conf_thresh:.2f}")
    elif query_complexity_level == QueryComplexity.MEDIUM:
        performance_report_parts.append(f"  - Confidence (for medium query early stop): {final_effective_medium_conf_thresh:.2f}")
    else: # COMPLEX
        performance_report_parts.append(f"  - Confidence (early stopping not applicable for COMPLEX queries based on this threshold type)")
    
    performance_report_parts.append(f"- **Performance Profile Applied:** {valves.performance_profile}")
    performance_report_parts.append(f"- **Adaptive Thresholds Enabled:** {'Yes' if valves.enable_adaptive_thresholds else 'No'}")
    performance_report_parts.append(f"- **Document Size Category:** {doc_size_category}")


    output_parts.extend(performance_report_parts)
    if valves.debug_mode:
        debug_log.append("\n--- Performance Report (Debug Copy) ---")
        debug_log.extend(performance_report_parts)
        debug_log.append("--- End Performance Report (Debug Copy) ---")

    output_parts.append(f"\n## 📊 MinionS Efficiency Stats (v0.2.0)")
    output_parts.append(f"- **Protocol:** MinionS (Multi-Round)")
    output_parts.append(f"- **Query Complexity:** {query_complexity_level.value}")
    output_parts.append(f"- **Rounds executed (Profile Max):** {actual_rounds_executed}/{current_run_max_rounds}") # Use current_run_max_rounds
    output_parts.append(f"- **Total tasks for local LLM:** {stats['total_tasks_executed_local']}")

    # --- Explicitly define variables for the MinionS Efficiency Stats block ---
    # Ensure 'all_round_results_aggregated' is the correct list of task results.
    # And 'valves' and 'debug_log' are assumed to be accessible in this scope for the warning.

    explicit_total_successful_tasks = 0
    explicit_tasks_with_any_timeout = 0 # Renamed for clarity from tasks_with_any_timeout

    if 'all_round_results_aggregated' in locals() and isinstance(all_round_results_aggregated, list):
        explicit_total_successful_tasks = len([
            r for r in all_round_results_aggregated if isinstance(r, dict) and r.get('status') == 'success'
        ])
        explicit_tasks_with_any_timeout = len([
            r for r in all_round_results_aggregated if isinstance(r, dict) and r.get('status') == 'timeout_all_chunks'
        ])
    else:
        # This case should ideally not happen if the protocol ran correctly.
        # Adding a log if debug_mode is on and valves is accessible.
        if 'valves' in locals() and hasattr(valves, 'debug_mode') and valves.debug_mode and 'debug_log' in locals():
            debug_log.append("⚠️ Warning: 'all_round_results_aggregated' not found or not a list when calculating final efficiency stats.")
    # --- End of explicit definitions ---

    output_parts.append(f"- **Successful tasks (local):** {explicit_total_successful_tasks}")
    output_parts.append(f"- **Tasks where all chunks timed out (local):** {explicit_tasks_with_any_timeout}")
    output_parts.append(f"- **Total individual chunk processing timeouts (local):** {total_chunk_processing_timeouts_accumulated}")
    output_parts.append(f"- **Chunks processed per task (local):** {stats['total_chunks_processed_local'] if stats['total_tasks_executed_local'] > 0 else 0}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    if early_stopping_reason_for_output:
        output_parts.append(f"- **Early Stopping Triggered:** {early_stopping_reason_for_output}")
    output_parts.append(f"\n## 💰 Token Savings Analysis (Claude: {valves.remote_model})")
    output_parts.append(f"- **Traditional single call (est.):** ~{stats['traditional_tokens_claude']:,} tokens")
    output_parts.append(f"- **MinionS multi-round (Claude only):** ~{stats['minions_tokens_claude']:,} tokens")
    output_parts.append(f"- **💰 Est. Claude Token savings:** ~{stats['percentage_savings_claude']:.1f}%")
    
    return "\n".join(output_parts)

async def minions_pipe_method(
    pipe_self: Any,
    body: dict,
    __user__: dict,
    __request__: Request,
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "minions-claude",
) -> str:
    """Execute the MinionS protocol with Claude"""
    try:
        # Validate configuration
        if not pipe_self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key (and Ollama settings if applicable) in the function settings."

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

        # If no context, make a direct call to Claude
        if not context:
            direct_response = await _call_claude_directly(pipe_self.valves, user_query, call_claude_func=call_claude)
            return (
                "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                + direct_response
            )

        # Execute the MinionS protocol with correct parameter names
        result: str = await _execute_minions_protocol(
            pipe_self.valves, 
            user_query, 
            context, 
            call_claude,    # Changed from call_claude_func
            call_ollama,    # Changed from call_ollama_func
            TaskResult      # Changed from TaskResultModel
        )
        return result

    except Exception as e:
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in MinionS protocol:** {error_details}"


class Pipe:
    class Valves(MinionsValves):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "MinionS v0.3.7 (Modular)"

    def pipes(self):
        """Define the available models"""
        return [
            {
                "id": "minions-claude-v037",
                "name": f" ({self.valves.local_model} + {self.valves.remote_model})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "minions-claude-v037",
    ) -> str:
        """Execute the MinionS protocol with Claude"""
        return await minions_pipe_method(self, body, __user__, __request__, __files__, __pipe_id__)