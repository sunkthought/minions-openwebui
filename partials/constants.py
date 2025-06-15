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
    CURRENT_VERSION = "v0.3.9b"


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