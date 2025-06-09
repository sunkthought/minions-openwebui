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