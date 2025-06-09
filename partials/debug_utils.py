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