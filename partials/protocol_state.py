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