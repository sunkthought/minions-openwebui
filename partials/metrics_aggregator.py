# Partials File: partials/metrics_aggregator.py
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
import json


class OperationType(Enum):
    """Enumeration of operation types for consistent tracking."""
    TASK_DECOMPOSITION = "task_decomposition"
    TASK_EXECUTION = "task_execution"
    WEB_SEARCH = "web_search"
    DOCUMENT_PROCESSING = "document_processing"
    API_CALL = "api_call"
    CONVERSATION_ROUND = "conversation_round"
    SYNTHESIS = "synthesis"
    CITATION_PROCESSING = "citation_processing"
    STREAMING_UPDATE = "streaming_update"
    ERROR_HANDLING = "error_handling"
    CACHE_OPERATION = "cache_operation"


@dataclass
class OperationMetric:
    """Single operation metric with comprehensive tracking."""
    operation_type: str
    duration: float
    success: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional performance metrics
    tokens_used: Optional[int] = None
    api_calls_made: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for serialization."""
        return {
            "operation_type": self.operation_type,
            "duration": self.duration,
            "success": self.success,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tokens_used": self.tokens_used,
            "api_calls_made": self.api_calls_made,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "error_message": self.error_message
        }


@dataclass
class PhaseMetrics:
    """Metrics for a complete phase of operation."""
    phase_name: str
    start_time: float
    end_time: Optional[float] = None
    operations: List[OperationMetric] = field(default_factory=list)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Calculate total phase duration."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def operation_count(self) -> int:
        """Get total number of operations in this phase."""
        return len(self.operations)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for operations in this phase."""
        if not self.operations:
            return 1.0
        successful = sum(1 for op in self.operations if op.success)
        return successful / len(self.operations)


class MetricsAggregator:
    """
    Centralized metrics collection and reporting for both protocols.
    Provides comprehensive performance tracking and analysis.
    """
    
    def __init__(self, protocol_name: str, debug_mode: bool = False):
        self.protocol_name = protocol_name
        self.debug_mode = debug_mode
        self.metrics: List[OperationMetric] = []
        self.phases: Dict[str, PhaseMetrics] = {}
        self.start_time = time.time()
        self.session_metadata = {}
        
        # Performance counters
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0
        
    def set_session_metadata(self, metadata: Dict[str, Any]):
        """Set metadata for the current session."""
        self.session_metadata.update(metadata)
        
    def start_phase(self, phase_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start tracking a new phase.
        
        Args:
            phase_name: Name of the phase
            metadata: Optional metadata for the phase
            
        Returns:
            str: Phase ID for reference
        """
        phase_id = f"{phase_name}_{len(self.phases)}"
        self.phases[phase_id] = PhaseMetrics(
            phase_name=phase_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        if self.debug_mode:
            print(f"DEBUG [MetricsAggregator]: Started phase '{phase_name}' (ID: {phase_id})")
        
        return phase_id
    
    def end_phase(self, phase_id: str, success: bool = True, metadata: Dict[str, Any] = None):
        """
        End tracking for a phase.
        
        Args:
            phase_id: Phase ID returned from start_phase
            success: Whether the phase completed successfully
            metadata: Additional metadata for the phase
        """
        if phase_id in self.phases:
            phase = self.phases[phase_id]
            phase.end_time = time.time()
            phase.success = success
            if metadata:
                phase.metadata.update(metadata)
            
            if self.debug_mode:
                print(f"DEBUG [MetricsAggregator]: Ended phase '{phase.phase_name}' "
                      f"(Duration: {phase.duration:.2f}s, Success: {success})")
    
    def track_operation(self,
                       operation_type: Union[str, OperationType],
                       duration: float,
                       success: bool,
                       metadata: Dict[str, Any] = None,
                       phase_id: str = None,
                       **kwargs) -> OperationMetric:
        """
        Track a single operation.
        
        Args:
            operation_type: Type of operation
            duration: Duration in seconds
            success: Whether operation succeeded
            metadata: Additional metadata
            phase_id: Optional phase ID to associate with
            **kwargs: Additional metric fields
            
        Returns:
            OperationMetric: The created metric
        """
        # Convert enum to string if needed
        if isinstance(operation_type, OperationType):
            operation_type = operation_type.value
        
        metric = OperationMetric(
            operation_type=operation_type,
            duration=duration,
            success=success,
            timestamp=time.time(),
            metadata=metadata or {},
            **kwargs
        )
        
        self.metrics.append(metric)
        
        # Add to phase if specified
        if phase_id and phase_id in self.phases:
            self.phases[phase_id].operations.append(metric)
        
        # Update counters
        if not success:
            self.error_count += 1
        
        if metric.api_calls_made:
            self.total_api_calls += metric.api_calls_made
        
        if metric.tokens_used:
            self.total_tokens_used += metric.tokens_used
        
        if metric.cache_hits:
            self.cache_hits += metric.cache_hits
        
        if metric.cache_misses:
            self.cache_misses += metric.cache_misses
        
        if self.debug_mode:
            self._log_metric(metric)
        
        return metric
    
    @contextmanager
    def track_operation_context(self,
                              operation_type: Union[str, OperationType],
                              metadata: Dict[str, Any] = None,
                              phase_id: str = None,
                              **kwargs):
        """
        Context manager for automatic operation tracking.
        
        Args:
            operation_type: Type of operation
            metadata: Additional metadata
            phase_id: Optional phase ID to associate with
            **kwargs: Additional metric fields
        """
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            if metadata is None:
                metadata = {}
            metadata['error'] = error_message
            raise
        finally:
            duration = time.time() - start_time
            self.track_operation(
                operation_type=operation_type,
                duration=duration,
                success=success,
                metadata=metadata,
                phase_id=phase_id,
                error_message=error_message,
                **kwargs
            )
    
    def track_api_call(self,
                      api_name: str,
                      duration: float,
                      success: bool,
                      tokens_used: int = None,
                      model_name: str = None,
                      phase_id: str = None) -> OperationMetric:
        """
        Track an API call with specific metrics.
        
        Args:
            api_name: Name of the API called
            duration: Duration of the call
            success: Whether the call succeeded
            tokens_used: Number of tokens used
            model_name: Name of the model used
            phase_id: Optional phase ID
            
        Returns:
            OperationMetric: The created metric
        """
        metadata = {"api_name": api_name}
        if model_name:
            metadata["model_name"] = model_name
        
        return self.track_operation(
            operation_type=OperationType.API_CALL,
            duration=duration,
            success=success,
            metadata=metadata,
            tokens_used=tokens_used,
            api_calls_made=1,
            phase_id=phase_id
        )
    
    def track_cache_operation(self,
                            operation: str,
                            hit: bool,
                            duration: float = 0.0,
                            phase_id: str = None) -> OperationMetric:
        """
        Track a cache operation.
        
        Args:
            operation: Type of cache operation (get, set, etc.)
            hit: Whether it was a cache hit
            duration: Duration of the operation
            phase_id: Optional phase ID
            
        Returns:
            OperationMetric: The created metric
        """
        metadata = {"cache_operation": operation, "cache_hit": hit}
        
        return self.track_operation(
            operation_type=OperationType.CACHE_OPERATION,
            duration=duration,
            success=True,
            metadata=metadata,
            cache_hits=1 if hit else 0,
            cache_misses=0 if hit else 1,
            phase_id=phase_id
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        total_duration = time.time() - self.start_time
        
        # Group metrics by operation type
        operations_by_type = {}
        for metric in self.metrics:
            if metric.operation_type not in operations_by_type:
                operations_by_type[metric.operation_type] = []
            operations_by_type[metric.operation_type].append(metric)
        
        # Calculate statistics
        summary = {
            "protocol": self.protocol_name,
            "session_metadata": self.session_metadata,
            "timing": {
                "total_duration": total_duration,
                "start_time": self.start_time,
                "end_time": time.time()
            },
            "operations": {
                "total_operations": len(self.metrics),
                "successful_operations": sum(1 for m in self.metrics if m.success),
                "failed_operations": self.error_count,
                "success_rate": (len(self.metrics) - self.error_count) / len(self.metrics) if self.metrics else 1.0
            },
            "performance": {
                "total_api_calls": self.total_api_calls,
                "total_tokens_used": self.total_tokens_used,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
            },
            "operations_by_type": {},
            "phases": {}
        }
        
        # Add operation type statistics
        for op_type, metrics in operations_by_type.items():
            successful = [m for m in metrics if m.success]
            total_duration_for_type = sum(m.duration for m in metrics)
            
            summary["operations_by_type"][op_type] = {
                "count": len(metrics),
                "success_count": len(successful),
                "success_rate": len(successful) / len(metrics) if metrics else 0,
                "avg_duration": total_duration_for_type / len(metrics) if metrics else 0,
                "total_duration": total_duration_for_type,
                "min_duration": min(m.duration for m in metrics) if metrics else 0,
                "max_duration": max(m.duration for m in metrics) if metrics else 0
            }
        
        # Add phase statistics
        for phase_id, phase in self.phases.items():
            summary["phases"][phase_id] = {
                "phase_name": phase.phase_name,
                "duration": phase.duration,
                "success": phase.success,
                "operation_count": phase.operation_count,
                "success_rate": phase.success_rate,
                "metadata": phase.metadata
            }
        
        return summary
    
    def format_summary_report(self, include_details: bool = False) -> str:
        """
        Format a human-readable summary report.
        
        Args:
            include_details: Whether to include detailed operation breakdowns
            
        Returns:
            str: Formatted report
        """
        summary = self.get_summary()
        
        lines = [
            f"## 📊 {self.protocol_name.title()} Protocol Performance Report",
            "",
            f"**Total Duration:** {summary['timing']['total_duration']:.2f}s",
            f"**Operations:** {summary['operations']['total_operations']} total, "
            f"{summary['operations']['successful_operations']} successful "
            f"({summary['operations']['success_rate']:.1%} success rate)",
            ""
        ]
        
        # Performance metrics
        perf = summary['performance']
        if perf['total_api_calls'] > 0:
            lines.extend([
                "### 🔌 API Performance",
                f"- **API Calls:** {perf['total_api_calls']}",
                f"- **Tokens Used:** {perf['total_tokens_used']:,}",
                f"- **Cache Hit Rate:** {perf['cache_hit_rate']:.1%} "
                f"({perf['cache_hits']} hits, {perf['cache_misses']} misses)",
                ""
            ])
        
        # Phase breakdown
        if summary['phases']:
            lines.extend([
                "### 📋 Phase Breakdown",
                ""
            ])
            
            for phase_id, phase in summary['phases'].items():
                status_emoji = "✅" if phase['success'] else "❌"
                lines.append(
                    f"- **{phase['phase_name']}** {status_emoji}: "
                    f"{phase['duration']:.2f}s, {phase['operation_count']} operations "
                    f"({phase['success_rate']:.1%} success)"
                )
            
            lines.append("")
        
        # Operation type breakdown
        if include_details and summary['operations_by_type']:
            lines.extend([
                "### 🔧 Operation Details",
                ""
            ])
            
            for op_type, stats in summary['operations_by_type'].items():
                lines.append(
                    f"- **{op_type.replace('_', ' ').title()}**: "
                    f"{stats['count']} ops, {stats['avg_duration']:.3f}s avg, "
                    f"{stats['success_rate']:.1%} success"
                )
        
        # Session metadata
        if summary['session_metadata']:
            lines.extend([
                "",
                "### 📝 Session Info",
                ""
            ])
            
            for key, value in summary['session_metadata'].items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        return "\n".join(lines)
    
    def format_performance_summary(self) -> str:
        """Format a concise performance summary for inclusion in responses."""
        summary = self.get_summary()
        
        # Calculate key metrics
        duration = summary['timing']['total_duration']
        operations = summary['operations']['total_operations']
        success_rate = summary['operations']['success_rate']
        api_calls = summary['performance']['total_api_calls']
        tokens = summary['performance']['total_tokens_used']
        
        parts = [
            f"⏱️ {duration:.1f}s",
            f"🔧 {operations} ops",
            f"✅ {success_rate:.0%}"
        ]
        
        if api_calls > 0:
            parts.append(f"🔌 {api_calls} API calls")
        
        if tokens > 0:
            parts.append(f"🎯 {tokens:,} tokens")
        
        return " | ".join(parts)
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export metrics in various formats.
        
        Args:
            format: Export format ("json", "dict", "csv")
            
        Returns:
            Exported metrics in requested format
        """
        if format.lower() == "dict":
            return self.get_summary()
        elif format.lower() == "json":
            return json.dumps(self.get_summary(), indent=2, default=str)
        elif format.lower() == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv(self) -> str:
        """Export metrics as CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "timestamp", "operation_type", "duration", "success",
            "tokens_used", "api_calls_made", "cache_hits", "cache_misses",
            "error_message", "metadata"
        ])
        
        # Write metrics
        for metric in self.metrics:
            writer.writerow([
                metric.timestamp,
                metric.operation_type,
                metric.duration,
                metric.success,
                metric.tokens_used or 0,
                metric.api_calls_made or 0,
                metric.cache_hits or 0,
                metric.cache_misses or 0,
                metric.error_message or "",
                json.dumps(metric.metadata) if metric.metadata else ""
            ])
        
        return output.getvalue()
    
    def _log_metric(self, metric: OperationMetric):
        """Log a metric for debugging purposes."""
        status = "SUCCESS" if metric.success else "FAILED"
        print(f"DEBUG [MetricsAggregator]: {metric.operation_type} - "
              f"{status} in {metric.duration:.3f}s")
        
        if metric.error_message:
            print(f"  Error: {metric.error_message}")
        
        if metric.metadata:
            print(f"  Metadata: {metric.metadata}")
    
    def reset(self):
        """Reset all metrics and start fresh."""
        self.metrics.clear()
        self.phases.clear()
        self.start_time = time.time()
        self.session_metadata.clear()
        
        # Reset counters
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0