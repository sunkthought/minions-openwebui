# Partials File: partials/streaming_mixins.py
import asyncio
import time
from typing import Any, Callable, List, Dict, Optional, AsyncGenerator, Union
from contextlib import asynccontextmanager


class ProgressTrackingMixin:
    """Mixin for consistent progress tracking across protocols."""
    
    async def track_phase_progress(self,
                                 phase_name: str,
                                 operation: Callable,
                                 total_steps: int,
                                 streaming_manager: Any,
                                 step_descriptions: List[str] = None) -> Any:
        """
        Track progress for any phase with automatic updates.
        
        Args:
            phase_name: Name of the phase (e.g., "task_decomposition")
            operation: Async operation to track
            total_steps: Total number of steps
            streaming_manager: Manager for streaming updates
            step_descriptions: Optional descriptions for each step
            
        Returns:
            Result of the operation
        """
        if not streaming_manager or not streaming_manager.is_streaming_enabled():
            return await operation()
        
        phase_tracker = PhaseTracker(streaming_manager, phase_name, total_steps)
        
        # If we have step descriptions, use them for sub-phase naming
        if step_descriptions and len(step_descriptions) == total_steps:
            phase_tracker.set_step_descriptions(step_descriptions)
        
        # Start the phase
        await phase_tracker.start_phase()
        
        try:
            # Execute the operation with phase tracking context
            result = await self._execute_with_phase_tracking(operation, phase_tracker)
            await phase_tracker.complete_phase()
            return result
        except Exception as e:
            await phase_tracker.error_occurred(str(e))
            raise
    
    async def _execute_with_phase_tracking(self, operation: Callable, phase_tracker: 'PhaseTracker') -> Any:
        """Execute operation with phase tracking support."""
        # For now, just execute the operation
        # This can be enhanced to provide more granular tracking
        result = await operation()
        return result
    
    async def track_iterative_progress(self,
                                     items: List[Any],
                                     processor: Callable,
                                     phase_name: str,
                                     streaming_manager: Any,
                                     parallel: bool = False,
                                     item_name_extractor: Callable[[Any], str] = None) -> List[Any]:
        """
        Track progress for iterative operations.
        
        Args:
            items: List of items to process
            processor: Function to process each item
            phase_name: Name of the phase
            streaming_manager: Manager for streaming updates
            parallel: Whether to process items in parallel
            item_name_extractor: Function to extract readable name from item
            
        Returns:
            List of processed results
        """
        if not streaming_manager or not streaming_manager.is_streaming_enabled():
            if parallel:
                return await asyncio.gather(*[processor(item) for item in items])
            else:
                return [await processor(item) for item in items]
        
        total_items = len(items)
        phase_tracker = PhaseTracker(streaming_manager, phase_name, total_items)
        
        await phase_tracker.start_phase()
        results = []
        
        try:
            if parallel:
                # Process items in parallel with progress tracking
                tasks = []
                for i, item in enumerate(items):
                    item_name = item_name_extractor(item) if item_name_extractor else f"item_{i+1}"
                    task = self._process_item_with_tracking(
                        processor, item, phase_tracker, i, item_name
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
            else:
                # Process items sequentially with progress tracking
                for i, item in enumerate(items):
                    item_name = item_name_extractor(item) if item_name_extractor else f"item_{i+1}"
                    await phase_tracker.update_step(i, f"Processing {item_name}")
                    
                    result = await processor(item)
                    results.append(result)
            
            await phase_tracker.complete_phase()
            return results
            
        except Exception as e:
            await phase_tracker.error_occurred(str(e))
            raise
    
    async def _process_item_with_tracking(self,
                                        processor: Callable,
                                        item: Any,
                                        phase_tracker: 'PhaseTracker',
                                        index: int,
                                        item_name: str) -> Any:
        """Process a single item with progress tracking."""
        await phase_tracker.update_step(index, f"Processing {item_name}")
        result = await processor(item)
        return result


class StreamingPatterns:
    """Common streaming patterns for both protocols."""
    
    @staticmethod
    async def stream_with_retry(operation: Callable,
                               streaming_manager: Any,
                               max_attempts: int = 3,
                               phase_name: str = "operation",
                               backoff_factor: float = 1.0) -> Any:
        """
        Execute operation with retry logic and progress updates.
        
        Args:
            operation: Async operation to execute
            streaming_manager: Manager for streaming updates
            max_attempts: Maximum number of retry attempts
            phase_name: Name of the phase for progress tracking
            backoff_factor: Backoff factor for retry delays
            
        Returns:
            Result of the operation
        """
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                if streaming_manager and streaming_manager.is_streaming_enabled():
                    if attempt > 0:
                        await streaming_manager.stream_granular_update(
                            phase_name,
                            f"retry_attempt_{attempt + 1}",
                            attempt / max_attempts,
                            f"Retrying operation (attempt {attempt + 1}/{max_attempts})"
                        )
                    else:
                        await streaming_manager.stream_granular_update(
                            phase_name,
                            "executing",
                            0.0,
                            "Starting operation"
                        )
                
                result = await operation()
                
                if streaming_manager and streaming_manager.is_streaming_enabled():
                    await streaming_manager.stream_granular_update(
                        phase_name,
                        "completed",
                        1.0,
                        "Operation completed successfully"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if streaming_manager and streaming_manager.is_streaming_enabled():
                    await streaming_manager.stream_granular_update(
                        phase_name,
                        f"error_attempt_{attempt + 1}",
                        (attempt + 1) / max_attempts,
                        f"Operation failed: {str(e)}"
                    )
                
                if attempt < max_attempts - 1:
                    # Calculate backoff delay
                    delay = backoff_factor * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    break
        
        # All attempts failed
        if streaming_manager and streaming_manager.is_streaming_enabled():
            await streaming_manager.stream_granular_update(
                phase_name,
                "failed",
                1.0,
                f"All {max_attempts} attempts failed. Last error: {str(last_exception)}"
            )
        
        raise last_exception
    
    @staticmethod
    async def stream_document_processing(chunks: List[str],
                                       processor: Callable,
                                       streaming_manager: Any,
                                       phase_name: str = "document_processing") -> List[Any]:
        """
        Standard pattern for streaming document chunk processing.
        
        Args:
            chunks: List of document chunks to process
            processor: Function to process each chunk
            streaming_manager: Manager for streaming updates
            phase_name: Name of the phase
            
        Returns:
            List of processed chunk results
        """
        if not streaming_manager or not streaming_manager.is_streaming_enabled():
            return [await processor(chunk, i) for i, chunk in enumerate(chunks)]
        
        total_chunks = len(chunks)
        results = []
        
        await streaming_manager.stream_granular_update(
            phase_name,
            "initializing",
            0.0,
            f"Starting processing of {total_chunks} document chunks"
        )
        
        for i, chunk in enumerate(chunks):
            progress = i / total_chunks
            
            await streaming_manager.stream_granular_update(
                phase_name,
                f"chunk_{i+1}_of_{total_chunks}",
                progress,
                f"Processing chunk {i+1}/{total_chunks} ({len(chunk)} characters)"
            )
            
            try:
                result = await processor(chunk, i)
                results.append(result)
                
                # Update progress after completion
                completion_progress = (i + 1) / total_chunks
                await streaming_manager.stream_granular_update(
                    phase_name,
                    f"chunk_{i+1}_completed",
                    completion_progress,
                    f"Completed chunk {i+1}/{total_chunks}"
                )
                
            except Exception as e:
                await streaming_manager.stream_granular_update(
                    phase_name,
                    f"chunk_{i+1}_error",
                    progress,
                    f"Error processing chunk {i+1}: {str(e)}"
                )
                results.append(None)  # or handle error differently
        
        await streaming_manager.stream_granular_update(
            phase_name,
            "completed",
            1.0,
            f"Completed processing all {total_chunks} chunks"
        )
        
        return results
    
    @staticmethod
    @asynccontextmanager
    async def stream_operation_context(streaming_manager: Any,
                                     phase_name: str,
                                     operation_description: str = "",
                                     total_steps: int = 1):
        """
        Context manager for automatic streaming operation tracking.
        
        Args:
            streaming_manager: Manager for streaming updates
            phase_name: Name of the phase
            operation_description: Description of the operation
            total_steps: Total number of steps (for progress calculation)
        """
        if not streaming_manager or not streaming_manager.is_streaming_enabled():
            yield None
            return
        
        # Start the operation
        await streaming_manager.stream_granular_update(
            phase_name,
            "starting",
            0.0,
            operation_description or "Starting operation"
        )
        
        try:
            # Create a simple tracker object for the context
            tracker = SimpleOperationTracker(streaming_manager, phase_name, total_steps)
            yield tracker
            
            # Operation completed successfully
            await streaming_manager.stream_granular_update(
                phase_name,
                "completed",
                1.0,
                "Operation completed successfully"
            )
            
        except Exception as e:
            # Operation failed
            await streaming_manager.stream_granular_update(
                phase_name,
                "error",
                1.0,
                f"Operation failed: {str(e)}"
            )
            raise


class SimpleOperationTracker:
    """Simple tracker for operations within a streaming context."""
    
    def __init__(self, streaming_manager: Any, phase_name: str, total_steps: int):
        self.streaming_manager = streaming_manager
        self.phase_name = phase_name
        self.total_steps = total_steps
        self.current_step = 0
    
    async def update_progress(self, step_name: str, details: str = ""):
        """Update progress for the current step."""
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0.0
        await self.streaming_manager.stream_granular_update(
            self.phase_name,
            step_name,
            progress,
            details
        )
    
    async def next_step(self, step_name: str, details: str = ""):
        """Move to the next step and update progress."""
        self.current_step += 1
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 1.0
        await self.streaming_manager.stream_granular_update(
            self.phase_name,
            step_name,
            progress,
            details
        )


class PhaseTracker:
    """Tracks progress for a specific phase with detailed step management."""
    
    def __init__(self, streaming_manager: Any, phase_name: str, total_steps: int):
        self.streaming_manager = streaming_manager
        self.phase_name = phase_name
        self.total_steps = total_steps
        self.current_step = 0
        self.step_descriptions = []
        self.start_time = None
        self.completed = False
    
    def set_step_descriptions(self, descriptions: List[str]):
        """Set descriptions for each step."""
        self.step_descriptions = descriptions
    
    async def start_phase(self):
        """Start the phase tracking."""
        self.start_time = time.time()
        await self.streaming_manager.stream_granular_update(
            self.phase_name,
            "initializing",
            0.0,
            f"Starting {self.phase_name.replace('_', ' ')} ({self.total_steps} steps)"
        )
    
    async def update_step(self, step_index: int, details: str = ""):
        """Update progress for a specific step."""
        self.current_step = step_index + 1
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0.0
        
        step_name = (
            self.step_descriptions[step_index] 
            if step_index < len(self.step_descriptions) 
            else f"step_{step_index + 1}"
        )
        
        await self.streaming_manager.stream_granular_update(
            self.phase_name,
            step_name,
            progress,
            details
        )
    
    async def next_step(self, step_name: str = "", details: str = ""):
        """Move to the next step."""
        await self.update_step(self.current_step, details)
    
    async def complete_phase(self):
        """Complete the phase tracking."""
        if not self.completed:
            self.completed = True
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            await self.streaming_manager.stream_granular_update(
                self.phase_name,
                "completed",
                1.0,
                f"Phase completed in {elapsed_time:.2f}s"
            )
    
    async def error_occurred(self, error_message: str):
        """Handle error during phase execution."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        await self.streaming_manager.stream_granular_update(
            self.phase_name,
            "error",
            self.current_step / self.total_steps if self.total_steps > 0 else 1.0,
            f"Error after {elapsed_time:.2f}s: {error_message}"
        )


class EnhancedStreamingMixin(ProgressTrackingMixin):
    """Enhanced streaming mixin that combines multiple streaming patterns."""
    
    async def stream_multi_phase_operation(self,
                                         phases: List[Dict[str, Any]],
                                         streaming_manager: Any) -> List[Any]:
        """
        Execute multiple phases with streaming progress.
        
        Args:
            phases: List of phase dictionaries with 'name', 'operation', 'steps' keys
            streaming_manager: Manager for streaming updates
            
        Returns:
            List of results from each phase
        """
        total_phases = len(phases)
        results = []
        
        if not streaming_manager or not streaming_manager.is_streaming_enabled():
            return [await phase['operation']() for phase in phases]
        
        for i, phase in enumerate(phases):
            phase_name = phase.get('name', f'phase_{i+1}')
            operation = phase['operation']
            steps = phase.get('steps', 1)
            step_descriptions = phase.get('step_descriptions', [])
            
            # Update overall progress
            overall_progress = i / total_phases
            await streaming_manager.stream_granular_update(
                "multi_phase_operation",
                f"starting_{phase_name}",
                overall_progress,
                f"Starting {phase_name} (phase {i+1}/{total_phases})"
            )
            
            # Execute the phase with detailed tracking
            result = await self.track_phase_progress(
                phase_name,
                operation,
                steps,
                streaming_manager,
                step_descriptions
            )
            
            results.append(result)
        
        # Final completion update
        await streaming_manager.stream_granular_update(
            "multi_phase_operation",
            "all_phases_completed",
            1.0,
            f"All {total_phases} phases completed successfully"
        )
        
        return results