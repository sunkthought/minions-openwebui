# Partials File: partials/streaming_support.py

import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
import json
import time

class StreamingResponseManager:
    """
    Streaming Response Support for MinionS providing real-time updates during operations.
    Converts the main pipe method to use async generators for live progress updates.
    """
    
    def __init__(self, valves, debug_mode: bool = False):
        self.valves = valves
        self.debug_mode = debug_mode
        self.current_phase = ""
        self.total_tasks = 0
        self.completed_tasks = 0
        self.error_occurred = False
        self.last_update_time = 0
        self.min_update_interval = 0.1  # Minimum seconds between updates
        self.update_queue = []
    
    def is_streaming_enabled(self) -> bool:
        """Check if streaming responses are enabled via valves."""
        return getattr(self.valves, 'enable_streaming_responses', True)
    
    async def _rate_limited_update(self, message: str, force: bool = False) -> str:
        """
        Apply rate limiting to streaming updates to prevent flooding.
        
        Args:
            message: The update message
            force: Force immediate update regardless of rate limit
            
        Returns:
            str: The message if it should be sent, empty string otherwise
        """
        current_time = time.time()
        time_since_last = current_time - self.last_update_time
        
        if force or time_since_last >= self.min_update_interval:
            self.last_update_time = current_time
            # Flush any queued updates
            if self.update_queue:
                combined = "\n".join(self.update_queue) + "\n" + message
                self.update_queue.clear()
                return combined
            return message
        else:
            # Queue the update
            self.update_queue.append(message.strip())
            return ""
    
    async def stream_granular_update(self, 
                                   phase: str, 
                                   sub_phase: str,
                                   progress: float,
                                   details: str = "") -> str:
        """
        Stream granular updates with progress percentages and sub-phases.
        
        Args:
            phase: Main phase (e.g., "task_decomposition")
            sub_phase: Sub-phase (e.g., "analyzing_complexity", "generating_tasks")
            progress: Progress percentage (0.0 to 1.0)
            details: Optional details about current operation
            
        Returns:
            str: Formatted granular update message
        """
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Phase emoji mapping
        phase_emojis = {
            "task_decomposition": "🔍",
            "task_execution": "⚙️",
            "web_search": "🌐",
            "synthesis": "🔄",
            "conversation": "💬",
            "query_analysis": "🔍",
            "document_retrieval": "📄",
            "answer_synthesis": "🧠",
            "citation_processing": "📚"
        }
        
        emoji = phase_emojis.get(phase.lower().replace(" ", "_"), "📊")
        progress_bar = self.format_progress_bar(progress)
        percentage = int(progress * 100)
        
        # Format the update message
        message_parts = [
            f"{emoji} {phase.replace('_', ' ').title()} {progress_bar}",
            f"   └─ {sub_phase}: {details}" if details else f"   └─ {sub_phase}"
        ]
        
        message = "\n".join(message_parts) + "\n"
        
        if self.debug_mode:
            print(f"[Streaming] Granular update: {phase}/{sub_phase} - {percentage}%")
        
        # Apply rate limiting
        return await self._rate_limited_update(message)
    
    def format_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a text-based progress bar."""
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(progress * 100)
        return f"[{bar}] {percentage}%"
    
    async def stream_phase_update(self, phase_name: str, details: str = "") -> str:
        """
        Generate a streaming update for a new phase.
        
        Args:
            phase_name: Name of the current phase
            details: Optional additional details
            
        Returns:
            str: Formatted phase update message
        """
        self.current_phase = phase_name
        
        # Phase emoji mapping
        phase_emojis = {
            "query_analysis": "🔍",
            "task_decomposition": "📋",
            "document_retrieval": "📄",
            "web_search": "🌐",
            "task_execution": "⚙️",
            "answer_synthesis": "🧠",
            "citation_processing": "📚",
            "completion": "✅",
            "error": "❌"
        }
        
        emoji = phase_emojis.get(phase_name.lower().replace(" ", "_"), "📍")
        
        if details:
            message = f"{emoji} {phase_name}: {details}\n"
        else:
            message = f"{emoji} {phase_name}...\n"
        
        if self.debug_mode:
            print(f"[Streaming] Phase update: {phase_name}")
        
        return message
    
    async def stream_task_progress(self, task_number: int, total_tasks: int, 
                                 task_name: str, status: str = "executing") -> str:
        """
        Generate a streaming update for task progress.
        
        Args:
            task_number: Current task number (1-indexed)
            total_tasks: Total number of tasks
            task_name: Name or description of the current task
            status: Status of the task (executing, completed, failed)
            
        Returns:
            str: Formatted task progress message
        """
        self.total_tasks = total_tasks
        
        if status == "completed":
            self.completed_tasks = task_number
            emoji = "✅"
        elif status == "failed":
            emoji = "❌"
            self.error_occurred = True
        else:
            emoji = "🔄"
        
        # Truncate task name for readability
        display_name = task_name[:50] + "..." if len(task_name) > 50 else task_name
        
        progress_bar = self._generate_progress_bar(task_number, total_tasks)
        
        message = f"{emoji} Task {task_number}/{total_tasks}: {display_name}\n{progress_bar}\n"
        
        if self.debug_mode:
            print(f"[Streaming] Task progress: {task_number}/{total_tasks} - {status}")
        
        return message
    
    async def stream_search_update(self, search_type: str, query: str, 
                                 results_count: int = None) -> str:
        """
        Generate a streaming update for search operations.
        
        Args:
            search_type: Type of search (web_search, rag_retrieval, document_search)
            query: The search query
            results_count: Number of results found (optional)
            
        Returns:
            str: Formatted search update message
        """
        type_emojis = {
            "web_search": "🌐",
            "rag_retrieval": "🔍",
            "document_search": "📄"
        }
        
        emoji = type_emojis.get(search_type, "🔍")
        display_query = query[:40] + "..." if len(query) > 40 else query
        
        if results_count is not None:
            message = f"{emoji} {search_type.replace('_', ' ').title()}: \"{display_query}\" ({results_count} results)\n"
        else:
            message = f"{emoji} {search_type.replace('_', ' ').title()}: \"{display_query}\"\n"
        
        return message
    
    async def stream_error_update(self, error_message: str, 
                                error_type: str = "general") -> str:
        """
        Generate a streaming update for errors.
        
        Args:
            error_message: The error message
            error_type: Type of error (task_error, search_error, general)
            
        Returns:
            str: Formatted error update message
        """
        self.error_occurred = True
        
        # Truncate long error messages
        display_error = error_message[:100] + "..." if len(error_message) > 100 else error_message
        
        message = f"❌ Error ({error_type}): {display_error}\n"
        
        if self.debug_mode:
            print(f"[Streaming] Error: {error_type} - {error_message}")
        
        return message
    
    async def stream_metrics_update(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a streaming update with performance metrics.
        
        Args:
            metrics: Dictionary containing metrics data
            
        Returns:
            str: Formatted metrics update message
        """
        message_parts = ["📊 Performance Metrics:\n"]
        
        # Format common metrics
        if "execution_time" in metrics:
            time_sec = metrics["execution_time"] / 1000 if metrics["execution_time"] > 1000 else metrics["execution_time"]
            message_parts.append(f"   ⏱️ Execution time: {time_sec:.1f}s\n")
        
        if "tasks_executed" in metrics:
            message_parts.append(f"   ✅ Tasks completed: {metrics['tasks_executed']}\n")
        
        if "success_rate" in metrics:
            success_rate = metrics["success_rate"] * 100
            message_parts.append(f"   📈 Success rate: {success_rate:.1f}%\n")
        
        if "tokens_saved" in metrics:
            message_parts.append(f"   💰 Tokens saved: {metrics['tokens_saved']}\n")
        
        return "".join(message_parts)
    
    def _generate_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """
        Generate a text-based progress bar.
        
        Args:
            current: Current progress value
            total: Total value
            width: Width of the progress bar in characters
            
        Returns:
            str: Text progress bar
        """
        if total == 0:
            return "[" + "=" * width + "] 100%"
        
        percentage = min(current / total, 1.0)
        filled_width = int(width * percentage)
        
        bar = "=" * filled_width + "-" * (width - filled_width)
        percentage_text = f"{percentage * 100:.0f}%"
        
        return f"[{bar}] {percentage_text}"
    
    async def stream_task_decomposition_progress(self, 
                                                stage: str,
                                                current_step: int,
                                                total_steps: int = 5,
                                                details: str = "") -> str:
        """
        Stream granular updates for task decomposition phase.
        
        Args:
            stage: Current stage (analyzing_complexity, generating_tasks, etc.)
            current_step: Current step number
            total_steps: Total steps in decomposition
            details: Additional details
            
        Returns:
            str: Formatted update
        """
        progress = current_step / total_steps
        
        stage_details = {
            "analyzing_complexity": "Analyzing query complexity",
            "document_structure": "Analyzing document structure",
            "generating_tasks": "Generating task list",
            "task_validation": "Validating tasks",
            "complete": "Decomposition complete"
        }
        
        detail_text = stage_details.get(stage, stage)
        if details:
            detail_text = f"{detail_text} - {details}"
        
        return await self.stream_granular_update(
            "task_decomposition",
            stage,
            progress,
            detail_text
        )
    
    async def stream_task_execution_progress(self,
                                           task_idx: int,
                                           total_tasks: int,
                                           chunk_idx: int = None,
                                           total_chunks: int = None,
                                           task_description: str = "") -> str:
        """
        Stream granular updates for task execution phase.
        
        Args:
            task_idx: Current task index (0-based)
            total_tasks: Total number of tasks
            chunk_idx: Current chunk index (optional)
            total_chunks: Total chunks (optional)
            task_description: Task being executed
            
        Returns:
            str: Formatted update
        """
        # Calculate overall progress
        if chunk_idx is not None and total_chunks is not None:
            # Progress within current task considering chunks
            task_progress = (chunk_idx + 1) / total_chunks
            overall_progress = (task_idx + task_progress) / total_tasks
            sub_phase = f"task_{task_idx + 1}_chunk_{chunk_idx + 1}"
            details = f"Task {task_idx + 1}/{total_tasks}, Chunk {chunk_idx + 1}/{total_chunks}"
        else:
            # Simple task progress
            overall_progress = (task_idx + 1) / total_tasks
            sub_phase = f"task_{task_idx + 1}"
            details = f"Task {task_idx + 1}/{total_tasks}"
        
        if task_description:
            # Truncate long task descriptions
            truncated = task_description[:50] + "..." if len(task_description) > 50 else task_description
            details = f"{details}: {truncated}"
        
        return await self.stream_granular_update(
            "task_execution",
            sub_phase,
            overall_progress,
            details
        )
    
    async def stream_web_search_progress(self,
                                       stage: str,
                                       query: str = "",
                                       results_count: int = None) -> str:
        """
        Stream granular updates for web search phase.
        
        Args:
            stage: Search stage (formulation, execution, parsing, etc.)
            query: Search query
            results_count: Number of results found
            
        Returns:
            str: Formatted update
        """
        progress_map = {
            "formulation": 0.2,
            "execution": 0.4,
            "parsing": 0.6,
            "citation": 0.8,
            "complete": 1.0
        }
        
        progress = progress_map.get(stage, 0.5)
        
        stage_details = {
            "formulation": "Formulating search query",
            "execution": "Executing web search",
            "parsing": f"Parsing {results_count} results" if results_count else "Parsing results",
            "citation": "Generating citations",
            "complete": "Search complete"
        }
        
        details = stage_details.get(stage, stage)
        if query and stage in ["formulation", "execution"]:
            truncated_query = query[:40] + "..." if len(query) > 40 else query
            details = f'{details}: "{truncated_query}"'
        
        return await self.stream_granular_update(
            "web_search",
            stage,
            progress,
            details
        )
    
    async def stream_synthesis_progress(self,
                                      stage: str,
                                      processed_tasks: int = None,
                                      total_tasks: int = None) -> str:
        """
        Stream granular updates for synthesis phase.
        
        Args:
            stage: Synthesis stage
            processed_tasks: Number of tasks processed
            total_tasks: Total tasks to process
            
        Returns:
            str: Formatted update
        """
        progress_map = {
            "collecting": 0.2,
            "generating": 0.5,
            "formatting": 0.8,
            "complete": 1.0
        }
        
        # Adjust progress based on task processing
        if stage == "generating" and processed_tasks is not None and total_tasks:
            base_progress = 0.4
            task_progress = 0.4 * (processed_tasks / total_tasks)
            progress = base_progress + task_progress
        else:
            progress = progress_map.get(stage, 0.5)
        
        stage_details = {
            "collecting": "Collecting task results",
            "generating": f"Generating answer ({processed_tasks}/{total_tasks} tasks)" if processed_tasks else "Generating answer",
            "formatting": "Formatting citations",
            "complete": "Synthesis complete"
        }
        
        details = stage_details.get(stage, stage)
        
        return await self.stream_granular_update(
            "synthesis",
            stage,
            progress,
            details
        )
    
    async def stream_conversation_progress(self,
                                         round_num: int,
                                         max_rounds: int,
                                         stage: str = "questioning") -> str:
        """
        Stream granular updates for conversation rounds (Minion protocol).
        
        Args:
            round_num: Current conversation round
            max_rounds: Maximum rounds
            stage: Current stage (questioning, processing, etc.)
            
        Returns:
            str: Formatted update
        """
        progress = round_num / max_rounds
        
        stage_map = {
            "questioning": f"Claude asking question {round_num}",
            "processing": f"Processing response for round {round_num}",
            "analyzing": f"Analyzing sufficiency after round {round_num}"
        }
        
        details = stage_map.get(stage, f"Round {round_num}/{max_rounds}")
        
        return await self.stream_granular_update(
            "conversation",
            f"round_{round_num}",
            progress,
            details
        )
    
    async def stream_visualization_update(self, visualization_content: str) -> str:
        """
        Stream visualization content if enabled.
        
        Args:
            visualization_content: Mermaid diagram or other visualization content
            
        Returns:
            str: Formatted visualization update
        """
        if not visualization_content:
            return ""
        
        message = "📊 Task Visualization:\n\n" + visualization_content + "\n\n"
        return message
    
    async def create_streaming_wrapper(self, operation_name: str, 
                                     operation_func, 
                                     *args, **kwargs) -> AsyncGenerator[str, None]:
        """
        Wrap a synchronous operation to provide streaming updates.
        
        Args:
            operation_name: Name of the operation for status updates
            operation_func: The function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Yields:
            str: Streaming status messages
        """
        try:
            # Send start notification
            yield await self.stream_phase_update(operation_name, "Starting")
            
            # Execute the operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            # Send completion notification
            yield await self.stream_phase_update(operation_name, "Completed")
            
            # Return the result in the final message
            if isinstance(result, str):
                yield result
            else:
                yield json.dumps(result, indent=2)
                
        except Exception as e:
            error_message = await self.stream_error_update(str(e), operation_name)
            yield error_message
            raise
    
    async def stream_complete_pipeline(self, pipeline_steps: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """
        Stream updates for a complete MinionS pipeline execution.
        
        Args:
            pipeline_steps: List of pipeline steps with names and functions
            
        Yields:
            str: Streaming updates for each step
        """
        total_steps = len(pipeline_steps)
        
        try:
            for i, step in enumerate(pipeline_steps, 1):
                step_name = step.get("name", f"Step {i}")
                step_func = step.get("function")
                step_args = step.get("args", [])
                step_kwargs = step.get("kwargs", {})
                
                # Stream step start
                yield await self.stream_phase_update(
                    step_name, 
                    f"Step {i}/{total_steps}"
                )
                
                # Execute step
                try:
                    if asyncio.iscoroutinefunction(step_func):
                        result = await step_func(*step_args, **step_kwargs)
                    else:
                        result = step_func(*step_args, **step_kwargs)
                    
                    # Stream step completion
                    yield await self.stream_task_progress(i, total_steps, step_name, "completed")
                    
                    # If this is the final step, include the result
                    if i == total_steps and isinstance(result, str):
                        yield f"\n{result}"
                        
                except Exception as e:
                    yield await self.stream_error_update(str(e), step_name)
                    yield await self.stream_task_progress(i, total_steps, step_name, "failed")
                    raise
            
            # Final completion message
            yield await self.stream_phase_update("completion", "Pipeline execution completed successfully")
            
        except Exception as e:
            yield await self.stream_error_update(str(e), "pipeline")
            raise
    
    def create_final_response_with_metadata(self, main_response: str, 
                                          metadata: Dict[str, Any]) -> str:
        """
        Create a final response that includes both the main content and metadata.
        
        Args:
            main_response: The main response content
            metadata: Additional metadata to include
            
        Returns:
            str: Complete response with metadata
        """
        response_parts = [main_response]
        
        if metadata and self.debug_mode:
            response_parts.append("\n\n---\n**Debug Information:**")
            
            if "execution_time" in metadata:
                response_parts.append(f"- Execution time: {metadata['execution_time']:.2f}s")
            
            if "total_tasks" in metadata:
                response_parts.append(f"- Total tasks: {metadata['total_tasks']}")
            
            if "citations_count" in metadata:
                response_parts.append(f"- Citations: {metadata['citations_count']}")
            
            if "tokens_used" in metadata:
                response_parts.append(f"- Tokens used: {metadata['tokens_used']}")
        
        return "\n".join(response_parts)
    
    def reset_state(self) -> None:
        """Reset the streaming state for a new request."""
        self.current_phase = ""
        self.total_tasks = 0
        self.completed_tasks = 0
        self.error_occurred = False
        
        if self.debug_mode:
            print("[Streaming] State reset for new request")


class AsyncPipelineExecutor:
    """
    Async pipeline executor for streaming support.
    Converts synchronous operations to async streaming operations.
    """
    
    def __init__(self, streaming_manager: StreamingResponseManager):
        self.streaming_manager = streaming_manager
    
    async def execute_with_streaming(self, operation_name: str, 
                                   sync_function, 
                                   *args, **kwargs) -> AsyncGenerator[str, None]:
        """
        Execute a synchronous function with streaming progress updates.
        
        Args:
            operation_name: Name for progress tracking
            sync_function: Synchronous function to execute
            *args, **kwargs: Arguments for the function
            
        Yields:
            str: Progress updates and final result
        """
        async for update in self.streaming_manager.create_streaming_wrapper(
            operation_name, sync_function, *args, **kwargs
        ):
            yield update
    
    async def execute_task_batch_with_streaming(self, tasks: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """
        Execute a batch of tasks with individual progress updates.
        
        Args:
            tasks: List of task dictionaries with 'name' and 'function' keys
            
        Yields:
            str: Progress updates for each task
        """
        total_tasks = len(tasks)
        results = []
        
        for i, task in enumerate(tasks, 1):
            task_name = task.get("name", f"Task {i}")
            task_func = task.get("function")
            task_args = task.get("args", [])
            task_kwargs = task.get("kwargs", {})
            
            # Stream task start
            yield await self.streaming_manager.stream_task_progress(
                i, total_tasks, task_name, "executing"
            )
            
            try:
                # Execute task
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*task_args, **task_kwargs)
                else:
                    result = task_func(*task_args, **task_kwargs)
                
                results.append(result)
                
                # Stream task completion
                yield await self.streaming_manager.stream_task_progress(
                    i, total_tasks, task_name, "completed"
                )
                
            except Exception as e:
                yield await self.streaming_manager.stream_error_update(
                    str(e), f"task_{i}"
                )
                yield await self.streaming_manager.stream_task_progress(
                    i, total_tasks, task_name, "failed"
                )
                # Continue with other tasks instead of stopping
                results.append(None)
        
        # Return combined results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            yield f"\n**Completed {len(valid_results)}/{total_tasks} tasks successfully.**\n"