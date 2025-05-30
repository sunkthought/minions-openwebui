import traceback
from typing import List, Dict, Any, Callable, Awaitable

# Placeholder for actual Valves type (e.g., from .minions_valves import MinionsValves)
ValvesType = Any 
# Placeholder for actual TaskResult model (e.g., from .minions_models import TaskResult)
TaskResultModelType = Any

# Define a more specific type for the 'self' object if possible,
# to hint at available methods and 'valves' attribute for MinionS.
class MinionsPipeInstance:
    valves: ValvesType
    # Context extraction methods (placeholders for functions from common_context_utils)
    extract_context_from_messages: Callable[[List[Dict[str, Any]]], str]
    extract_context_from_files: Callable[[ValvesType, List[Dict[str, Any]]], Awaitable[str]]
    # Protocol execution method (placeholder for function from minions_protocol_logic)
    execute_minions_protocol: Callable[..., Awaitable[str]] # Args: valves, query, context, api_calls, models, logic_helpers
    # API call function (placeholder for function from common_api_calls)
    call_claude_api: Callable[[ValvesType, str], Awaitable[str]]
    call_ollama_api: Callable # Placeholder
    # Models
    task_result_model: TaskResultModelType
    # Protocol logic helpers (placeholders for functions from minions_protocol_logic)
    parse_tasks_func: Callable
    create_chunks_func: Callable
    execute_tasks_on_chunks_func: Callable
    parse_local_response_func: Callable
    calculate_token_savings_func: Callable


async def _call_claude_directly_minions_helper(
    pipe_self: MinionsPipeInstance, # Provides access to self.valves and call_claude_api
    query: str
) -> str:
    """
    Fallback to direct Claude call when no context is available (MinionS version).
    """
    # Assumes call_claude_api is attached to pipe_self and is the refactored call_claude
    return await pipe_self.call_claude_api(pipe_self.valves, f"Please answer this question: {query}")

async def minions_pipe(
    self: MinionsPipeInstance, # Instance of the specific MinionS Pipe class
    body: Dict[str, Any],
    __user__: Dict[str, Any], 
    __request__: Any, # fastapi.Request, type hinted as Any
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "" 
) -> str:
    """
    Executes the MinionS (multi-task, multi-round) protocol.
    This function is intended to be the 'pipe' method of a class that has 'valves'
    and access to helper functions for context extraction, protocol execution, etc.
    """
    try:
        if not self.valves.anthropic_api_key: # type: ignore
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

        messages = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."
        user_query = messages[-1]["content"]

        # Context extraction using helpers from common_context_utils.py (expected on self)
        context_from_messages = self.extract_context_from_messages(messages[:-1])
        context_from_files = await self.extract_context_from_files(self.valves, __files__)

        all_context_parts = []
        if context_from_messages: all_context_parts.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files: all_context_parts.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")
        context = "\n\n".join(all_context_parts) if all_context_parts else ""

        if not context:
            return (
                "ℹ️ **Note:** No significant context detected. Using standard remote model response.\n\n"
                + await _call_claude_directly_minions_helper(self, user_query)
            )

        # Execute the MinionS protocol using the main logic function from minions_protocol_logic.py
        # This function expects its dependencies (other helpers, API calls, models) to be passed.
        result = await self.execute_minions_protocol(
            self.valves,
            user_query,
            context,
            self.call_claude_api, 
            self.call_ollama_api, 
            self.task_result_model,
            # Pass MinionS specific logic functions (expected to be on self)
            self.parse_tasks_func,
            self.create_chunks_func,
            self.execute_tasks_on_chunks_func,
            self.parse_local_response_func, # MinionS specific parser
            self.calculate_token_savings_func # MinionS specific calculator
        )
        return result

    except Exception as e:
        error_details = traceback.format_exc() if (hasattr(self.valves, 'debug_mode') and self.valves.debug_mode) else str(e) # type: ignore
        return f"❌ **Error in MinionS protocol:** {error_details}"
