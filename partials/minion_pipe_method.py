import traceback
from typing import List, Dict, Any, Callable, Awaitable

# Placeholder for actual Valves type, e.g., from .minion_valves import MinionValves
ValvesType = Any 
# Placeholder for actual LocalAssistantResponse model, e.g., from .minion_models import LocalAssistantResponse
LocalAssistantResponseModelType = Any

# Define a more specific type for the 'self' object if possible,
# to hint at available methods and 'valves' attribute.
class PipeInstance:
    valves: ValvesType
    # Context extraction methods (placeholders for functions from common_context_utils)
    extract_context_from_messages: Callable[[List[Dict[str, Any]]], str]
    extract_context_from_files: Callable[[ValvesType, List[Dict[str, Any]]], Awaitable[str]]
    # Protocol execution method (placeholder for function from minion_protocol_logic)
    execute_minion_protocol: Callable[[ValvesType, str, str, Any, Any, Any], Awaitable[str]]
    # API call function (placeholder for function from common_api_calls)
    call_claude_api: Callable[[ValvesType, str], Awaitable[str]]
    # Local Assistant Response Model (placeholder for model from minion_models)
    local_assistant_response_model: LocalAssistantResponseModelType


async def _call_claude_directly_helper(
    pipe_self: PipeInstance,  # Provides access to self.valves and call_claude_api
    query: str
) -> str:
    """
    Fallback to direct Claude call when no context is available.
    This is a helper function for minion_pipe.
    """
    # Assumes call_claude_api is attached to pipe_self and is the refactored call_claude
    return await pipe_self.call_claude_api(pipe_self.valves, f"Please answer this question: {query}")

async def minion_pipe(
    self: PipeInstance, # Instance of the specific Pipe class
    body: Dict[str, Any],
    __user__: Dict[str, Any], # Included as per original, though not used in this specific logic
    __request__: Any, # fastapi.Request, type hinted as Any for broader compatibility
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "" # Included as per original
) -> str:
    """
    Executes the Minion (conversational) protocol.
    This function is intended to be the 'pipe' method of a class that has 'valves'
    and access to helper functions for context extraction and protocol execution.
    """
    try:
        # Validate configuration
        if not self.valves.anthropic_api_key: # type: ignore
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

        messages = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."

        user_query = messages[-1]["content"]

        # Extract context using helper functions expected to be on 'self'
        # These helpers are from common_context_utils.py
        context_from_messages = self.extract_context_from_messages(messages[:-1])
        context_from_files = await self.extract_context_from_files(self.valves, __files__)

        all_context = []
        if context_from_messages:
            all_context.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files:
            all_context.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")
        context = "\n\n".join(all_context) if all_context else ""

        if not context:
            # Call the local helper which in turn calls the main call_claude API function
            return (
                "ℹ️ **Note:** No significant context detected. Using standard remote model response.\n\n"
                + await _call_claude_directly_helper(self, user_query)
            )

        # Execute the Minion protocol using the main logic function
        # This function is from minion_protocol_logic.py
        # It requires call_claude and call_ollama, and the response model, to be passed through 'self'
        result = await self.execute_minion_protocol(
            self.valves, 
            user_query, 
            context,
            self.call_claude_api, # Pass the actual API call function for Claude
            getattr(self, 'call_ollama_api', None), # Pass Ollama call if available
            self.local_assistant_response_model # Pass the Pydantic model for local assistant responses
        )
        return result

    except Exception as e:
        # Keep traceback import local as it's only used here
        error_details = traceback.format_exc() if (hasattr(self.valves, 'debug_mode') and self.valves.debug_mode) else str(e) # type: ignore
        return f"❌ **Error in Minion protocol:** {error_details}"
