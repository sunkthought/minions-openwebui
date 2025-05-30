import asyncio
from typing import Any, List, Callable, Dict
from fastapi import Request

from .common_api_calls import call_claude, call_ollama
from .minion_protocol_logic import _execute_minion_protocol
from .minion_models import LocalAssistantResponse # Assuming this is the correct model name
from .common_context_utils import extract_context_from_messages, extract_context_from_files

async def _call_claude_directly(valves: Any, query: str, call_claude_func: Callable) -> str: # Renamed for clarity
    """Fallback to direct Claude call when no context is available"""
    return await call_claude_func(valves, f"Please answer this question: {query}")

async def minion_pipe(
    pipe_self: Any,
    body: Dict[str, Any], # Typed body
    __user__: Dict[str, Any], # Typed __user__
    __request__: Request,
    __files__: List[Dict[str, Any]] = [], # Typed __files__
    __pipe_id__: str = "minion-claude",
) -> str:
    """Execute the Minion protocol with Claude"""
    try:
        # Validate configuration
        if not pipe_self.valves.anthropic_api_key: # Add ollama key check if necessary
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

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

        if not context:
            # Pass the imported call_claude to _call_claude_directly
            direct_response = await _call_claude_directly(pipe_self.valves, user_query, call_claude_func=call_claude)
            return (
                "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                + direct_response
            )

        # Execute the Minion protocol, passing the imported call_claude, call_ollama, and LocalAssistantResponse
        # The _execute_minion_protocol itself expects these as arguments.
        result: str = await _execute_minion_protocol(
            valves=pipe_self.valves, 
            query=user_query, 
            context=context, 
            call_claude_func=call_claude,  # Pass imported function
            call_ollama_func=call_ollama,  # Pass imported function
            LocalAssistantResponseModel=LocalAssistantResponse # Pass imported class
        )
        return result

    except Exception as e:
        import traceback # Keep import here as it's conditional
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"