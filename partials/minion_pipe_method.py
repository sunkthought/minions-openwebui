# Partials File: partials/minion_pipe_method.py
import asyncio
from typing import Any, List, Callable, Dict
from fastapi import Request

from .common_api_calls import call_claude, call_ollama
from .minion_protocol_logic import _execute_minion_protocol
from .minion_models import LocalAssistantResponse, ConversationState, QuestionDeduplicator, ConversationFlowController, AnswerValidator # Import all models
from .common_context_utils import extract_context_from_messages, extract_context_from_files
from .common_file_processing import create_chunks

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

        # Handle chunking for large documents
        chunks = create_chunks(context, pipe_self.valves.chunk_size, pipe_self.valves.max_chunks)
        if not chunks and context:
            return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size setting."
        
        # Process all chunks together in one protocol execution (like MinionS)
        # This prevents the performance issue of running full protocol per chunk
        combined_context = "\n\n".join([f"=== CHUNK {i+1} OF {len(chunks)} ===\n{chunk}" for i, chunk in enumerate(chunks)])
        
        result: str = await _execute_minion_protocol(
            valves=pipe_self.valves, 
            query=user_query, 
            context=combined_context, 
            call_claude_func=call_claude,
            call_ollama_func=call_ollama,
            LocalAssistantResponseModel=LocalAssistantResponse,
            ConversationStateModel=ConversationState,
            QuestionDeduplicatorModel=QuestionDeduplicator,
            ConversationFlowControllerModel=ConversationFlowController,
            AnswerValidatorModel=AnswerValidator
        )
        
        if len(chunks) > 1:
            # Add multi-chunk processing note to the result
            chunk_info = f"\n\n---\n\n## 📄 Multi-Chunk Processing Info\n**Document processed as {len(chunks)} chunks** (max {pipe_self.valves.chunk_size:,} characters each) in a single conversation session for optimal performance."
            result += chunk_info
        
        return result

    except Exception as e:
        import traceback # Keep import here as it's conditional
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"