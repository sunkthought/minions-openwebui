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

async def _execute_simplified_chunk_protocol(
    valves: Any,
    query: str,
    context: str,
    chunk_num: int,
    total_chunks: int,
    call_claude_func: Callable,
    call_ollama_func: Callable,
    LocalAssistantResponseModel: Any,
    shared_state: Any = None,
    shared_deduplicator: Any = None
) -> str:
    """Execute a simplified 1-2 round protocol for individual chunks"""
    conversation_log = []
    debug_log = []
    
    # Create a focused prompt for this chunk
    claude_prompt = f"""You are analyzing chunk {chunk_num} of {total_chunks} from a larger document to answer: "{query}"

Here is the chunk content:

<document_chunk>
{context}
</document_chunk>

This chunk contains a portion of the document. Your task is to:
1. Ask ONE focused question to extract the most relevant information from this chunk
2. Based on the response, either ask ONE follow-up question OR provide a final answer for this chunk

Be concise and focused since this is one chunk of a larger analysis.

Ask your first question about this chunk:"""

    try:
        # Round 1: Get Claude's question
        claude_response = await call_claude_func(valves, claude_prompt)
        
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Remote Model (Chunk {chunk_num}):** {claude_response}")
        
        # Get local response
        from .minion_prompts import get_minion_local_prompt
        local_prompt = get_minion_local_prompt(context, query, claude_response, valves)
        
        local_response_str = await call_ollama_func(
            valves,
            local_prompt,
            use_json=True,
            schema=LocalAssistantResponseModel
        )
        
        # Parse local response (simplified)
        from .minion_protocol_logic import _parse_local_response
        local_response_data = _parse_local_response(
            local_response_str,
            is_structured=True,
            use_structured_output=valves.use_structured_output,
            debug_mode=valves.debug_mode,
            LocalAssistantResponseModel=LocalAssistantResponseModel
        )
        
        local_answer = local_response_data.get("answer", "No answer provided")
        
        if valves.show_conversation:
            conversation_log.append(f"**💻 Local Model (Chunk {chunk_num}):** {local_answer}")
        
        # Round 2: Quick synthesis
        synthesis_prompt = f"""Based on the local assistant's response: "{local_answer}"

Provide a brief summary of what this chunk (#{chunk_num} of {total_chunks}) contributes to answering: "{query}"

Keep it concise since this is just one part of a larger document analysis."""

        final_response = await call_claude_func(valves, synthesis_prompt)
        
        if valves.show_conversation:
            conversation_log.append(f"**🎯 Chunk Summary:** {final_response}")
        
        # Build output
        output_parts = []
        if valves.show_conversation:
            output_parts.extend(conversation_log)
        output_parts.append(f"**Result:** {final_response}")
        
        return "\n\n".join(output_parts)
        
    except Exception as e:
        return f"❌ Error processing chunk {chunk_num}: {str(e)}"

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
        
        if len(chunks) <= 2:
            # For small number of chunks, combine them (performance optimization)
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
                chunk_info = f"\n\n---\n\n## 📄 Multi-Chunk Processing Info\n**Document processed as {len(chunks)} chunks** (max {pipe_self.valves.chunk_size:,} characters each) in a single conversation session."
                result += chunk_info
            
            return result
        else:
            # For many chunks, use lightweight chunk-by-chunk processing
            # This avoids overwhelming the local model while still being efficient
            chunk_results = []
            conversation_state = ConversationState() if pipe_self.valves.track_conversation_state else None
            deduplicator = QuestionDeduplicator(pipe_self.valves.deduplication_threshold) if pipe_self.valves.enable_deduplication else None
            
            for i, chunk in enumerate(chunks):
                chunk_header = f"## 📄 Chunk {i+1} of {len(chunks)}\n"
                
                try:
                    # Use a simplified single-round protocol for efficiency
                    chunk_result = await _execute_simplified_chunk_protocol(
                        valves=pipe_self.valves,
                        query=user_query,
                        context=chunk,
                        chunk_num=i+1,
                        total_chunks=len(chunks),
                        call_claude_func=call_claude,
                        call_ollama_func=call_ollama,
                        LocalAssistantResponseModel=LocalAssistantResponse,
                        shared_state=conversation_state,
                        shared_deduplicator=deduplicator
                    )
                    chunk_results.append(chunk_header + chunk_result)
                except Exception as e:
                    chunk_results.append(f"{chunk_header}❌ **Error processing chunk {i+1}:** {str(e)}")
            
            # Generate final synthesis from all chunk results
            synthesis_prompt = f"""You have analyzed a document in {len(chunks)} chunks to answer: "{user_query}"

Here are the results from each chunk:

{chr(10).join(chunk_results)}

Based on all the information gathered from these chunks, provide a comprehensive final answer to the user's original question: "{user_query}"

Your response should:
1. Synthesize the key information from all chunks
2. Address the specific question asked
3. Be well-organized and coherent
4. Include the most important findings and insights

Provide your final comprehensive answer:"""

            try:
                final_answer = await call_claude(valves=pipe_self.valves, prompt=synthesis_prompt)
            except Exception as e:
                final_answer = f"❌ Error generating final synthesis: {str(e)}"
            
            # Combine all chunk results with final synthesis
            combined_result = "\n\n---\n\n".join(chunk_results)
            
            # Add summary header with final answer
            summary_header = f"""# 🔗 Multi-Chunk Analysis Results
            
**Document processed in {len(chunks)} chunks** (max {pipe_self.valves.chunk_size:,} characters each)

{combined_result}

---

## 🎯 Final Answer

{final_answer}

---

## 📋 Summary
The document was automatically divided into {len(chunks)} chunks for efficient processing. Each chunk was analyzed using an optimized protocol to balance performance with thoroughness."""
            
            return summary_header

    except Exception as e:
        import traceback # Keep import here as it's conditional
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"