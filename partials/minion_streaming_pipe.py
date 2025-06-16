# Partials File: partials/minion_streaming_pipe.py

import asyncio
from typing import Any, List, Callable, Dict, AsyncGenerator
from fastapi import Request

async def minion_pipe(
    pipe_self: Any,
    body: Dict[str, Any],
    __user__: Dict[str, Any],
    __request__: Request,
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "minion-claude",
) -> AsyncGenerator[str, None]:
    """Execute the Minion protocol with streaming updates"""
    async for chunk in minion_pipe_streaming(pipe_self, body, __user__, __request__, __files__, __pipe_id__):
        yield chunk


async def _call_supervisor_directly(valves: Any, query: str) -> str:
    """Fallback to direct supervisor call when no context is available"""
    return await call_supervisor_model(valves, f"Please answer this question: {query}")


async def _execute_simplified_chunk_protocol(
    valves: Any,
    query: str,
    context: str,
    chunk_num: int,
    total_chunks: int,
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
        # Round 1: Get supervisor's question
        supervisor_response = await call_supervisor_model(valves, claude_prompt)
        
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Remote Model (Chunk {chunk_num}):** {supervisor_response}")
        
        # Get local response
        local_prompt = get_minion_local_prompt(context, query, supervisor_response, valves)
        
        local_response_str = await call_ollama_func(
            valves,
            local_prompt,
            use_json=True,
            schema=LocalAssistantResponseModel
        )
        
        # Parse local response (simplified)
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

        final_response = await call_supervisor_model(valves, synthesis_prompt)
        
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


async def minion_pipe_streaming(
    pipe_self: Any,
    body: Dict[str, Any],
    __user__: Dict[str, Any],
    __request__: Request,
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "minion-claude",
) -> AsyncGenerator[str, None]:
    """Execute the Minion protocol with streaming updates"""
    
    # Initialize streaming manager
    streaming_manager = StreamingResponseManager(pipe_self.valves, pipe_self.valves.debug_mode)
    
    try:
        # Validate configuration with streaming update
        yield await streaming_manager.stream_phase_update("configuration", "Validating API keys and settings")
        
        provider = getattr(pipe_self.valves, 'supervisor_provider', 'anthropic')
        if provider == 'anthropic' and not pipe_self.valves.anthropic_api_key:
            yield await streaming_manager.stream_error_update("Please configure your Anthropic API key in the function settings.", "configuration")
            return
        elif provider == 'openai' and not pipe_self.valves.openai_api_key:
            yield await streaming_manager.stream_error_update("Please configure your OpenAI API key in the function settings.", "configuration")
            return

        # Extract user message and context with progress
        yield await streaming_manager.stream_phase_update("query_analysis", "Processing user query and context")
        
        messages: List[Dict[str, Any]] = body.get("messages", [])
        if not messages:
            yield await streaming_manager.stream_error_update("No messages provided.", "query_analysis")
            return

        user_query: str = messages[-1]["content"]

        # Extract context from messages AND uploaded files
        yield await streaming_manager.stream_phase_update("document_retrieval", "Extracting context from messages and files")
        
        context_from_messages: str = extract_context_from_messages(messages[:-1])
        context_from_files: str = await extract_context_from_files(pipe_self.valves, __files__)

        # Combine all context sources
        all_context_parts: List[str] = []
        if context_from_messages:
            all_context_parts.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files:
            all_context_parts.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")

        context: str = "\n\n".join(all_context_parts) if all_context_parts else ""

        # If no context, make a direct call to supervisor
        if not context:
            yield await streaming_manager.stream_phase_update("answer_synthesis", "No context detected, calling supervisor directly")
            direct_response = await _call_supervisor_directly(pipe_self.valves, user_query)
            provider_name = getattr(pipe_self.valves, 'supervisor_provider', 'anthropic').title()
            
            final_response = (
                f"ℹ️ **Note:** No significant context detected. Using standard {provider_name} response.\n\n"
                + direct_response
            )
            
            yield f"\n## 🎯 Final Answer\n{final_response}"
            return

        # Execute with streaming progress updates
        yield await streaming_manager.stream_phase_update("conversation", f"Starting Minion protocol with {len(context)} characters of context")
        
        # Handle chunking for large documents
        chunks = create_chunks(context, pipe_self.valves.chunk_size, pipe_self.valves.max_chunks)
        if not chunks and context:
            yield await streaming_manager.stream_error_update("Context provided, but failed to create any processable chunks. Check chunk_size setting.", "chunking")
            return

        # Stream chunk analysis
        yield await streaming_manager.stream_granular_update(
            "conversation", "document_analysis", 0.2,
            f"Analyzing document structure ({len(chunks)} chunks)"
        )
        
        if len(chunks) <= 2:
            # For small number of chunks, combine them (performance optimization)
            combined_context = "\n\n".join([f"=== CHUNK {i+1} OF {len(chunks)} ===\n{chunk}" for i, chunk in enumerate(chunks)])
            
            # Execute streaming minion protocol
            async for update in _execute_minion_protocol_streaming(
                valves=pipe_self.valves,
                query=user_query,
                context=combined_context,
                call_ollama_func=call_ollama,
                LocalAssistantResponseModel=LocalAssistantResponse,
                ConversationStateModel=ConversationState,
                QuestionDeduplicatorModel=QuestionDeduplicator,
                ConversationFlowControllerModel=ConversationFlowController,
                AnswerValidatorModel=AnswerValidator,
                streaming_manager=streaming_manager
            ):
                yield update
            
        else:
            # For many chunks, use chunk-by-chunk processing with streaming
            conversation_state = ConversationState() if pipe_self.valves.track_conversation_state else None
            deduplicator = QuestionDeduplicator(pipe_self.valves.deduplication_threshold) if pipe_self.valves.enable_deduplication else None
            
            # Collect chunk results for synthesis
            chunk_results = []
            
            # Start the chunk analysis section
            yield f"\n## 📄 Chunk-by-Chunk Analysis\n"
            
            for i, chunk in enumerate(chunks):
                chunk_header = f"### Chunk {i+1} of {len(chunks)}\n"
                
                # Stream chunk processing progress
                progress = (i + 1) / len(chunks)
                yield await streaming_manager.stream_granular_update(
                    "conversation", f"chunk_{i+1}", progress,
                    f"Processing chunk {i+1}/{len(chunks)}"
                )
                
                # Execute simplified protocol for this chunk
                chunk_result = await _execute_simplified_chunk_protocol(
                    pipe_self.valves,
                    user_query,
                    chunk,
                    i+1,
                    len(chunks),
                    call_ollama,
                    LocalAssistantResponse,
                    conversation_state,
                    deduplicator
                )
                
                # Store result for synthesis
                chunk_results.append(chunk_result)
                
                # Yield this chunk's result immediately (progressive streaming)
                yield f"{chunk_header}{chunk_result}\n\n"
            
            # Synthesis phase
            yield await streaming_manager.stream_phase_update("synthesis", "Synthesizing final answer from all chunks")
            
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
                final_answer = await call_supervisor_model(pipe_self.valves, synthesis_prompt)
                
                # Yield the synthesized final answer
                yield f"\n---\n\n## 🎯 Final Answer\n\n{final_answer}\n"
                
            except Exception as e:
                # Fallback to basic synthesis if API call fails
                yield f"\n---\n\n## 🎯 Final Answer\n\n"
                yield "Based on the chunk analyses above, here's a summary of the key findings:\n\n"
                for i, result in enumerate(chunk_results):
                    if "**Result:**" in result:
                        summary = result.split("**Result:**")[1].strip()
                        yield f"• From Chunk {i+1}: {summary}\n\n"
            
            # Add multi-chunk processing info at the end
            chunk_info = f"\n---\n\n## 📄 Multi-Chunk Processing Info\n**Document processed as {len(chunks)} chunks** (max {pipe_self.valves.chunk_size:,} characters each) in {len(chunks)} separate conversation sessions."
            yield f"{chunk_info}"

    except Exception as e:
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        yield await streaming_manager.stream_error_update(f"Error in Minion protocol: {error_details}", "general")


async def _execute_minion_protocol_streaming(
    valves: Any,
    query: str,
    context: str,
    call_ollama_func: Callable,
    LocalAssistantResponseModel: Any,
    ConversationStateModel: Any = None,
    QuestionDeduplicatorModel: Any = None,
    ConversationFlowControllerModel: Any = None,
    AnswerValidatorModel: Any = None,
    streaming_manager: Any = None
) -> AsyncGenerator[str, None]:
    """Execute the Minion protocol with streaming updates"""
    
    conversation_log = []
    debug_log = []
    conversation_history = []
    final_response = "I was unable to generate a response."
    
    # Initialize conversation state
    conversation_state = None
    if ConversationStateModel and valves.track_conversation_state:
        conversation_state = ConversationStateModel()
    
    # Initialize question deduplicator
    deduplicator = None
    if valves.enable_deduplication and QuestionDeduplicatorModel:
        deduplicator = QuestionDeduplicatorModel(
            similarity_threshold=valves.deduplication_threshold
        )
    
    # Initialize flow controller
    flow_controller = None
    if valves.enable_flow_control and ConversationFlowControllerModel:
        flow_controller = ConversationFlowControllerModel()
    
    # Initialize answer validator
    validator = None
    if valves.enable_answer_validation and AnswerValidatorModel:
        validator = AnswerValidatorModel()
    
    # Initialize metrics tracking
    overall_start_time = asyncio.get_event_loop().time()
    metrics = {
        'confidence_scores': [],
        'confidence_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
        'rounds_completed': 0,
        'completion_via_detection': False,
        'estimated_tokens': 0,
        'chunk_size_used': valves.chunk_size,
        'context_size': len(context)
    }

    if streaming_manager:
        update = await streaming_manager.stream_granular_update(
            "conversation", "initialization", 0.1,
            f"Initializing conversation with {valves.max_rounds} max rounds"
        )
        if update:
            yield update

    for round_num in range(valves.max_rounds):
        if streaming_manager:
            update = await streaming_manager.stream_conversation_progress(
                round_num=round_num + 1,
                max_rounds=valves.max_rounds,
                stage="questioning"
            )
            if update:
                yield update

        # Get phase guidance if flow control is enabled
        phase_guidance = None
        if flow_controller and valves.enable_flow_control:
            phase_guidance = flow_controller.get_phase_guidance()

        # Generate Claude prompt for this round
        claude_prompt_for_this_round = ""
        if round_num == 0:
            if conversation_state and valves.track_conversation_state:
                claude_prompt_for_this_round = get_minion_initial_claude_prompt_with_state(
                    query, len(context), valves, conversation_state, phase_guidance
                )
            else:
                claude_prompt_for_this_round = get_minion_initial_claude_prompt(query, len(context), valves)
        else:
            # Check if this is the last round and force a final answer
            is_last_round = (round_num == valves.max_rounds - 1)
            if is_last_round:
                claude_prompt_for_this_round = f"""You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: "{query}"

The local assistant has full access to the source document and has been providing factual information extracted from it.

CONVERSATION SO FAR:
"""
                for role, message in conversation_history:
                    if role == "assistant":
                        claude_prompt_for_this_round += f"\nYou previously asked: \"{message}\""
                    else:
                        claude_prompt_for_this_round += f"\nLocal assistant responded: \"{message}\""
                
                claude_prompt_for_this_round += f"""

THIS IS YOUR FINAL OPPORTUNITY TO ANSWER. You have gathered sufficient information through {round_num} rounds of questions.

Based on ALL the information provided by the local assistant, you MUST now provide a comprehensive answer to the user's original question: "{query}"

Respond with "FINAL ANSWER READY." followed by your synthesized answer. Do NOT ask any more questions."""
            else:
                if conversation_state and valves.track_conversation_state:
                    previous_questions = deduplicator.get_all_questions() if deduplicator else None
                    claude_prompt_for_this_round = get_minion_conversation_claude_prompt_with_state(
                        conversation_history, query, valves, conversation_state, previous_questions, phase_guidance
                    )
                else:
                    claude_prompt_for_this_round = get_minion_conversation_claude_prompt(
                        conversation_history, query, valves
                    )

        # Call Claude
        if streaming_manager:
            update = await streaming_manager.stream_conversation_progress(
                round_num=round_num + 1,
                max_rounds=valves.max_rounds,
                stage="processing"
            )
            if update:
                yield update

        try:
            claude_response = await call_supervisor_model(valves, claude_prompt_for_this_round)
            conversation_history.append(("assistant", claude_response))
            
            # Check for final answer
            if _is_final_answer(claude_response):
                final_response = claude_response
                if streaming_manager:
                    update = await streaming_manager.stream_phase_update("completion", f"Final answer ready after {round_num + 1} rounds")
                    if update:
                        yield update
                break
            
            # Check completion detection
            if detect_completion(claude_response):
                final_response = claude_response
                metrics['completion_via_detection'] = True
                if streaming_manager:
                    update = await streaming_manager.stream_phase_update("completion", f"Completion detected after {round_num + 1} rounds")
                    if update:
                        yield update
                break

            # Extract question from Claude's response
            question = claude_response.strip()
            
            # Check for duplicate questions
            if deduplicator and valves.enable_deduplication:
                if deduplicator.is_duplicate(question):
                    continue
                deduplicator.add_question(question)

            # Call local model
            if streaming_manager:
                update = await streaming_manager.stream_conversation_progress(
                    round_num=round_num + 1,
                    max_rounds=valves.max_rounds,
                    stage="analyzing"
                )
                if update:
                    yield update

            local_prompt = get_minion_local_prompt(context, query, question, valves)
            
            local_response_str = await asyncio.wait_for(
                call_ollama_func(
                    valves,
                    local_prompt,
                    use_json=True,
                    schema=LocalAssistantResponseModel
                ),
                timeout=valves.timeout_local,
            )

            # Parse response
            local_response_data = _parse_local_response(
                local_response_str,
                is_structured=True,
                use_structured_output=valves.use_structured_output,
                debug_mode=valves.debug_mode,
                LocalAssistantResponseModel=LocalAssistantResponseModel
            )

            response_text = local_response_data.get('response', 'No response provided')
            conversation_history.append(("user", response_text))
            
            # Update metrics
            confidence = local_response_data.get('confidence', 'LOW')
            metrics['confidence_scores'].append(confidence)
            metrics['confidence_distribution'][confidence] += 1

        except Exception as e:
            if streaming_manager:
                update = await streaming_manager.stream_error_update(f"Error in round {round_num + 1}: {str(e)}", "conversation")
                if update:
                    yield update
            continue

    # Update metrics
    metrics['rounds_completed'] = min(round_num + 1, valves.max_rounds)
    
    # Calculate token savings
    token_savings = _calculate_token_savings(conversation_history, context, query)
    
    # Yield final result
    yield f"\n## 🎯 Final Answer\n{final_response}"