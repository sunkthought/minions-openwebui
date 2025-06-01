import asyncio
import json
from typing import List, Dict, Any, Callable # Removed Optional, Awaitable
from .minions_models import RoundMetrics # Import RoundMetrics

# parse_tasks function removed, will be part of minions_decomposition_logic.py

# Removed create_chunks function from here

def parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, TaskResultModel: Any) -> Dict:
    """Parse local model response, supporting both text and structured formats"""
    if is_structured and use_structured_output:
        try:
            parsed_json = json.loads(response)
            
            # Safety net: if answer is a dict/list, stringify it
            if 'answer' in parsed_json and not isinstance(parsed_json['answer'], (str, type(None))):
                if debug_mode:
                    print(f"DEBUG: Converting non-string answer to string: {type(parsed_json['answer'])}")
                parsed_json['answer'] = json.dumps(parsed_json['answer']) if parsed_json['answer'] else None
            
            validated_model = TaskResultModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            
            # Check if the structured response indicates "not found" via its 'answer' field
            if model_dict.get('answer') is None:
                model_dict['_is_none_equivalent'] = True
            else:
                model_dict['_is_none_equivalent'] = False
            return model_dict
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in MinionS: {e}. Response was: {response[:500]}")
            # Fallback for parsing failure
            is_none_equivalent_fallback = response.strip().upper() == "NONE"
            return {"answer": response, "explanation": response, "confidence": "LOW", "citation": None, "parse_error": str(e), "_is_none_equivalent": is_none_equivalent_fallback}
    
    # Fallback for non-structured processing
    is_none_equivalent_text = response.strip().upper() == "NONE"
    return {"answer": response, "explanation": response, "confidence": "MEDIUM", "citation": None, "parse_error": None, "_is_none_equivalent": is_none_equivalent_text}

async def execute_tasks_on_chunks(
    tasks: List[str], 
    chunks: List[str], 
    conversation_log: List[str], 
    current_round: int,
    valves: Any,
    call_ollama: Callable,
    TaskResult: Any
) -> Dict:
    """Execute tasks on chunks using local model"""
    overall_task_results = []
    total_attempts_this_call = 0
    total_timeouts_this_call = 0

    # Initialize Metrics
    round_start_time = asyncio.get_event_loop().time()
    tasks_executed_count = 0
    task_success_count = 0
    task_failure_count = 0
    chunk_processing_times = []

    for task_idx, task in enumerate(tasks):
        tasks_executed_count += 1 # Track Tasks Executed
        conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
        results_for_this_task_from_chunks = []
        chunk_timeout_count_for_task = 0
        num_relevant_chunks_found = 0

        for chunk_idx, chunk in enumerate(chunks):
            total_attempts_this_call += 1
            
            # Call the new function for local task prompt
            local_prompt = get_minions_local_task_prompt( # Ensure this function is defined or imported
                chunk=chunk,
                task=task,
                chunk_idx=chunk_idx,
                total_chunks=len(chunks),
                valves=valves
            )
            
            # Track Chunk Processing Time
            chunk_start_time = asyncio.get_event_loop().time()
            # start_time_ollama variable was previously used for debug,
            # let's ensure we use chunk_start_time for metrics consistently.
            # If start_time_ollama is still needed for debug, it can be kept separate.
            # For metrics, we'll use chunk_start_time and chunk_end_time.

            if valves.debug_mode:
                conversation_log.append(
                    f"   🔄 Task {task_idx + 1} - Trying chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} chars)... (Debug Mode)"
                )
                # start_time_ollama = asyncio.get_event_loop().time() # This was for debug, let's use chunk_start_time

            try:
                response_str = await asyncio.wait_for(
                    call_ollama(
                        valves,
                        local_prompt,
                        use_json=True,
                        schema=TaskResult
                    ),
                    timeout=valves.timeout_local,
                )
                chunk_end_time = asyncio.get_event_loop().time()
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)

                response_data = parse_local_response(
                    response_str,
                    is_structured=True,
                    use_structured_output=valves.use_structured_output,
                    debug_mode=valves.debug_mode,
                    TaskResultModel=TaskResult # Pass TaskResult to parse_local_response
                )
                
                if valves.debug_mode:
                    # end_time_ollama = asyncio.get_event_loop().time() # Already have chunk_end_time
                    time_taken_ollama = (chunk_end_time - chunk_start_time) # Use metric times for debug consistency
                    status_msg = ""
                    details_msg = ""
                    if response_data.get("parse_error"):
                        status_msg = "Parse Error"
                        details_msg = f"Error: {response_data['parse_error']}, Raw: {response_data.get('answer', '')[:70]}..."
                    elif response_data['_is_none_equivalent']:
                        status_msg = "No relevant info"
                        details_msg = f"Response indicates no info found. Confidence: {response_data.get('confidence', 'N/A')}"
                    else:
                        status_msg = "Relevant info found"
                        details_msg = f"Answer: {response_data.get('answer', '')[:70]}..., Confidence: {response_data.get('confidence', 'N/A')}"
        
                    conversation_log.append(
                         f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} processed by local LLM in {time_taken_ollama:.2f}s. Status: {status_msg}. Details: {details_msg} (Debug Mode)"
                    )

                if not response_data['_is_none_equivalent']:
                    extracted_info = response_data.get('answer') or response_data.get('explanation', 'Could not extract details.')
                    results_for_this_task_from_chunks.append(f"[Chunk {chunk_idx+1}]: {extracted_info}")
                    num_relevant_chunks_found += 1
                    
            except asyncio.TimeoutError:
                chunk_end_time = asyncio.get_event_loop().time() # Capture time even on timeout
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)
                total_timeouts_this_call += 1
                chunk_timeout_count_for_task += 1
                conversation_log.append(
                    f"   ⏰ Task {task_idx + 1} - Chunk {chunk_idx + 1} timed out after {valves.timeout_local}s"
                )
                if valves.debug_mode:
                    # end_time_ollama = asyncio.get_event_loop().time() # Already have chunk_end_time
                    time_taken_ollama = (chunk_end_time - chunk_start_time) # Use metric times
                    conversation_log.append(
                         f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} TIMEOUT after {time_taken_ollama:.2f}s. (Debug Mode)"
                    )
            except Exception as e:
                # It's good practice to also record chunk processing time if an unexpected exception occurs
                chunk_end_time = asyncio.get_event_loop().time()
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)
                conversation_log.append(
                    f"   ❌ Task {task_idx + 1} - Chunk {chunk_idx + 1} error: {str(e)}"
                )
        
        # Track Task Success/Failure
        if results_for_this_task_from_chunks:
            task_success_count += 1
            aggregated_result_for_task = "\n".join(results_for_this_task_from_chunks)
            overall_task_results.append({"task": task, "result": aggregated_result_for_task, "status": "success"})
            conversation_log.append(
                f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). First result snippet: {results_for_this_task_from_chunks[0][:100]}..."
            )
        elif chunk_timeout_count_for_task > 0 and chunk_timeout_count_for_task == len(chunks):
            task_failure_count += 1 # All chunks timed out
            overall_task_results.append({"task": task, "result": f"Timeout on all {len(chunks)} chunks", "status": "timeout_all_chunks"})
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out."
            )
        else:
            task_failure_count += 1 # No relevant info found or other errors
            overall_task_results.append(
                {"task": task, "result": "Information not found in any relevant chunk", "status": "not_found"}
            )
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found in any chunk."
            )
    
    # Calculate Final Metrics
    round_end_time = asyncio.get_event_loop().time()
    execution_time_ms = (round_end_time - round_start_time) * 1000
    avg_chunk_processing_time_ms = sum(chunk_processing_times) / len(chunk_processing_times) if chunk_processing_times else 0
    success_rate = task_success_count / tasks_executed_count if tasks_executed_count > 0 else 0

    # Prepare Metrics Object (as a dictionary for now, as per instructions)
    round_metrics_data = {
        "round_number": current_round,
        "tasks_executed": tasks_executed_count,
        "task_success_count": task_success_count,
        "task_failure_count": task_failure_count,
        "avg_chunk_processing_time_ms": avg_chunk_processing_time_ms,
        "execution_time_ms": execution_time_ms,
        "success_rate": success_rate,
        # total_unique_findings_count will be handled later, defaulting in the model
    }

    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_attempts_this_call,
        "total_chunk_processing_timeouts": total_timeouts_this_call,
        "round_metrics_data": round_metrics_data
    }

def calculate_token_savings(
    decomposition_prompts: List[str], 
    synthesis_prompts: List[str],
    all_results_summary: str, 
    final_response: str, 
    context_length: int, 
    query_length: int, 
    total_chunks: int,
    total_tasks: int
) -> dict:
    """Calculate token savings for MinionS protocol"""
    chars_per_token = 3.5
    
    # Traditional approach: entire context + query sent to Claude
    traditional_tokens = int((context_length + query_length) / chars_per_token)
    
    # MinionS approach: only prompts and summaries sent to Claude
    minions_tokens = 0
    for p in decomposition_prompts:
        minions_tokens += int(len(p) / chars_per_token)
    for p in synthesis_prompts:
        minions_tokens += int(len(p) / chars_per_token)
    minions_tokens += int(len(all_results_summary) / chars_per_token)
    minions_tokens += int(len(final_response) / chars_per_token)
    
    token_savings = traditional_tokens - minions_tokens
    percentage_savings = (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    
    return {
        'traditional_tokens_claude': traditional_tokens,
        'minions_tokens_claude': minions_tokens,
        'token_savings_claude': token_savings,
        'percentage_savings_claude': percentage_savings,
        'total_rounds': len(decomposition_prompts),
        'total_chunks_processed_local': total_chunks,
        'total_tasks_executed_local': total_tasks,
    }