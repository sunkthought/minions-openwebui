import asyncio
import json
from typing import List, Optional, Dict, Any, Callable, Awaitable

def parse_tasks(claude_response: str, max_tasks: int) -> List[str]:
    """Parse tasks from Claude's response"""
    lines = claude_response.split("\n")
    tasks = []
    for line in lines:
        line = line.strip()
        # More robust parsing for numbered or bulleted lists
        if line.startswith(tuple(f"{i}." for i in range(1, 10))) or \
           line.startswith(tuple(f"{i})" for i in range(1, 10))) or \
           line.startswith(("- ", "* ", "+ ")):
            task = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            if len(task) > 10:  # Keep simple task filter
                tasks.append(task)
    return tasks[:max_tasks]

def create_chunks(context: str, chunk_size: int, max_chunks: int) -> List[str]:
    """Create chunks from context"""
    if not context:
        return []
    actual_chunk_size = max(1, min(chunk_size, len(context)))
    chunks = [
        context[i : i + actual_chunk_size]
        for i in range(0, len(context), actual_chunk_size)
    ]
    return chunks[:max_chunks] if max_chunks > 0 else chunks

def parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool) -> Dict:
    """Parse local model response, supporting both text and structured formats"""
    if is_structured and use_structured_output:
        try:
            parsed_json = json.loads(response)
            validated_model = TaskResult(**parsed_json)
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

    for task_idx, task in enumerate(tasks):
        conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
        results_for_this_task_from_chunks = []
        chunk_timeout_count_for_task = 0
        num_relevant_chunks_found = 0

        for chunk_idx, chunk in enumerate(chunks):
            total_attempts_this_call += 1
            local_prompt = f'''Text to analyze (Chunk {chunk_idx + 1}/{len(chunks)} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}'''

            if valves.use_structured_output:
                local_prompt += f"\n\nProvide your answer ONLY as a valid JSON object matching the specified schema. If no relevant information is found in THIS SPECIFIC TEXT, ensure the 'answer' field in your JSON response is explicitly set to null (or None)."
            else:
                local_prompt += "\n\nProvide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\"."
            
            start_time_ollama = 0
            if valves.debug_mode:
                conversation_log.append(
                    f"   🔄 Task {task_idx + 1} - Trying chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} chars)... (Debug Mode)"
                )
                start_time_ollama = asyncio.get_event_loop().time()

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
                response_data = parse_local_response(
                    response_str,
                    is_structured=True,
                    use_structured_output=valves.use_structured_output,
                    debug_mode=valves.debug_mode
                )
                
                if valves.debug_mode:
                    end_time_ollama = asyncio.get_event_loop().time()
                    time_taken_ollama = end_time_ollama - start_time_ollama
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
                total_timeouts_this_call += 1
                chunk_timeout_count_for_task += 1
                conversation_log.append(
                    f"   ⏰ Task {task_idx + 1} - Chunk {chunk_idx + 1} timed out after {valves.timeout_local}s"
                )
                if valves.debug_mode:
                    end_time_ollama = asyncio.get_event_loop().time()
                    time_taken_ollama = end_time_ollama - start_time_ollama
                    conversation_log.append(
                         f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} TIMEOUT after {time_taken_ollama:.2f}s. (Debug Mode)"
                    )
            except Exception as e:
                conversation_log.append(
                    f"   ❌ Task {task_idx + 1} - Chunk {chunk_idx + 1} error: {str(e)}"
                )
        
        if results_for_this_task_from_chunks:
            aggregated_result_for_task = "\n".join(results_for_this_task_from_chunks)
            overall_task_results.append({"task": task, "result": aggregated_result_for_task, "status": "success"})
            conversation_log.append(
                f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). First result snippet: {results_for_this_task_from_chunks[0][:100]}..."
            )
        elif chunk_timeout_count_for_task > 0 and chunk_timeout_count_for_task == len(chunks):
             overall_task_results.append({"task": task, "result": f"Timeout on all {len(chunks)} chunks", "status": "timeout_all_chunks"})
             conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out."
            )
        else:
            overall_task_results.append(
                {"task": task, "result": "Information not found in any relevant chunk", "status": "not_found"}
            )
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found in any chunk."
            )
    
    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_attempts_this_call,
        "total_chunk_processing_timeouts": total_timeouts_this_call
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