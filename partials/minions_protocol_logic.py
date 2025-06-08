# Partials File: partials/minions_protocol_logic.py
import asyncio
import json
import hashlib # Import hashlib
from typing import List, Dict, Any, Callable # Removed Optional, Awaitable
from .minions_models import RoundMetrics # Import RoundMetrics

# parse_tasks function removed, will be part of minions_decomposition_logic.py

# Removed create_chunks function from here

def parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, TaskResultModel: Any, structured_output_fallback_enabled: bool = True) -> Dict:
    """Parse local model response, supporting both text and structured formats"""
    confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
    default_numeric_confidence = 0.3 # Corresponds to LOW

    if is_structured and use_structured_output:
        # Clean up common formatting issues
        cleaned_response = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[7:-3].strip()
        elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
        
        # Try to extract JSON from response with explanatory text
        if not cleaned_response.startswith("{"):
            # Look for JSON object in the response
            json_start = cleaned_response.find("{")
            json_end = cleaned_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_response = cleaned_response[json_start:json_end+1]
        
        try:
            parsed_json = json.loads(cleaned_response)
            
            # Handle missing confidence field with default
            if 'confidence' not in parsed_json:
                parsed_json['confidence'] = 'LOW'
            
            # Safety net: if answer is a dict/list, stringify it
            if 'answer' in parsed_json and not isinstance(parsed_json['answer'], (str, type(None))):
                if debug_mode:
                    print(f"DEBUG: Converting non-string answer to string: {type(parsed_json['answer'])}")
                parsed_json['answer'] = json.dumps(parsed_json['answer']) if parsed_json['answer'] else None
            
            # Ensure required fields have defaults if missing
            if 'explanation' not in parsed_json:
                parsed_json['explanation'] = parsed_json.get('answer', '') or "No explanation provided"
            if 'citation' not in parsed_json:
                parsed_json['citation'] = None
            
            validated_model = TaskResultModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            
            text_confidence = model_dict.get('confidence', 'LOW').upper()
            model_dict['numeric_confidence'] = confidence_map.get(text_confidence, default_numeric_confidence)

            # Check if the structured response indicates "not found" via its 'answer' field
            if model_dict.get('answer') is None:
                model_dict['_is_none_equivalent'] = True
            else:
                model_dict['_is_none_equivalent'] = False
            return model_dict
            
        except json.JSONDecodeError as e:
            if debug_mode:
                print(f"DEBUG: JSON decode error in MinionS: {e}. Cleaned response was: {cleaned_response[:500]}")
            
            if not structured_output_fallback_enabled:
                # Re-raise the error if fallback is disabled
                raise e
            
            # Try regex fallback to extract key information
            import re
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
            confidence_match = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', response, re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1)
                confidence = confidence_match.group(1).upper() if confidence_match else "LOW"
                return {
                    "answer": answer,
                    "explanation": f"Extracted from malformed JSON: {answer}",
                    "confidence": confidence,
                    "numeric_confidence": confidence_map.get(confidence, default_numeric_confidence),
                    "citation": None,
                    "parse_error": f"JSON parse error (recovered): {str(e)}",
                    "_is_none_equivalent": answer.strip().upper() == "NONE"
                }
            
            # Complete fallback
            is_none_equivalent_fallback = response.strip().upper() == "NONE"
            return {
                "answer": response, 
                "explanation": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence, 
                "parse_error": f"JSON parse error: {str(e)}", 
                "_is_none_equivalent": is_none_equivalent_fallback
            }
            
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in MinionS: {e}. Response was: {response[:500]}")
            # Fallback for parsing failure
            is_none_equivalent_fallback = response.strip().upper() == "NONE"
            return {
                "answer": response, 
                "explanation": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence, 
                "parse_error": str(e), 
                "_is_none_equivalent": is_none_equivalent_fallback
            }
    
    # Fallback for non-structured processing
    is_none_equivalent_text = response.strip().upper() == "NONE"
    # Confidence is MEDIUM by default in this path
    return {
        "answer": response, 
        "explanation": response, 
        "confidence": "MEDIUM", 
        "numeric_confidence": confidence_map['MEDIUM'], 
        "citation": None, 
        "parse_error": None, 
        "_is_none_equivalent": is_none_equivalent_text
    }

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
    # Initialize Confidence Accumulators
    round_confidence_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    aggregated_task_confidences = []
    
    # Structured output metrics
    structured_output_attempts = 0
    structured_output_successes = 0

    for task_idx, task in enumerate(tasks):
        tasks_executed_count += 1 # Track Tasks Executed
        conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
        results_for_this_task_from_chunks = []
        current_task_chunk_confidences = [] # Initialize for current task
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
                    TaskResultModel=TaskResult, # Pass TaskResult to parse_local_response
                    structured_output_fallback_enabled=getattr(valves, 'structured_output_fallback_enabled', True)
                )
                
                # Track structured output success
                if valves.use_structured_output:
                    structured_output_attempts += 1
                    if not response_data.get('parse_error'):
                        structured_output_successes += 1

                # Collect Confidence per Chunk
                numeric_confidence = response_data.get('numeric_confidence', 0.3) # Default to LOW numeric
                text_confidence = response_data.get('confidence', 'LOW').upper()
                response_data['fingerprint'] = None # Initialize fingerprint

                if not response_data.get('_is_none_equivalent') and not response_data.get('parse_error'):
                    if text_confidence in round_confidence_distribution:
                        round_confidence_distribution[text_confidence] += 1
                    current_task_chunk_confidences.append(numeric_confidence)

                    # Fingerprint Generation Logic
                    answer_text = response_data.get('answer')
                    if answer_text: # Ensure answer_text is not None and not empty
                        normalized_text = answer_text.lower().strip()
                        if normalized_text: # Ensure normalized_text is not empty
                            response_data['fingerprint'] = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
                
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

                if not response_data.get('_is_none_equivalent'): # Check with .get for safety
                    extracted_info = response_data.get('answer') or response_data.get('explanation', 'Could not extract details.')
                    # Store as dict with fingerprint
                    results_for_this_task_from_chunks.append({
                        "text": f"[Chunk {chunk_idx+1}]: {extracted_info}",
                        "fingerprint": response_data.get('fingerprint')
                    })
                    num_relevant_chunks_found += 1
                    # Note: current_task_chunk_confidences is already appended if valid (based on earlier logic)
                    
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
        
        # Track Task Success/Failure and Aggregate Confidence per Task
        if results_for_this_task_from_chunks:
            task_success_count += 1
            avg_task_confidence = sum(current_task_chunk_confidences) / len(current_task_chunk_confidences) if current_task_chunk_confidences else 0.0
            aggregated_task_confidences.append({
                "task": task,
                "avg_numeric_confidence": avg_task_confidence,
                "contributing_successful_chunks": len(current_task_chunk_confidences)
            })
            # Modify overall_task_results for successful tasks
            detailed_results = [{"text": res["text"], "fingerprint": res["fingerprint"]} for res in results_for_this_task_from_chunks if isinstance(res, dict)]
            aggregated_text_result = "\n".join([res["text"] for res in detailed_results])
            overall_task_results.append({
                "task": task,
                "result": aggregated_text_result,
                "status": "success",
                "detailed_findings": detailed_results
            })
            conversation_log.append(
                f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). Avg Confidence: {avg_task_confidence:.2f}. First result snippet: {detailed_results[0]['text'][:100] if detailed_results else 'N/A'}..."
            )
        elif chunk_timeout_count_for_task > 0 and chunk_timeout_count_for_task == len(chunks):
            task_failure_count += 1 # All chunks timed out
            aggregated_task_confidences.append({"task": task, "avg_numeric_confidence": 0.0, "contributing_successful_chunks": 0})
            overall_task_results.append({
                "task": task,
                "result": f"Timeout on all {len(chunks)} chunks",
                "status": "timeout_all_chunks",
                "detailed_findings": [] # Add empty list for consistency
            })
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out."
            )
        else: # No relevant info found or other errors
            task_failure_count += 1
            aggregated_task_confidences.append({"task": task, "avg_numeric_confidence": 0.0, "contributing_successful_chunks": 0})
            overall_task_results.append({
                "task": task,
                "result": "Information not found in any relevant chunk",
                "status": "not_found",
                "detailed_findings": [] # Add empty list for consistency
            })
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found in any chunk."
            )
        # current_task_chunk_confidences is implicitly reset at the start of the task loop

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

    # Calculate structured output success rate if applicable
    structured_output_success_rate = None
    if structured_output_attempts > 0:
        structured_output_success_rate = structured_output_successes / structured_output_attempts
    
    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_attempts_this_call,
        "total_chunk_processing_timeouts": total_timeouts_this_call,
        "round_metrics_data": round_metrics_data,
        "confidence_metrics_data": { # New confidence data
            "task_confidences": aggregated_task_confidences,
            "round_confidence_distribution": round_confidence_distribution
        },
        "structured_output_metrics": {
            "attempts": structured_output_attempts,
            "successes": structured_output_successes,
            "success_rate": structured_output_success_rate
        }
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