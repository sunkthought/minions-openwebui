import asyncio
import json
import logging # Added logging
from typing import List, Dict, Any, Callable, Type, Awaitable # Added Type, Awaitable

# Helper function for truncating text
def _truncate_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

# parse_local_response and calculate_token_savings are preserved from original file

def parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, TaskResultModel: Any, logger: logging.Logger) -> Dict:
    """Parse local model response, supporting both text and structured formats"""
    if is_structured and use_structured_output:
        try:
            parsed_json = json.loads(response)

            answer_content = parsed_json.get('answer')

            if answer_content is not None and not isinstance(answer_content, str):
                logger.warning(f"Local LLM provided non-string for 'answer' (type: {type(answer_content).__name__}); attempting conversion.")
                try:
                    parsed_json['answer'] = json.dumps(answer_content)
                    logger.info(f"Successfully converted 'answer' of type {type(answer_content).__name__} to JSON string.")
                except TypeError as te:
                    logger.error(f"Could not json.dumps answer_content of type {type(answer_content).__name__}: {te}. Falling back to str() conversion.")
                    parsed_json['answer'] = str(answer_content)
            elif answer_content is None: # Explicitly ensure 'answer' key exists if it was None or not present
                parsed_json['answer'] = None
            # If answer_content is already a string, it will pass through unmodified.

            validated_model = TaskResultModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            if model_dict.get('answer') is None:
                model_dict['_is_none_equivalent'] = True
            else:
                model_dict['_is_none_equivalent'] = False
            return model_dict
        except Exception as e:
            if debug_mode:
                # Using logger now.
                logger.debug(f"Failed to parse structured output or validation error: {e}. Response was: {response[:500]}")
            is_none_equivalent_fallback = response.strip().upper() == "NONE"
            return {"answer": response, "explanation": response, "confidence": "LOW", "citation": None, "parse_error": str(e), "_is_none_equivalent": is_none_equivalent_fallback}
    
    is_none_equivalent_text = response.strip().upper() == "NONE"
    return {"answer": response, "explanation": response, "confidence": "MEDIUM", "citation": None, "parse_error": None, "_is_none_equivalent": is_none_equivalent_text}

async def execute_tasks_on_chunks(
    tasks: List[Any],  # List of JobManifest instances
    chunks: List[str],
    conversation_log: List[str], 
    # JobManifest_cls: Type, # Not strictly needed if tasks are already instances
    valves: Any,
    call_ollama_func: Callable[..., Awaitable[str]],
    logger: logging.Logger,
    TaskResult_cls: Type # Added for structured output
) -> List[Dict[str, str]]:
    """
    Execute tasks defined by JobManifest objects in parallel across document chunks.
    This version is aligned with the refactoring of minions-fn-claude.py,
    expecting JobManifest instances and returning simple string results.
    """
    task_results = []
    total_chunks_in_document = len(chunks)

    for task_idx, manifest in enumerate(tasks):
        task_id = getattr(manifest, 'task_id', f'unknown_task_{task_idx}')
        task_description = getattr(manifest, 'task_description', 'No description')
        advice = getattr(manifest, 'advice', None)
        chunk_id_from_manifest = getattr(manifest, 'chunk_id', None)

        conversation_log.append(f"**📋 Task {task_idx + 1} (ID: {task_id}):** {task_description}")
        logger.info(f"Executing task ID {task_id}: {task_description}")

        best_result_for_task = "Information not found"
        processed_at_least_one_chunk = False

        selected_chunks_with_indices: List[tuple[int, str]] = []
        if chunk_id_from_manifest is not None and 0 <= chunk_id_from_manifest < total_chunks_in_document:
            selected_chunks_with_indices = [(chunk_id_from_manifest, chunks[chunk_id_from_manifest])]
            if valves.debug_mode:
                logger.debug(f"Task {task_id} targeting specific chunk_id: {chunk_id_from_manifest}.")
            conversation_log.append(f"   ℹ️ Task targets specific chunk: {chunk_id_from_manifest + 1}")
        else:
            if chunk_id_from_manifest is not None:
                logger.warning(f"Task {task_id} had invalid chunk_id: {chunk_id_from_manifest}. Applying to all {total_chunks_in_document} chunks.")
                conversation_log.append(f"   ⚠️ Task had invalid chunk_id {chunk_id_from_manifest}, applying to all chunks.")
            selected_chunks_with_indices = list(enumerate(chunks)) # Process all chunks if no specific valid one
            if valves.debug_mode:
                logger.debug(f"Task {task_id} applying to all {len(selected_chunks_with_indices)} chunks.")

        for original_chunk_idx, chunk_content in selected_chunks_with_indices:
            processed_at_least_one_chunk = True

            prompt_intro = f"Text to analyze (Chunk {original_chunk_idx + 1}/{total_chunks_in_document} of document):"
            if chunk_id_from_manifest is not None and chunk_id_from_manifest == original_chunk_idx: # check if current chunk is the targeted one
                prompt_intro = f"Text to analyze (Specifically targeted Chunk {original_chunk_idx + 1}/{total_chunks_in_document} of document based on task assignment):"

            max_chars_for_prompt = getattr(valves, 'max_prompt_chars_local', 4000)
            
            local_prompt = f'''{prompt_intro}
---BEGIN TEXT---
{chunk_content[:max_chars_for_prompt]}
---END TEXT---

Task: {task_description}
'''
            if advice:
                local_prompt += f"\nHint: {advice}"

            # Adjust prompt if structured output is expected
            if valves.use_structured_output:
                local_prompt += f"\n\nProvide your answer ONLY as a valid JSON object matching the specified schema. If no relevant information is found in THIS SPECIFIC TEXT, ensure the 'answer' field in your JSON response is explicitly set to null."
            else:
                local_prompt += "\n\nProvide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\"."

            try:
                task_desc_summary = _truncate_text(task_description, 30)
                if valves.debug_mode:
                    logger.debug(f"   Task {task_id} trying chunk {original_chunk_idx + 1} (size: {len(chunk_content)} chars). Prompt: {_truncate_text(local_prompt,200)}...")
                    conversation_log.append(f"   🔄 Task '{task_desc_summary}' on chunk {original_chunk_idx + 1}...")

                response_str = await asyncio.wait_for(
                    call_ollama_func(
                        valves,
                        local_prompt,
                        use_json=valves.use_structured_output,
                        schema=TaskResult_cls if valves.use_structured_output else None
                        ),
                    timeout=valves.timeout_local
                )

                current_result_text = "Information not found" # Default before parsing

                if valves.use_structured_output:
                    response_data = parse_local_response(
                        response_str,
                        is_structured=True, # We requested structured if use_structured_output is True
                        use_structured_output=True, # Pass it along
                        debug_mode=valves.debug_mode,
                        TaskResultModel=TaskResult_cls,
                        logger=logger
                    )
                    if valves.debug_mode:
                        logger.debug(f"   Task {task_id} chunk {original_chunk_idx + 1} structured response parsed: {response_data}")

                    if not response_data['_is_none_equivalent']:
                        # Use 'answer' if available, otherwise 'explanation' as fallback text.
                        # The 'result' for aggregation should be a simple string.
                        current_result_text = response_data.get('answer') if response_data.get('answer') is not None else response_data.get('explanation', 'No explanation provided.')
                        # Ensure current_result_text is a string, as 'answer' could have been (list,dict) then stringified by parse_local_response
                        if not isinstance(current_result_text, str):
                            current_result_text = str(current_result_text)

                        logger.info(f"   Task {task_id} found relevant info (structured) in chunk {original_chunk_idx + 1}.")
                        conversation_log.append(f"**💻 Local Model (Task '{task_desc_summary}', Chunk {original_chunk_idx+1}):** {_truncate_text(current_result_text, 100)}...")
                        best_result_for_task = current_result_text # Update best result for this task
                        if chunk_id_from_manifest is not None:
                            break # Found result for specific chunk task
                    else:
                        # Structured response indicated no relevant info
                        logger.info(f"   Task {task_id} (structured) found no relevant info in chunk {original_chunk_idx + 1}.")
                else: # Plain text processing
                    response_str_summary = _truncate_text(response_str, 100)
                    if valves.debug_mode:
                        logger.debug(f"   Task {task_id} chunk {original_chunk_idx + 1} plain text response: {response_str_summary}...")
                        conversation_log.append(f"   ✅ Chunk {original_chunk_idx + 1} plain response: {response_str_summary}...")

                    if "NONE" not in response_str.upper() and len(response_str.strip()) > 2:
                        current_result_text = response_str
                        logger.info(f"   Task {task_id} found relevant info (plain text) in chunk {original_chunk_idx + 1}.")
                        conversation_log.append(f"**💻 Local Model (Task '{task_desc_summary}', Chunk {original_chunk_idx+1}):** {response_str_summary}...")
                        best_result_for_task = current_result_text # Update best result for this task
                        if chunk_id_from_manifest is not None:
                            break # Found result for specific chunk task

            except asyncio.TimeoutError:
                logger.warning(f"   Task {task_id} on chunk {original_chunk_idx + 1} timed out after {valves.timeout_local}s.")
                conversation_log.append(f"   ⏰ Task '{task_desc_summary}' on chunk {original_chunk_idx + 1} timed out.")
                if best_result_for_task == "Information not found":
                     best_result_for_task = "Timeout"
            except Exception as e:
                logger.error(f"   Task {task_id} on chunk {original_chunk_idx + 1} error: {e}", exc_info=valves.debug_mode)
                conversation_log.append(f"   ❌ Task '{task_desc_summary}' on chunk {original_chunk_idx + 1} error: {e}")
                if best_result_for_task == "Information not found":
                    best_result_for_task = f"Error: {str(e)}"

        if not processed_at_least_one_chunk and chunk_id_from_manifest is not None:
            logger.warning(f"Task {task_id} requested specific chunk {chunk_id_from_manifest} which was not processed (total chunks: {total_chunks_in_document}).")
            conversation_log.append(f"   ⚠️ Task {task_id} could not be processed as requested chunk {chunk_id_from_manifest} is invalid.")
            best_result_for_task = "Invalid chunk requested"

        task_results.append({
            "task_id": task_id,
            "task_description": task_description,
            "result": best_result_for_task
        })
        
        task_desc_summary_final = _truncate_text(task_description, 30)
        if best_result_for_task == "Information not found" and processed_at_least_one_chunk:
             conversation_log.append(f"**💻 Local Model (Task '{task_desc_summary_final}...'):** No relevant information found across applicable chunks.")
        elif best_result_for_task == "Timeout" and processed_at_least_one_chunk:
             conversation_log.append(f"**💻 Local Model (Task '{task_desc_summary_final}...'):** Timed out without finding information.")

    return task_results

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
    chars_per_token = 3.5
    
    traditional_tokens = int((context_length + query_length) / chars_per_token)
    
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