import asyncio
import json
from typing import List, Optional, Dict, Any, Callable

# Placeholder for actual Valves type from minions_valves.py
# from .minions_valves import MinionsValves as ValvesType
ValvesType = Any 

# Placeholder for actual TaskResult model from minions_models.py
# from .minions_models import TaskResult
class TaskResultModel(Dict): # Basic Dict as placeholder
    pass

# Placeholder for API call functions from common_api_calls.py
# async def call_claude(valves: ValvesType, prompt: str) -> str: ...
# async def call_ollama(valves: ValvesType, prompt: str, use_json: bool, schema: Optional[Any]) -> str: ...


def parse_minions_tasks(valves: ValvesType, claude_response: str) -> List[str]:
    """Parses tasks from the remote model's (Claude) response for MinionS protocol."""
    lines = claude_response.split("\n")
    tasks = []
    for line in lines:
        line = line.strip()
        if line.startswith(tuple(f"{i}." for i in range(1, 10))) or \
           line.startswith(tuple(f"{i})" for i in range(1, 10))) or \
           line.startswith(("- ", "* ", "+ ")):
            task_content = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            if len(task_content) > 10:  # Keep simple task filter
                tasks.append(task_content)
    return tasks[:valves.max_tasks_per_round]

def create_minions_chunks(valves: ValvesType, context: str) -> List[str]:
    """Creates chunks from the context string based on chunk_size valve."""
    if not context:
        return []
    # Ensure chunk_size is at least 1, and not larger than the context itself if context is very small.
    actual_chunk_size = max(1, min(valves.chunk_size, len(context))) 
    chunks = [
        context[i : i + actual_chunk_size] 
        for i in range(0, len(context), actual_chunk_size)
    ]
    # Limit number of chunks if max_chunks is set and positive
    if hasattr(valves, 'max_chunks') and valves.max_chunks > 0:
        chunks = chunks[:valves.max_chunks]
    return chunks

def parse_minions_local_response(
    response_text: str, 
    valves: ValvesType, # For debug_mode and use_structured_output flags
    is_structured: bool = False,
    task_result_model: Optional[Any] = None # e.g., TaskResult from minions_models
) -> Dict[str, Any]:
    """
    Parses the local model's response for MinionS, supporting both text and structured (JSON) formats.
    Includes logic to determine if the response signifies "no relevant information found".
    """
    if is_structured and valves.use_structured_output and task_result_model:
        try:
            parsed_json = json.loads(response_text)
            # validated_model = task_result_model(**parsed_json) # Actual Pydantic validation
            # model_dict = validated_model.dict()
            # For now, as task_result_model is Any (placeholder)
            if isinstance(parsed_json, dict):
                model_dict = parsed_json
                model_dict['parse_error'] = None
                # Crucial for MinionS: Check if the structured response indicates "not found"
                if model_dict.get('answer') is None and model_dict.get('explanation') is not None : # Check if answer is explicitly null
                    model_dict['_is_none_equivalent'] = True
                else: # If answer has content, or if explanation itself is missing (treat as non-conclusive for "none")
                    model_dict['_is_none_equivalent'] = False
                return model_dict
            else:
                raise ValueError("Parsed JSON is not a dictionary")
        except Exception as e:
            if valves.debug_mode:
                print(f"DEBUG: Failed to parse structured output in MinionS: {e}. Response was: {response_text[:500]}")
            is_none_equivalent_fallback = response_text.strip().upper() == "NONE"
            return {"answer": response_text, "explanation": response_text, "confidence": "LOW", "citation": None, "parse_error": str(e), "_is_none_equivalent": is_none_equivalent_fallback}
    
    is_none_equivalent_text = response_text.strip().upper() == "NONE"
    return {"answer": response_text, "explanation": response_text, "confidence": "MEDIUM", "citation": None, "parse_error": None, "_is_none_equivalent": is_none_equivalent_text}

async def execute_minions_tasks_on_chunks(
    valves: ValvesType,
    tasks: List[str], 
    chunks: List[str], 
    conversation_log: List[str], # For logging progress directly
    current_round: int,
    call_ollama_func: Callable, # common_api_calls.call_ollama
    task_result_model: Any, # minions_models.TaskResult
    parse_local_response_func: Callable # parse_minions_local_response
) -> Dict[str, Any]:
    """Executes sub-tasks on document chunks using the local model."""
    overall_task_results = []
    total_chunk_processing_attempts = 0
    total_chunk_processing_timeouts = 0

    for task_idx, task in enumerate(tasks):
        if valves.show_conversation:
            conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
        
        results_for_this_task_from_chunks = []
        chunk_timeout_count_for_task = 0
        num_relevant_chunks_found = 0

        for chunk_idx, chunk in enumerate(chunks):
            total_chunk_processing_attempts += 1
            
            local_prompt_text = f'''Text to analyze (Chunk {chunk_idx + 1}/{len(chunks)} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}'''

            if valves.use_structured_output:
                local_prompt_text += f"\n\nProvide your answer ONLY as a valid JSON object matching the specified schema. If no relevant information is found in THIS SPECIFIC TEXT, ensure the 'answer' field in your JSON response is explicitly set to null (or None)."
            else:
                local_prompt_text += "\n\nProvide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\"."
            
            start_time_ollama_local = 0 # Renamed to avoid conflict
            if valves.debug_mode:
                if valves.show_conversation: # Avoid double logging if not showing full convo
                    conversation_log.append(f"   🔄 Task {task_idx + 1} - Trying chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} chars)... (Debug Mode)")
                start_time_ollama_local = asyncio.get_event_loop().time()

            try:
                response_str = await asyncio.wait_for(
                    call_ollama_func(
                        valves,
                        local_prompt_text, # Use the fully formed prompt
                        use_json=True, 
                        schema=task_result_model 
                    ),
                    timeout=valves.timeout_local,
                )
                response_data = parse_local_response_func(
                    response_str,
                    valves,
                    is_structured=True,
                    task_result_model=task_result_model
                )

                if valves.debug_mode:
                    end_time_ollama_local = asyncio.get_event_loop().time()
                    time_taken_ollama_local = end_time_ollama_local - start_time_ollama_local
                    status_msg = "Parse Error" if response_data.get("parse_error") else ("No relevant info" if response_data.get('_is_none_equivalent') else "Relevant info found")
                    details_msg = f"Error: {response_data['parse_error']}, Raw: {response_data.get('answer', '')[:70]}..." if response_data.get("parse_error") else \
                                  (f"Response indicates no info. Confidence: {response_data.get('confidence', 'N/A')}" if response_data.get('_is_none_equivalent') else \
                                   f"Answer: {response_data.get('answer', '')[:70]}..., Confidence: {response_data.get('confidence', 'N/A')}")
                    if valves.show_conversation:
                         conversation_log.append(f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} processed by local model in {time_taken_ollama_local:.2f}s. Status: {status_msg}. Details: {details_msg} (Debug Mode)")

                if not response_data.get('_is_none_equivalent'):
                    extracted_info = response_data.get('answer') or response_data.get('explanation', 'Could not extract details.')
                    results_for_this_task_from_chunks.append(f"[Chunk {chunk_idx+1}]: {extracted_info}")
                    num_relevant_chunks_found += 1
            
            except asyncio.TimeoutError:
                total_chunk_processing_timeouts += 1
                chunk_timeout_count_for_task +=1
                if valves.show_conversation:
                    conversation_log.append(f"   ⏰ Task {task_idx + 1} - Chunk {chunk_idx + 1} timed out after {valves.timeout_local}s")
                if valves.debug_mode:
                    end_time_ollama_local = asyncio.get_event_loop().time() # type: ignore
                    time_taken_ollama_local = end_time_ollama_local - start_time_ollama_local # type: ignore
                    if valves.show_conversation:
                         conversation_log.append(f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} TIMEOUT after {time_taken_ollama_local:.2f}s. (Debug Mode)")
            except Exception as e:
                if valves.show_conversation:
                    conversation_log.append(f"   ❌ Task {task_idx + 1} - Chunk {chunk_idx + 1} error: {str(e)}")
        
        if results_for_this_task_from_chunks:
            aggregated_result_for_task = "\n".join(results_for_this_task_from_chunks)
            overall_task_results.append({"task": task, "result": aggregated_result_for_task, "status": "success"})
            if valves.show_conversation:
                conversation_log.append(f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). First result snippet: {results_for_this_task_from_chunks[0][:100]}...")
        elif chunk_timeout_count_for_task == len(chunks) and len(chunks) > 0 : # Only if all chunks for this task timed out
            overall_task_results.append({"task": task, "result": f"Timeout on all {len(chunks)} chunks", "status": "timeout_all_chunks"})
            if valves.show_conversation:
                conversation_log.append(f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out.")
        else:
            overall_task_results.append({"task": task, "result": "Information not found or not extracted from any relevant chunk", "status": "not_found"})
            if valves.show_conversation:
                conversation_log.append(f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found/extracted in any chunk.")
    
    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_chunk_processing_attempts,
        "total_chunk_processing_timeouts": total_chunk_processing_timeouts
    }

def calculate_minions_token_savings(
    valves: ValvesType,
    decomposition_prompts: List[str], 
    synthesis_prompts: List[str],
    all_results_summary_for_claude: str, 
    final_response_claude: str, 
    context_length: int, 
    query_length: int
    # total_chunks_processed_local and total_tasks_executed_local are more for operational stats, not direct token cost of Claude
) -> Dict[str, Any]:
    """Calculates token savings for the MinionS protocol, focusing on remote model calls."""
    chars_per_token = 3.5 
    traditional_tokens_claude = int((context_length + query_length) / chars_per_token)
    
    minions_tokens_claude = 0
    for p_list in [decomposition_prompts, synthesis_prompts]:
        for p_content in p_list:
            minions_tokens_claude += int(len(p_content) / chars_per_token)
            
    minions_tokens_claude += int(len(all_results_summary_for_claude) / chars_per_token)
    minions_tokens_claude += int(len(final_response_claude) / chars_per_token)
    
    token_savings_claude = traditional_tokens_claude - minions_tokens_claude
    percentage_savings_claude = (token_savings_claude / traditional_tokens_claude * 100) if traditional_tokens_claude > 0 else 0
    
    return {
        'traditional_tokens_claude': traditional_tokens_claude,
        'minions_tokens_claude': minions_tokens_claude,
        'token_savings_claude': token_savings_claude,
        'percentage_savings_claude': percentage_savings_claude,
        'total_decomposition_rounds': len(decomposition_prompts)
    }

async def execute_minions_protocol(
    valves: ValvesType,
    query: str, 
    context: str,
    call_claude_func: Callable, # common_api_calls.call_claude
    call_ollama_func: Callable, # common_api_calls.call_ollama
    task_result_model: Any,     # minions_models.TaskResult
    # Functions from this file, passed for clarity or if they were in different modules
    parse_tasks_func: Callable[[ValvesType, str], List[str]],
    create_chunks_func: Callable[[ValvesType, str], List[str]],
    execute_tasks_on_chunks_func: Callable[..., Awaitable[Dict[str, Any]]],
    parse_local_response_func: Callable[..., Dict[str, Any]],
    calculate_token_savings_func: Callable[..., Dict[str, Any]]
) -> str:
    """Executes the MinionS (multi-task, multi-round) protocol."""
    conversation_log: List[str] = []
    debug_log: List[str] = []
    scratchpad_content = ""
    all_round_results_aggregated: List[Dict[str, Any]] = []
    decomposition_prompts_history: List[str] = []
    synthesis_prompts_history: List[str] = []
    final_answer = "No answer could be synthesized."
    claude_provided_final_answer = False
    total_tasks_executed_local = 0
    total_chunk_processing_timeouts_accumulated = 0

    overall_start_time = 0.0
    if valves.debug_mode:
        overall_start_time = asyncio.get_event_loop().time()
        debug_log.append(f"🔍 **Debug Info (MinionS Protocol v0.2.0):**\n- Query: {query[:100]}...\n- Context length: {len(context)} chars")
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**")

    chunks = create_chunks_func(valves, context)
    if not chunks and context:
        return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size/max_chunks."
    
    # Direct call if no context/chunks - This part can be handled by the main pipe method before calling this execute function
    # For now, keeping it to show full logic.
    if not context: # Simplified from original, as chunk creation handles empty context
        if valves.show_conversation: conversation_log.append("ℹ️ No context to process with MinionS. Attempting direct call to remote model.")
        try:
            # This direct call logic might better reside in the main pipe method
            final_answer = await call_claude_func(valves, f"Please answer this question: {query}") # Direct query
            # ... (build simple output for direct call)
            return f"## 🎯 Final Answer (Direct)\n{final_answer}"
        except Exception as e:
            return f"❌ **Error in direct remote model call:** {str(e)}"


    for current_round in range(valves.max_rounds):
        if valves.debug_mode: 
            debug_log.append(f"**⚙️ Starting Round {current_round + 1}/{valves.max_rounds}... (Debug Mode)**")
        if valves.show_conversation:
            conversation_log.append(f"### 🎯 Round {current_round + 1}/{valves.max_rounds} - Task Decomposition Phase")
        
        decomposition_prompt = f'''You are a supervisor LLM in a multi-round process. Your goal is to answer: "{query}"
Context has been split into {len(chunks)} chunks. A local LLM will process these chunks for each task you define.
Scratchpad (previous findings): {scratchpad_content if scratchpad_content else "Nothing yet."}

Based on the scratchpad and the original query, identify up to {valves.max_tasks_per_round} specific, simple tasks for the local assistant.
If the information in the scratchpad is sufficient to answer the query, respond ONLY with the exact phrase "FINAL ANSWER READY." followed by the comprehensive answer.
Otherwise, list the new tasks clearly. Ensure tasks are actionable. Avoid redundant tasks.
Format tasks as a simple list (e.g., 1. Task A, 2. Task B).'''
        decomposition_prompts_history.append(decomposition_prompt)
        
        claude_response_text = ""
        try:
            if valves.debug_mode: start_time_claude_decomp = asyncio.get_event_loop().time()
            claude_response_text = await call_claude_func(valves, decomposition_prompt)
            if valves.debug_mode:
                debug_log.append(f"⏱️ Remote model call (Decomposition Round {current_round+1}) took {asyncio.get_event_loop().time() - start_time_claude_decomp:.2f}s. (Debug Mode)")
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Remote Model (Decomposition - Round {current_round + 1}):**\n{claude_response_text}\n")
        except Exception as e:
            if valves.show_conversation: conversation_log.append(f"❌ Error calling remote model for decomposition in round {current_round + 1}: {e}")
            break 

        if "FINAL ANSWER READY." in claude_response_text:
            final_answer = claude_response_text.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_provided_final_answer = True
            if valves.show_conversation: conversation_log.append(f"**🤖 Remote model indicates final answer is ready in round {current_round + 1}.**")
            scratchpad_content += f"\n\n**Round {current_round + 1}:** Remote model provided final answer."
            break 

        tasks = parse_tasks_func(valves, claude_response_text)
        if valves.debug_mode:
            debug_log.append(f"   Identified {len(tasks)} tasks for Round {current_round + 1}. (Debug Mode)")
            # ... (logging individual tasks)

        if not tasks:
            if valves.show_conversation: conversation_log.append(f"**🤖 Remote model provided no new tasks in round {current_round + 1}. Proceeding to final synthesis.**")
            break
        
        total_tasks_executed_local += len(tasks)
        if valves.show_conversation:
            conversation_log.append(f"### ⚡ Round {current_round + 1} - Parallel Execution Phase (Processing {len(chunks)} chunks for {len(tasks)} tasks)")
        
        execution_details = await execute_tasks_on_chunks_func(
            valves, tasks, chunks, conversation_log, current_round + 1, 
            call_ollama_func, task_result_model, parse_local_response_func
        )
        current_round_task_results = execution_details["results"]
        # ... (handle timeout stats and warnings as in original)
        if execution_details["total_chunk_processing_attempts"] > 0:
            timeout_percentage = (execution_details["total_chunk_processing_timeouts"] / execution_details["total_chunk_processing_attempts"]) * 100
            log_msg = f"**📈 Round {current_round + 1} Local Model Timeout Stats:** {execution_details['total_chunk_processing_timeouts']}/{execution_details['total_chunk_processing_attempts']} chunk calls timed out ({timeout_percentage:.1f}%)."
            if valves.show_conversation: conversation_log.append(log_msg)
            if valves.debug_mode: debug_log.append(log_msg)
            if timeout_percentage >= valves.max_round_timeout_failure_threshold_percent:
                warn_msg = f"⚠️ **Warning:** Round {current_round + 1} exceeded local model timeout threshold. Results may be incomplete."
                if valves.show_conversation: conversation_log.append(warn_msg)
                if valves.debug_mode: debug_log.append(warn_msg)
                scratchpad_content += f"\n\n**Note from Round {current_round + 1}:** High local model timeout rate ({timeout_percentage:.1f}%)."


        round_summary_parts = [f"- {'✅' if r['status'] == 'success' else '❓'} Task: {r['task']}, Result: {r['result'][:100]}..." for r in current_round_task_results]
        if round_summary_parts: scratchpad_content += f"\n\n**Results from Round {current_round + 1}:**\n" + "\n".join(round_summary_parts)
        
        all_round_results_aggregated.extend(current_round_task_results)
        total_chunk_processing_timeouts_accumulated += execution_details["total_chunk_processing_timeouts"]

        if valves.debug_mode: debug_log.append(f"**🏁 Completed Round {current_round + 1}. Cumulative time: {asyncio.get_event_loop().time() - overall_start_time:.2f}s. (Debug Mode)**")
        if current_round == valves.max_rounds - 1 and valves.show_conversation:
            conversation_log.append(f"**🏁 Reached max rounds ({valves.max_rounds}). Proceeding to final synthesis.**")

    if not claude_provided_final_answer:
        if valves.show_conversation: conversation_log.append("\n### 🔄 Final Synthesis Phase")
        successful_results = [r for r in all_round_results_aggregated if r['status'] == 'success']
        if not successful_results:
            final_answer = "No information was successfully gathered by local models across the rounds."
            if valves.show_conversation: conversation_log.append(f"**🤖 Remote Model (Synthesis):** {final_answer}")
        else:
            synthesis_input_summary = "\n".join([f"- Task: {r['task']}\n  Result: {r['result']}" for r in successful_results])
            synthesis_prompt = f'''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary}

If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.
Final Answer:'''
            synthesis_prompts_history.append(synthesis_prompt)
            try:
                if valves.debug_mode: start_time_synth = asyncio.get_event_loop().time()
                final_answer = await call_claude_func(valves, synthesis_prompt)
                if valves.debug_mode: debug_log.append(f"⏱️ Remote model call (Final Synthesis) took {asyncio.get_event_loop().time() - start_time_synth:.2f}s. (Debug Mode)")
                if valves.show_conversation: conversation_log.append(f"**🤖 Remote Model (Final Synthesis):**\n{final_answer}")
            except Exception as e:
                if valves.show_conversation: conversation_log.append(f"❌ Error during final synthesis: {e}")
                final_answer = "Error during final synthesis. Raw findings might be in conversation log."
    
    output_parts = []
    if valves.show_conversation:
        output_parts.append("## 🗣️ MinionS Collaboration (Multi-Round)")
        output_parts.extend(conversation_log)
        output_parts.append("---")
    if valves.debug_mode:
        output_parts.append("### 🔍 Debug Log")
        output_parts.extend(debug_log)
        output_parts.append("---")
    output_parts.append(f"## 🎯 Final Answer")
    output_parts.append(final_answer)

    stats = calculate_token_savings_func(
        valves, decomposition_prompts_history, synthesis_prompts_history,
        scratchpad_content, # Using scratchpad as summary for Claude if it provided final answer early
        final_answer, 
        len(context), len(query)
    )
    
    successful_local_tasks = len([r for r in all_round_results_aggregated if r['status'] == 'success'])
    tasks_all_chunks_timed_out = len([r for r in all_round_results_aggregated if r['status'] == 'timeout_all_chunks'])

    output_parts.extend([
        f"\n## 📊 MinionS Efficiency Stats (v0.2.0)",
        f"- **Protocol:** MinionS (Multi-Round)",
        f"- **Rounds executed:** {stats.get('total_decomposition_rounds', 0)}/{valves.max_rounds}",
        f"- **Total tasks for local model:** {total_tasks_executed_local}",
        f"- **Successful tasks (local):** {successful_local_tasks}",
        f"- **Tasks where all chunks timed out (local):** {tasks_all_chunks_timed_out}",
        f"- **Total individual chunk processing timeouts (local):** {total_chunk_processing_timeouts_accumulated}",
        f"- **Context size:** {len(context):,} characters",
        f"\n## 💰 Token Savings Analysis (Remote Model: {valves.remote_model})",
        f"- **Traditional single call (est.):** ~{stats.get('traditional_tokens_claude', 0):,} tokens",
        f"- **MinionS multi-round (Remote Model only):** ~{stats.get('minions_tokens_claude', 0):,} tokens",
        f"- **💰 Est. Remote Model Token savings:** ~{stats.get('percentage_savings_claude', 0.0):.1f}%"
    ])
            
    return "\n".join(output_parts)
