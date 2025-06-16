# Partials File: partials/minions_streaming_protocol.py

import asyncio
from typing import Any, Callable, AsyncGenerator, List

async def _execute_minions_protocol_with_streaming_updates(
    valves: Any,
    query: str,
    context: str,
    call_claude: Callable,
    call_ollama_func: Callable,
    TaskResultModel: Any,
    streaming_manager: Any
) -> AsyncGenerator[str, None]:
    """Execute the MinionS protocol with real-time streaming updates"""
    
    # Initialize protocol state
    conversation_log = []
    debug_log = []
    scratchpad_content = ""
    all_round_results_aggregated = []
    all_round_metrics: List[RoundMetrics] = []
    global_unique_fingerprints_seen = set()
    decomposition_prompts_history = []
    synthesis_prompts_history = []
    final_response = "No answer could be synthesized."
    claude_provided_final_answer = False
    total_tasks_executed_local = 0
    total_chunks_processed_for_stats = 0
    total_chunk_processing_timeouts_accumulated = 0
    synthesis_input_summary = ""
    early_stopping_reason_for_output = None

    overall_start_time = asyncio.get_event_loop().time()
    user_query = query

    # Performance Profile Logic
    current_run_max_rounds = valves.max_rounds
    current_run_base_sufficiency_thresh = valves.convergence_sufficiency_threshold
    current_run_base_novelty_thresh = valves.convergence_novelty_threshold
    current_run_simple_query_confidence_thresh = valves.simple_query_confidence_threshold
    current_run_medium_query_confidence_thresh = valves.medium_query_confidence_threshold

    profile_applied_details = [f"🧠 Performance Profile selected: {valves.performance_profile}"]

    if valves.performance_profile == "high_quality":
        current_run_max_rounds = min(valves.max_rounds + 1, 10)
        current_run_base_sufficiency_thresh = min(0.95, valves.convergence_sufficiency_threshold + 0.1)
        current_run_base_novelty_thresh = max(0.03, valves.convergence_novelty_threshold - 0.03)
        current_run_simple_query_confidence_thresh = min(0.95, valves.simple_query_confidence_threshold + 0.1)
        current_run_medium_query_confidence_thresh = min(0.95, valves.medium_query_confidence_threshold + 0.1)
        profile_applied_details.append(f"   - Applied 'high_quality' adjustments: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")
    elif valves.performance_profile == "fastest_results":
        current_run_max_rounds = max(1, valves.max_rounds - 1)
        current_run_base_sufficiency_thresh = max(0.05, valves.convergence_sufficiency_threshold - 0.1)
        current_run_base_novelty_thresh = min(0.95, valves.convergence_novelty_threshold + 0.05)
        current_run_simple_query_confidence_thresh = max(0.05, valves.simple_query_confidence_threshold - 0.1)
        current_run_medium_query_confidence_thresh = max(0.05, valves.medium_query_confidence_threshold - 0.1)
        profile_applied_details.append(f"   - Applied 'fastest_results' adjustments: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")
    else: # balanced
        profile_applied_details.append(f"   - Using 'balanced' profile: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")

    # Yield initial profile info
    if streaming_manager:
        for detail in profile_applied_details:
            yield f"📋 {detail}\n"

    # Initialize Sufficiency Analyzer
    analyzer = InformationSufficiencyAnalyzer(query=user_query, debug_mode=valves.debug_mode)
    convergence_detector = ConvergenceDetector(debug_mode=valves.debug_mode)
    
    # Initialize Query Complexity Classifier
    query_classifier = QueryComplexityClassifier(debug_mode=valves.debug_mode)
    query_complexity_level = query_classifier.classify_query(query)

    if streaming_manager:
        update = await streaming_manager.stream_granular_update(
            "query_analysis", "complexity_classification", 0.3, 
            f"Query classified as {query_complexity_level.value}"
        )
        if update:
            yield update

    # Document size analysis
    doc_size_category = "medium"
    context_len = len(context)
    if context_len < valves.doc_size_small_char_limit:
        doc_size_category = "small"
    elif context_len > valves.doc_size_large_char_start:
        doc_size_category = "large"

    if streaming_manager:
        update = await streaming_manager.stream_granular_update(
            "query_analysis", "document_analysis", 0.6,
            f"Document size: {doc_size_category} ({context_len:,} characters)"
        )
        if update:
            yield update

    # Initialize effective thresholds
    effective_sufficiency_threshold = current_run_base_sufficiency_thresh
    effective_novelty_threshold = current_run_base_novelty_thresh

    # Chunking
    chunks = create_chunks(context, valves.chunk_size, valves.max_chunks)
    if not chunks and context:
        error_msg = "Context provided, but failed to create any processable chunks. Check chunk_size setting."
        if streaming_manager:
            yield await streaming_manager.stream_error_update(error_msg, "chunking")
        return

    if streaming_manager:
        update = await streaming_manager.stream_granular_update(
            "query_analysis", "chunking_complete", 1.0,
            f"Created {len(chunks)} chunks"
        )
        if update:
            yield update

    # Execute rounds with streaming updates
    for current_round in range(current_run_max_rounds):
        if streaming_manager:
            # Task decomposition progress
            update = await streaming_manager.stream_task_decomposition_progress(
                "analyzing_complexity", 1, 5, f"Starting round {current_round + 1}/{current_run_max_rounds}"
            )
            if update:
                yield update

        # Call decompose_task
        tasks, claude_response_for_decomposition, decomposition_prompt = await decompose_task(
            valves=valves,
            query=query,
            scratchpad_content=scratchpad_content,
            num_chunks=len(chunks),
            max_tasks_per_round=valves.max_tasks_per_round,
            current_round=current_round + 1,
            conversation_log=conversation_log,
            debug_log=debug_log
        )

        if streaming_manager:
            update = await streaming_manager.stream_task_decomposition_progress(
                "generating_tasks", 3, 5, f"Generated {len(tasks)} tasks"
            )
            if update:
                yield update

        # Handle FINAL ANSWER READY
        if "FINAL ANSWER READY." in claude_response_for_decomposition:
            answer_parts = claude_response_for_decomposition.split("FINAL ANSWER READY.", 1)
            if len(answer_parts) > 1:
                final_response = answer_parts[1].strip()
                if final_response.startswith('"') and final_response.endswith('"'):
                    final_response = final_response[1:-1]
                claude_provided_final_answer = True
                if streaming_manager:
                    yield await streaming_manager.stream_phase_update("completion", "Final answer ready from Claude")
                break

        # Execute tasks with streaming progress
        if streaming_manager:
            update = await streaming_manager.stream_task_decomposition_progress(
                "complete", 5, 5, "Task decomposition complete"
            )
            if update:
                yield update

        # Execute tasks on chunks with detailed progress
        execution_details = await execute_tasks_on_chunks_with_streaming(
            tasks, chunks, current_round + 1, valves, call_ollama_func, TaskResultModel, streaming_manager
        )
        
        # Yield each execution update
        for update in execution_details.get("streaming_updates", []):
            yield update

        # Process results
        current_round_task_results = execution_details["results"]
        all_round_results_aggregated.extend(current_round_task_results)

        # Update scratchpad
        round_summary = f"\n**Results from Round {current_round + 1}:**\n"
        for result in current_round_task_results:
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            truncated_task = result['task'][:80] + "..." if len(result['task']) > 80 else result['task']
            truncated_result = result['result'][:100] + "..." if len(result['result']) > 100 else result['result']
            round_summary += f"- {status_emoji} Task: {truncated_task}, Result: {truncated_result}\n"
        
        scratchpad_content += round_summary

    # Final synthesis with streaming
    if not claude_provided_final_answer:
        if streaming_manager:
            update = await streaming_manager.stream_synthesis_progress(
                "collecting", total_tasks=len(all_round_results_aggregated)
            )
            if update:
                yield update

        if not all_round_results_aggregated:
            final_response = "No information was gathered from the document by local models across the rounds."
        else:
            synthesis_input_summary = "\n".join([f"- Task: {r['task']}\n  Result: {r['result']}" for r in all_round_results_aggregated if r['status'] == 'success'])
            if not synthesis_input_summary:
                synthesis_input_summary = "No definitive information was found by local models. The original query was: " + query
            
            if streaming_manager:
                update = await streaming_manager.stream_synthesis_progress(
                    "generating", processed_tasks=len(all_round_results_aggregated), 
                    total_tasks=len(all_round_results_aggregated)
                )
                if update:
                    yield update

            synthesis_prompt = get_minions_synthesis_claude_prompt(query, synthesis_input_summary, valves)
            final_response = await call_supervisor_model(valves, synthesis_prompt)

            if streaming_manager:
                update = await streaming_manager.stream_synthesis_progress("complete")
                if update:
                    yield update

    # Yield final result
    if streaming_manager:
        yield await streaming_manager.stream_phase_update("completion", "Protocol execution completed")
    
    yield f"\n## 🎯 Final Answer\n{final_response}"


async def execute_tasks_on_chunks_with_streaming(
    tasks: List[str],
    chunks: List[str],
    current_round: int,
    valves: Any,
    call_ollama: Callable,
    TaskResult: Any,
    streaming_manager: Any = None
) -> Dict:
    """Execute tasks on chunks with detailed streaming updates"""
    
    overall_task_results = []
    total_attempts_this_call = 0
    total_timeouts_this_call = 0
    streaming_updates = []

    for task_idx, task in enumerate(tasks):
        if streaming_manager:
            update = await streaming_manager.stream_task_execution_progress(
                task_idx=task_idx,
                total_tasks=len(tasks),
                task_description=task
            )
            if update:
                streaming_updates.append(update)

        results_for_this_task_from_chunks = []
        
        for chunk_idx, chunk in enumerate(chunks):
            if streaming_manager:
                update = await streaming_manager.stream_task_execution_progress(
                    task_idx=task_idx,
                    total_tasks=len(tasks),
                    chunk_idx=chunk_idx,
                    total_chunks=len(chunks),
                    task_description=task
                )
                if update:
                    streaming_updates.append(update)

            total_attempts_this_call += 1
            
            # Generate local prompt
            local_prompt = get_minions_local_task_prompt(
                chunk=chunk,
                task=task,
                chunk_idx=chunk_idx,
                total_chunks=len(chunks),
                valves=valves
            )

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
                    debug_mode=valves.debug_mode,
                    TaskResultModel=TaskResult,
                    structured_output_fallback_enabled=getattr(valves, 'structured_output_fallback_enabled', True)
                )

                if not response_data.get('_is_none_equivalent'):
                    extracted_info = response_data.get('answer') or response_data.get('explanation', 'Could not extract details.')
                    results_for_this_task_from_chunks.append({
                        "text": f"[Chunk {chunk_idx+1}]: {extracted_info}",
                        "fingerprint": response_data.get('fingerprint')
                    })

            except asyncio.TimeoutError:
                total_timeouts_this_call += 1

        # Aggregate results for this task
        if results_for_this_task_from_chunks:
            aggregated_result = "\n".join([r["text"] for r in results_for_this_task_from_chunks])
            overall_task_results.append({
                "task": task,
                "result": aggregated_result,
                "status": "success"
            })
        else:
            overall_task_results.append({
                "task": task,
                "result": "No relevant information found",
                "status": "no_info"
            })

    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_attempts_this_call,
        "total_chunk_processing_timeouts": total_timeouts_this_call,
        "streaming_updates": streaming_updates
    }