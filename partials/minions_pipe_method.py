import asyncio
import re # Added re
from enum import Enum # Ensured Enum is present
from typing import Any, List, Callable, Dict
from fastapi import Request

from .common_api_calls import call_claude, call_ollama
from .minions_protocol_logic import execute_tasks_on_chunks, calculate_token_savings
from .common_file_processing import create_chunks
from .minions_models import TaskResult, RoundMetrics # Import RoundMetrics
from .common_context_utils import extract_context_from_messages, extract_context_from_files
from .minions_decomposition_logic import decompose_task
from .minions_prompts import get_minions_synthesis_claude_prompt
# Removed: from .common_query_utils import QueryComplexityClassifier, QueryComplexity

# --- Content from common_query_utils.py START ---
class QueryComplexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"

class QueryComplexityClassifier:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        # Keywords indicating complexity
        self.complex_keywords = [
            "analyze", "compare", "contrast", "summarize", "explain in detail",
            "discuss", "critique", "evaluate", "recommend", "predict", "what if",
            "how does", "why does", "implications"
        ]
        self.medium_keywords = [
            "list", "describe", "details of", "tell me about", "what are the"
        ]
        # Question words (simple ones often start fact-based questions)
        self.simple_question_starters = ["what is", "who is", "when was", "where is", "define"]

    def classify_query(self, query: str) -> QueryComplexity:
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Query='{query_lower}', WordCount={word_count}")

        # Rule 1: Complex Keywords
        for keyword in self.complex_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched complex keyword '{keyword}'")
                return QueryComplexity.COMPLEX

        # Rule 2: Word Count for Complex
        if word_count > 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched complex by word count (>25)")
            return QueryComplexity.COMPLEX

        # Rule 3: Word Count for Simple (and simple question starters)
        if word_count < 10:
            is_simple_starter = any(query_lower.startswith(starter) for starter in self.simple_question_starters)
            if is_simple_starter:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched simple by word count (<10) and starter.")
                return QueryComplexity.SIMPLE

        # Rule 4: Medium Keywords
        for keyword in self.medium_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched medium keyword '{keyword}'")
                return QueryComplexity.MEDIUM

        if word_count >= 10 and word_count <= 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched medium by word count (10-25)")
            return QueryComplexity.MEDIUM

        if word_count < 10: # Default for short queries not caught by simple_question_starters
             if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Defaulting short query to MEDIUM (no simple starter)")
             return QueryComplexity.MEDIUM

        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Defaulting to MEDIUM (no other rules matched clearly)")
        return QueryComplexity.MEDIUM
# --- Content from common_query_utils.py END ---


async def _call_claude_directly(valves: Any, query: str, call_claude_func: Callable) -> str:
    """Fallback to direct Claude call when no context is available"""
    return await call_claude_func(valves, f"Please answer this question: {query}")

async def _execute_minions_protocol(
    valves: Any,
    query: str,
    context: str,
    call_claude: Callable,
    call_ollama_func: Callable,
    TaskResultModel: Any
) -> str:
    """Execute the MinionS protocol"""
    conversation_log = []
    debug_log = []
    scratchpad_content = ""
    all_round_results_aggregated = []
    all_round_metrics: List[RoundMetrics] = [] # Initialize Metrics List
    global_unique_fingerprints_seen = set() # Initialize Global Fingerprint Set
    decomposition_prompts_history = []
    synthesis_prompts_history = []
    final_response = "No answer could be synthesized."
    claude_provided_final_answer = False
    total_tasks_executed_local = 0
    total_chunks_processed_for_stats = 0
    total_chunk_processing_timeouts_accumulated = 0
    synthesis_input_summary = ""

    overall_start_time = asyncio.get_event_loop().time()
    if valves.debug_mode:
        debug_log.append(f"🔍 **Debug Info (MinionS v0.2.0):**\n- Query: {query[:100]}...\n- Context length: {len(context)} chars")
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**")

    # Initialize Query Complexity Classifier and Classify Query
    query_classifier = QueryComplexityClassifier(debug_mode=valves.debug_mode)
    query_complexity_level = query_classifier.classify_query(query)

    if valves.debug_mode:
        debug_log.append(f"🧠 Query classified as: {query_complexity_level.value} (Debug Mode)")
    # Optional: Add to conversation_log if you want user to see it always
    # if valves.show_conversation:
    #     conversation_log.append(f"🧠 Initial query classified as complexity: {query_complexity_level.value}")

    chunks = create_chunks(context, valves.chunk_size, valves.max_chunks)
    if not chunks and context:
        return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size."
    if not chunks and not context:
        conversation_log.append("ℹ️ No context or chunks to process with MinionS. Attempting direct call.")
        start_time_claude = 0
        if valves.debug_mode: 
            start_time_claude = asyncio.get_event_loop().time()
        try:
            final_response = await _call_claude_directly(valves, query, call_claude_func=call_claude)
            if valves.debug_mode:
                end_time_claude = asyncio.get_event_loop().time()
                time_taken_claude = end_time_claude - start_time_claude
                debug_log.append(f"⏱️ Claude direct call took {time_taken_claude:.2f}s. (Debug Mode)")
            output_parts = []
            if valves.show_conversation:
                output_parts.append("## 🗣️ MinionS Collaboration (Direct Call)")
                output_parts.extend(conversation_log)
                output_parts.append("---")
            if valves.debug_mode:
                output_parts.append("### 🔍 Debug Log")
                output_parts.extend(debug_log)
                output_parts.append("---")
            output_parts.append(f"## 🎯 Final Answer (Direct)\n{final_response}")
            return "\n".join(output_parts)
        except Exception as e:
            return f"❌ **Error in direct Claude call:** {str(e)}"

    total_chunks_processed_for_stats = len(chunks)

    for current_round in range(valves.max_rounds):
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {current_round + 1}/{valves.max_rounds}... (Debug Mode)**")
        
        if valves.show_conversation:
            conversation_log.append(f"### 🎯 Round {current_round + 1}/{valves.max_rounds} - Task Decomposition Phase")

        # Call the new decompose_task function
        # Note: now returns three values instead of two
        tasks, claude_response_for_decomposition, decomposition_prompt = await decompose_task(
            valves=valves,
            call_claude_func=call_claude,
            query=query,
            scratchpad_content=scratchpad_content,
            num_chunks=len(chunks),
            max_tasks_per_round=valves.max_tasks_per_round,
            current_round=current_round + 1,
            conversation_log=conversation_log,
            debug_log=debug_log
        )
        
        # Store the decomposition prompt in history
        if decomposition_prompt:  # Only add if not empty (error case)
            decomposition_prompts_history.append(decomposition_prompt)
        
        # Handle Claude communication errors from decompose_task
        if claude_response_for_decomposition.startswith("CLAUDE_ERROR:"):
            error_message = claude_response_for_decomposition.replace("CLAUDE_ERROR: ", "")
            final_response = f"MinionS protocol failed during task decomposition: {error_message}"
            break

        # Log the raw Claude response if conversation is shown
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round + 1}):**\n{claude_response_for_decomposition}\n")

        # Check for "FINAL ANSWER READY."
        if "FINAL ANSWER READY." in claude_response_for_decomposition:
            final_response = claude_response_for_decomposition.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_provided_final_answer = True
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Claude indicates final answer is ready in round {current_round + 1}.**")
            scratchpad_content += f"\n\n**Round {current_round + 1}:** Claude provided final answer."
            break

        if not tasks:
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Claude provided no new tasks in round {current_round + 1}. Proceeding to final synthesis.**")
            break
        
        total_tasks_executed_local += len(tasks)
        
        if valves.show_conversation:
            conversation_log.append(f"### ⚡ Round {current_round + 1} - Parallel Execution Phase (Processing {len(chunks)} chunks for {len(tasks)} tasks)")
        
        execution_details = await execute_tasks_on_chunks(
            tasks, chunks, conversation_log if valves.show_conversation else debug_log, 
            current_round + 1, valves, call_ollama_func, TaskResultModel
        )
        current_round_task_results = execution_details["results"]
        round_chunk_attempts = execution_details["total_chunk_processing_attempts"]
        round_chunk_timeouts = execution_details["total_chunk_processing_timeouts"]

        # Process Metrics After execute_tasks_on_chunks
        raw_metrics_data = execution_details.get("round_metrics_data")

        # Extract and Calculate Confidence Metrics
        confidence_data = execution_details.get("confidence_metrics_data")
        task_confidences = confidence_data.get("task_confidences", []) if confidence_data else []

        round_avg_numeric_confidence = 0.0
        if task_confidences:
            total_confidence_sum = sum(tc['avg_numeric_confidence'] for tc in task_confidences if tc.get('contributing_successful_chunks', 0) > 0)
            num_successful_tasks_with_confidence = sum(1 for tc in task_confidences if tc.get('contributing_successful_chunks', 0) > 0)
            if num_successful_tasks_with_confidence > 0:
                round_avg_numeric_confidence = total_confidence_sum / num_successful_tasks_with_confidence

        round_confidence_distribution = confidence_data.get("round_confidence_distribution", {"HIGH": 0, "MEDIUM": 0, "LOW": 0}) if confidence_data else {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        # Determine Confidence Trend (before creating current RoundMetrics object)
        confidence_trend = "N/A"
        if all_round_metrics: # Check if there are previous rounds' metrics
            previous_round_metric = all_round_metrics[-1]
            previous_avg_confidence = previous_round_metric.avg_confidence_score
            diff = round_avg_numeric_confidence - previous_avg_confidence
            if diff > 0.05:
                confidence_trend = "improving"
            elif diff < -0.05:
                confidence_trend = "declining"
            else:
                confidence_trend = "stable"

        if raw_metrics_data:
            # Process Findings for Redundancy Metrics
            current_round_new_findings = 0
            current_round_duplicate_findings = 0
            # current_round_fingerprints_seen_this_round = set() # Not strictly needed for plan's definition

            task_execution_results = execution_details.get("results", [])
            for task_result in task_execution_results:
                if task_result.get("status") == "success" and task_result.get("detailed_findings"):
                    for finding in task_result["detailed_findings"]:
                        fingerprint = finding.get("fingerprint")
                        if fingerprint:
                            # Check against global set first for duplicates from previous rounds or earlier in this round
                            if fingerprint in global_unique_fingerprints_seen:
                                current_round_duplicate_findings += 1
                            else:
                                current_round_new_findings += 1
                                global_unique_fingerprints_seen.add(fingerprint)
                            # current_round_fingerprints_seen_this_round.add(fingerprint)


            total_findings_this_round = current_round_new_findings + current_round_duplicate_findings
            redundancy_percentage_this_round = (current_round_duplicate_findings / total_findings_this_round) * 100 if total_findings_this_round > 0 else 0.0

            try: # Add a try-except block for robustness when creating RoundMetrics
                round_metric = RoundMetrics(
                    round_number=raw_metrics_data["round_number"],
                    tasks_executed=raw_metrics_data["tasks_executed"],
                    task_success_count=raw_metrics_data["task_success_count"],
                    task_failure_count=raw_metrics_data["task_failure_count"],
                    avg_chunk_processing_time_ms=raw_metrics_data["avg_chunk_processing_time_ms"],
                    # total_unique_findings_count=0,  # Placeholder for Iteration 1 REMOVED
                    execution_time_ms=raw_metrics_data["execution_time_ms"],
                    success_rate=raw_metrics_data["success_rate"],
                    # Add new confidence fields
                    avg_confidence_score=round_avg_numeric_confidence,
                    confidence_distribution=round_confidence_distribution,
                    confidence_trend=confidence_trend,
                    # Add new redundancy fields
                    new_findings_count_this_round=current_round_new_findings,
                    duplicate_findings_count_this_round=current_round_duplicate_findings,
                    redundancy_percentage_this_round=redundancy_percentage_this_round,
                    total_unique_findings_count=len(global_unique_fingerprints_seen)
                )
                all_round_metrics.append(round_metric)

                # Format and append metrics summary (now includes redundancy)
                metrics_summary = (
                    f"**📊 Round {round_metric.round_number} Metrics:**\n"
                    f"  - Tasks Executed: {round_metric.tasks_executed}, Success Rate: {round_metric.success_rate:.2%}\n"
                    f"  - Task Counts (S/F): {round_metric.task_success_count}/{round_metric.task_failure_count}\n"
                    f"  - Findings (New/Dup): {round_metric.new_findings_count_this_round}/{round_metric.duplicate_findings_count_this_round}, Total Unique: {round_metric.total_unique_findings_count}\n"
                    f"  - Redundancy This Round: {round_metric.redundancy_percentage_this_round:.1f}%\n"
                    f"  - Avg Confidence: {round_metric.avg_confidence_score:.2f} ({round_metric.confidence_trend})\n"
                    f"  - Confidence Dist (H/M/L): {round_metric.confidence_distribution.get('HIGH',0)}/{round_metric.confidence_distribution.get('MEDIUM',0)}/{round_metric.confidence_distribution.get('LOW',0)}\n"
                    f"  - Round Time: {round_metric.execution_time_ms:.0f} ms, Avg Chunk Time: {round_metric.avg_chunk_processing_time_ms:.0f} ms"
                )
                scratchpad_content += f"\n\n{metrics_summary}"
                if valves.show_conversation:
                    conversation_log.append(metrics_summary)

            except KeyError as e:
                if valves.debug_mode:
                    debug_log.append(f"⚠️ **Metrics Error:** Missing key {e} in round_metrics_data for round {current_round + 1}. Skipping metrics for this round.")
            except Exception as e: # Catch any other validation error from Pydantic
                 if valves.debug_mode:
                    debug_log.append(f"⚠️ **Metrics Error:** Could not create RoundMetrics object for round {current_round + 1} due to {type(e).__name__}: {e}. Skipping metrics for this round.")


        if round_chunk_attempts > 0:
            timeout_percentage_this_round = (round_chunk_timeouts / round_chunk_attempts) * 100
            log_msg_timeout_stat = f"**📈 Round {current_round + 1} Local LLM Timeout Stats:** {round_chunk_timeouts}/{round_chunk_attempts} chunk calls timed out ({timeout_percentage_this_round:.1f}%)."
            if valves.show_conversation: 
                conversation_log.append(log_msg_timeout_stat)
            if valves.debug_mode: 
                debug_log.append(log_msg_timeout_stat)

            if timeout_percentage_this_round >= valves.max_round_timeout_failure_threshold_percent:
                warning_msg = f"⚠️ **Warning:** Round {current_round + 1} exceeded local LLM timeout threshold of {valves.max_round_timeout_failure_threshold_percent}%. Results from this round may be incomplete or unreliable."
                if valves.show_conversation: 
                    conversation_log.append(warning_msg)
                if valves.debug_mode: 
                    debug_log.append(warning_msg)
                scratchpad_content += f"\n\n**Note from Round {current_round + 1}:** High percentage of local model timeouts ({timeout_percentage_this_round:.1f}%) occurred, results for this round may be partial."
        
        round_summary_for_scratchpad_parts = []
        for r_val in current_round_task_results:
            status_icon = "✅" if r_val['status'] == 'success' else ("⏰" if 'timeout' in r_val['status'] else "❓")
            summary_text = f"- {status_icon} Task: {r_val['task']}, Result: {r_val['result'][:200]}..." if r_val['status'] == 'success' else f"- {status_icon} Task: {r_val['task']}, Status: {r_val['result']}"
            round_summary_for_scratchpad_parts.append(summary_text)
        
        if round_summary_for_scratchpad_parts:
            scratchpad_content += f"\n\n**Results from Round {current_round + 1}:**\n" + "\n".join(round_summary_for_scratchpad_parts)
        
        all_round_results_aggregated.extend(current_round_task_results)
        total_chunk_processing_timeouts_accumulated += round_chunk_timeouts

        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {current_round + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**")

        if current_round == valves.max_rounds - 1:
            if valves.show_conversation:
                conversation_log.append(f"**🏁 Reached max rounds ({valves.max_rounds}). Proceeding to final synthesis.**")

    if not claude_provided_final_answer:
        if valves.show_conversation:
            conversation_log.append("\n### 🔄 Final Synthesis Phase")
        if not all_round_results_aggregated:
            final_response = "No information was gathered from the document by local models across the rounds."
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Claude (Synthesis):** {final_response}")
        else:
            synthesis_input_summary = "\n".join([f"- Task: {r['task']}\n  Result: {r['result']}" for r in all_round_results_aggregated if r['status'] == 'success'])
            if not synthesis_input_summary:
                synthesis_input_summary = "No definitive information was found by local models. The original query was: " + query
            
            synthesis_prompt = get_minions_synthesis_claude_prompt(query, synthesis_input_summary, valves)
            synthesis_prompts_history.append(synthesis_prompt)
            
            start_time_claude_synth = 0
            if valves.debug_mode:
                start_time_claude_synth = asyncio.get_event_loop().time()
            try:
                final_response = await call_claude(valves, synthesis_prompt)
                if valves.debug_mode:
                    end_time_claude_synth = asyncio.get_event_loop().time()
                    time_taken_claude_synth = end_time_claude_synth - start_time_claude_synth
                    debug_log.append(f"⏱️ Claude call (Final Synthesis) took {time_taken_claude_synth:.2f}s. (Debug Mode)")
                if valves.show_conversation:
                    conversation_log.append(f"**🤖 Claude (Final Synthesis):**\n{final_response}")
            except Exception as e:
                if valves.show_conversation:
                    conversation_log.append(f"❌ Error during final synthesis: {e}")
                final_response = "Error during final synthesis. Raw findings might be available in conversation log."
    
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
    output_parts.append(final_response)

    summary_for_stats = synthesis_input_summary if not claude_provided_final_answer else scratchpad_content

    stats = calculate_token_savings(
        decomposition_prompts_history, synthesis_prompts_history,
        summary_for_stats, final_response,
        len(context), len(query), total_chunks_processed_for_stats, total_tasks_executed_local
    )
    
    # Override the total_rounds if needed to show actual rounds executed
    actual_rounds_executed = len(decomposition_prompts_history)
    if actual_rounds_executed == 0 and current_round >= 0:
        # Fallback: if history wasn't populated properly, at least show rounds attempted
        actual_rounds_executed = min(current_round + 1, valves.max_rounds)
    
    total_successful_tasks = len([r for r in all_round_results_aggregated if r['status'] == 'success'])
    tasks_with_any_timeout = len([r for r in all_round_results_aggregated if r['status'] == 'timeout_all_chunks'])

    output_parts.append(f"\n## 📊 MinionS Efficiency Stats (v0.2.0)")
    output_parts.append(f"- **Protocol:** MinionS (Multi-Round)")
    output_parts.append(f"- **Query Complexity:** {query_complexity_level.value}") # Display Query Complexity
    output_parts.append(f"- **Rounds executed:** {actual_rounds_executed}/{valves.max_rounds}")
    output_parts.append(f"- **Total tasks for local LLM:** {stats['total_tasks_executed_local']}")
    output_parts.append(f"- **Successful tasks (local):** {total_successful_tasks}")
    output_parts.append(f"- **Tasks where all chunks timed out (local):** {tasks_with_any_timeout}")
    output_parts.append(f"- **Total individual chunk processing timeouts (local):** {total_chunk_processing_timeouts_accumulated}")
    output_parts.append(f"- **Chunks processed per task (local):** {stats['total_chunks_processed_local'] if stats['total_tasks_executed_local'] > 0 else 0}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    output_parts.append(f"\n## 💰 Token Savings Analysis (Claude: {valves.remote_model})")
    output_parts.append(f"- **Traditional single call (est.):** ~{stats['traditional_tokens_claude']:,} tokens")
    output_parts.append(f"- **MinionS multi-round (Claude only):** ~{stats['minions_tokens_claude']:,} tokens")
    output_parts.append(f"- **💰 Est. Claude Token savings:** ~{stats['percentage_savings_claude']:.1f}%")
    
    return "\n".join(output_parts)

async def minions_pipe_method(
    pipe_self: Any,
    body: dict,
    __user__: dict,
    __request__: Request,
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "minions-claude",
) -> str:
    """Execute the MinionS protocol with Claude"""
    try:
        # Validate configuration
        if not pipe_self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key (and Ollama settings if applicable) in the function settings."

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

        # If no context, make a direct call to Claude
        if not context:
            direct_response = await _call_claude_directly(pipe_self.valves, user_query, call_claude_func=call_claude)
            return (
                "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                + direct_response
            )

        # Execute the MinionS protocol with correct parameter names
        result: str = await _execute_minions_protocol(
            pipe_self.valves, 
            user_query, 
            context, 
            call_claude,    # Changed from call_claude_func
            call_ollama,    # Changed from call_ollama_func
            TaskResult      # Changed from TaskResultModel
        )
        return result

    except Exception as e:
        import traceback
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in MinionS protocol:** {error_details}"