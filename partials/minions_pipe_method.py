# Partials File: partials/minions_pipe_method.py
import asyncio
import re # Added re
from enum import Enum # Ensured Enum is present
from typing import Any, List, Callable, Dict
from fastapi import Request

from .common_api_calls import call_claude, call_ollama, call_supervisor_model
from .minions_protocol_logic import execute_tasks_on_chunks, calculate_token_savings
from .common_file_processing import create_chunks
from .minions_models import TaskResult, RoundMetrics # Import RoundMetrics
from .common_context_utils import extract_context_from_messages, extract_context_from_files
from .minions_decomposition_logic import decompose_task
from .minions_prompts import get_minions_synthesis_claude_prompt
from .minion_sufficiency_analyzer import InformationSufficiencyAnalyzer # Added import
from .minion_convergence_detector import ConvergenceDetector # Added import
# Note: TaskVisualizer, TaskStatus, TaskType, and StreamingResponseManager are imported from other partials
# Removed: from .common_query_utils import QueryComplexityClassifier, QueryComplexity

# --- Content from common_query_utils.py START ---
class QueryComplexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"

class QueryComplexityClassifier:
    """Classifies query complexity based on keywords and length."""
    def __init__(self, debug_mode: bool = False):
        """Initializes the classifier with debug mode."""
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
        """
        Classifies the given query into SIMPLE, MEDIUM, or COMPLEX.
        Uses rules based on keywords and word count.
        """
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


async def _call_supervisor_directly(valves: Any, query: str) -> str:
    """Fallback to direct supervisor call when no context is available"""
    return await call_supervisor_model(valves, f"Please answer this question: {query}")

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
    early_stopping_reason_for_output = None # Initialize for storing stopping reason

    overall_start_time = asyncio.get_event_loop().time()

    # Initialize streaming manager if enabled
    streaming_manager = None
    if getattr(valves, 'enable_streaming_responses', True):
        streaming_manager = StreamingResponseManager(valves, valves.debug_mode)

    # User query is passed directly to _execute_minions_protocol
    user_query = query # Use the passed 'query' as user_query for clarity if needed elsewhere

    # --- Performance Profile Logic ---
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

    if valves.debug_mode:
        debug_log.extend(profile_applied_details)
        debug_log.append(f"🔍 **Debug Info (MinionS v0.2.0):**\n- Query: {user_query[:100]}...\n- Context length: {len(context)} chars") # Original debug line moved after profile logic
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**")


    # Instantiate Sufficiency Analyzer
    analyzer = InformationSufficiencyAnalyzer(query=user_query, debug_mode=valves.debug_mode)
    if valves.debug_mode:
        debug_log.append(f"🧠 Sufficiency Analyzer initialized for query: {user_query[:100]}...")
        debug_log.append(f"   Identified components: {list(analyzer.components.keys())}")

    # Instantiate Convergence Detector
    convergence_detector = ConvergenceDetector(debug_mode=valves.debug_mode)
    if valves.debug_mode:
        debug_log.append(f"🧠 Convergence Detector initialized.")

    # Initialize Query Complexity Classifier and Classify Query
    query_classifier = QueryComplexityClassifier(debug_mode=valves.debug_mode)
    query_complexity_level = query_classifier.classify_query(query)

    if valves.debug_mode:
        debug_log.append(f"🧠 Query classified as: {query_complexity_level.value} (Debug Mode)")
    # Optional: Add to conversation_log if you want user to see it always
    # if valves.show_conversation:
    #     conversation_log.append(f"🧠 Initial query classified as complexity: {query_complexity_level.value}")

    # --- Dynamic Threshold Initialization ---
    doc_size_category = "medium" # Default
    context_len = len(context)
    if context_len < valves.doc_size_small_char_limit:
        doc_size_category = "small"
    elif context_len > valves.doc_size_large_char_start:
        doc_size_category = "large"

    if valves.debug_mode:
        debug_log.append(f"🧠 Document size category: {doc_size_category} (Length: {context_len} chars)")

    # Initialize effective thresholds with base values (now from current_run_... variables)
    effective_sufficiency_threshold = current_run_base_sufficiency_thresh
    effective_novelty_threshold = current_run_base_novelty_thresh
    # Base confidence thresholds for simple/medium queries will use current_run_... variables where they are applied.

    if valves.debug_mode:
        debug_log.append(f"🧠 Initial effective thresholds (after profile adjustments): Sufficiency={effective_sufficiency_threshold:.2f}, Novelty={effective_novelty_threshold:.2f}")
        debug_log.append(f"   Effective Simple Confidence Thresh (base for adaptation)={current_run_simple_query_confidence_thresh:.2f}, Medium Confidence Thresh (base for adaptation)={current_run_medium_query_confidence_thresh:.2f}")

    if valves.enable_adaptive_thresholds:
        if valves.debug_mode:
            debug_log.append(f"🧠 Adaptive thresholds ENABLED. Applying query/doc modifiers...")

        # Apply query complexity modifiers to sufficiency and novelty
        if query_complexity_level == QueryComplexity.SIMPLE:
            effective_sufficiency_threshold += valves.sufficiency_modifier_simple_query
            effective_novelty_threshold += valves.novelty_modifier_simple_query
        elif query_complexity_level == QueryComplexity.COMPLEX:
            effective_sufficiency_threshold += valves.sufficiency_modifier_complex_query
            effective_novelty_threshold += valves.novelty_modifier_complex_query

        # Clamp all thresholds to sensible ranges (e.g., 0.05 to 0.95)
        effective_sufficiency_threshold = max(0.05, min(0.95, effective_sufficiency_threshold))
        effective_novelty_threshold = max(0.05, min(0.95, effective_novelty_threshold))

        if valves.debug_mode:
            debug_log.append(f"   After query complexity mods: Eff.Sufficiency={effective_sufficiency_threshold:.2f}, Eff.Novelty={effective_novelty_threshold:.2f}")

    chunks = create_chunks(context, valves.chunk_size, valves.max_chunks)
    if not chunks and context:
        return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size."
    if not chunks and not context:
        conversation_log.append("ℹ️ No context or chunks to process with MinionS. Attempting direct call.")
        start_time_claude = 0
        if valves.debug_mode: 
            start_time_claude = asyncio.get_event_loop().time()
        try:
            final_response = await _call_supervisor_directly(valves, query)
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

    # Initialize effective confidence threshold variables to store them for the performance report
    final_effective_simple_conf_thresh = current_run_simple_query_confidence_thresh
    final_effective_medium_conf_thresh = current_run_medium_query_confidence_thresh

    for current_round in range(current_run_max_rounds): # Use current_run_max_rounds
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {current_round + 1}/{current_run_max_rounds}... (Debug Mode)**") # Use current_run_max_rounds
        
        if valves.show_conversation:
            conversation_log.append(f"### 🎯 Round {current_round + 1}/{current_run_max_rounds} - Task Decomposition Phase") # Use current_run_max_rounds

        # Stream task decomposition progress if streaming is enabled
        if streaming_manager and hasattr(streaming_manager, 'stream_task_decomposition_progress'):
            # Analyzing complexity
            update = await streaming_manager.stream_task_decomposition_progress(
                "analyzing_complexity", 1, 5, f"Analyzing query for round {current_round + 1}"
            )
            if update:
                conversation_log.append(update)

        # Call the new decompose_task function
        # Note: now returns three values instead of two
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
        
        # Stream task generation completion
        if streaming_manager and hasattr(streaming_manager, 'stream_task_decomposition_progress'):
            update = await streaming_manager.stream_task_decomposition_progress(
                "generating_tasks", 3, 5, f"Generated {len(tasks)} tasks"
            )
            if update:
                conversation_log.append(update)

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
            # Extract content after "FINAL ANSWER READY."
            answer_parts = claude_response_for_decomposition.split("FINAL ANSWER READY.", 1)
            if len(answer_parts) > 1:
                final_response = answer_parts[1].strip()
                # Clean up any remaining formatting
                if final_response.startswith('"') and final_response.endswith('"'):
                    final_response = final_response[1:-1]
            else:
                final_response = "Final answer was indicated but content could not be extracted."
            
            claude_provided_final_answer = True
            early_stopping_reason_for_output = "Claude provided FINAL ANSWER READY." # Explicitly set reason
            if valves.show_conversation: # This log already exists
                conversation_log.append(f"**🤖 Claude indicates final answer is ready in round {current_round + 1}.**")
                conversation_log.append(f"**🤖 Claude (Final Answer):**\n{final_response}")
            scratchpad_content += f"\n\n**Round {current_round + 1}:** Claude provided final answer. Stopping." # Added "Stopping."
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
            current_round + 1, valves, call_ollama_func, TaskResultModel, streaming_manager
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
                    # Sufficiency fields will be added below
                )
                all_round_metrics.append(round_metric) # Add before sufficiency update

                # --> Sufficiency Analysis <--
                metric_to_update = round_metric # This is the one we just added
                if valves.debug_mode:
                    debug_log.append(f"   [Debug] Updating Sufficiency Analyzer with scratchpad content for round {current_round + 1}...")

                analyzer.update_components(text_to_analyze=scratchpad_content, round_avg_confidence=metric_to_update.avg_confidence_score)
                sufficiency_details = analyzer.get_analysis_details()

                metric_to_update.sufficiency_score = sufficiency_details["sufficiency_score"]
                metric_to_update.component_coverage_percentage = sufficiency_details["component_coverage_percentage"]
                metric_to_update.information_components = sufficiency_details["information_components_status"]

                if valves.debug_mode:
                    debug_log.append(f"   [Debug] Sufficiency for round {current_round + 1}: Score={metric_to_update.sufficiency_score:.2f}, Coverage={metric_to_update.component_coverage_percentage:.2f}")
                    debug_log.append(f"   [Debug] Component Status: {metric_to_update.information_components}")

                # --> Convergence Detection Calculations (after sufficiency is updated) <--
                if metric_to_update: # Ensure we have the current round's metric object
                    previous_round_metric_obj = all_round_metrics[-2] if len(all_round_metrics) > 1 else None

                    convergence_calcs = convergence_detector.calculate_round_convergence_metrics(
                        current_round_metric=metric_to_update,
                        previous_round_metric=previous_round_metric_obj
                    )

                    metric_to_update.information_gain_rate = convergence_calcs.get("information_gain_rate", 0.0)
                    metric_to_update.novel_findings_percentage_this_round = convergence_calcs.get("novel_findings_percentage_this_round", 0.0)
                    metric_to_update.task_failure_rate_trend = convergence_calcs.get("task_failure_rate_trend", "N/A")
                    metric_to_update.predicted_value_of_next_round = convergence_calcs.get("predicted_value_of_next_round", "N/A")
                    # convergence_detected_this_round is set by check_for_convergence below

                    if valves.debug_mode:
                        debug_log.append(f"   [Debug] Convergence Detector calculated for round {metric_to_update.round_number}: InfoGain={metric_to_update.information_gain_rate:.0f}, Novelty={metric_to_update.novel_findings_percentage_this_round:.2%}, FailTrend={metric_to_update.task_failure_rate_trend}, NextRoundValue={metric_to_update.predicted_value_of_next_round}")

                # Format and append metrics summary (now includes redundancy, sufficiency, AND convergence calcs)
                component_status_summary = {k: ('Met' if v else 'Not Met') for k,v in metric_to_update.information_components.items()}
                metrics_summary = (
                    f"**📊 Round {metric_to_update.round_number} Metrics:**\n"
                    f"  - Tasks Executed: {metric_to_update.tasks_executed}, Success Rate: {metric_to_update.success_rate:.2%}\n"
                    f"  - Task Counts (S/F): {metric_to_update.task_success_count}/{metric_to_update.task_failure_count}\n"
                    f"  - Findings (New/Dup): {metric_to_update.new_findings_count_this_round}/{metric_to_update.duplicate_findings_count_this_round}, Total Unique: {metric_to_update.total_unique_findings_count}\n"
                    f"  - Redundancy This Round: {metric_to_update.redundancy_percentage_this_round:.1f}%\n"
                    f"  - Avg Confidence: {metric_to_update.avg_confidence_score:.2f} ({metric_to_update.confidence_trend})\n"
                    f"  - Confidence Dist (H/M/L): {metric_to_update.confidence_distribution.get('HIGH',0)}/{metric_to_update.confidence_distribution.get('MEDIUM',0)}/{metric_to_update.confidence_distribution.get('LOW',0)}\n"
                    f"  - Sufficiency Score: {metric_to_update.sufficiency_score:.2f}, Info Coverage: {metric_to_update.component_coverage_percentage:.2%}\n"
                    f"  - Components Status: {component_status_summary}\n"
                    f"  - Info Gain Rate: {metric_to_update.information_gain_rate:.0f}, Novelty This Round: {metric_to_update.novel_findings_percentage_this_round:.1%}\n"
                    f"  - Task Fail Trend: {metric_to_update.task_failure_rate_trend}, Predicted Next Round Value: {metric_to_update.predicted_value_of_next_round}\n"
                    f"  - Converged This Round: {'Yes' if metric_to_update.convergence_detected_this_round else 'No'}\n" # Will be updated by convergence check later
                    f"  - Round Time: {metric_to_update.execution_time_ms:.0f} ms, Avg Chunk Time: {metric_to_update.avg_chunk_processing_time_ms:.0f} ms"
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

        # Placeholder for Sufficiency-Based Stopping Logic (Debug)
        # This is checked *before* other early stopping conditions like confidence thresholds.
        # The 'metric_to_update' variable should be the most up-to-date version of the current round's metrics.
        # It now includes sufficiency and initial convergence calculation fields.

        # --- First Round Novelty Adjustment (occurs only after round 0 processing) ---
        if current_round == 0 and valves.enable_adaptive_thresholds and 'metric_to_update' in locals() and metric_to_update:
            first_round_novelty_perc = metric_to_update.novel_findings_percentage_this_round
            if first_round_novelty_perc > valves.first_round_high_novelty_threshold:
                original_eff_sufficiency_before_1st_round_adj = effective_sufficiency_threshold
                effective_sufficiency_threshold += valves.sufficiency_modifier_high_first_round_novelty
                effective_sufficiency_threshold = max(0.05, min(0.95, effective_sufficiency_threshold)) # Clamp again
                if valves.debug_mode:
                    debug_log.append(
                        f"🧠 High first round novelty ({first_round_novelty_perc:.2%}) detected. "
                        f"Adjusting effective sufficiency threshold from {original_eff_sufficiency_before_1st_round_adj:.2f} to {effective_sufficiency_threshold:.2f}."
                    )

        if valves.debug_mode and 'metric_to_update' in locals() and metric_to_update:
            # Placeholder Debug for Sufficiency (already exists, using the dynamically adjusted threshold now)
            # This hypothetical threshold is just for this debug log, actual check uses effective_sufficiency_threshold
            debug_hypothetical_sufficiency_thresh_for_log = 0.75
            if metric_to_update.sufficiency_score >= debug_hypothetical_sufficiency_thresh_for_log:
                debug_log.append(
                    f"   [Debug Placeholder] Sufficiency score {metric_to_update.sufficiency_score:.2f} >= "
                    f"{debug_hypothetical_sufficiency_thresh_for_log} (hypothetical debug value). "
                    f"Effective sufficiency for convergence check is {effective_sufficiency_threshold:.2f}."
                )
            else:
                debug_log.append(
                    f"   [Debug Placeholder] Sufficiency score {metric_to_update.sufficiency_score:.2f} < "
                    f"{debug_hypothetical_sufficiency_thresh_for_log} (hypothetical debug value). "
                    f"Effective sufficiency for convergence check is {effective_sufficiency_threshold:.2f}."
                )

        # --> Convergence Check (for early stopping) <--
        # This comes before the original early stopping logic. If convergence is met, we stop.
        if 'metric_to_update' in locals() and metric_to_update and valves.enable_early_stopping:
            converged, convergence_reason = convergence_detector.check_for_convergence(
                current_round_metric=metric_to_update,
                sufficiency_score=metric_to_update.sufficiency_score, # Base sufficiency from analyzer
                total_rounds_executed=current_round + 1,
                effective_novelty_to_use=effective_novelty_threshold, # Pass calculated value
                effective_sufficiency_to_use=effective_sufficiency_threshold, # Pass calculated value
                valves=valves,
                all_round_metrics=all_round_metrics
            )
            if converged:
                metric_to_update.convergence_detected_this_round = True
                # Update the metrics_summary in scratchpad and conversation_log one last time with Converged=Yes
                # This is a bit repetitive but ensures the log reflects the final state that caused the stop.
                component_status_summary = {k: ('Met' if v else 'Not Met') for k,v in metric_to_update.information_components.items()}
                updated_metrics_summary_for_convergence_stop = (
                    f"**📊 Round {metric_to_update.round_number} Metrics (Final Update Before Convergence Stop):**\n"
                    f"  - Tasks Executed: {metric_to_update.tasks_executed}, Success Rate: {metric_to_update.success_rate:.2%}\n"
                    f"  - Task Counts (S/F): {metric_to_update.task_success_count}/{metric_to_update.task_failure_count}\n"
                    f"  - Findings (New/Dup): {metric_to_update.new_findings_count_this_round}/{metric_to_update.duplicate_findings_count_this_round}, Total Unique: {metric_to_update.total_unique_findings_count}\n"
                    f"  - Redundancy This Round: {metric_to_update.redundancy_percentage_this_round:.1f}%\n"
                    f"  - Avg Confidence: {metric_to_update.avg_confidence_score:.2f} ({metric_to_update.confidence_trend})\n"
                    f"  - Confidence Dist (H/M/L): {metric_to_update.confidence_distribution.get('HIGH',0)}/{metric_to_update.confidence_distribution.get('MEDIUM',0)}/{metric_to_update.confidence_distribution.get('LOW',0)}\n"
                    f"  - Sufficiency Score: {metric_to_update.sufficiency_score:.2f}, Info Coverage: {metric_to_update.component_coverage_percentage:.2%}\n"
                    f"  - Components Status: {component_status_summary}\n"
                    f"  - Info Gain Rate: {metric_to_update.information_gain_rate:.0f}, Novelty This Round: {metric_to_update.novel_findings_percentage_this_round:.1%}\n"
                    f"  - Task Fail Trend: {metric_to_update.task_failure_rate_trend}, Predicted Next Round Value: {metric_to_update.predicted_value_of_next_round}\n"
                    f"  - Converged This Round: {'Yes' if metric_to_update.convergence_detected_this_round else 'No'}\n"
                    f"  - Round Time: {metric_to_update.execution_time_ms:.0f} ms, Avg Chunk Time: {metric_to_update.avg_chunk_processing_time_ms:.0f} ms"
                )
                scratchpad_content += f"\n\n{updated_metrics_summary_for_convergence_stop}" # Append the final metrics to scratchpad
                if valves.show_conversation: # Replace the last metrics log with the fully updated one
                    if conversation_log and conversation_log[-1].startswith("**📊 Round"): conversation_log[-1] = updated_metrics_summary_for_convergence_stop
                    else: conversation_log.append(updated_metrics_summary_for_convergence_stop)

                early_stopping_reason_for_output = convergence_reason
                if valves.show_conversation:
                    conversation_log.append(f"**⚠️ Early Stopping Triggered (Convergence):** {convergence_reason}")
                if valves.debug_mode:
                    debug_log.append(f"**⚠️ Early Stopping Triggered (Convergence):** {convergence_reason} (Debug Mode)")
                scratchpad_content += f"\n\n**EARLY STOPPING (Convergence Round {current_round + 1}):** {convergence_reason}"
                if valves.debug_mode:
                     debug_log.append(f"**🏁 Breaking loop due to convergence in Round {current_round + 1}. (Debug Mode)**")
                break # Exit the round loop

        # v0.3.8 Adaptive Round Control
        if hasattr(valves, 'adaptive_rounds') and valves.adaptive_rounds and len(all_round_results_aggregated) >= 2:
            try:
                # Get current and previous round results for analysis
                current_round_results = [r for r in all_round_results_aggregated if r.get('round') == current_round + 1]
                previous_round_results = [r for r in all_round_results_aggregated if r.get('round') == current_round]
                
                # Simple adaptive analysis based on confidence and information gain
                if current_round_results and previous_round_results:
                    current_avg_conf = sum(1 for r in current_round_results if r.get('status') == 'success') / len(current_round_results) if current_round_results else 0.0
                    should_stop_adaptive = (current_avg_conf >= getattr(valves, 'confidence_threshold_adaptive', 0.8) and 
                                          len(current_round_results) > 0)
                    
                    if should_stop_adaptive and (current_round + 1) >= getattr(valves, 'min_rounds', 1):
                        early_stopping_reason_for_output = f"Adaptive round control: High confidence ({current_avg_conf:.2f}) reached"
                        if valves.show_conversation:
                            conversation_log.append(f"**⚠️ Adaptive Early Stopping:** {early_stopping_reason_for_output}")
                        if valves.debug_mode:
                            debug_log.append(f"**⚠️ Adaptive Early Stopping:** {early_stopping_reason_for_output} (Debug Mode)")
                        scratchpad_content += f"\n\n**ADAPTIVE STOPPING (Round {current_round + 1}):** {early_stopping_reason_for_output}"
                        break
                        
            except Exception as e:
                if valves.debug_mode:
                    debug_log.append(f"**⚠️ Adaptive round control error:** {e} (Debug Mode)")

        # Original Early Stopping Logic (Confidence-based)
        # This will only be reached if convergence was NOT met and we didn't break above.
        # Note: 'round_metric' is the original metric object from raw_metrics_data,
        # 'metric_to_update' is the same object, but after being updated with sufficiency.
        # So, using 'metric_to_update' here for consistency if we were to integrate sufficiency into this logic.
        # However, the current early stopping is based on avg_confidence_score which is set before sufficiency.
        # For now, we keep `round_metric` for the existing logic as it was originally.
        # If sufficiency were to be a primary driver for early stopping, this would need refactoring.
        if valves.enable_early_stopping and round_metric: # Ensure round_metric exists
            stop_early = False
            stopping_reason = ""

            # Ensure we've met the minimum number of rounds
            if (current_round + 1) >= valves.min_rounds_before_stopping:
                current_avg_confidence = round_metric.avg_confidence_score # This is from the current round's raw metrics

                # Determine effective confidence threshold for current query type
                current_query_type_base_confidence_threshold = 0.0 # This will be set to the current_run_... value
                threshold_name_for_log = "N/A"
                # Store the calculated effective confidence threshold for the performance report
                # Initialize with base, then adapt if needed
                effective_confidence_threshold_for_query_this_check = 0.0


                if query_complexity_level == QueryComplexity.SIMPLE:
                    current_query_type_base_confidence_threshold = current_run_simple_query_confidence_thresh # Use current_run_
                    threshold_name_for_log = "Simple Query Confidence"
                    effective_confidence_threshold_for_query_this_check = current_query_type_base_confidence_threshold
                    if valves.enable_adaptive_thresholds:
                        if doc_size_category == "small":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_small_doc
                        elif doc_size_category == "large":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_large_doc
                        effective_confidence_threshold_for_query_this_check = max(0.05, min(0.95, effective_confidence_threshold_for_query_this_check))
                    final_effective_simple_conf_thresh = effective_confidence_threshold_for_query_this_check # Store for report

                elif query_complexity_level == QueryComplexity.MEDIUM:
                    current_query_type_base_confidence_threshold = current_run_medium_query_confidence_thresh # Use current_run_
                    threshold_name_for_log = "Medium Query Confidence"
                    effective_confidence_threshold_for_query_this_check = current_query_type_base_confidence_threshold
                    if valves.enable_adaptive_thresholds:
                        if doc_size_category == "small":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_small_doc
                        elif doc_size_category == "large":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_large_doc
                        effective_confidence_threshold_for_query_this_check = max(0.05, min(0.95, effective_confidence_threshold_for_query_this_check))
                    final_effective_medium_conf_thresh = effective_confidence_threshold_for_query_this_check # Store for report

                if valves.debug_mode and query_complexity_level != QueryComplexity.COMPLEX:
                     debug_log.append(f"   [Debug] Adaptive Confidence Check: QueryType={query_complexity_level.value}, BaseThreshForProfile={current_query_type_base_confidence_threshold:.2f}, EffectiveThreshForStopCheck={effective_confidence_threshold_for_query_this_check:.2f} (DocSize: {doc_size_category})")

                if query_complexity_level != QueryComplexity.COMPLEX and current_avg_confidence >= effective_confidence_threshold_for_query_this_check:
                    stop_early = True
                    stopping_reason = (
                        f"{query_complexity_level.value} query confidence ({current_avg_confidence:.2f}) "
                        f"met/exceeded effective threshold ({effective_confidence_threshold_for_query_this_check:.2f}) "
                        f"after round {current_round + 1}."
                    )
                # No specific confidence-based early stopping rule for COMPLEX queries; they run max_rounds or until convergence.

            if stop_early:
                if valves.show_conversation:
                    conversation_log.append(f"**⚠️ Early Stopping Triggered:** {stopping_reason}")
                if valves.debug_mode:
                    debug_log.append(f"**⚠️ Early Stopping Triggered:** {stopping_reason} (Debug Mode)")

                early_stopping_reason_for_output = stopping_reason # Store it for final output
                scratchpad_content += f"\n\n**EARLY STOPPING TRIGGERED (Round {current_round + 1}):** {stopping_reason}"
                # Add a final log message before breaking, as the "Completed Round" message will be skipped
                if valves.debug_mode:
                     debug_log.append(f"**🏁 Breaking loop due to early stopping in Round {current_round + 1}. (Debug Mode)**")
                break # Exit the round loop

        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {current_round + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**")

        if current_round == current_run_max_rounds - 1: # Use current_run_max_rounds
            if valves.show_conversation:
                conversation_log.append(f"**🏁 Reached max rounds ({current_run_max_rounds}). Proceeding to final synthesis.**") # Use current_run_max_rounds

    if not claude_provided_final_answer:
        if valves.show_conversation:
            conversation_log.append("\n### 🔄 Final Synthesis Phase")
        
        # Stream synthesis progress
        if streaming_manager and hasattr(streaming_manager, 'stream_synthesis_progress'):
            update = await streaming_manager.stream_synthesis_progress(
                "collecting", total_tasks=len(all_round_results_aggregated)
            )
            if update:
                conversation_log.append(update)
        
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
            
            # Stream synthesis generation progress
            if streaming_manager and hasattr(streaming_manager, 'stream_synthesis_progress'):
                update = await streaming_manager.stream_synthesis_progress(
                    "generating", processed_tasks=len(all_round_results_aggregated), 
                    total_tasks=len(all_round_results_aggregated)
                )
                if update:
                    conversation_log.append(update)
            
            start_time_claude_synth = 0
            if valves.debug_mode:
                start_time_claude_synth = asyncio.get_event_loop().time()
            try:
                final_response = await call_supervisor_model(valves, synthesis_prompt)
                if valves.debug_mode:
                    end_time_claude_synth = asyncio.get_event_loop().time()
                    time_taken_claude_synth = end_time_claude_synth - start_time_claude_synth
                    debug_log.append(f"⏱️ Claude call (Final Synthesis) took {time_taken_claude_synth:.2f}s. (Debug Mode)")
                # Stream synthesis completion
                if streaming_manager and hasattr(streaming_manager, 'stream_synthesis_progress'):
                    update = await streaming_manager.stream_synthesis_progress("complete")
                    if update:
                        conversation_log.append(update)
                
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
    actual_rounds_executed = len(decomposition_prompts_history) # Number of rounds for which decomposition prompts were made
    # If loop broke early, current_round might be less than actual_rounds_executed -1
    # If no decomposition prompts (e.g. direct call, or error before first decomp), then 0.
    # If loop completed, actual_rounds_executed should be current_run_max_rounds (if prompts were made each round)
    # Or, if loop broke, it's the number of rounds that *started* decomposition.
    # A simple way: if all_round_metrics exists, it's len(all_round_metrics)
    if all_round_metrics: # This is a more reliable count of rounds that completed metric generation
        actual_rounds_executed = len(all_round_metrics)
    elif actual_rounds_executed == 0 and current_round >=0: # Fallback if decomp history is empty but loop ran
         actual_rounds_executed = min(current_round + 1, current_run_max_rounds)


    # --- Performance Report Section ---
    performance_report_parts = ["\n## 📝 Performance Report"]
    performance_report_parts.append(f"- **Total rounds executed:** {actual_rounds_executed} / {current_run_max_rounds} (Profile Max)")
    performance_report_parts.append(f"- **Stopping reason:** {early_stopping_reason_for_output if early_stopping_reason_for_output else 'Max rounds reached or no further tasks.'}")

    last_metric = all_round_metrics[-1] if all_round_metrics else None
    if last_metric:
        performance_report_parts.append(f"- **Final Sufficiency Score:** {getattr(last_metric, 'sufficiency_score', 'N/A'):.2f}")
        performance_report_parts.append(f"- **Final Component Coverage:** {getattr(last_metric, 'component_coverage_percentage', 'N/A'):.2%}")
        performance_report_parts.append(f"- **Final Information Components Status:** {str(getattr(last_metric, 'information_components', {}))}")
        performance_report_parts.append(f"- **Final Convergence Detected:** {'Yes' if getattr(last_metric, 'convergence_detected_this_round', False) else 'No'}")
    else:
        performance_report_parts.append("- *No round metrics available for final values.*")

    performance_report_parts.append("- **Effective Thresholds Used (Final Values):**")
    performance_report_parts.append(f"  - Sufficiency (for convergence): {effective_sufficiency_threshold:.2f}")
    performance_report_parts.append(f"  - Novelty (for convergence): {effective_novelty_threshold:.2f}")
    if query_complexity_level == QueryComplexity.SIMPLE:
        performance_report_parts.append(f"  - Confidence (for simple query early stop): {final_effective_simple_conf_thresh:.2f}")
    elif query_complexity_level == QueryComplexity.MEDIUM:
        performance_report_parts.append(f"  - Confidence (for medium query early stop): {final_effective_medium_conf_thresh:.2f}")
    else: # COMPLEX
        performance_report_parts.append(f"  - Confidence (early stopping not applicable for COMPLEX queries based on this threshold type)")
    
    performance_report_parts.append(f"- **Performance Profile Applied:** {valves.performance_profile}")
    performance_report_parts.append(f"- **Adaptive Thresholds Enabled:** {'Yes' if valves.enable_adaptive_thresholds else 'No'}")
    performance_report_parts.append(f"- **Document Size Category:** {doc_size_category}")


    output_parts.extend(performance_report_parts)
    if valves.debug_mode:
        debug_log.append("\n--- Performance Report (Debug Copy) ---")
        debug_log.extend(performance_report_parts)
        debug_log.append("--- End Performance Report (Debug Copy) ---")

    output_parts.append(f"\n## 📊 MinionS Efficiency Stats (v0.2.0)")
    output_parts.append(f"- **Protocol:** MinionS (Multi-Round)")
    output_parts.append(f"- **Query Complexity:** {query_complexity_level.value}")
    output_parts.append(f"- **Rounds executed (Profile Max):** {actual_rounds_executed}/{current_run_max_rounds}") # Use current_run_max_rounds
    output_parts.append(f"- **Total tasks for local LLM:** {stats['total_tasks_executed_local']}")

    # --- Explicitly define variables for the MinionS Efficiency Stats block ---
    # Ensure 'all_round_results_aggregated' is the correct list of task results.
    # And 'valves' and 'debug_log' are assumed to be accessible in this scope for the warning.

    explicit_total_successful_tasks = 0
    explicit_tasks_with_any_timeout = 0 # Renamed for clarity from tasks_with_any_timeout

    if 'all_round_results_aggregated' in locals() and isinstance(all_round_results_aggregated, list):
        explicit_total_successful_tasks = len([
            r for r in all_round_results_aggregated if isinstance(r, dict) and r.get('status') == 'success'
        ])
        explicit_tasks_with_any_timeout = len([
            r for r in all_round_results_aggregated if isinstance(r, dict) and r.get('status') == 'timeout_all_chunks'
        ])
    else:
        # This case should ideally not happen if the protocol ran correctly.
        # Adding a log if debug_mode is on and valves is accessible.
        if 'valves' in locals() and hasattr(valves, 'debug_mode') and valves.debug_mode and 'debug_log' in locals():
            debug_log.append("⚠️ Warning: 'all_round_results_aggregated' not found or not a list when calculating final efficiency stats.")
    # --- End of explicit definitions ---

    output_parts.append(f"- **Successful tasks (local):** {explicit_total_successful_tasks}")
    output_parts.append(f"- **Tasks where all chunks timed out (local):** {explicit_tasks_with_any_timeout}")
    output_parts.append(f"- **Total individual chunk processing timeouts (local):** {total_chunk_processing_timeouts_accumulated}")
    output_parts.append(f"- **Chunks processed per task (local):** {stats['total_chunks_processed_local'] if stats['total_tasks_executed_local'] > 0 else 0}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    if early_stopping_reason_for_output:
        output_parts.append(f"- **Early Stopping Triggered:** {early_stopping_reason_for_output}")
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
):
    """Execute the MinionS protocol with Claude - supports both streaming and traditional modes"""
    
    # Check if streaming is enabled
    streaming_enabled = getattr(pipe_self.valves, 'enable_streaming_responses', True)
    
    if streaming_enabled:
        # Return async generator for streaming
        async for chunk in _execute_minions_protocol_streaming(
            pipe_self, body, __user__, __request__, __files__, __pipe_id__
        ):
            yield chunk
    else:
        # Return string for traditional mode
        result = await _execute_minions_protocol_traditional(
            pipe_self, body, __user__, __request__, __files__, __pipe_id__
        )
        yield result

async def _execute_minions_protocol_traditional(
    pipe_self: Any,
    body: dict,
    __user__: dict,
    __request__: Request,
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "minions-claude",
) -> str:
    """Traditional non-streaming execution"""
    try:
        # Validate configuration
        provider = getattr(pipe_self.valves, 'supervisor_provider', 'anthropic')
        if provider == 'anthropic' and not pipe_self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."
        elif provider == 'openai' and not pipe_self.valves.openai_api_key:
            return "❌ **Error:** Please configure your OpenAI API key in the function settings."

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

        # If no context, make a direct call to supervisor
        if not context:
            direct_response = await _call_supervisor_directly(pipe_self.valves, user_query)
            provider_name = getattr(pipe_self.valves, 'supervisor_provider', 'anthropic').title()
            return (
                f"ℹ️ **Note:** No significant context detected. Using standard {provider_name} response.\n\n"
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

async def _execute_minions_protocol_streaming(
    pipe_self: Any,
    body: dict,
    __user__: dict,
    __request__: Request,
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "minions-claude",
):
    """Streaming execution with real-time updates"""
    
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
            
            yield await streaming_manager.stream_phase_update("completion", "Direct response completed")
            yield f"\n## 🎯 Final Answer\n{final_response}"
            return

        # Execute the MinionS protocol with streaming updates
        yield await streaming_manager.stream_phase_update("task_decomposition", f"Starting MinionS protocol with {len(context)} characters of context")
        
        # Initialize task visualizer if enabled
        task_visualizer = None
        if getattr(pipe_self.valves, 'show_task_visualization', True):
            task_visualizer = TaskVisualizer(pipe_self.valves, pipe_self.valves.debug_mode)
            if task_visualizer.is_visualization_enabled():
                yield await streaming_manager.stream_phase_update("task_visualization", "Initializing task decomposition diagram")
                
                # Add initial tasks to visualizer and show diagram immediately
                task_visualizer.add_task("task_1", "Document analysis", TaskType.DOCUMENT_ANALYSIS, TaskStatus.PENDING)
                task_visualizer.add_task("task_2", "Information extraction", TaskType.DOCUMENT_ANALYSIS, TaskStatus.PENDING) 
                task_visualizer.add_task("task_3", "Results synthesis", TaskType.SYNTHESIS, TaskStatus.PENDING)
                
                initial_diagram = task_visualizer.generate_mermaid_diagram(include_status_colors=True)
                if initial_diagram:
                    yield f"\n## 📊 Initial Task Decomposition\n\n{initial_diagram}\n\n"
        
        # Execute with streaming progress updates
        async for progress_chunk in _execute_minions_protocol_with_streaming_generator(
            pipe_self.valves, 
            user_query, 
            context, 
            call_claude,
            call_ollama,
            TaskResult,
            streaming_manager,
            task_visualizer
        ):
            yield progress_chunk

    except Exception as e:
        import traceback
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        yield await streaming_manager.stream_error_update(f"Error in MinionS protocol: {error_details}", "general")

async def _execute_minions_protocol_with_streaming(
    valves: Any,
    query: str,
    context: str,
    call_claude: Callable,
    call_ollama_func: Callable,
    TaskResultModel: Any,
    streaming_manager: StreamingResponseManager,
    task_visualizer: Any = None
) -> str:
    """Execute MinionS protocol with streaming support - this is a wrapper that adds streaming to the existing protocol"""
    
    # Execute the existing protocol with a timeout wrapper for periodic updates
    import asyncio
    
    # Create a task for the main execution
    main_task = asyncio.create_task(_execute_minions_protocol(
        valves, 
        query, 
        context, 
        call_claude,
        call_ollama_func,
        TaskResultModel
    ))
    
    # Create periodic update indicators
    update_count = 0
    indicators = ["⏳", "🔄", "⚙️", "🧠", "📊"]
    
    try:
        while not main_task.done():
            await asyncio.sleep(3)  # Update every 3 seconds
            if not main_task.done():
                indicator = indicators[update_count % len(indicators)]
                # Note: We can't yield from this function, so we'll track in a way that the result shows progress
                update_count += 1
                
        # Get the result
        result = await main_task
        
    except Exception as e:
        if not main_task.done():
            main_task.cancel()
        raise e
    
    # Add visualization to result if enabled
    if task_visualizer and task_visualizer.is_visualization_enabled():
        # Update task statuses to completed (the tasks were already added in the calling function)
        task_visualizer.update_task_status("task_1", TaskStatus.COMPLETED)
        task_visualizer.update_task_status("task_2", TaskStatus.COMPLETED)
        task_visualizer.update_task_status("task_3", TaskStatus.COMPLETED)
        
        # Generate final diagram
        final_diagram = task_visualizer.generate_mermaid_diagram(include_status_colors=True)
        
        if final_diagram:
            # Prepend the final visualization to the result
            result = f"\n## 📊 Final Task Status\n\n{final_diagram}\n\n{result}"
    
    return result

async def _execute_minions_protocol_with_streaming_generator(
    valves: Any,
    query: str,
    context: str,
    call_claude: Callable,
    call_ollama_func: Callable,
    TaskResultModel: Any,
    streaming_manager: StreamingResponseManager,
    task_visualizer: Any = None
):
    """Execute MinionS protocol with streaming progress updates as an async generator"""
    
    import asyncio
    
    # Show working indicator
    yield "🔄 **Executing MinionS protocol...** ⏳\n\n"
    
    # Create a task for the main execution
    main_task = asyncio.create_task(_execute_minions_protocol(
        valves, 
        query, 
        context, 
        call_claude,
        call_ollama_func,
        TaskResultModel
    ))
    
    # Create periodic update indicators
    update_count = 0
    indicators = ["⏳ Working", "🔄 Processing", "⚙️ Analyzing", "🧠 Thinking", "📊 Computing"]
    
    try:
        while not main_task.done():
            await asyncio.sleep(30)  # Update every 30 seconds
            if not main_task.done():
                update_count += 1
                if update_count == 1:
                    yield "🔄 **Still working...** ⏳\n"
                else:
                    yield f"🔄 **Still working...** (⏱️ {update_count * 30}s)\n"
                
        # Get the result
        result = await main_task
        
        # Show completion
        yield await streaming_manager.stream_phase_update("completion", "MinionS protocol execution completed")
        
        # Add final visualization if enabled
        if task_visualizer and task_visualizer.is_visualization_enabled():
            # Update task statuses to completed
            task_visualizer.update_task_status("task_1", TaskStatus.COMPLETED)
            task_visualizer.update_task_status("task_2", TaskStatus.COMPLETED)
            task_visualizer.update_task_status("task_3", TaskStatus.COMPLETED)
            
            # Generate final diagram
            final_diagram = task_visualizer.generate_mermaid_diagram(include_status_colors=True)
            
            if final_diagram:
                yield f"\n## 📊 Final Task Status\n\n{final_diagram}\n\n"
        
        # Final completion message and result
        yield "✅ **Processing complete!**\n\n"
        yield result
        
    except Exception as e:
        if not main_task.done():
            main_task.cancel()
        yield await streaming_manager.stream_error_update(f"Error during execution: {str(e)}", "execution")
        raise e