```markdown
# Query Optimization & Reformulation - Part 6: Intelligent Query Reformulation (v0.3.5)

This document summarizes the work completed in Iteration 6 of the Query Optimization and Reformulation system. This iteration introduces an LLM-based (Claude) intelligent query reformulation step to further refine queries before they enter the main MinionS protocol.

## Core Objective Achieved (Iteration 6)

An `IntelligentQueryReformulator` has been added to the MinionS pipeline. This system:
1.  Operates after the initial query expansion and template-based reformulation steps.
2.  Uses an LLM (Claude) to intelligently reformulate the current query based on its content, `QueryMetadata` (like ambiguity and decomposability scores), and conversation history.
3.  Applies several reformulation strategies via prompting, including:
    *   **Specificity Enhancement**: Making vague queries more precise.
    *   **Scope Optimization**: Adjusting query scope (broadening or narrowing) based on query type and implied goals.
    *   **Task Alignment**: Restructuring queries to better suit MinionS task decomposition patterns (e.g., more actionable steps).
    *   **Hierarchical Breakdown**: Converting complex multi-part queries into a primary question, potentially with implicitly suggested follow-ups.
4.  Leverages few-shot prompting by providing examples of good reformulations to the LLM.
5.  Is controlled by new valves for enabling/disabling the feature, selecting the LLM model, and setting thresholds for triggering based on ambiguity and decomposability scores.

This aims to improve task decomposition accuracy, reduce processing rounds, and enhance overall answer quality by leveraging advanced LLM capabilities for query preprocessing.

## Key Components and Their Locations

### 1. Intelligent Query Reformulator (New)

*   **File:** `partials/intelligent_query_reformulator.py`
*   **Purpose:** Implements LLM-based query reformulation using strategies tailored to query characteristics.
*   **Key Class:** `IntelligentQueryReformulator`
    *   **Initialization:** `__init__(self, debug_mode: bool = False, valves: Optional[Any] = None, call_claude_func: Optional[Callable] = None)`
        *   Takes `debug_mode`, the MinionS `valves` object, and the `call_claude` function.
    *   **Main Method:** `async reformulate(self, query: str, query_metadata: QueryMetadata, conversation_history: Optional[List[str]] = None) -> str`
        *   Receives the current query (potentially modified by prior steps), `QueryMetadata` from `QueryAnalyzer`, and recent `conversation_history`.
        *   **Strategy Selection (Heuristic-based):**
            *   Checks `query_metadata['ambiguity_score']` against `valves.min_ambiguity_for_intelligent_reformulation`. If above, prioritizes "Specificity Enhancement".
            *   Checks `query_metadata['decomposability_score']` against `valves.min_decomposability_score_for_intelligent_reformulation`. If below, prioritizes "Hierarchical Breakdown & Task Alignment".
            *   Considers `query_metadata['query_type']` for "Scope Optimization" for certain types like "ANALYSIS_REQUEST" or "COMPARISON" if other conditions aren't met.
            *   If no conditions trigger reformulation, the original query is returned.
        *   **Prompt Construction:**
            *   Builds a detailed prompt for Claude, including:
                *   The role of an expert query reformulator.
                *   The original query and its full metadata.
                *   Recent conversation history (last 3 turns).
                *   Specific instructions based on selected strategies.
                *   Few-shot examples demonstrating each key reformulation strategy (Specificity Enhancement, Hierarchical Breakdown, Scope Optimization, Task Alignment).
            *   Instructs Claude to return only the reformulated query or the original query if no changes are needed.
        *   **LLM Call:**
            *   Uses the provided `call_claude_func` and `valves`.
            *   The LLM model is determined by `valves.intelligent_reformulation_model` (fallback to `valves.remote_model`).
        *   **Response Handling:**
            *   Cleans up the LLM response (strips whitespace, removes common preambles like "Reformulated Query:").
            *   Returns the original query if the LLM response is empty, unchanged, or indicates no changes are necessary.
*   **Debug Logging:** Includes detailed logs if `debug_mode` is enabled.

### 2. MinionS Pipe Method (Updated)

*   **File:** `partials/minions_pipe_method.py` (within `_execute_minions_protocol`)
*   **Integration:**
    *   Imports `IntelligentQueryReformulator`.
    *   A new section for intelligent query reformulation is added *after* the template-based `QueryReformulator` step.
    *   This step is conditional on the `enable_intelligent_query_reformulation` valve.
    *   `IntelligentQueryReformulator` is instantiated with `valves.debug_mode`, the `valves` object, and the `call_claude` function.
    *   Its `reformulate` method is called with the current `user_query`, `query_metadata`, and `conversation_log`.
    *   The `user_query` variable is updated with the result of this step.
    *   Appropriate debug and conversation logs are added to show the outcome of this step.

### 3. Valves (Updated)

*   **File:** `partials/minions_valves.py`
*   **New Valves:**
    *   `enable_intelligent_query_reformulation: bool = Field(default=True, ...)`: Master toggle for this feature.
    *   `intelligent_reformulation_model: str = Field(default="claude-3-5-haiku-20241022", ...)`: Specifies the Claude model for this step.
    *   `min_ambiguity_for_intelligent_reformulation: float = Field(default=0.5, ...)`: Threshold for `query_metadata['ambiguity_score']` to trigger this feature.
    *   `min_decomposability_score_for_intelligent_reformulation: float = Field(default=0.5, ...)`: Reformulation is triggered if `query_metadata['decomposability_score']` is *below* this threshold.

### 4. Generation Config (Updated)

*   **File:** `generation_config.json`
*   **Changes:** `intelligent_query_reformulator.py` has been added to `partials_concat_order` in the `minions_default` profile. It's placed after `query_reformulator.py` and before `minions_pipe_method.py` to ensure class availability.

## How to Use and Test

1.  **Generate the Function:** Ensure `generator_script.py` is run to include the new partial and changes.
2.  **Enable the Feature:**
    *   In Open WebUI function settings for the MinionS pipe, ensure `enable_intelligent_query_reformulation` is `True` (it defaults to `True`).
    *   Adjust `intelligent_reformulation_model` if needed (e.g., to `claude-3-5-sonnet-20241022` for potentially higher quality reformulations at a higher cost).
    *   Configure `min_ambiguity_for_intelligent_reformulation` and `min_decomposability_score_for_intelligent_reformulation` to control when the feature activates. For example:
        *   Setting `min_ambiguity_for_intelligent_reformulation` to `0.0` and `min_decomposability_score_for_intelligent_reformulation` to `1.0` would make it attempt reformulation on almost all queries.
        *   Setting `min_ambiguity_for_intelligent_reformulation` to `1.0` and `min_decomposability_score_for_intelligent_reformulation` to `0.0` would make it trigger less often.
3.  **Activate Debug Mode:** Set `debug_mode` to `True` in function settings to see detailed logs from `IntelligentQueryReformulator`, including selected strategies and the prompt sent to Claude.
4.  **Test Queries:**
    *   **Vague/Ambiguous Queries:**
        *   Original: "Tell me about recent financial developments."
        *   Expected (if ambiguity > threshold): "What are the key recent developments in global financial markets, focusing on stock market trends, interest rate changes, and cryptocurrency news from the last quarter?"
    *   **Complex/Multi-Part Queries:**
        *   Original: "Explain quantum computing and its main applications and also tell me about the leading companies in the field."
        *   Expected (if decomposability < threshold): "What are the fundamental principles of quantum computing and its primary applications in fields like medicine, materials science, and cryptography? (Follow up: Who are the leading companies currently developing quantum computing technology?)"
    *   **Broad Scope Queries:**
        *   Original: "Information on space exploration."
        *   Expected (if query type triggers scope optimization): "What are the latest missions and key discoveries in space exploration, particularly concerning Mars and lunar exploration efforts by NASA and SpaceX?"
    *   **Queries that are already good:**
        *   Original: "What was the closing price of Apple stock yesterday?"
        *   Expected: (No change, or minimal refinement if any) "What was the closing price of Apple (AAPL) stock on [Yesterday's Date]?"
5.  **Observe Logs and Behavior:**
    *   If `debug_mode` is on, MinionS logs will show:
        *   Messages from `IntelligentQueryReformulator` indicating if it's attempting reformulation, the strategies chosen, and the final prompt.
        *   The reformulated query returned by Claude.
    *   If `show_conversation` is on and the query is changed, a message like "ℹ️ Query has been intelligently refined to: ..." will appear in the main output.
    *   Observe if the downstream behavior (task decomposition, final answer quality) improves for queries that are intelligently reformulated.

## Next Steps (Beyond Iteration 6)

This iteration provides a significant enhancement to query understanding. Future work could involve:
*   **Dynamic Few-Shot Example Library:** Building a mechanism to store and retrieve successful reformulations to dynamically construct few-shot examples, rather than hardcoding them in the prompt.
*   **User Feedback Loop:** Allowing users to rate or correct reformulations, feeding this data back to improve the system (e.g., by fine-tuning a model or updating the example library).
*   **More Sophisticated Strategy Selection:** Using more advanced logic or even a small ML model to select the optimal reformulation strategy or combination of strategies.
*   **Direct Handling of Multiple Sub-Queries:** If the hierarchical breakdown suggests multiple distinct sub-queries, explore modifying the MinionS protocol to process them systematically.
*   **Confidence Scoring for Reformulations:** Developing a way to score the quality or confidence of the LLM's reformulation.
```
