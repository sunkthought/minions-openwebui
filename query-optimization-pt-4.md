# Query Optimization & Reformulation - Part 4: Query Decomposition Templates (v0.3.5)

This document summarizes the work completed in Iteration 4 of the Query Optimization and Reformulation system. This iteration focuses on implementing template-based query reformulation to improve the decomposability of common query patterns.

## Core Objective Achieved (Iteration 4)

A `QueryReformulator` system has been added to the MinionS pipeline. This system:
1.  Identifies user queries matching predefined templates (Comparative, Temporal Analysis, Comprehensive Analysis, Causal Analysis).
2.  Extracts key variables from these queries.
3.  Reformulates the original query into a list of more specific, decomposable sub-queries based on the matched template.
4.  Currently, the MinionS protocol uses the *first* sub-query from this list for subsequent processing. The full list is available in debug logs for future enhancements.

This aims to improve task decomposition accuracy, reduce processing rounds, and enhance overall answer quality by breaking down complex queries before they enter the main MinionS protocol.

## Key Components and Their Locations

### 1. Query Reformulator

*   **File:** `partials/query_reformulator.py`
*   **Purpose:** Implements template-based matching and reformulation of user queries.
*   **Key Class:** `QueryReformulator`
    *   **Initialization:** `QueryReformulator(debug_mode: bool = False)`
    *   **Main Method:** `reformulate(original_query: str, query_metadata: Optional[Dict] = None) -> List[str]`
        *   Takes the (potentially expanded) user query and optional metadata from `QueryAnalyzer`.
        *   Returns a list of strings, which are the sub-queries. If no template matches or reformulation fails, it returns a list containing the original query.
*   **Template Library (`_load_templates` method):**
    *   **Comparative:**
        *   Pattern Examples: "Compare X and Y [regarding Z]", "What is the difference between X and Y [regarding Z]?"
        *   Reformulation: ["What is X?", "What is Y?", "What are the differences between X and Y [regarding Z]?"]
    *   **Temporal Analysis:**
        *   Pattern Examples: "How has X changed [from period1 to period2]?", "Trend of X [from period1 to period2]?"
        *   Reformulation: ["What was X in {period1}?", "What is X in {period2}?", "What is the trend for X between {period1} and {period2}?"] (Sub-queries involving periods are skipped if periods are not specified)
    *   **Comprehensive Analysis:**
        *   Pattern Examples: "Analyze X", "Tell me everything about X"
        *   Reformulation: ["What is X?", "What are the key metrics for X?", "What are the recent trends for X?", "What are the implications or future outlook for X?"]
    *   **Causal Analysis:**
        *   Pattern Examples: "Why did X happen?", "What caused X?"
        *   Reformulation: ["What is X?", "What factors influenced X?", "What was the primary cause of X?"]
*   **Template Matching (`_match_template`):**
    *   Uses regex to match queries against templates.
    *   Includes a basic confidence score (currently fixed at 0.9 for a match, logged in debug mode).
*   **Variable Extraction (`_extract_variables`):**
    *   Extracts variables (e.g., X, Y, period1) from the query based on named capture groups in the regex.
*   **Sub-query Generation (`_generate_sub_queries`):**
    *   Constructs sub-queries using f-strings and extracted variables.
    *   Skips sub-queries if their optional variables are not available.

### 2. Integration into MinionS Protocol

*   **File:** `partials/minions_pipe_method.py` (within `_execute_minions_protocol`)
*   **Order of Operations:**
    1.  `QueryAnalyzer` runs.
    2.  `QueryExpander` runs on the original query.
    3.  `QueryReformulator` runs on the (potentially expanded) query and `query_metadata`.
        *   If `debug_mode` is true, logs show the matched template, confidence, extracted variables, and all generated sub-queries.
    4.  The `user_query` variable (used for downstream processing) is updated with the *first* sub-query from the `reformulated_queries` list.
        *   If multiple sub-queries are generated, a debug message (and conversation log message if `show_conversation` is on) indicates that only the first is being used in this iteration.
    5.  The (potentially reformulated) `user_query` proceeds to Query Complexity Classification and task decomposition.

### 3. Configuration & Valves

*   **Valve:**
    *   File: `partials/minions_valves.py`
    *   `enable_query_reformulation: bool = Field(default=True, ...)`: Controls whether the query reformulation step is active. Defaults to `True`.
*   **Build Configuration:**
    *   File: `generation_config.json`
    *   `query_reformulator.py` has been added to the `partials_concat_order` in the `minions_default` profile, placed after `query_expander.py` and before `minions_prompts.py`.

## How to Use and Test

1.  **Enable the Feature:** The feature is enabled by default (`enable_query_reformulation` valve is `True`).
2.  **Activate Debug Mode:** To see the reformulation in action, set the `debug_mode` valve to `True` in the Open WebUI function settings for the MinionS pipe.
3.  **Test Queries:** Try queries that match the defined patterns:
    *   "Compare product A and product B"
    *   "What is the difference between the new proposal and the old one regarding cost?"
    *   "How have our sales figures changed from Q1 to Q2?"
    *   "Analyze the current market trends for AI."
    *   "Why did the system outage occur last night?"
4.  **Observe Logs:**
    *   In `debug_mode`, the MinionS logs will show:
        *   Messages from `QueryReformulator` detailing the matched template (if any), confidence, extracted variables, and all generated sub-queries.
        *   A message in `minions_pipe_method.py` indicating if the query was reformulated and if multiple sub-queries were generated (noting that only the first is used).
    *   If `show_conversation` is enabled, some high-level messages about reformulation might also appear in the conversation log.

## Next Steps (Beyond Iteration 4)

This iteration lays the groundwork for more advanced query decomposition. Future enhancements could include:
*   **Processing Multiple Sub-Queries:** Modifying the MinionS protocol to intelligently handle and process *all* generated sub-queries, potentially in parallel or sequentially, and then synthesize their results.
*   **Refined Confidence Scoring:** Implementing more dynamic confidence scoring for template matches.
*   **User Feedback for Clarification:** If a query matches a template with low confidence or is still ambiguous, the system could ask the user for clarification based on the template.
*   **Expanding the Template Library:** Adding more templates for other common query patterns.
*   **Dynamic Template Generation/Selection:** Exploring methods to learn or select templates based on context or domain.

This summary provides an overview of the v0.3.5 enhancements.
```
