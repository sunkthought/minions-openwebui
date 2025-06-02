# Query Optimization & Reformulation - Parts 1-3 Summary

This document summarizes the work completed in Iterations 1, 2, and 3 for the Query Optimization and Reformulation system designed for the Minion/MinionS protocols.

## Core Objective Achieved (Iterations 1-3)

A query analysis and reformulation pipeline has been established that preprocesses user queries before they enter the main MinionS protocol. This pipeline:
1.  Analyzes the query to extract metadata and understand its structure (Iteration 1).
2.  Detects and scores various types of ambiguities within the query (Iteration 2).
3.  Applies simple expansion techniques to add context and specificity (Iteration 3).

These enhancements aim to improve task decomposition, reduce processing rounds, and enhance overall answer quality.

## Key Components and Their Locations

### 1. Query Analyzer

*   **File:** `partials/query_analyzer.py`
*   **Purpose:** Extracts detailed metadata from the user's query and detects ambiguities.
*   **Key Class:** `QueryAnalyzer`
    *   **Initialization:** `QueryAnalyzer(query: str, debug_mode: bool = False)`
    *   **Main Method:** `analyze() -> QueryMetadata`
*   **Key Data Structure:** `QueryMetadata` (TypedDict)
    *   `original_query: str`
    *   `query_type: QueryType` (Enum: QUESTION, COMMAND, ANALYSIS_REQUEST, COMPARISON, UNKNOWN)
    *   `entities: List[Entity]` (TypedDict: text, label, start_char, end_char)
    *   `temporal_refs: List[TemporalReference]` (TypedDict: text, type, start_char, end_char)
    *   `action_verbs: List[str]`
    *   `scope: ScopeIndicator` (Enum: SPECIFIC, BROAD, COMPREHENSIVE, UNKNOWN)
    *   `ambiguity_markers: List[str]` (General list of potentially ambiguous words)
    *   `detected_patterns: List[QueryPattern]` (TypedDict: type, text; e.g., MULTI_PART, NESTED_QUERY)
    *   `ambiguity_score: float` (0-1, calculated based on detected ambiguities)
    *   `decomposability_score: float` (Initial estimation based on patterns and ambiguity)
    *   `detailed_ambiguity_report: List[AmbiguityDetail]` (TypedDict: type, text, suggestion)
*   **Key Methods in `QueryAnalyzer` (Internal):**
    *   **Iteration 1 (Metadata Extraction):**
        *   `extract_query_type()`
        *   `extract_entities()` (uses spaCy if available, else regex)
        *   `extract_temporal_references()`
        *   `extract_action_verbs()` (uses spaCy for lemmatization if available)
        *   `extract_scope_indicator()`
        *   `extract_ambiguity_markers()` (general word list)
        *   `detect_patterns()`
    *   **Iteration 2 (Ambiguity Detection & Scoring):**
        *   `_detect_pronoun_ambiguity()`
        *   `_detect_temporal_ambiguity()`
        *   `_detect_scope_ambiguity()`
        *   `_detect_comparative_ambiguity()`
        *   `_detect_entity_ambiguity()`
        *   `_calculate_ambiguity_score()`

### 2. Query Expander

*   **File:** `partials/query_expander.py`
*   **Purpose:** Applies simple query expansion techniques to add context and specificity.
*   **Key Class:** `QueryExpander`
    *   **Initialization:** `QueryExpander(debug_mode: bool = False)`
    *   **Main Method:** `expand(query: str, apply_context_injection: bool, apply_completion: bool, apply_synonyms: bool, domain_hint: Optional[str]) -> str`
*   **Key Methods in `QueryExpander` (Internal):**
    *   **Iteration 3 (Query Expansion Techniques):**
        *   `add_document_type_context()`: Adds domain-specific prepending/appending phrases.
            *   Example data: `self.default_doc_type_context_examples` (inline) for domains: default, research, finance, writing, coding.
        *   `complete_query()`: Completes common incomplete query patterns using regex.
            *   Example data: `self.default_completion_templates` (inline).
        *   `expand_synonyms()`: Augments keywords with synonyms from domain-aware sets (uses spaCy for lemmatization if available).
            *   Example data: `self.default_synonym_sets` (inline) for domains: default, research, finance, writing, coding.
        *   `_infer_domain()`: Basic keyword-based domain inference.

### 3. Integration into MinionS Protocol

*   **File:** `partials/minions_pipe_method.py` (primarily within the `_execute_minions_protocol` function)
*   **Order of Operations:**
    1.  The original user query is received (`user_query`).
    2.  `QueryAnalyzer` is instantiated with the original query. Its `analyze()` method is called to produce `query_metadata`.
        *   This metadata (including ambiguity score and report) is logged if `debug_mode` is true.
        *   A warning is logged if `ambiguity_score` exceeds `high_ambiguity_threshold`.
    3.  The original query is stored in `user_query_original`.
    4.  `QueryExpander` is instantiated. Its `expand()` method is called (conditionally based on valves) on `user_query_original` to produce `processed_query`.
        *   The original and expanded queries are logged if `debug_mode` is true.
    5.  The `processed_query` (which may or may not be modified from the original) is then used by:
        *   `QueryComplexityClassifier`
        *   `decompose_task` function
        *   `get_minions_synthesis_claude_prompt` for the main synthesis prompt.
    6.  `InformationSufficiencyAnalyzer` is initialized with `user_query_original` to analyze components based on the user's actual input.
    7.  The fallback message for empty `synthesis_input_summary` uses `user_query_original`.
    8.  Token savings calculations in `calculate_token_savings` use the length of `user_query_original`.

*   **Key Variables in `_execute_minions_protocol`:**
    *   `user_query_original: str`: Stores the initial query from the user.
    *   `user_query: str`: This variable is updated with the (potentially) expanded query (`processed_query`) and used by most downstream parts of the protocol.
    *   `query_metadata: QueryMetadata`: Output of the `QueryAnalyzer`.
    *   `processed_query: str`: The query after it has gone through the `QueryExpander`. This becomes the new `user_query`.

*   **Key Valves (configurable via Function Settings in OpenWebUI, defined in `minions_valves.py`):**
    *   `debug_mode: bool`: Enables detailed logging from `QueryAnalyzer` and `QueryExpander`.
    *   `high_ambiguity_threshold: float` (Default: 0.7): Threshold above which a warning is logged for high query ambiguity. (Used by `minions_pipe_method.py` when checking `query_metadata['ambiguity_score']`).
    *   `enable_query_expansion: bool` (Default: True in `minions_pipe_method.py` if valve not present): Master toggle for the entire query expansion step.
    *   `qe_apply_context_injection: bool` (Default: True): Enables/disables the context injection step.
    *   `qe_apply_completion: bool` (Default: True): Enables/disables the query completion step.
    *   `qe_apply_synonyms: bool` (Default: True): Enables/disables the synonym expansion step.
    *   `domain_hint: Optional[str]` (Default: None): Allows users to provide a domain hint (e.g., "finance", "coding") to guide query expansion.

### 4. Build Configuration

*   **File:** `generation_config.json`
*   **Changes:** The `minions_default` profile (and any other relevant profiles) in `partials_concat_order` has been updated to include:
    *   `query_analyzer.py`
    *   `query_expander.py`
    *   These are placed before `minions_pipe_method.py` to ensure they are available during the generation of the final function file.

## Next Steps (Iterations 4-8)

Future iterations will build upon this foundation to implement more advanced query reformulation techniques, potentially including:
*   Clarification question generation for highly ambiguous queries.
*   More sophisticated context utilization from conversation history or documents.
*   Advanced query decomposition strategies based on the refined understanding of the query.
*   User feedback mechanisms for query reformulation.

This summary should provide a good overview for continuing development.
