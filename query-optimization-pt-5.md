# Query Optimization & Reformulation - Part 5: Entity and Reference Resolution Framework (v0.3.5)

This document summarizes the work completed in Iteration 5 of the Query Optimization and Reformulation system. This iteration focused on establishing the framework for entity and reference resolution within the MinionS protocol.

## Core Objective Achieved (Iteration 5)

A structural framework has been integrated into the MinionS pipeline to enable future implementation of sophisticated entity and reference resolution. This includes:
1.  **New Partials:** `entity_resolver.py` and `reference_resolver.py` have been created with placeholder classes (`EntityResolver`, `ReferenceResolver`) and methods.
2.  **`QueryAnalyzer` Integration:** The `QueryAnalyzer` (`partials/query_analyzer.py`) has been updated to:
    *   Initialize and call methods from `EntityResolver` and `ReferenceResolver`.
    *   Introduce new fields in `QueryMetadata` (`resolved_query`, `initial_resolved_entities`, `resolved_references`, and renamed `entities` to `extracted_entities`).
3.  **Pipeline Integration:** The main pipeline in `partials/minions_pipe_method.py` now:
    *   Conditionally calls the enhanced `QueryAnalyzer.analyze()` method with document and conversation context (if the new valve is enabled).
    *   Logs the (currently placeholder) outputs of the resolution process in debug mode.
    *   Continues to use the query from `QueryExpander` for downstream tasks, ensuring no change in existing behavior.
4.  **Configuration:**
    *   A new valve `enable_entity_reference_resolution` (defaulting to `True`) in `partials/minions_valves.py` controls this new functionality.
    *   `generation_config.json` has been updated to include the new partials in the correct order.

While the core resolution logic within the new partials is placeholder, this iteration successfully sets up all the necessary connections and data flow for future implementation.

## Key Components and Their Locations

### 1. Entity Resolver (New)
*   **File:** `partials/entity_resolver.py`
*   **Purpose:** (Future) To extract and resolve entities using document metadata, conversation context, and co-reference resolution.
*   **Key Class:** `EntityResolver`
    *   `Entity` TypedDict: Defines the structure for entity data.
    *   Placeholder methods: `extract_entities_from_metadata`, `build_entity_registry`, `resolve_coreferences`, `resolve_acronyms_aliases`, `resolve_entities_in_query`.

### 2. Reference Resolver (New)
*   **File:** `partials/reference_resolver.py`
*   **Purpose:** (Future) To resolve pronouns and indirect references within the query.
*   **Key Class:** `ReferenceResolver`
    *   Placeholder methods: `resolve_pronouns`, `resolve_indirect_references`, `resolve_references_in_query`.

### 3. Query Analyzer (Updated)
*   **File:** `partials/query_analyzer.py`
*   **Changes:**
    *   Imports and initializes `EntityResolver` and `ReferenceResolver`.
    *   `QueryMetadata` TypedDict: Added `resolved_query: str`, `initial_resolved_entities: List[Entity]`, `resolved_references: List[ResolvedEntity]`. Renamed `entities` to `extracted_entities`.
    *   `ResolvedEntity` TypedDict: New structure for holding details of a resolved reference.
    *   `analyze()` method: Orchestrates calls to the new resolvers. Passes `document_metadata` and `conversation_history`.

### 4. MinionS Pipe Method (Updated)
*   **File:** `partials/minions_pipe_method.py`
*   **Changes:**
    *   Uses the `enable_entity_reference_resolution` valve to control the new functionality.
    *   Passes context (currently `conversation_log` for history, `None` for document metadata due to type mismatch) to `QueryAnalyzer.analyze()`.
    *   Logs new fields from `query_metadata` in debug mode.
    *   The query used for downstream processing (`QueryComplexityClassifier`, `decompose_task`) remains unchanged (`processed_query` from `QueryExpander`).

### 5. Valves (Updated)
*   **File:** `partials/minions_valves.py`
*   **New Valve:** `enable_entity_reference_resolution: bool` (default: `True`).

### 6. Generation Config (Updated)
*   **File:** `generation_config.json`
*   **Changes:** `entity_resolver.py` and `reference_resolver.py` added before `query_analyzer.py` in `partials_concat_order`.

## How to Test This Framework Iteration

1.  **Generate the Function:** Run `python generator_script.py` to build the MinionS function with the new partials.
2.  **Configure in Open WebUI:**
    *   Ensure the MinionS function is loaded/updated in Open WebUI.
    *   In the function settings:
        *   Set `debug_mode` to `True`.
        *   Ensure `enable_entity_reference_resolution` is `True` (it defaults to True).
3.  **Send a Query:** Use any query. The content of the query is not critical for this framework test, as the resolution logic is not yet implemented.
4.  **Observe Logs (MinionS Output/Docker Logs):**
    *   Look for the new debug log messages:
        *   `DEBUG: Resolved Query: ...` (should show the original query for now)
        *   `DEBUG: Initial Resolved Entities: []` (should be an empty list for now)
        *   `DEBUG: Resolved References: []` (should be an empty list for now)
    *   Confirm that the rest of the MinionS protocol (expansion, reformulation, decomposition if applicable) proceeds as it did before this iteration.
    *   Confirm no errors related to the new components.

## Next Steps (Implementing Resolution Logic for v0.3.5)

This framework sets the stage. The immediate next steps involve populating the placeholder methods with actual logic:

1.  **Implement `EntityResolver.extract_entities_from_metadata`:**
    *   Define how document metadata is structured (e.g., expected keys for entities).
    *   Process `document_metadata: List[Dict]` to extract `Entity` objects.
2.  **Implement `EntityResolver.build_entity_registry`:**
    *   Develop logic to create a registry (e.g., `Dict[str, Entity]`) from entities found in `document_entities` and potentially from named entities in `conversation_history`.
    *   Consider how to handle conflicting entities or merge information.
3.  **Implement `EntityResolver.resolve_acronyms_aliases`:**
    *   Requires a predefined or dynamically built dictionary of acronyms/aliases and their expansions.
    *   Use the `entity_registry` to look up known entities.
4.  **Implement `EntityResolver.resolve_coreferences`:**
    *   This is a complex NLP task. Start with simple rule-based approaches (e.g., pronoun resolution based on proximity to named entities in recent conversation history or query).
    *   Consider integrating a lightweight coreference resolution library if feasible within constraints.
    *   Update the `query` string by replacing coreferent expressions with their resolved antecedents.
    *   Populate the `resolved_references` list in `QueryMetadata` with `ResolvedEntity` objects detailing what was changed.
5.  **Implement `ReferenceResolver.resolve_pronouns`:**
    *   Similar to coreference, but specifically for pronouns ("it", "they", "he", "she", etc.).
    *   Use `conversation_history`, `entity_registry`, and entities from the current query.
    *   Update the `query` string and populate `resolved_references`.
6.  **Implement `ReferenceResolver.resolve_indirect_references`:**
    *   Handle phrases like "the company," "the CEO" by looking them up in the `entity_registry` or document context.
    *   For temporal references like "last year," use document publication dates or current date.
    *   Update the `query` string and populate `resolved_references`.
7.  **Refine `QueryAnalyzer.analyze`:**
    *   Ensure `ResolvedEntity` objects are correctly created and added to `query_metadata['resolved_references']` after calls to the resolvers.
    *   The `resolved_query_final` should reflect all changes made by the resolvers.
8.  **Decision Point:** Determine when and how to use `query_metadata['resolved_query']` in the downstream MinionS protocol (i.e., should it replace `processed_query` after this stage?). This will likely be after thorough testing of the resolution logic.
9.  **Confidence Scoring:** Implement actual confidence scoring for each resolution.

This iterative approach allows for gradual implementation and testing of each resolution component.
