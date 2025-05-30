## New Partials Documentation

This document outlines the new partial files created during the recent refactoring effort and provides brief notes on their purpose and relevant OpenWebUI file handling.

### New Partial Files and Their Roles:

1.  **`partials/common_file_processing.py`**:
    *   **Purpose**: Contains common utilities related to file content processing, specifically for chunking large contexts.
    *   **Key Functions**:
        *   `create_chunks(context: str, chunk_size: int, max_chunks: int) -> List[str]`: Splits a large text context into smaller, manageable chunks. (Used by MinionS)

2.  **`partials/minions_decomposition_logic.py`**:
    *   **Purpose**: Manages the task decomposition step specific to the MinionS protocol. It handles the interaction with the remote model (Claude) to break down a user query into sub-tasks.
    *   **Key Functions**:
        *   `decompose_task(valves, call_claude_func, query, context_len, scratchpad_content, num_chunks, max_tasks_per_round, current_round, conversation_log, debug_log)`: Orchestrates the decomposition prompt generation, call to Claude, and parsing of the response into tasks.

3.  **`partials/minion_prompts.py`**:
    *   **Purpose**: Centralizes the generation of prompts used by the Minion protocol for interacting with both the remote (Claude) and local (Ollama) models.
    *   **Key Functions**:
        *   `get_minion_initial_claude_prompt(query, context_len, valves)`: Generates the first prompt sent to Claude.
        *   `get_minion_conversation_claude_prompt(history, original_query, valves)`: Generates subsequent prompts for Claude during the conversation.
        *   `get_minion_local_prompt(context, query, claude_request, valves)`: Generates the prompt for the local Ollama model.

4.  **`partials/minions_prompts.py`**:
    *   **Purpose**: Centralizes the generation of prompts used by the MinionS protocol.
    *   **Key Functions**:
        *   `get_minions_synthesis_claude_prompt(query, results_summary, valves)`: Generates the prompt for Claude to synthesize an answer from task results.
        *   `get_minions_local_task_prompt(chunk, task, chunk_idx, total_chunks, valves)`: Generates the prompt for the local Ollama model to execute a specific task on a text chunk.

### OpenWebUI File Handling Notes:

*   **File Input**: The OpenWebUI functions (Minion and MinionS) receive file information via the `__files__: List[dict]` parameter in their main `pipe` method.
*   **Context Extraction**:
    *   The `partials/common_context_utils.py` file contains functions to extract text content from these files:
        *   `extract_context_from_files(__files__)`
        *   `extract_file_content(file_info)`
    *   **Current Implementation of `extract_file_content`**: As of this refactoring, the `extract_file_content` function primarily checks if the file content is directly available in the `file_info` dictionary (e.g., for short text snippets passed this way) or provides debug information about the file.
    *   **Assumption/Further Work**: For handling larger, actual uploaded files in OpenWebUI, the `extract_file_content` function might need to be enhanced to use specific OpenWebUI APIs or mechanisms to fetch the full content of files based on their `id` or path, if such mechanisms are provided by OpenWebUI for functions/pipelines. The current version is a basic placeholder for this interaction and might require further development for robust RAG capabilities with various file types. The URLs explored for OpenWebUI documentation on this (e.g., `/integration/pipelines/`, `/getting-started/functions/`) did not yield immediate, specific details on a file content retrieval API for use within custom functions at the time of this refactoring.

This refactoring aims to make the codebase more modular, maintainable, and easier to extend for future enhancements.
