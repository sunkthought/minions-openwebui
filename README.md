## MinionS Open WebUI Function - v0.2.0 (Multi-Round)

### Overview
This code implements an enhanced version of the MinionS protocol from HazyResearch's paper, now as **MinionS v0.2.0 (Multi-Round)**, an Open WebUI function. It marries the academic research of cost-efficient collaboration between local and cloud Language Models with Open WebUI's practical interface for users, now with more robust and iterative document analysis capabilities.

### Protocol Implementation
This function supports two primary modes of operation:

*   **Minion Protocol (Legacy):** A simple conversational approach where Claude (cloud) and a local Ollama model chat back-and-forth. This mode is less common now with the advanced MinionS.
*   **MinionS Protocol (v0.2.0 - Multi-Round):** An advanced, iterative task decomposition approach where Claude breaks down complex queries into subtasks over multiple rounds, leveraging a "scratchpad" to build comprehensive answers.

## How It Works

### Context Handling:
*   Extracts context from conversation history (messages longer than 200 chars are considered potential context).
*   Processes uploaded files through the `__files__` parameter, extracting their text content.
*   Combines both sources to create a comprehensive context for the local model.

### Minion Protocol Flow:
*(Legacy - Less emphasized in v0.2.0)*
*   Claude receives the query but NOT the full context.
*   Claude asks specific questions to the local model.
*   Local model (with full context access) answers Claude's questions.
*   Process continues for multiple rounds until Claude has enough info.
*   Claude provides the final answer.

### MinionS Protocol Flow (v0.2.0 - Multi-Round):
The MinionS Protocol (v0.2.0) has been significantly enhanced for more thorough and iterative analysis:
- **Multi-Round Task Decomposition:** Claude, acting as a supervisor, can now break down complex queries into subtasks over multiple rounds (configurable by `max_rounds`).
- **Iterative Refinement with Scratchpad:** Findings from each round (results from local model processing) are stored in a "scratchpad." Claude consults this scratchpad in subsequent rounds to ask more informed follow-up questions or determine if the main query can be answered.
- **Comprehensive Chunk Processing:** For each sub-task defined by Claude in a given round, the local Ollama model processes *all* available document chunks (derived from the full context). This ensures each task benefits from a complete view of the document.
- **Task-Level Aggregation:** Results from all chunks for a single task are aggregated before being added to the scratchpad and presented to Claude.
- **Intelligent Loop Termination:** The iterative process continues until:
    1.  The configured `max_rounds` is reached.
    2.  Claude indicates the answer is complete by responding with "FINAL ANSWER READY."
    3.  Claude generates no new constructive tasks in a round.
- **Final Synthesis:** After the loop terminates, Claude synthesizes the final comprehensive answer based on the accumulated information from all rounds and tasks stored in the scratchpad. If Claude provided "FINAL ANSWER READY.", that answer is used directly.

### Cost Efficiency:
*   The MinionS protocol reduces cloud model costs by primarily "reading" and processing the full data locally. Only task definitions, aggregated results, and final synthesis prompts are exchanged with the remote (Claude) model.
*   The function calculates and displays estimated token savings compared to sending the entire context directly to the cloud model.

### Open WebUI Integration Features:

*   **Valves (Configuration Settings):**
    *   `anthropic_api_key: str`: Your Anthropic API key for Claude.
    *   `remote_model: str` (Default: "claude-3-5-haiku-20241022"): The Claude model for supervision and synthesis.
    *   `ollama_base_url: str` (Default: "http://localhost:11434"): URL of your Ollama server.
    *   `local_model: str` (Default: "llama3.2"): The local Ollama model for chunk processing.
    *   `max_rounds: int` (Default: 2): Maximum number of task decomposition rounds.
    *   `max_tasks_per_round: int` (Default: 3): Maximum tasks Claude can create per round.
    *   `chunk_size: int` (Default: 5000): Target character size for document chunks.
    *   `show_conversation: bool` (Default: True): Display the full decomposition and execution details.
    *   `timeout_local: int` (Default: 30): **(Default Changed from 45s)** Timeout for local Ollama model calls *per chunk* in seconds. Adjust based on your local model's speed and typical document complexity.
    *   `max_round_timeout_failure_threshold_percent: int` (Default: 50): If this percentage of local model calls (per chunk) in a single round time out, a warning is issued in the conversation log, suggesting that results from that round may be incomplete or unreliable.
    *   `max_tokens_claude: int` (Default: 2000): Maximum tokens for Claude's API responses (decomposition, synthesis).
    *   `timeout_claude: int` (Default: 60): Timeout for Claude API calls in seconds.
    *   `ollama_num_predict: int` (Default: 1000): `num_predict` for Ollama generation (max output length for local model task responses).
    *   `debug_mode: bool` (Default: False): Show additional technical details and logs. (See "Enhanced Debug Mode" section for details).
    *   **Note on `max_chunks`:** The `max_chunks` valve, while potentially still visible in settings from older versions, is **no longer used** to limit chunk creation in v0.2.0. All chunks derived from the context are processed by each task to ensure thoroughness.

*   **File Handling:** Processes uploaded documents via Open WebUI's file system integration.
*   **Conversation Display:** Optionally shows the detailed interaction between Claude and the local model, including tasks, chunk processing summaries, scratchpad updates, and timeout statistics per round.
*   **Enhanced Debug Mode (`debug_mode = true`):**
    *   Provides significantly more detailed performance and progress logs, including:
        *   Start and end timestamps for the overall process.
        *   Start and end of each processing round, with cumulative processing time.
        *   Initiation and details of each task within a round.
        *   Start of processing for each document chunk by the local LLM.
        *   Time taken for individual calls to both local (Ollama) and remote (Claude) LLMs, including for timeouts.
    *   This verbose logging is invaluable for diagnosing bottlenecks, understanding where time is spent during complex multi-round analyses, and fine-tuning performance settings.

### Key Optimizations and Enhancements in v0.2.0:
*   **Iterative Analysis:** The multi-round approach allows for a deeper, more comprehensive analysis of documents, as Claude can refine its understanding and task list based on previous findings.
*   **Thorough Chunk Processing:** By ensuring each task processes all chunks, the risk of missing information spread across a document is minimized.
*   **Improved Robustness:** Enhanced error handling for API calls and better management of the interaction flow.
*   **Detailed Performance Logging:** New debug mode features offer granular insight into processing times.
*   The core principles of addressing small LM limitations (confusion with long contexts, difficulty with multi-step instructions) are maintained and enhanced by the iterative, focused task execution.

### Performance Guidance and Tuning:
The MinionS protocol, especially with multiple rounds, can involve many LLM calls and significant processing time, particularly with large documents or slower local hardware. Here's how to manage and optimize performance:

*   **Critical Valves for Performance:**
    *   `max_rounds`: The most significant factor. Each round involves a Claude call for task decomposition and multiple local LLM calls for chunk processing. For very large documents or initial exploration, **start with `max_rounds = 1`**.
    *   `timeout_local`: (Default 30s) If your local model is slow or tasks are complex, it might frequently time out on chunks. Monitor the per-round timeout statistics (logged if `show_conversation` is true). If you see many timeouts or the warning triggered by `max_round_timeout_failure_threshold_percent` (default 50%), consider:
        *   Increasing `timeout_local` (e.g., to 60s, 90s, or more).
        *   Using a faster local model if available.
        *   Reducing `chunk_size` to give the local model less text to process per call (though this increases the number of calls).
    *   `chunk_size`: (Default 5000 chars) Smaller chunks mean more local LLM calls but faster processing per call. Larger chunks mean fewer calls but each takes longer and is more prone to timeout if `timeout_local` is too short.
    *   `local_model`: The choice of local model (e.g., `llama3.1:8b` vs. a smaller, faster one like `phi3` or `gemma:2b`) dramatically impacts speed.

*   **Interpreting Timeout Warnings:**
    *   The function logs the percentage of chunk processing calls that timed out in each round.
    *   If this exceeds `max_round_timeout_failure_threshold_percent`, a warning is issued. Frequent warnings suggest that `timeout_local` is too aggressive for your setup/document, or your local model is struggling. While the process will continue, results from rounds with high timeouts might be incomplete.

*   **Using Enhanced Debug Mode:**
    *   Set `debug_mode = true` to get detailed timings for each API call (Claude, Ollama) and cumulative times per round. This is crucial for identifying specific bottlenecks.

*   **`max_chunks` Valve Note:**
    *   As mentioned in the Valves section, the `max_chunks` setting is **no longer used** to limit chunk processing in v0.2.0. All document chunks are processed by each task in each round to ensure analytical thoroughness. This makes `chunk_size` and `timeout_local` the primary levers for managing local processing load.

### Error Handling:
*   Timeout management for both local and remote model calls, including per-round timeout threshold warnings.
*   Graceful degradation if no context is available (attempts a direct call to Claude).
*   Clear error messages for API issues or unexpected responses.

### Conclusion:
MinionS v0.2.0 (Multi-Round) further advances the goal of making cutting-edge AI research accessible and practical. It allows users to:
*   Leverage expensive cloud models (like Claude) more efficiently and for more complex analytical tasks.
*   Keep sensitive documents local while still benefiting from powerful cloud AI for reasoning and synthesis.
*   Achieve high-quality, comprehensive answers from large documents at a fraction of the cost of purely cloud-based processing.

The function refines the "supervisor-worker" relationship: Claude acts as an intelligent supervisor, iteratively refining its understanding and task definitions, while the local model diligently processes the entire available context for each specific query posed by the supervisor.
This version focuses on maximizing the quality and depth of analysis by ensuring comprehensive local processing and intelligent, iterative task management by the remote supervisor model.