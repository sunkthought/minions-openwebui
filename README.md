## Minion Family Open WebUI Pipes - Minion & MinionS (v0.2.0)

### Overview
This repository contains two Open WebUI pipe functions, **Minion v0.2.0** (conversational) and **MinionS v0.2.0 (Multi-Round)** (task decomposition), which implement protocols inspired by HazyResearch's Minion work. They enable cost-efficient collaboration between local Ollama Language Models (LLMs) and powerful cloud LLMs (Anthropic's Claude series) directly within Open WebUI.

These pipes allow users to leverage the reasoning capabilities of advanced models like Claude while processing large documents or sensitive data locally, aiming to reduce costs and enhance privacy.

## Installation Guide for Open WebUI Pipes

This guide provides general steps for installing custom pipe functions like Minion or MinionS into your Open WebUI instance.

1.  **Locate or Create your Open WebUI `pipelines` Directory:**
    *   Open WebUI loads custom pipe functions from a specific directory. The exact path and configuration can depend on your Open WebUI setup (e.g., Docker, bare-metal, specific version).
    *   **Docker:** If you are running Open WebUI via Docker, you need to map a local directory on your host machine to the directory Open WebUI uses for pipelines. This is typically `/app/backend/data/pipelines` inside the container.
        *   You can do this by adding a volume mount to your `docker run` command, for example: `-v /path/to/your/local/pipelines_folder:/app/backend/data/pipelines`.
        *   If you are using Docker Compose, add a similar mapping to your `docker-compose.yml` file under the `volumes` section for the Open WebUI service.
    *   **Environment Variable:** Some setups might use an `OPENAI_PIPELINES_DIR` environment variable to specify the location of the pipelines directory.
    *   If you're unsure, refer to the latest official Open WebUI documentation or your specific installation guide for custom pipe loading.

2.  **Download Pipe Files:**
    *   Download the pipe Python files (e.g., `minion-fn-claude.py` for Minion v0.2.0, `minions-fn-claude.py` for MinionS v0.2.0) from this repository.

3.  **Place Files in Directory:**
    *   Place the downloaded `.py` files directly into your local `pipelines_folder` (which is mapped to Open WebUI's pipeline directory as per step 1).

4.  **Restart Open WebUI:**
    *   Restart your Open WebUI Docker container (or the Open WebUI service if not using Docker) for it to detect and load the new pipe functions.

5.  **Verify Installation:**
    *   Once restarted, the new pipes (e.g., "Minion v0.2.0", "MinionS v0.2.0") should appear in the model selection list within Open WebUI (often prefixed by their filename or the `name` attribute defined in the script).

**Important Note on Dependencies:**
*   The provided pipe scripts use common Python libraries like `aiohttp` and `pydantic`, which are typically included in standard Open WebUI Docker images. If you encounter import errors (e.g., "ModuleNotFoundError"), you might need to ensure these dependencies are present in Open WebUI's Python environment. This could involve creating a custom Docker image that includes these dependencies or installing them into the running container if your setup allows and persists such changes.

**General Disclaimer:**
*   The exact method for installing community-provided pipes can vary based on your Open WebUI setup and version. Always refer to the latest official Open WebUI documentation for the most accurate instructions on installing and managing custom pipeline functions.

## Supported Protocols

This repository offers two distinct pipes:

*   **Minion Protocol (v0.2.0):** Implemented in `minion-fn-claude.py`. A conversational approach where Claude (cloud) and a local Ollama model chat back-and-forth to answer a query using a local document.
*   **MinionS Protocol (v0.2.0 - Multi-Round):** Implemented in `minions-fn-claude.py`. An advanced, iterative task decomposition approach where Claude breaks down complex queries into subtasks over multiple rounds, leveraging a "scratchpad" to build comprehensive answers from local documents.

## How It Works

### Context Handling:
*   Extracts context from conversation history (messages longer than 200 chars are considered potential context).
*   Processes uploaded files through the `__files__` parameter, extracting their text content.
*   Combines both sources to create a comprehensive context for the local model.

### Minion Protocol Flow (v0.2.0):
*This is the conversational, simpler Minion pipe, distinct from the multi-round MinionS pipe.*
*   **Version:** Now Minion v0.2.0.
*   Claude (remote model) receives the user's query but NOT the full document context.
*   Claude asks specific questions to the local Ollama model to gather information.
*   The local model, having access to the full context, answers Claude's questions.
*   This conversational process continues for a configured number of rounds (`max_rounds` for this pipe).
*   **Improved Final Answer Detection:** Claude is prompted to use an explicit "FINAL ANSWER READY." marker when it believes it has sufficient information, making the process more robust.
*   The local model processes the *entire document context* with each call, which can be time-consuming for large documents.

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

## Valve Configuration Guide
Valves are special parameters that allow you to customize the behavior of the Minion and MinionS pipes directly from the Open WebUI interface (usually found under "Function Calling" > "Valves" when a pipe model is selected).

### A. Valves Common to Both Minion & MinionS Pipes:
*   **`anthropic_api_key: str`**
    *   **Applies to:** Both Minion v0.2.0, MinionS v0.2.0
    *   **Default Value:** `""` (empty string)
    *   **Description:** Your Anthropic API key for accessing Claude models.
    *   **Ramifications:** Essential for functionality. An incorrect or missing key will cause API errors and prevent the pipe from working.
*   **`remote_model: str`**
    *   **Applies to:** Both
    *   **Default Value:** `claude-3-5-haiku-20241022`
    *   **Description:** Specifies the Claude model to be used (e.g., `claude-3-5-haiku-20241022` for cost-efficiency, `claude-3-5-sonnet-20241022` for quality, `claude-3-opus-20240229` for maximum power).
    *   **Ramifications:** Directly impacts cost (different models have different pricing), output quality, and API response time. Ensure the chosen model is compatible with your Anthropic API access level and desired balance of performance/cost.
*   **`ollama_base_url: str`**
    *   **Applies to:** Both
    *   **Default Value:** `http://localhost:11434`
    *   **Description:** The base URL for your Ollama server. Change this if your Ollama instance is running on a different host or port accessible from your Open WebUI backend.
    *   **Ramifications:** Must point to a running and accessible Ollama instance for local model processing to work. Incorrect URL will lead to connection errors.
*   **`local_model: str`**
    *   **Applies to:** Both
    *   **Default Value:** `llama3.2`
    *   **Description:** The name of the local model hosted on your Ollama server (e.g., `llama3.1`, `phi3`, `gemma:2b`).
    *   **Ramifications:** Affects the speed and quality of local processing (chunk analysis in MinionS, full-context responses in Minion). A less capable local model might provide poorer intermediate results, affecting the overall outcome. Ensure the model is available in your Ollama instance (`ollama list`).
*   **`show_conversation: bool`**
    *   **Applies to:** Both
    *   **Default Value:** `True`
    *   **Description:** If true, the full conversation log (Claude's prompts/reasoning, local model responses, round progression, warnings) is included in the final output.
    *   **Ramifications:** Increases output length significantly. Very useful for debugging, understanding the process, and seeing performance statistics. Turn off for cleaner final answers when not troubleshooting.
*   **`debug_mode: bool`**
    *   **Applies to:** Both
    *   **Default Value:** `False`
    *   **Description:** Enables very verbose logging within the `show_conversation` output, including detailed timings for API calls, round progression, cumulative time, and other technical details.
    *   **Ramifications:** Produces a large amount of log data. Essential for troubleshooting performance issues or unexpected behavior. Should generally be off for normal use. See "Enhanced Debug Mode" section for specific logs.
*   **`timeout_claude: int`**
    *   **Applies to:** Both
    *   **Default Value:** 60 seconds
    *   **Description:** Timeout in seconds for API calls to the remote Claude model.
    *   **Ramifications:** Setting too low might cause frequent errors if Claude takes longer for complex prompts or due to network latency. Setting too high can lead to long waits if the API is unresponsive.
*   **`max_tokens_claude: int`** 
    *   **Applies to:** Both (MinionS Default: 2000, Minion Default: 4000 - *Note: These defaults are set in the respective pipe scripts.*)
    *   **Description:** Maximum number of tokens to allow for Claude's responses.
    *   **Ramifications:** Controls the length of Claude's output (decompositions, questions, final synthesis). Too low might truncate responses, affecting quality. Higher values allow for more detailed responses but can increase costs if the model generates more tokens.
*   **`ollama_num_predict: int`**
    *   **Applies to:** Both
    *   **Default Value:** 1000 tokens
    *   **Description:** Controls the `num_predict` parameter (maximum output tokens) for local Ollama model calls.
    *   **Ramifications:** Limits the length of responses from the local model. Too low might cut off useful information extracted by the local model.

### B. MinionS (Multi-Round Task Decomposition) Specific Valves - `minions-fn-claude.py` (v0.2.0):
*   **`max_rounds: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 2
    *   **Description:** Maximum number of task decomposition rounds Claude will perform.
    *   **Ramifications:** More rounds can lead to more thorough analysis and better answers for complex queries but significantly increase processing time (more Claude calls, more local LLM calls per task) and cost. Start with 1 for large documents or initial testing.
*   **`timeout_local: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 30 seconds
    *   **Description:** Timeout in seconds for each local Ollama model call *per document chunk, per task*.
    *   **Ramifications:** Critical for performance. If too low for your local model's speed or chunk complexity, many calls might time out, leading to incomplete task results. If too high, you might wait unnecessarily for unresponsive chunks. The per-round timeout logging helps diagnose this.
*   **`max_tasks_per_round: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 3
    *   **Description:** Maximum number of new sub-tasks Claude can define in each decomposition round.
    *   **Ramifications:** More tasks can increase parallelism and detail but also increase the number of local LLM calls per round, impacting speed and local processing load.
*   **`chunk_size: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 5000 characters
    *   **Description:** The target size for splitting the input document into chunks for local LLM processing.
    *   **Ramifications:** Affects the number of chunks. Smaller chunks mean more calls to the local LLM (can be slower overall but might be better for models with small context windows or for finding very specific details). Larger chunks mean fewer calls but might overwhelm smaller local models or cause them to miss details.
*   **`max_chunks: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 2 (effectively UNUSED for processing limit in v0.2.0)
    *   **Description:** This valve is still present in settings but is **NO LONGER USED** in MinionS v0.2.0 to limit chunk processing during task execution; all chunks derived from the context are processed by each task to ensure thoroughness. It may be deprecated or repurposed in future versions.
    *   **Ramifications:** None on processing logic in MinionS v0.2.0.
*   **`max_round_timeout_failure_threshold_percent: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 50
    *   **Description:** If this percentage of local model chunk calls in a round time out, a warning is logged about potentially incomplete results for that round.
    *   **Ramifications:** Does not stop processing but alerts the user to potential issues with local model performance or an overly aggressive `timeout_local` setting for the given content/model.

### C. Minion (Conversational) Specific Valves - `minion-fn-claude.py` (v0.2.0):
*   **`max_rounds: int`**
    *   **Applies to:** Minion
    *   **Default Value:** 2
    *   **Description:** Maximum number of conversational turns between Claude and the local model.
    *   **Ramifications:** More turns allow for more detailed back-and-forth, potentially leading to better answers, but increase time and cost as each turn involves a Claude call and a full-context local LLM call.
*   **`timeout_local: int`**
    *   **Applies to:** Minion
    *   **Default Value:** 60 seconds
    *   **Description:** Timeout in seconds for local Ollama model calls. **Important:** In this protocol, the local model processes the *entire document context* with each call from Claude.
    *   **Ramifications:** This is a critical setting. For large documents, 60 seconds might still be too short if the local model is slow. If too low, calls will fail. If too high, long waits if the model struggles. Adjust based on document size and local model speed.

### General Advice for Valve Tuning:
*   **Start with Defaults:** Especially for `max_rounds`, begin with lower values (e.g., 1 for MinionS on large documents) to gauge performance on your specific setup and documents.
*   **Use `debug_mode` and `show_conversation`:** These are invaluable for understanding the process flow, seeing timings, and identifying bottlenecks or areas for improvement before extensive tuning.
*   **Adjust Timeouts Systematically:** If you see frequent timeouts for local or remote LLMs, incrementally increase the respective timeout values (e.g., `timeout_local`, `timeout_claude`). If calls are very fast, you might slightly decrease them to fail faster on problematic calls, but be cautious.
*   **Balance Thoroughness vs. Resources:** Features like more `max_rounds` (in MinionS and Minion) and MinionS's full-chunk processing aim for thoroughness but consume more time and potentially cost. Adjust based on your needs for speed, depth of analysis, and budget.
*   **Local Model Choice Matters:** The speed and capability of your chosen `local_model` will heavily influence overall performance and the quality of results from both Minion and MinionS protocols.

### Cost Efficiency:
*   The MinionS protocol reduces cloud model costs by primarily "reading" and processing the full data locally. Only task definitions, aggregated results, and final synthesis prompts are exchanged with the remote (Claude) model.
*   The function calculates and displays estimated token savings compared to sending the entire context directly to the cloud model.

### Open WebUI Integration Features:

*   **Valves (Configuration Settings):**
    *   `anthropic_api_key: str`: Your Anthropic API key for Claude. (Applies to both Minion and MinionS)
    *   `remote_model: str` (Default: "claude-3-5-haiku-20241022"): The Claude model used. (Applies to both)
    *   `ollama_base_url: str` (Default: "http://localhost:11434"): URL of your Ollama server. (Applies to both)
    *   `local_model: str` (Default: "llama3.2"): The local Ollama model. (Applies to both)
    *   `show_conversation: bool` (Default: True): Display the full interaction. (Applies to both)
    *   `debug_mode: bool` (Default: False): Show additional technical logs. (Applies to both, see specific debug sections)

    *   **Minion v0.2.0 Specific Valves:**
        *   `max_rounds: int` (Default: 2 for Minion): Maximum number of conversational rounds for the Minion pipe.
        *   `timeout_local: int` (Default: 60s for Minion): Timeout for local Ollama calls. **Note:** For Minion, the local model processes the *entire context* each time. Default is 60s. May need significant increase (e.g., 120s+) for large documents or slower local models.
        *   `timeout_claude: int` (Default: 60s for Minion): Timeout for API calls to the remote Claude model.
        *   `max_tokens_claude: int` (Default: 4000 for Minion): Max tokens for Claude's responses.
        *   `ollama_num_predict: int` (Default: 1000 for Minion): `num_predict` (max output tokens) for local Ollama model responses.

    *   **MinionS v0.2.0 (Multi-Round) Specific Valves:**
        *   `max_rounds: int` (Default: 2 for MinionS): Maximum number of task decomposition rounds.
        *   `max_tasks_per_round: int` (Default: 3): Maximum tasks Claude can create per round (MinionS only).
        *   `chunk_size: int` (Default: 5000): Target character size for document chunks (MinionS only).
        *   `timeout_local: int` (Default: 30s for MinionS): Timeout for local Ollama calls *per chunk* (MinionS only).
        *   `max_round_timeout_failure_threshold_percent: int` (Default: 50): Timeout threshold for warnings in MinionS.
        *   `max_tokens_claude: int` (Default: 2000 for MinionS): Max tokens for Claude API responses (MinionS).
        *   `timeout_claude: int` (Default: 60s for MinionS): Timeout for Claude API calls (MinionS).
        *   `ollama_num_predict: int` (Default: 1000 for MinionS): `num_predict` for local Ollama calls (MinionS).
        *   **Note on `max_chunks` (MinionS):** This valve is no longer used in MinionS v0.2.0.

*   **File Handling:** Processes uploaded documents via Open WebUI's file system integration. (Applies to both)
*   **Conversation Display:** Optionally shows the detailed interaction between Claude and the local model, including tasks, chunk processing summaries, scratchpad updates, and timeout statistics per round.
*   **Enhanced Debug Mode (`debug_mode = true`):**
    *   **For MinionS (Multi-Round):** Provides significantly more detailed performance and progress logs, including:
        *   Start and end timestamps for the overall process.
        *   Start and end of each processing round (MinionS task decomposition rounds), with cumulative processing time.
        *   Initiation and details of each task within a round.
        *   Start of processing for each document chunk by the local LLM.
        *   Time taken for individual calls to both local (Ollama) and remote (Claude) LLMs, including for timeouts.
    *   **For Minion (Conversational v0.2.0):** Also provides enhanced debug logging:
        *   Overall process start time.
        *   Start and end of each conversational round.
        *   Timings for each individual call to Claude and the local Ollama model.
        *   Total execution time for the Minion protocol.
    *   This verbose logging (for both pipes) is invaluable for diagnosing bottlenecks, understanding where time is spent, and fine-tuning performance settings.

## Valve Configuration Guide
Valves are special parameters that allow you to customize the behavior of the Minion and MinionS pipes directly from the Open WebUI interface (usually found under "Function Calling" > "Valves" when a pipe model is selected).

### A. Valves Common to Both Minion & MinionS Pipes:
*   **`anthropic_api_key: str`**
    *   **Applies to:** Both Minion v0.2.0, MinionS v0.2.0
    *   **Default Value:** `""` (empty string)
    *   **Description:** Your Anthropic API key for accessing Claude models.
    *   **Ramifications:** Essential for functionality. An incorrect or missing key will cause API errors and prevent the pipe from working.
*   **`remote_model: str`**
    *   **Applies to:** Both
    *   **Default Value:** `claude-3-5-haiku-20241022`
    *   **Description:** Specifies the Claude model to be used (e.g., `claude-3-5-haiku-20241022` for cost-efficiency, `claude-3-5-sonnet-20241022` for quality, `claude-3-opus-20240229` for maximum power).
    *   **Ramifications:** Directly impacts cost (different models have different pricing), output quality, and API response time. Ensure the chosen model is compatible with your Anthropic API access level and desired balance of performance/cost.
*   **`ollama_base_url: str`**
    *   **Applies to:** Both
    *   **Default Value:** `http://localhost:11434`
    *   **Description:** The base URL for your Ollama server. Change this if your Ollama instance is running on a different host or port accessible from your Open WebUI backend.
    *   **Ramifications:** Must point to a running and accessible Ollama instance for local model processing to work. Incorrect URL will lead to connection errors.
*   **`local_model: str`**
    *   **Applies to:** Both
    *   **Default Value:** `llama3.2`
    *   **Description:** The name of the local model hosted on your Ollama server (e.g., `llama3.1`, `phi3`, `gemma:2b`).
    *   **Ramifications:** Affects the speed and quality of local processing (chunk analysis in MinionS, full-context responses in Minion). A less capable local model might provide poorer intermediate results, affecting the overall outcome. Ensure the model is available in your Ollama instance (`ollama list`).
*   **`show_conversation: bool`**
    *   **Applies to:** Both
    *   **Default Value:** `True`
    *   **Description:** If true, the full conversation log (Claude's prompts/reasoning, local model responses, round progression, warnings) is included in the final output.
    *   **Ramifications:** Increases output length significantly. Very useful for debugging, understanding the process, and seeing performance statistics. Turn off for cleaner final answers when not troubleshooting.
*   **`debug_mode: bool`**
    *   **Applies to:** Both
    *   **Default Value:** `False`
    *   **Description:** Enables very verbose logging within the `show_conversation` output, including detailed timings for API calls, round progression, cumulative time, and other technical details.
    *   **Ramifications:** Produces a large amount of log data. Essential for troubleshooting performance issues or unexpected behavior. Should generally be off for normal use. See "Enhanced Debug Mode" section for specific logs.
*   **`timeout_claude: int`**
    *   **Applies to:** Both
    *   **Default Value:** 60 seconds
    *   **Description:** Timeout in seconds for API calls to the remote Claude model.
    *   **Ramifications:** Setting too low might cause frequent errors if Claude takes longer for complex prompts or due to network latency. Setting too high can lead to long waits if the API is unresponsive.
*   **`max_tokens_claude: int`** 
    *   **Applies to:** Both (MinionS Default: 2000, Minion Default: 4000 - *Note: These defaults are set in the respective pipe scripts.*)
    *   **Description:** Maximum number of tokens to allow for Claude's responses.
    *   **Ramifications:** Controls the length of Claude's output (decompositions, questions, final synthesis). Too low might truncate responses, affecting quality. Higher values allow for more detailed responses but can increase costs if the model generates more tokens.
*   **`ollama_num_predict: int`**
    *   **Applies to:** Both
    *   **Default Value:** 1000 tokens
    *   **Description:** Controls the `num_predict` parameter (maximum output tokens) for local Ollama model calls.
    *   **Ramifications:** Limits the length of responses from the local model. Too low might cut off useful information extracted by the local model.

### B. MinionS (Multi-Round Task Decomposition) Specific Valves - `minions-fn-claude.py` (v0.2.0):
*   **`max_rounds: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 2
    *   **Description:** Maximum number of task decomposition rounds Claude will perform.
    *   **Ramifications:** More rounds can lead to more thorough analysis and better answers for complex queries but significantly increase processing time (more Claude calls, more local LLM calls per task) and cost. Start with 1 for large documents or initial testing.
*   **`timeout_local: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 30 seconds
    *   **Description:** Timeout in seconds for each local Ollama model call *per document chunk, per task*.
    *   **Ramifications:** Critical for performance. If too low for your local model's speed or chunk complexity, many calls might time out, leading to incomplete task results. If too high, you might wait unnecessarily for unresponsive chunks. The per-round timeout logging helps diagnose this.
*   **`max_tasks_per_round: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 3
    *   **Description:** Maximum number of new sub-tasks Claude can define in each decomposition round.
    *   **Ramifications:** More tasks can increase parallelism and detail but also increase the number of local LLM calls per round, impacting speed and local processing load.
*   **`chunk_size: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 5000 characters
    *   **Description:** The target size for splitting the input document into chunks for local LLM processing.
    *   **Ramifications:** Affects the number of chunks. Smaller chunks mean more calls to the local LLM (can be slower overall but might be better for models with small context windows or for finding very specific details). Larger chunks mean fewer calls but might overwhelm smaller local models or cause them to miss details.
*   **`max_chunks: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 2 (effectively UNUSED for processing limit in v0.2.0)
    *   **Description:** This valve is still present in settings but is **NO LONGER USED** in MinionS v0.2.0 to limit chunk processing during task execution; all chunks derived from the context are processed by each task to ensure thoroughness. It may be deprecated or repurposed in future versions.
    *   **Ramifications:** None on processing logic in MinionS v0.2.0.
*   **`max_round_timeout_failure_threshold_percent: int`**
    *   **Applies to:** MinionS
    *   **Default Value:** 50
    *   **Description:** If this percentage of local model chunk calls in a round time out, a warning is logged about potentially incomplete results for that round.
    *   **Ramifications:** Does not stop processing but alerts the user to potential issues with local model performance or an overly aggressive `timeout_local` setting for the given content/model.

### C. Minion (Conversational) Specific Valves - `minion-fn-claude.py` (v0.2.0):
*   **`max_rounds: int`**
    *   **Applies to:** Minion
    *   **Default Value:** 2
    *   **Description:** Maximum number of conversational turns between Claude and the local model.
    *   **Ramifications:** More turns allow for more detailed back-and-forth, potentially leading to better answers, but increase time and cost as each turn involves a Claude call and a full-context local LLM call.
*   **`timeout_local: int`**
    *   **Applies to:** Minion
    *   **Default Value:** 60 seconds
    *   **Description:** Timeout in seconds for local Ollama model calls. **Important:** In this protocol, the local model processes the *entire document context* with each call from Claude.
    *   **Ramifications:** This is a critical setting. For large documents, 60 seconds might still be too short if the local model is slow. If too low, calls will fail. If too high, long waits if the model struggles. Adjust based on document size and local model speed.

### General Advice for Valve Tuning:
*   **Start with Defaults:** Especially for `max_rounds`, begin with lower values (e.g., 1 for MinionS on large documents) to gauge performance on your specific setup and documents.
*   **Use `debug_mode` and `show_conversation`:** These are invaluable for understanding the process flow, seeing timings, and identifying bottlenecks or areas for improvement before extensive tuning.
*   **Adjust Timeouts Systematically:** If you see frequent timeouts for local or remote LLMs, incrementally increase the respective timeout values (e.g., `timeout_local`, `timeout_claude`). If calls are very fast, you might slightly decrease them to fail faster on problematic calls, but be cautious.
*   **Balance Thoroughness vs. Resources:** Features like more `max_rounds` (in MinionS and Minion) and MinionS's full-chunk processing aim for thoroughness but consume more time and potentially cost. Adjust based on your needs for speed, depth of analysis, and budget.
*   **Local Model Choice Matters:** The speed and capability of your chosen `local_model` will heavily influence overall performance and the quality of results from both Minion and MinionS protocols.

### Key Optimizations and Enhancements in v0.2.0:
*   **Iterative Analysis (MinionS):** The multi-round task decomposition approach in MinionS allows for deeper document analysis.
*   **Improved Final Answer Detection (Minion v0.2.0):** The conversational Minion pipe now uses a more explicit "FINAL ANSWER READY." marker, making its conclusion more reliable.
*   **Thorough Chunk Processing (MinionS):** Ensures tasks in MinionS benefit from a complete document view.
*   **Improved Robustness (Both Pipes):** Enhanced error handling for API calls and better management of interaction flows.
*   **Detailed Performance Logging (Both Pipes):** Debug mode offers granular insight into processing times for both Minion and MinionS.
*   The core principles of addressing small LM limitations are maintained and enhanced across both protocols.

### Performance Guidance and Tuning:
Both Minion protocols can involve many LLM calls. Here's how to manage performance:

*   **General:**
    *   `local_model`: The choice of local model (e.g., `llama3.1:8b` vs. a smaller, faster one like `phi3` or `gemma:2b`) dramatically impacts speed for both pipes.
*   **For MinionS (Multi-Round):**
    *   `max_rounds` (MinionS): The most significant factor. Each round involves a Claude call for task decomposition and multiple local LLM calls for chunk processing. For very large documents or initial exploration, **start with `max_rounds = 1`**.
    *   `timeout_local` (MinionS - default 30s per chunk): If your local model frequently times out on chunks in MinionS, monitor per-round timeout stats. Consider increasing this, using a faster local model, or reducing `chunk_size`.
    *   `chunk_size`: Smaller chunks mean more local LLM calls but faster processing per call. Larger chunks mean fewer calls but each takes longer.
*   **For Minion (Conversational v0.2.0):**
    *   `max_rounds` (Minion - default 2 rounds): Controls the number of conversational turns.
    *   `timeout_local` (Minion - default 60s): Crucial for the Minion pipe as the local model processes the *entire document context* each time it's called. For large documents or slower hardware, this may need to be significantly increased (e.g., 120s, 180s, or more) to prevent timeouts. Monitor total execution time in debug mode.

*   **Interpreting Timeout Warnings (MinionS):**
    *   For MinionS, the function logs the percentage of chunk processing calls that timed out in each round.
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