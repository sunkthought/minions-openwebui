# Warning: THIS SOFTWARE WAS HEAVILY AI GENERATED
Use at your own risk. Any voluntary submissions by humans greatly accepted.

# Collaborative AI with Minion and MinionS for Open WebUI

This repository provides an implementation of the Minion and MinionS protocols, designed by HazyResearch, tailored for use as **Functions** within the [Open WebUI](https://github.com/open-webui/open-webui) platform. These protocols enable sophisticated collaboration between local AI models (e.g., Ollama) and powerful remote AI models (e.g., Anthropic's Claude) to tackle complex tasks.

## Understanding Minion and MinionS: Collaborative AI Agents

MinionS (and its predecessor, Minion) is a protocol designed by HazyResearch to enable sophisticated collaboration between different AI models, often a smaller, efficient local model and a larger, more powerful remote model. The core idea is to create a "team" of AI agents that can work together to solve complex queries or perform tasks that might be too large or inefficient for a single model to handle alone.

Key concepts include:

*   **Task Decomposition (MinionS)**: A complex user query is broken down into smaller, manageable sub-tasks by a capable remote "supervisor" model. **Version 0.3.0+** now features advanced code-based task decomposition, where the supervisor generates Python code to dynamically create tasks based on document structure. **Version 0.3.6b** introduces advanced conversation intelligence with state tracking, question deduplication, flow control, and answer validation.
*   **Specialized Roles**: Different models take on specific roles. In this implementation:
    *   A remote "supervisor" model (e.g., Claude) decomposes tasks (MinionS) or guides the conversation (Minion), and synthesizes final answers.
    *   Local "worker" models (e.g., Ollama-based models) execute specific sub-tasks on portions of data (MinionS) or provide information from the full context (Minion).
*   **Efficient Resource Use**: Local models process large local contexts (like uploaded documents), minimizing data sent to remote (often more expensive) APIs. Only tasks, summaries, and key information are exchanged with the remote model.
*   **Conversational Collaboration (Minion Protocol)**: A simpler, iterative form where a local model acts as a knowledgeable assistant to a remote model. The remote model asks questions, the local model answers based on the context, and this dialogue continues until the remote model can answer the user's original query.
*   **Parallel Processing & Multi-Round (MinionS Protocol)**: The supervisor model defines multiple tasks. These tasks are then executed by local models over chunks of the context. The results are gathered, summarized, and presented back to the supervisor, which can then decide to ask for more tasks in subsequent rounds or synthesize a final answer. (Note: True parallelism in task execution depends on the underlying infrastructure; the current implementation processes tasks sequentially within each round).

This approach aims to improve efficiency, reduce costs, and enhance AI capabilities by combining the strengths of different models, especially for querying large documents or codebases.

**Learn More:**
*   **Original Research Paper**: [MinionS: A Protocol for Scalable and Cost-Effective AI Collaboration (PDF)](https://arxiv.org/pdf/2502.15964)
*   **HazyResearch GitHub Repository**: [github.com/HazyResearch/minions](https://github.com/HazyResearch/minions)

## Minion/MinionS for Open WebUI

This repository provides an implementation of the Minion and MinionS protocols specifically designed to be used as **Functions** within the [Open WebUI](https://github.com/open-webui/open-webui) platform. This allows users to leverage these advanced collaborative AI techniques directly from their Open WebUI interface when interacting with compatible models.

**Benefits of this Implementation:**

*   **Seamless Model Collaboration**: Easily orchestrate workflows that combine the strengths of local models (e.g., Ollama models accessible to your Open WebUI instance) and powerful remote APIs (e.g., Anthropic's Claude). The local model can efficiently process large documents or provide specific context based on supervisor requests, while the remote model can handle complex reasoning, task decomposition, or synthesis of the final answer.
*   **Enhanced In-Chat Capabilities**: Perform sophisticated tasks like in-depth document analysis, context-aware question answering over provided text, and multi-step query resolution directly within your Open WebUI chat, simply by using the Function calling capability.
*   **User-Friendly Interface**: Manage and interact with these protocols through Open WebUI's existing chat and Function features, without needing to run separate scripts or services for the orchestration logic. Configuration is handled through the Function settings panel in Open WebUI.
*   **Modular and Customizable**: Built using a system of "partials" (modular Python code blocks concatenated by a generator script), this implementation allows for easier customization of prompts, API choices (for Claude and Ollama endpoints/models), and core logic. Users can regenerate the Minion/MinionS functions with their modifications using the provided generator script. The recent refactoring has further enhanced this modularity by separating concerns into dedicated partials for API calls, prompt generation, context utilities, file processing, and protocol-specific logic.

By integrating Minion and MinionS as Open WebUI Functions, this project aims to make advanced AI collaboration techniques more accessible and practical for a wider range of users, enabling more powerful interactions with their documents and data.

## Quick Start Guide

This guide will help you get the Minion/MinionS functions up and running with your Open WebUI installation.

### 1. Install Open WebUI

If you don't have Open WebUI installed yet, you'll need to set it up first. It's commonly run using Docker.

A typical Docker command to install Open WebUI is:
```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```
*After running, access Open WebUI by navigating to `http://localhost:3000` in your web browser.*

**Important**: This is a basic command. Open WebUI offers various installation options (e.g., for GPU support, bundled Ollama, Kubernetes, pip). For detailed and up-to-date installation instructions, please refer to the **[official Open WebUI GitHub repository](https://github.com/open-webui/open-webui)**.

### 2. Add Minion/MinionS Function to Open WebUI

Once Open WebUI is running:
1.  Navigate to **Workspace** > **Functions**.
2.  Click the **Import Functions** button.
3.  **Paste Function Code**:
    *   In this repository (`SunkThought/minions-openwebui`), navigate to the `generated_functions/` directory.
    *   Choose the function file based on your preference:
        *   **Latest v0.3.8 (Current)**:
            *   For Minion protocol: `minion_v037_function.py` 
            *   For MinionS protocol: `minions_v037_function.py`
        *   **Default v0.3.7 (Enhanced)**:
            *   For Minion protocol: `minion_default_function.py`
            *   For MinionS protocol: `minions_default_function.py`
        *   **Legacy v0.3.6 (Stable)**:
            *   For Minion protocol: `minion_default_old_function.py`
            *   For MinionS protocol: `minions_default_old_function.py`
    *   Copy the **entire content** of this file.
    *   Paste it into the import dialog in Open WebUI.
4.  Click **Import** to add the function.

### 3. Configure Function Valves

"Valves" are the settings you can adjust for the Minion/MinionS function. After importing the function, you can configure these settings in the Function editor or when enabling the function for a model.

Key valves to configure:

#### Essential Settings
*   `supervisor_provider`: Choose between 'anthropic' or 'openai' for the supervisor model (default: 'anthropic').
*   `anthropic_api_key`: Your API key for Anthropic Claude (required if using 'anthropic' provider).
*   `openai_api_key`: Your API key for OpenAI (required if using 'openai' provider).
*   `remote_model`: The Claude model you wish to use (e.g., `claude-3-5-haiku-20241022` for cost efficiency, `claude-3-5-sonnet-20241022` for quality). Note: When using OpenAI provider, see `openai_model` setting.
*   `openai_model`: The OpenAI model to use when supervisor_provider is 'openai' (e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-4`). Default is `gpt-4o`.
*   `ollama_base_url`: The base URL of your Ollama server. Default is `http://localhost:11434`.
*   `local_model`: The name of the local Ollama model you want to use (e.g., `llama3.2`, `mistral`). Ensure this model is available in your Ollama instance (e.g., via `ollama pull llama3.2`).

#### Protocol Settings (Minion)
*   `max_rounds`: Maximum conversation rounds between models (default: 2).
*   `show_conversation`: Show detailed interaction log (default: true).
*   `timeout_local`: Timeout for local model calls in seconds (default: 60).
*   `timeout_claude`: Timeout for Claude API calls in seconds (default: 60).
*   `max_tokens_claude`: Maximum tokens for Claude's responses (default: 4000).
*   `use_structured_output`: Enable JSON structured output for local model responses (default: **true** as of v0.3.6).
*   `enable_completion_detection`: Enable detection of when the remote model has gathered sufficient information (default: true).
*   `chunk_size`: Maximum chunk size in characters for context fed to local models (default: 5000).
*   `max_chunks`: Maximum number of document chunks to process (default: 2).

#### Protocol Settings (MinionS)
*   `max_rounds`: Maximum task decomposition rounds (default: 2).
*   `max_tasks_per_round`: Maximum sub-tasks per round (default: 3).
*   `chunk_size`: Maximum chunk size in characters (default: 5000).
*   `max_chunks`: Maximum chunks to process per task (default: 2).
*   `show_conversation`: Show full task decomposition and execution details (default: true).
*   `timeout_local`: Timeout for each local model call in seconds (default: 30).
*   `max_round_timeout_failure_threshold_percent`: Warning threshold for timeouts (default: 50).
*   `use_structured_output`: Enable JSON structured output (default: **true** as of v0.3.5).
*   `structured_output_fallback_enabled`: Enable fallback parsing when structured output fails (default: true).

##### Scaling Strategy Settings (v0.3.8)
*   `scaling_strategy`: Choose scaling strategy for task execution: 'none', 'repeated_sampling', 'finer_decomposition', 'context_chunking' (default: 'none').
*   `repeated_samples`: Number of samples to take when using repeated_sampling strategy (default: 3).
*   `adaptive_rounds`: Enable adaptive round control based on information gain (default: true).
*   `min_info_gain`: Minimum information gain (0.0-1.0) required to continue to the next round (default: 0.1).

##### Performance Profile
*   `performance_profile`: Overall performance profile: 'high_quality', 'balanced', 'fastest_results'. Affects base thresholds and max_rounds before other adaptive modifiers. (default: "balanced", options: ["high_quality", "balanced", "fastest_results"])

##### Early Stopping & Convergence Settings
*   `enable_early_stopping`: Enable early stopping of rounds based on confidence, query complexity, or convergence. (default: false)
*   `simple_query_confidence_threshold`: Confidence threshold (0.0-1.0) to stop early for SIMPLE queries. (default: 0.85)
*   `medium_query_confidence_threshold`: Confidence threshold (0.0-1.0) to stop early for MEDIUM queries. (default: 0.75)
*   `min_rounds_before_stopping`: Minimum number of rounds to execute before early stopping (confidence or convergence) can be triggered. (default: 1)
*   `convergence_novelty_threshold`: Minimum percentage of novel findings required per round to consider it non-convergent. E.g., 0.10 means less than 10% new findings might indicate convergence if other criteria met. (default: 0.10)
*   `convergence_rounds_min_novelty`: Number of consecutive rounds novelty must be below `convergence_novelty_threshold` to trigger convergence. (default: 2)
*   `convergence_sufficiency_threshold`: Minimum sufficiency score required for convergence-based early stopping. E.g., 0.7 means 70% sufficiency needed. (default: 0.7)

##### Adaptive Threshold Settings
*   `enable_adaptive_thresholds`: Allow the system to dynamically adjust confidence, sufficiency, and novelty thresholds based on document size, query complexity, and first-round performance. (default: true)
*   `doc_size_small_char_limit`: Documents with character count below this are considered 'small' for threshold adjustments. (default: 5000)
*   `doc_size_large_char_start`: Documents with character count above this are considered 'large' for threshold adjustments. (default: 50000)
*   `confidence_modifier_small_doc`: Value added to base confidence thresholds if document is small (e.g., -0.05 to be less strict). (default: 0.0)
*   `confidence_modifier_large_doc`: Value added to base confidence thresholds if document is large (e.g., +0.05 to be more strict). (default: 0.0)
*   `sufficiency_modifier_simple_query`: Value added to base sufficiency thresholds for simple queries (e.g., -0.1 to require less sufficiency). (default: 0.0)
*   `sufficiency_modifier_complex_query`: Value added to base sufficiency thresholds for complex queries (e.g., +0.1 to require more). (default: 0.0)
*   `novelty_modifier_simple_query`: Value added to the base novelty threshold for simple queries. (default: 0.0)
*   `novelty_modifier_complex_query`: Value added to the base novelty threshold for complex queries. (default: 0.0)
*   `first_round_high_novelty_threshold`: If first round's novel findings percentage is above this (e.g., 0.75 for 75%), it's a high novelty first round. (default: 0.75)
*   `sufficiency_modifier_high_first_round_novelty`: Value added to sufficiency thresholds if first round novelty is high (e.g., -0.05 to relax sufficiency requirement). (default: -0.05)

#### Advanced Settings
*   `debug_mode`: Enable verbose logging and technical details (default: false).
*   `ollama_num_predict`: Maximum output tokens for local model (default: 1000).

**⚠️ Important: `ollama_base_url` for Docker Users**

If you are running Open WebUI in a Docker container (which is common) and your Ollama service is running on your host machine (i.e., *not* inside another Docker container on the same Docker network):
*   The default `http://localhost:11434` for `ollama_base_url` **will not work** from inside the Open WebUI container. This is because `localhost` inside the container refers to the container itself, not your host machine.
*   **Solutions**:
    *   **Docker Desktop (Mac/Windows/Linux with Host Networking Support)**: Try `http://host.docker.internal:11434`. This special DNS name resolves to the host's IP address.
    *   **Linux Host (Bridge Network)**: You might need to use the actual IP address of your host machine on your local network (e.g., `http://192.168.1.100:11434`). You can find this IP using commands like `ifconfig` or `ip addr show docker0` (for the Docker bridge).
    *   **Ollama in Docker**: If Ollama is also running as a Docker container, ensure both Open WebUI and Ollama containers are on the same custom Docker network. You can then use the Ollama container's name as the hostname (e.g., `http://ollama:11434`, assuming your Ollama container is named `ollama`).
*   Refer to Docker networking documentation or Open WebUI's troubleshooting guides for more comprehensive help.

### 4. Enable the Function for a Model

1.  Navigate to **Workspace** > **Models**.
2.  Select the model you want to use with Minion/MinionS.
3.  Click the pencil icon to edit the model.
4.  Scroll down to the **Functions** section.
5.  Check the box next to the Minion or MinionS function you imported.
6.  Click **Save**.

### 5. Use the Function in a Chat

1.  Start a new chat or select an existing one.
2.  Select the model you configured with the Minion/MinionS function.
3.  The function will automatically activate when you provide context (documents, long text) along with your query.
4.  Type your query and send the message. The Minion/MinionS protocol will execute, and you should see the results in the chat.

**Tips for best results:**
- Provide clear context by pasting documents or uploading files
- Ask specific questions about the provided context
- Use `show_conversation` valve to see the detailed collaboration process
- Adjust timeout settings if you're working with particularly long documents

### Structured Output Support (v0.3.5+)

Both Minion and MinionS protocols now include robust structured output support for improved reliability when working with local models. As of v0.3.6, structured output is enabled by default for both protocols, ensuring consistent, parseable responses from local models and making the protocols more reliable and easier to debug.

**Key Features:**
- **Automatic Model Detection**: The system automatically detects if your local model supports JSON mode
- **Smart Fallback**: If JSON parsing fails, the system attempts to extract information using regex patterns
- **Clear Schema**: Local models receive explicit JSON schema with examples for consistent formatting

**Supported Models:**
The following models are known to support structured output:
- Llama family: llama3.2, llama3.1, llama3, llama2
- Mistral family: mistral, mixtral, mistral-nemo
- Qwen family: qwen2, qwen2.5
- Others: gemma2, phi3, command-r, codellama, and many more

**Configuration:**
- `use_structured_output`: Enable/disable structured output (default: true)
- `structured_output_fallback_enabled`: Enable/disable fallback parsing (default: true)

**Benefits:**
- More reliable parsing of local model responses
- Better confidence scoring (HIGH/MEDIUM/LOW mapped to numeric values)
- Easier debugging with clear parse error messages
- Improved success rates for task execution

### Conversation Analytics (v0.3.6+)

The Minion protocol now provides comprehensive conversation metrics to help you understand and optimize the collaborative process:

**Metrics Tracked:**
- Round usage efficiency (actual vs maximum allowed)
- Average confidence levels across all responses
- Confidence distribution breakdown (HIGH/MEDIUM/LOW counts)
- Completion method detection (early detection vs full rounds)
- Total conversation duration and token estimates
- Document chunking information

**Benefits:**
- Identify optimal settings for your use cases
- Monitor conversation efficiency and quality
- Debug issues with confidence levels or completion detection
- Understand the impact of different chunk sizes on processing

## Choosing Your Minion: Minion vs. MinionS

Both Minion and MinionS are designed for collaborative AI, but they employ different strategies and are suited for different types of tasks.

### Minion Protocol: Conversational Context Expert

*   **Approach**: The Minion protocol facilitates a direct, iterative conversation between a remote "supervisor" model (e.g., Claude) and a local "assistant" model (e.g., an Ollama model like Llama 3). The local model is assumed to have full access to the provided context (e.g., uploaded documents or pasted text). The remote model asks specific questions to the local model to gather the information it needs to answer the user's overall query. Think of it as the remote model "interviewing" the local model about the document.
*   **Strengths**:
    *   Effective for focused, deep dives into a specific topic within the context.
    *   Allows for iterative refinement; if the initial answer isn't sufficient, the remote model can ask follow-up questions.
    *   Can be more efficient if the query requires only a small, specific piece of information from a large document, as only the relevant Q&A snippets are processed by the remote model.
    *   Simpler interaction flow compared to MinionS.
*   **Best Suited For**:
    *   Answering specific, targeted questions about a document (e.g., "What was the main conclusion of the study described in section 3?").
    *   Interactive summarization where you might guide the process by clarifying what the remote model should ask the local model.
    *   When the remote model needs to "interview" the local model to understand nuanced aspects or extract very specific details from the provided text.
    *   Tasks where the path to the answer is relatively clear but requires sequential information extraction.

### MinionS Protocol: Parallel Task Decomposer & Synthesizer

*   **Approach**: The MinionS (Minion Supervisor) protocol takes a more structured, multi-step approach, particularly useful for complex queries over large contexts:
    1.  **Code-Based Decomposition (v0.3.0+)**: The remote supervisor model analyzes the query and generates Python code that dynamically creates sub-tasks based on document structure.
    2.  **Chunking & Task Execution**: The provided context is divided into smaller chunks. Each sub-task is then executed by local models on relevant chunks of the document.
    3.  **Aggregation & Synthesis**: The results from all sub-tasks across all chunks are collected. This aggregated information is then sent back to the remote supervisor model, which synthesizes it into a comprehensive final answer. The supervisor might also initiate further rounds of decomposition and execution if the initial results are insufficient.
*   **Strengths**:
    *   Excellent for broad analysis of large documents or complex queries that can be broken down into parallelizable sub-parts.
    *   Can be significantly more efficient for tasks that benefit from parallel processing of information across different document segments.
    *   Handles multifaceted queries well (e.g., "Summarize this document, list all key personnel mentioned, and identify potential future risks discussed.").
    *   Reduces the amount of data sent to the expensive remote model by processing chunks locally.
    *   **v0.3.0+**: Dynamic task generation based on actual document content and structure.
*   **Best Suited For**:
    *   Comprehensive analysis of one or more large documents where multiple pieces of information need to be found and correlated.
    *   Queries that require identifying and correlating different types of information from various parts of a text (e.g., "Find all mentions of project Alpha, who was involved, and what were the outcomes from each phase described in this report.").
    *   Situations where the initial query is broad and needs a structured breakdown to be effectively addressed (e.g., "Tell me everything important about this lengthy project proposal.").
    *   When you want to apply a consistent set of questions/tasks to multiple document chunks.

### Quick Comparison

| Feature          | Minion                                     | MinionS                                            |
|------------------|--------------------------------------------|----------------------------------------------------|
| **Primary Method** | Iterative Conversational Q&A             | Code-Based Task Decomposition, Chunk Execution, Synthesis |
| **Complexity**   | Simpler, direct interaction flow           | More complex, multi-step orchestration             |
| **Context Handling**| Full context or automatic chunking (v0.3.6+) | Local models see chunks of context per sub-task    |
| **Best For**     | Focused Q&A, iterative refinement         | Broad analysis, multifaceted queries, large docs   |
| **Task Generation** | Natural language questions              | Python code generating dynamic tasks (v0.3.0+)     |
| **Efficiency**   | Good for specific queries                  | Better for comprehensive analysis                  |
| **Document Chunking** | Automatic chunking for large docs (v0.3.6+) | Built-in chunking and parallel processing         |

By understanding these differences, you can choose the protocol that best fits the complexity and nature of your task when using these functions in Open WebUI.

### Choosing Between v0.3.7 and v0.3.6

#### v0.3.7 (Default/Recommended)
- **Enhanced architecture** with modular design for better maintainability
- **Improved error handling** with context-aware messages and troubleshooting hints
- **Structured debugging** with hierarchical logging and performance metrics
- **Centralized constants** for easier customization
- **Better code organization** for future development and modifications
- Use profiles: `minion_default` and `minions_default` (default)

#### v0.3.6 (Legacy/Stable)
- **Proven stability** with extensive testing in production environments
- **Simpler architecture** for users who prefer the traditional approach
- **Smaller file sizes** due to less comprehensive utility modules
- **Established workflows** for users already familiar with the codebase
- Use profiles: `minion_default_old` and `minions_default_old`

**Recommendation**: v0.3.7 is now the default for all new function generation. Use v0.3.6 only if you need compatibility with existing deployments or prefer the simpler architecture.

### What's New in Version 0.3.8

Version 0.3.8 represents a major advancement in collaborative AI capabilities:

1. **Multi-Provider API Support**: Full support for both Anthropic Claude and OpenAI GPT models as supervisor models, providing flexibility in model selection and reducing dependency on a single provider.

2. **Intelligent Scaling Strategies**: Implementation of research-backed scaling strategies including repeated sampling for higher confidence, finer decomposition for complex tasks, and intelligent context chunking for large documents.

3. **Adaptive Round Control**: Smart termination based on information gain analysis that balances thoroughness with efficiency, automatically detecting when additional rounds provide diminishing returns.

4. **Model Capability Detection**: Automatic detection and optimization based on model capabilities, ensuring optimal performance regardless of the specific models being used.

These enhancements make v0.3.8 the most robust and intelligent version of the MinionS/Minion protocols, suitable for production use across a wide variety of scenarios and model combinations.

### What's New in Version 0.3.6b

Version 0.3.6b introduces four major improvements to the Minion protocol:

1. **Conversation State Tracking**: Maintains comprehensive conversation state including Q&A pairs, topics covered, key findings, and information gaps for better context awareness.

2. **Question Deduplication**: Prevents the remote model from asking semantically similar questions by detecting duplicates and requesting alternative questions.

3. **Conversation Flow Control**: Guides conversations through logical phases (exploration → deep dive → gap filling → synthesis) for more efficient and structured interactions.

4. **Answer Validation Loop**: Validates answer quality and automatically requests clarification for unclear or incomplete responses, improving overall answer quality.

These improvements work together to create more intelligent, efficient, and higher-quality conversations between the remote and local models.

## Advanced: Configuration and Custom Function Generation

The Minion and MinionS functions in this repository are not static; they are dynamically generated from modular code pieces called "partials." This design allows for flexibility and customization.

### Understanding the Components

*   **`partials/` Directory**:
    This directory is the heart of the modular system. It contains numerous Python files (`.py`), each representing a "partial" piece of the final function's code (e.g., API call logic, prompt construction, protocol execution steps, valve definitions). Advanced users can modify these partials or even create new ones to alter functionality.
    
    **v0.3.7 Enhancement**: The partials directory now includes centralized utility modules:
    - `imports_registry.py`: Manages all imports and dependencies
    - `constants.py`: Centralizes configuration values and magic numbers
    - `error_handling.py`: Provides consistent error handling across protocols
    - `debug_utils.py`: Offers structured debugging and logging utilities
    - `protocol_base.py`: Contains shared patterns between Minion and MinionS
    - `protocol_state.py`: Manages execution state and metrics tracking

*   **`generation_config.json`**:
    This JSON file defines "profiles" for generating functions. Each profile acts as a blueprint for a specific function variant. Key fields in a profile include:
    *   `description`: A human-readable description of the profile.
    *   `output_filename_template`: A template for naming the generated Python file (e.g., `{profile_name}_function.py`).
    *   `header_placeholders`: Allows you to set the metadata (title, version, description, etc.) that appears in the comment block at the top of the generated function file. This metadata is often displayed by Open WebUI in the function management interface.
    *   `partials_concat_order`: This is a critical list that specifies which files from the `partials/` directory are included and the exact order in which they will be concatenated to form the complete function code for that profile. The order defines the Python code structure and how dependencies are resolved, so it must be logical.

*   **`generator_script.py`**:
    This Python script is the tool that builds the functions. It takes a function type (`minion` or `minions`) and an optional profile name (e.g., `--profile my_custom_profile`) as input. It then reads the specified profile from `generation_config.json` and assembles the partials in the defined order, outputting a single, runnable Python file compatible with Open WebUI's function system into the `generated_functions/` directory.

### Generating Functions

You can generate different versions of Minion and MinionS functions:

#### Latest Functions (v0.3.8 with OpenAI Support)
```bash
# Generates minion_v037_function.py using v0.3.8 features including OpenAI API support
python generator_script.py minion --profile minion_v037

# Generates minions_v037_function.py using v0.3.8 features including scaling strategies
python generator_script.py minions --profile minions_v037
```

#### Default Functions (v0.3.7 Enhanced)
```bash
# Generates minion_default_function.py using the enhanced v0.3.7 modular architecture
python generator_script.py minion

# Generates minions_default_function.py using the enhanced v0.3.7 modular architecture
python generator_script.py minions
```

#### Legacy Functions (v0.3.6 Stable)
```bash
# Generates minion_default_old_function.py with legacy architecture
python generator_script.py minion --profile minion_default_old

# Generates minions_default_old_function.py with legacy architecture
python generator_script.py minions --profile minions_default_old
```

The output files will be placed in the `generated_functions/` directory.

### Creating Custom Function Versions

You can tailor the generated functions to your specific needs:

**1. Simple Customization (Editing `generation_config.json`)**

*   **Modify Existing Profiles**: You can directly edit the `minion_default` or `minions_default` profiles. For example, you could change the `header_placeholders.TITLE` or `header_placeholders.DESCRIPTION` to better suit your Open WebUI setup or to note your own modifications. You might also tweak the default values for some valves by editing the respective `minion_valves.py` or `minions_valves.py` files (though this is a step towards modifying partials).
*   **Create New Profiles**:
    1.  Copy an existing profile block in `generation_config.json` (e.g., copy the entire `minion_default` object).
    2.  Rename the new profile key (e.g., from `"minion_default"` to `"minion_experimental"`). Make sure this new key is unique.
    3.  Customize its `header_placeholders` or, more importantly, its `partials_concat_order`. For instance, you could remove a partial if you don't need its functionality, or add a new custom partial you've created (see example below).
    4.  Generate your custom function by specifying the profile name:
        ```bash
        python generator_script.py minion --profile minion_experimental
        ```
        *(This will create `minion_experimental_function.py` in `generated_functions/`)*.

**2. Advanced Customization (Modifying Partials)**

*   Users comfortable with Python can directly modify the code within the `.py` files in the `partials/` directory. This allows for fine-grained control over the logic of each component (e.g., changing prompt wording, altering API call parameters, modifying the protocol steps).
*   **Caution**: Modifying partials directly can have significant effects on the generated functions. It's advisable to:
    *   Back up original partials before making changes.
    *   Use version control (like Git branches) to manage your modifications.
    *   Test thoroughly after changes.

**Example Customization Scenario:**

Let's say you want a version of the Minion function that uses a slightly different set of initial prompts for Claude, and you want to keep the default version as well.

1.  **Copy & Edit Partial**:
    *   Duplicate `partials/minion_prompts.py` and name the copy `partials/my_custom_minion_prompts.py`.
    *   Modify the prompt generation logic (e.g., the text returned by `get_minion_initial_claude_prompt`) within `my_custom_minion_prompts.py` as desired.
2.  **Create New Profile in `generation_config.json`**:
    *   Copy the entire `minion_default` profile object.
    *   Rename the key to `"minion_custom_prompt"`.
    *   Update `header_placeholders.TITLE` to something like "Minion - Custom Prompts".
    *   In the `partials_concat_order` list for your new `"minion_custom_prompt"` profile, find the line `"minion_prompts.py"` and change it to `"my_custom_minion_prompts.py"`.
3.  **Generate the New Function**:
    ```bash
    python generator_script.py minion --profile minion_custom_prompt
    ```
    This will create `minion_custom_prompt_function.py` in the `generated_functions/` directory. You can now add this new function to Open WebUI alongside the default one.

This modular approach provides a powerful way to adapt and evolve the Minion/MinionS functions to fit a wide variety of use cases and preferences.

## Version History

### v0.3.8 - Multi-Provider API Support & Intelligent Scaling (Latest)
- **🔄 Multi-Provider API Support**: Added OpenAI API support as an alternative to Anthropic Claude
  - **Provider Selection**: Choose between 'anthropic' or 'openai' for the supervisor model via `supervisor_provider` valve
  - **Unified API Interface**: `call_supervisor_model()` function provides seamless switching between providers
  - **Provider-Specific Configuration**: Separate API keys and model selection for each provider
  - **Model Support**: Compatible with GPT-4o, GPT-4-turbo, Claude-3.5-Sonnet, and other models
- **📈 Scaling Strategies (MinionS)**: Implemented three scaling strategies from the HazyResearch paper
  - **Repeated Sampling**: Execute tasks multiple times and aggregate results for higher confidence
  - **Finer Decomposition**: Break down complex tasks into smaller, more focused sub-tasks
  - **Context Chunking**: Process large documents by intelligently splitting content across chunks
  - **Strategy Selection**: Configure via `scaling_strategy` valve with options: none, repeated_sampling, finer_decomposition, context_chunking
- **🧠 Adaptive Round Control**: Intelligent stopping based on information gain analysis
  - **Information Gain Tracking**: Monitor the novelty and value of information gathered each round
  - **Dynamic Convergence**: Automatically detect when additional rounds provide diminishing returns
  - **Configurable Thresholds**: Fine-tune sensitivity with `min_info_gain` setting (0.0-1.0)
  - **Smart Termination**: Balance thoroughness with efficiency by stopping when sufficient information is gathered
- **🔍 Model Capability Detection**: Automatic parameter adjustment based on model capabilities
  - **Capability Database**: Comprehensive database of model capabilities (context limits, JSON support, function calling)
  - **Automatic Detection**: Detect model capabilities for both supervisor and worker models
  - **Dynamic Adjustment**: Automatically adjust chunk sizes, token limits, and processing strategies
  - **Performance Optimization**: Optimize processing based on each model's strengths and limitations
- **🐛 Critical Bug Fixes**: Resolved JSON parsing errors that prevented local model execution
  - **Escape Sequence Handling**: Fixed invalid JSON escape sequences generated by local models (e.g., `\a`, `\c`)
  - **Robust Parsing**: Improved JSON parsing with better error recovery and fallback mechanisms
  - **Preventive Fixes**: Applied fixes to both Minion and MinionS protocols for consistency

### v0.3.7 - Modular Architecture & Code Quality Improvements
- **🏗️ Modular Architecture**: Complete redesign using centralized utility modules for better maintainability and consistency
  - **Centralized Import Management**: `imports_registry.py` eliminates duplicate imports and organizes dependencies
  - **Constants Extraction**: `constants.py` centralizes 200+ magic numbers, timeouts, model names, and configuration values
  - **Unified Error Handling**: `error_handling.py` provides consistent error formatting, context-aware messages, and troubleshooting hints
  - **Structured Debug Logging**: `debug_utils.py` offers hierarchical context management, timing utilities, and multiple debug levels
  - **Protocol Base Classes**: `protocol_base.py` extracts common patterns between Minion and MinionS for code reuse
  - **State Management**: `protocol_state.py` centralizes round tracking, metrics collection, and execution context
- **🔧 Enhanced Generator Script**: Updated to support both legacy and modular architectures with backward compatibility
- **📊 New v0.3.7 Profiles**: Added `minion_v037` and `minions_v037` profiles leveraging the new modular structure
- **🛠️ Developer Experience**: Improved code organization, consistent patterns, and easier customization through modular design
- **🔄 Backward Compatibility**: Legacy profiles (`minion_default`, `minions_default`) continue to work unchanged
- **📈 Performance Benefits**: Better error recovery, consistent debugging, and reduced code duplication across protocols

### v0.3.6 - Enhanced Minion Protocol
- **Structured Output by Default**: Minion protocol now enables structured output by default for improved reliability and consistency
- **Intelligent Completion Detection**: Added detection of when the remote model has sufficient information using natural language cues like "I now have sufficient information" or "I can now answer"
- **Enhanced Question Generation**: Improved remote model prompting with strategic question guidelines, examples of good vs poor questions, and context-aware tips
- **Better Local Model Prompting**: Enhanced local assistant prompts with clearer role definition, citation guidance, and confidence level criteria
- **Comprehensive Conversation Metrics**: New metrics tracking including round usage, confidence distribution, completion method, duration, and token estimates
- **Document Chunking Support**: Mirrored MinionS chunking capabilities for handling large documents efficiently
  - Automatic document splitting for large files
  - Individual chunk processing with combined results
  - Configurable chunk size and maximum chunks
  - Clear multi-chunk result presentation

### v0.3.5 - Robust Structured Output Support
- **JSON Mode by Default**: Structured output is now enabled by default for improved reliability with compatible local models
- **Model Capability Detection**: Automatically detects if local models support JSON mode (includes llama3.2, mistral, mixtral, qwen2, gemma2, and many others)
- **Enhanced Prompt Engineering**: Clear JSON schema with comprehensive examples of correct and incorrect formats
- **Robust Parsing**: Handles markdown-wrapped JSON, extracts JSON from responses with explanatory text, and includes regex fallback for malformed responses
- **Configurable Fallback**: New `structured_output_fallback_enabled` valve allows control over fallback behavior
- **Success Metrics**: Tracks structured output success rate for monitoring and debugging
- **Better Confidence Scoring**: Consistent confidence values (HIGH/MEDIUM/LOW) mapped to numeric scores for better decision making

### v0.3.4 - Advanced Adaptive Round Management & Performance Insights
- **Smart Information Sufficiency**: Implemented information sufficiency scoring that considers query component coverage and confidence of addressed components, moving beyond simple confidence metrics.
- **Dynamic Convergence Detection**: MinionS can now detect diminishing returns by tracking per-round information gain, novelty of findings, and task failure trends. This allows the system to stop early if further rounds are unlikely to yield significant new information, especially when sufficiency is met.
- **Adaptive Thresholds**: Key decision thresholds (for confidence-based early stopping, sufficiency requirements, and novelty sensitivity in convergence) are now dynamically adjusted based on:
    - Document size (small, medium, large).
    - Query complexity (simple, medium, complex).
    - First-round performance (high novelty can relax subsequent sufficiency needs).
- **Performance Profiles**: Introduced 'high_quality', 'balanced', and 'fastest_results' profiles to allow users to easily tune the base operational parameters (max rounds, base thresholds) before adaptive adjustments.
- **Comprehensive Performance Report**: The final output now includes a detailed report summarizing total rounds, stopping reasons, final sufficiency and convergence scores, and the effective thresholds used during the run.
- **Numerous new valves** added to configure these adaptive behaviors, convergence criteria, and performance profiles.

### v0.3.3 - Adaptive Round Management
- **Smart iteration control**: The system now dynamically adjusts the number of rounds based on task complexity and progress
- **Early termination logic**: Automatically stops when sufficient information is gathered, saving costs
- **Improved efficiency**: Reduces unnecessary API calls while maintaining answer quality

### v0.3.2 - Custom Prompts
- **User-defined prompts**: Added support for custom task instructions and synthesis prompts
- **Enhanced flexibility**: Users can now fine-tune how the supervisor decomposes tasks and synthesizes results
- **Better domain adaptation**: Custom prompts allow optimization for specific document types or query patterns

### v0.3.1 - Task-Specific Instructions and Advanced Prompts
- **Context-aware task generation**: Tasks now include specific instructions based on document content
- **Improved local model guidance**: Better prompting strategies for local models to extract relevant information
- **Enhanced accuracy**: More precise task execution leads to better overall results

### v0.3.0 - Code-Based Task Decomposition
- **Dynamic task generation**: The supervisor now generates Python code to create tasks programmatically
- **Document-aware decomposition**: Tasks are created based on actual document structure and content
- **Scalable approach**: Can handle documents of varying sizes and structures more effectively
- **Improved Minion protocol**: Enhanced conversation flow and better final answer detection

### v0.2.1 - Refactored Architecture
- **Modular design**: Separated concerns into dedicated partials for better maintainability
- **Enhanced error handling**: Improved timeout management and error recovery
- **Better token savings calculation**: More accurate cost estimation

### v0.2.0 - Initial Release
- Basic Minion and MinionS protocol implementation
- Support for Claude and Ollama integration
- Token savings analysis

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The @HazyResearch team for creating the original Minion and MinionS protocols
- The @OpenWebUI community for providing an excellent platform for AI interactions
- The @SunkThought Team: Jules by GoogleLabs, Anthropic's Claude 4 family, and Wil Everts 😇
