# Collaborative AI with Minion and MinionS for Open WebUI

This repository provides an implementation of the Minion and MinionS protocols, designed by HazyResearch, tailored for use as **Functions** within the [Open WebUI](https://github.com/open-webui/open-webui) platform. These protocols enable sophisticated collaboration between local AI models (e.g., Ollama) and powerful remote AI models (e.g., Anthropic's Claude) to tackle complex tasks.

## Understanding Minion and MinionS: Collaborative AI Agents

MinionS (and its predecessor, Minion) is a protocol designed by HazyResearch to enable sophisticated collaboration between different AI models, often a smaller, efficient local model and a larger, more powerful remote model. The core idea is to create a "team" of AI agents that can work together to solve complex queries or perform tasks that might be too large or inefficient for a single model to handle alone.

Key concepts include:

*   **Task Decomposition (MinionS)**: A complex user query is broken down into smaller, manageable sub-tasks by a capable remote "supervisor" model.
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
1.  Navigate to **Settings** (usually a gear icon in the sidebar).
2.  Under the "Admin" section, click on **Functions**.
3.  Click the **+ Add Function** button.
4.  **Set a Title**: Give your function a descriptive name, for example:
    *   `Minion Protocol - Claude/Ollama`
    *   `MinionS Protocol - Claude/Ollama`
5.  **Paste Function Code**:
    *   In this repository (`SunkThought/minions-openwebui`), navigate to the `generated_functions/` directory.
    *   Open the desired default function file:
        *   For the Minion protocol: `minion_default_function.py`
        *   For the MinionS protocol: `minions_default_function.py`
    *   Copy the **entire content** of this file.
    *   Paste it into the large text area labeled "Function Code" in Open WebUI.
6.  Click **Save** (or "Create Function").

### 3. Configure Function Valves

"Valves" are the settings you can adjust for the Minion/MinionS function each time you use it in Open WebUI. After adding the function and selecting it for a model, these settings will appear in the right-hand panel under "Function Calling" > "Valves".

Key valves to configure:

*   `anthropic_api_key`: **Required**. Your API key for Anthropic Claude.
*   `remote_model`: The Claude model you wish to use (e.g., `claude-3-5-sonnet-20240620`, `claude-3-haiku-20240307`).
*   `ollama_base_url`: The base URL of your Ollama server. Default is `http://localhost:11434`.
*   `local_model`: The name of the local Ollama model you want to use (e.g., `llama3.1`, `mistral`). Ensure this model is available in your Ollama instance (e.g., via `ollama pull llama3.1`).
*   `max_rounds` (Minion/MinionS): Maximum number of conversation rounds (Minion) or decomposition/execution rounds (MinionS).
*   `show_conversation` (Minion/MinionS): Set to `true` to see the detailed interaction log between models in the output, or `false` to only see the final answer. This is very useful for debugging and understanding the process.
*   `timeout_local`, `timeout_claude`: Timeouts in seconds for API calls to the local and remote models, respectively.

**⚠️ Important: `ollama_base_url` for Docker Users**

If you are running Open WebUI in a Docker container (which is common) and your Ollama service is running on your host machine (i.e., *not* inside another Docker container on the same Docker network):
*   The default `http://localhost:11434` for `ollama_base_url` **will not work** from inside the Open WebUI container. This is because `localhost` inside the container refers to the container itself, not your host machine.
*   **Solutions**:
    *   **Docker Desktop (Mac/Windows/Linux with Host Networking Support)**: Try `http://host.docker.internal:11434`. This special DNS name resolves to the host's IP address.
    *   **Linux Host (Bridge Network)**: You might need to use the actual IP address of your host machine on your local network (e.g., `http://192.168.1.100:11434`). You can find this IP using commands like `ifconfig` or `ip addr show docker0` (for the Docker bridge).
    *   **Ollama in Docker**: If Ollama is also running as a Docker container, ensure both Open WebUI and Ollama containers are on the same custom Docker network. You can then use the Ollama container's name as the hostname (e.g., `http://ollama:11434`, assuming your Ollama container is named `ollama`).
*   Refer to Docker networking documentation or Open WebUI's troubleshooting guides for more comprehensive help.

### 4. Run the Function in Open WebUI

1.  Start or select a chat in Open WebUI.
2.  In the chat settings (often accessible via a model selection dropdown or a settings icon near the model name):
    *   Choose a model that will act as the "entry point" or "main model" for the chat. This model's own capabilities are less important when a function is active, as the function dictates the primary logic.
    *   Enable **Function calling**.
    *   From the dropdown list of available functions, select the Minion or MinionS function you added (e.g., "Minion Protocol - Claude/Ollama").
3.  Once the function is selected, its configurable **Valves** should appear in the UI (typically in the right-hand panel). Adjust them as needed (e.g., paste your API key, verify model names).
4.  Type your query in the chat input. If you are providing context (like a document paste), do so along with your query. If you have uploaded files, the function will attempt to use them if you reference them or if your query implies their use.
5.  Send the message. The Minion/MinionS protocol will execute, and you should see the results in the chat. If `show_conversation` is true, you'll see a detailed log; otherwise, just the final answer.

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
    1.  **Decomposition**: The remote supervisor model (e.g., Claude) first analyzes the user's query and the overall context. It then breaks down the main query into several smaller, independent sub-tasks.
    2.  **Chunking & Task Execution**: The provided context is divided into smaller chunks. Each sub-task is then typically executed by local models (e.g., Ollama models) on each relevant chunk of the document. This step gathers information related to each sub-task from across the document.
    3.  **Aggregation & Synthesis**: The results from all sub-tasks across all chunks are collected. This aggregated information is then sent back to the remote supervisor model, which synthesizes it into a comprehensive final answer to the original query. The supervisor might also initiate further rounds of decomposition and execution if the initial results are insufficient.
*   **Strengths**:
    *   Excellent for broad analysis of large documents or complex queries that can be broken down into parallelizable sub-parts.
    *   Can be significantly more efficient for tasks that benefit from parallel processing of information across different document segments.
    *   Handles multifaceted queries well (e.g., "Summarize this document, list all key personnel mentioned, and identify potential future risks discussed.").
    *   Reduces the amount of data sent to the expensive remote model by processing chunks locally.
*   **Best Suited For**:
    *   Comprehensive analysis of one or more large documents where multiple pieces of information need to be found and correlated.
    *   Queries that require identifying and correlating different types of information from various parts of a text (e.g., "Find all mentions of project Alpha, who was involved, and what were the outcomes from each phase described in this report.").
    *   Situations where the initial query is broad and needs a structured breakdown to be effectively addressed (e.g., "Tell me everything important about this lengthy project proposal.").
    *   When you want to apply a consistent set of questions/tasks to multiple document chunks.

### Quick Comparison

| Feature          | Minion                                     | MinionS                                            |
|------------------|--------------------------------------------|----------------------------------------------------|
| **Primary Method** | Iterative Conversational Q&A             | Task Decomposition, Chunk-based Local Execution, Synthesis |
| **Complexity**   | Simpler, direct interaction flow           | More complex, multi-step orchestration             |
| **Context Handling**| Local model sees full context per query  | Local models see chunks of context per sub-task    |
| **Best For**     | Focused Q&A, iterative context refinement | Broad analysis, multifaceted queries, large docs   |

By understanding these differences, you can choose the protocol that best fits the complexity and nature of your task when using these functions in Open WebUI.

## Advanced: Configuration and Custom Function Generation

The Minion and MinionS functions in this repository are not static; they are dynamically generated from modular code pieces called "partials." This design allows for flexibility and customization.

### Understanding the Components

*   **`partials/` Directory**:
    This directory is the heart of the modular system. It contains numerous Python files (`.py`), each representing a "partial" piece of the final function's code (e.g., API call logic, prompt construction, protocol execution steps, valve definitions). Advanced users can modify these partials or even create new ones to alter functionality.

*   **`generation_config.json`**:
    This JSON file defines "profiles" for generating functions. Each profile acts as a blueprint for a specific function variant. Key fields in a profile include:
    *   `description`: A human-readable description of the profile.
    *   `output_filename_template`: A template for naming the generated Python file (e.g., `{profile_name}_function.py`).
    *   `header_placeholders`: Allows you to set the metadata (title, version, description, etc.) that appears in the comment block at the top of the generated function file. This metadata is often displayed by Open WebUI in the function management interface.
    *   `partials_concat_order`: This is a critical list that specifies which files from the `partials/` directory are included and the exact order in which they will be concatenated to form the complete function code for that profile. The order defines the Python code structure and how dependencies are resolved, so it must be logical.

*   **`generator_script.py`**:
    This Python script is the tool that builds the functions. It takes a function type (`minion` or `minions`) and an optional profile name (e.g., `--profile my_custom_profile`) as input. It then reads the specified profile from `generation_config.json` and assembles the partials in the defined order, outputting a single, runnable Python file compatible with Open WebUI's function system into the `generated_functions/` directory.

### Generating Default Functions

As mentioned in the Quick Start, you can generate the standard versions of Minion and MinionS using:

```bash
# Generates minion_default_function.py using the 'minion_default' profile
python generator_script.py minion

# Generates minions_default_function.py using the 'minions_default' profile
python generator_script.py minions
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
