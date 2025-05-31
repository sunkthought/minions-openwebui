from typing import List, Any, Optional
from .minions_models import JobManifest # Added import

# This file will store prompt generation functions for the MinionS (multi-turn, multi-task) protocol.

def get_minions_synthesis_claude_prompt(query: str, synthesis_input_summary: str, valves: Any) -> str:
    """
    Returns the synthesis prompt for Claude in the MinionS protocol.
    Logic moved from _execute_minions_protocol in minions_pipe_method.py.
    'synthesis_input_summary' is the aggregation of successful task results.
    """
    # valves might be used for model name or other minor adjustments in future, kept for consistency.
    return f'''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary if synthesis_input_summary else "No specific information was extracted by local models."}

If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.
Final Answer:'''

def get_minions_local_task_prompt(
    chunk: str, 
    task: str, 
    chunk_idx: int, 
    total_chunks: int, 
    valves: Any, 
    # schema_json: Optional[str] = None # Not directly used as schema is handled by call_ollama, but prompt notes structure.
) -> str:
    """
    Returns the prompt for the local Ollama model for a specific task on a chunk 
    in the MinionS protocol.
    Logic moved from execute_tasks_on_chunks in minions_protocol_logic.py.
    """
    prompt = f'''Text to analyze (Chunk {chunk_idx + 1}/{total_chunks} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}'''

    if valves.use_structured_output:
        # The schema_json parameter was considered, but the actual schema is passed to call_ollama,
        # so the prompt only needs to instruct about JSON format generally if structured output is used.
        prompt += f"\n\nProvide your answer ONLY as a valid JSON object matching the specified schema. If no relevant information is found in THIS SPECIFIC TEXT, ensure the 'answer' field in your JSON response is explicitly set to null (or None)."
    else:
        prompt += "\n\nProvide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\"."
    
    return prompt

def get_minions_code_generation_claude_prompt(query: str, context_summary: str, valves: Any, job_manifest_model_name: str) -> str:
    """
    Generates a prompt for Claude to create Python code that defines a list of JobManifest objects.
    """
    # Note: The example in the prompt uses JobManifest(...) directly.
    # The job_manifest_model_name variable is used to dynamically refer to the class name in the prompt's instructions.
    prompt = f'''You are an expert task decomposition agent. Your goal is to help answer the user's query: "{query}"
Based on the available context (summarized as: {context_summary}), generate a Python list of `{job_manifest_model_name}` objects to break down the query into specific, actionable tasks.

You must follow these guidelines:
- The output must be a Python list of `{job_manifest_model_name}` objects. Ensure you import `{job_manifest_model_name}` from `minions_models` if your execution environment requires it for the generated code.
- Each task must have a unique `task_id` (e.g., "task_1", "task_2", ...).
- Each task should be simple and focused (e.g., "Extract the revenue for Q3 2023").
- Tasks should NOT require multiple steps or complex reasoning.
- You can assign a task to a specific chunk by setting `chunk_id` (0-indexed integer). If `chunk_id` is omitted or set to `None`, the task may be applied to all relevant chunks or by a system that assigns it globally.
- Use the `advice` field in each `{job_manifest_model_name}` to provide hints to the local model for executing the task. This is crucial for guiding the local model effectively.
- Generate no more than {valves.max_tasks_per_round} unique tasks in this list.
- Ensure the generated Python code is a valid list of `{job_manifest_model_name}` objects. Do not include any other text or explanation outside the Python code block.

Example of a single `{job_manifest_model_name}` instance:
`{job_manifest_model_name}(task_id="unique_task_id", task_description="Describe the task here.", advice="Optional advice for the local model.", chunk_id=None)`

Begin your Python code block now. The code should start with `[` and end with `]`.
```python
[
    # {job_manifest_model_name}(task_id="task_1", task_description="Extract the CEO's name.", advice="Look for titles like 'Chief Executive Officer'."),
    # {job_manifest_model_name}(task_id="task_2", task_description="What was the total revenue in 2023?", chunk_id=0, advice="Focus on financial statements if available in this chunk.")
]
```
'''
    return prompt
