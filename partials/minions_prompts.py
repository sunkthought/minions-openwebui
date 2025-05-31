from typing import List, Any, Optional

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
