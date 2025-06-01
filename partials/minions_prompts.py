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
        # Enhanced instructions for structured output
        prompt += f'''

IMPORTANT: Provide your answer as a valid JSON object with the following structure:
{{
    "explanation": "Brief explanation of your findings",
    "citation": "Direct quote from the text if applicable, or null",
    "answer": "Your complete answer as a SINGLE STRING"
}}

CRITICAL RULES:
1. The "answer" field MUST be a plain text string, NOT an object or array
2. If you need to list multiple items, format them as a single string with clear separators (e.g., "Item 1: Description. Item 2: Description.")
3. Do NOT create nested JSON structures within any field
4. If no relevant information is found, set "answer" to null

EXAMPLE of CORRECT format:
{{
    "explanation": "Found information about Parts I and II in the text",
    "citation": "Part I discusses foundations...",
    "answer": "Part I: Foundations of AI including Turing's work. Part II: Early AI systems like ELIZA."
}}

EXAMPLE of INCORRECT format (DO NOT DO THIS):
{{
    "answer": {{"Part I": "...", "Part II": "..."}}  // WRONG - answer must be a string!
}}'''
    else:
        prompt += "\n\nProvide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\"."
    
    return prompt