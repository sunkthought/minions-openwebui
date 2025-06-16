# Partials File: partials/minions_prompts.py
from typing import List, Any, Optional
from .prompt_templates import PromptTemplates

# This file will store prompt generation functions for the MinionS (multi-turn, multi-task) protocol.

def get_minions_synthesis_claude_prompt(query: str, synthesis_input_summary: str, valves: Any) -> str:
    """
    Returns the synthesis prompt for Claude in the MinionS protocol.
    Logic moved from _execute_minions_protocol in minions_pipe_method.py.
    'synthesis_input_summary' is the aggregation of successful task results.
    """
    # Build synthesis guidelines from valves
    guidelines = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        guidelines.append(f"When synthesizing the final answer, consider these overall instructions: {valves.extraction_instructions}")
    if hasattr(valves, 'expected_format') and valves.expected_format:
        guidelines.append(f"Format the final synthesized answer as {valves.expected_format}.")
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0:
        guidelines.append(f"Aim for an overall confidence level of at least {valves.confidence_threshold} in your synthesized answer.")

    return PromptTemplates.get_minions_synthesis_claude_prompt(
        query=query,
        synthesis_input_summary=synthesis_input_summary,
        guidelines=guidelines
    )

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
    # Build task-specific instructions from valves
    task_instructions = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        task_instructions.append(f"Follow these specific extraction instructions: {valves.extraction_instructions}")
    
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0:
        task_instructions.append(f"Aim for a confidence level of at least {valves.confidence_threshold} in your findings for this task.")

    expected_format = getattr(valves, 'expected_format', None)
    
    return PromptTemplates.get_minions_local_task_prompt(
        chunk=chunk,
        task=task,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        use_structured_output=valves.use_structured_output,
        task_instructions=task_instructions,
        expected_format=expected_format
    )