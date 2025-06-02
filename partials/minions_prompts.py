# Partials File: partials/minions_prompts.py
from typing import List, Any, Optional

# This file will store prompt generation functions for the MinionS (multi-turn, multi-task) protocol.

def get_minions_synthesis_claude_prompt(query: str, synthesis_input_summary: str, valves: Any) -> str:
    """
    Returns the synthesis prompt for Claude in the MinionS protocol.
    Logic moved from _execute_minions_protocol in minions_pipe_method.py.
    'synthesis_input_summary' is the aggregation of successful task results.
    """
    prompt_lines = [
        f'''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary if synthesis_input_summary else "No specific information was extracted by local models."}
'''
    ]

    # Add instructions from valves for synthesis guidance
    synthesis_instructions = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        synthesis_instructions.append(f"When synthesizing the final answer, consider these overall instructions: {valves.extraction_instructions}")
    if hasattr(valves, 'expected_format') and valves.expected_format:
        synthesis_instructions.append(f"Format the final synthesized answer as {valves.expected_format}.")
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0: # Assuming 0 is default/off
        synthesis_instructions.append(f"Aim for an overall confidence level of at least {valves.confidence_threshold} in your synthesized answer.")

    if synthesis_instructions:
        prompt_lines.append("\nSYNTHESIS GUIDELINES:")
        prompt_lines.extend(synthesis_instructions)
        prompt_lines.append("") # Add a newline for separation

    prompt_lines.append("If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.")
    prompt_lines.append("Final Answer:")
    return "\n".join(prompt_lines)

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
    prompt_lines = [
        f'''Text to analyze (Chunk {chunk_idx + 1}/{total_chunks} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}'''
    ]

    task_specific_instructions = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        task_specific_instructions.append(f"- Follow these specific extraction instructions: {valves.extraction_instructions}")

    # Confidence threshold is a general guideline for the task.
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0:
        task_specific_instructions.append(f"- Aim for a confidence level of at least {valves.confidence_threshold} in your findings for this task.")

    if task_specific_instructions:
        prompt_lines.append("\nTASK-SPECIFIC INSTRUCTIONS:")
        prompt_lines.extend(task_specific_instructions)

    if valves.use_structured_output:
        prompt_lines.append(f'''\n
IMPORTANT: Provide your answer as a valid JSON object with the following structure:
{{
    "explanation": "Brief explanation of your findings for this task",
    "citation": "Direct quote from the text if applicable to this task, or null",
    "answer": "Your complete answer to the task as a SINGLE STRING"
}}''')

        structured_output_rules = [
            "CRITICAL RULES FOR JSON:",
            "1. The \"answer\" field MUST be a plain text string, NOT an object or array.",
            "2. If you need to list multiple items in the \"answer\" field, format them as a single string with clear separators (e.g., \"Item 1: Description. Item 2: Description.\").",
            "3. Do NOT create nested JSON structures within any field.",
            "4. If you cannot confidently determine the information from the provided text to answer the task, ALL THREE fields (\"explanation\", \"citation\", \"answer\") in the JSON object must be null."
        ]
        prompt_lines.extend(structured_output_rules)

        # Rule 5 regarding expected_format needs to be renumbered if it was part of the list,
        # but it's added conditionally after extending structured_output_rules.
        # So, its conditional addition logic remains correct without renumbering.

        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() != "json":
            prompt_lines.append(f"5. Format the content WITHIN the \"answer\" field as {valves.expected_format.upper()}. For example, if \"bullet points\", the \"answer\" string should look like \"- Point 1\\n- Point 2\".")
        elif hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() == "json":
             prompt_lines.append("5. The overall response is already JSON. Ensure the content of the 'answer' field is a simple string, not further JSON encoded, unless the task specifically asks for a JSON string as the answer.")


        prompt_lines.append(f'''
EXAMPLE of CORRECT format:
{{
    "explanation": "Found information about X in the text for the task",
    "citation": "The text states 'X is Y'...",
    "answer": "X is Y according to the document."
}}''')
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() == "bullet points":
            prompt_lines.append(f'''
EXAMPLE for "answer" field formatted as BULLET POINTS:
{{
    "explanation": "Found several points for the task.",
    "citation": "Relevant quote...",
    "answer": "- Point 1: Details about point 1.\\n- Point 2: Details about point 2."
}}''')

        prompt_lines.append(f'''
EXAMPLE of INCORRECT format (DO NOT DO THIS):
{{
    "answer": {{"key": "value"}}  // WRONG - "answer" field must be a string!
}}''')

    else: # Not using structured output
        prompt_lines.append("\n\nProvide a brief, specific answer based ONLY on the text provided above.")
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() != "text":
            prompt_lines.append(f"Format your entire response as {valves.expected_format.upper()}.")
        prompt_lines.append("If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\".")
    
    return "\n".join(prompt_lines)