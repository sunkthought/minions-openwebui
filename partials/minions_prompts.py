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
IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any text before or after the JSON.

Required JSON structure:
{{
    "explanation": "Brief explanation of your findings for this task",
    "citation": "Direct quote from the text if applicable to this task, or null",
    "answer": "Your complete answer to the task as a SINGLE STRING",
    "confidence": "HIGH, MEDIUM, or LOW"
}}''')

        structured_output_rules = [
            "\nCRITICAL RULES FOR JSON OUTPUT:",
            "1. Output ONLY the JSON object - no markdown formatting, no explanatory text, no code blocks",
            "2. The \"answer\" field MUST be a plain text string, NOT an object or array",
            "3. If listing multiple items, format as a single string (e.g., \"Item 1: Description. Item 2: Description.\")",
            "4. Use proper JSON escaping for quotes within strings (\\\" for quotes inside string values)",
            "5. If information is not found, set \"answer\" to null and \"confidence\" to \"LOW\"",
            "6. The \"confidence\" field must be exactly one of: \"HIGH\", \"MEDIUM\", or \"LOW\"",
            "7. All string values must be properly quoted and escaped"
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
\nEXAMPLES OF CORRECT JSON OUTPUT:

Example 1 - Information found:
{{
    "explanation": "Found budget information in the financial section",
    "citation": "The total project budget is set at $2.5 million for fiscal year 2024",
    "answer": "$2.5 million",
    "confidence": "HIGH"
}}

Example 2 - Information NOT found:
{{
    "explanation": "Searched for revenue projections but this chunk only contains expense data",
    "citation": null,
    "answer": null,
    "confidence": "LOW"
}}

Example 3 - Multiple items found:
{{
    "explanation": "Identified three risk factors mentioned in the document",
    "citation": "Key risks include: market volatility, regulatory changes, and supply chain disruptions",
    "answer": "1. Market volatility 2. Regulatory changes 3. Supply chain disruptions",
    "confidence": "HIGH"
}}''')
        
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() == "bullet points":
            prompt_lines.append(f'''
\nExample with bullet points in answer field:
{{
    "explanation": "Found multiple implementation steps",
    "citation": "The implementation plan consists of three phases...",
    "answer": "- Phase 1: Initial setup and configuration\\n- Phase 2: Testing and validation\\n- Phase 3: Full deployment",
    "confidence": "MEDIUM"
}}''')

        prompt_lines.append(f'''
\nEXAMPLES OF INCORRECT OUTPUT (DO NOT DO THIS):

Wrong - Wrapped in markdown:
```json
{{"answer": "some value"}}
```

Wrong - Answer is not a string:
{{
    "answer": {{"key": "value"}},
    "confidence": "HIGH"
}}

Wrong - Missing required fields:
{{
    "answer": "some value"
}}

Wrong - Text outside JSON:
Here is my response:
{{"answer": "some value"}}''')

    else: # Not using structured output
        prompt_lines.append("\n\nProvide a brief, specific answer based ONLY on the text provided above.")
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() != "text":
            prompt_lines.append(f"Format your entire response as {valves.expected_format.upper()}.")
        prompt_lines.append("If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\".")
    
    return "\n".join(prompt_lines)