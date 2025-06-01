from typing import List, Tuple, Any

# This file will store prompt generation functions for the Minion (single-turn) protocol.

def get_minion_initial_claude_prompt(query: str, context_len: int, valves: Any) -> str:
    """
    Returns the initial prompt for Claude in the Minion protocol.
    """
    prompt_lines = [
        f"""Your primary goal is to answer the user's question: "{query}"

To achieve this, you will collaborate with a local AI assistant. This local assistant has ALREADY READ and has FULL ACCESS to the relevant document ({context_len} characters long). You will not be given the full document. Instead you are to ask the TRUSTED local assistant who can provide you with factual information, summaries, and direct extractions FROM THE DOCUMENT in response to your questions."""
    ]

Your role is to:
1. Ask SPECIFIC, TARGETED questions to extract concrete information from the document
2. Request specific examples, quotes, or detailed descriptions rather than high-level overviews
3. Build up enough specific information to answer the user's query comprehensively

IMPORTANT INSTRUCTIONS:
- DO NOT ask for general overviews or themes - ask for specific content
- Request concrete details: names, dates, technologies, specific advancements
- Ask for direct quotes or detailed descriptions from specific parts
- If you need information from multiple parts, ask for each one specifically

Example good questions:
- "What specific technological advancements are described in Part III?"
- "Can you provide the exact details about the AI winter mentioned in Part II, including dates and causes?"
- "What are the specific building blocks or technologies mentioned in Part V?" """,
        "Start by asking your first SPECIFIC question to begin gathering concrete information."
    ]

    # Add instructions from valves
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        prompt_lines.append(f"\nFollow these specific extraction instructions: {valves.extraction_instructions}")
    if hasattr(valves, 'expected_format') and valves.expected_format:
        prompt_lines.append(f"Please format your response as {valves.expected_format}.")
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0: # Assuming 0 is default/off
        prompt_lines.append(f"Aim for a confidence level of at least {valves.confidence_threshold}.")

    return "\n".join(prompt_lines)

def get_minion_conversation_claude_prompt(history: List[Tuple[str, str]], original_query: str, valves: Any) -> str:
    """
    Returns the prompt for Claude during subsequent conversation rounds in the Minion protocol.
    Moved from _build_conversation_context in minion_protocol_logic.py.
    """
    prompt_lines = [
        f"You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: \"{original_query}\"",
        "The local assistant has full access to the source document and has been providing factual information extracted from it.",
        "",
        "CONVERSATION SO FAR (Your questions, Local Assistant's factual responses from the document):",
    ]

    for role, message in history:
        if role == "assistant":  # Claude's previous message
            prompt_lines.append(f"You previously asked the local assistant: \"{message}\"")
        else:  # Local model's response
            prompt_lines.append(f"The local assistant responded with information from the document: \"{message}\"")

    prompt_lines.append("")

    # Add instructions from valves as a reminder
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        prompt_lines.append(f"Remember to follow these specific extraction instructions: {valves.extraction_instructions}")
    if hasattr(valves, 'expected_format') and valves.expected_format:
        prompt_lines.append(f"Remember to format your response as {valves.expected_format}.")
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0: # Assuming 0 is default/off
        prompt_lines.append(f"Remember to aim for a confidence level of at least {valves.confidence_threshold}.")
    if any([hasattr(valves, attr) and getattr(valves, attr) for attr in ['extraction_instructions', 'expected_format', 'confidence_threshold'] if (attr == 'confidence_threshold' and getattr(valves, attr, 0) > 0) or (attr != 'confidence_threshold' and getattr(valves, attr))]):
        prompt_lines.append("") # Add a newline if any valve instruction was added

    prompt_lines.extend(
        [
            "REMINDER: The local assistant's responses are factual information extracted directly from the document.",
            "Based on ALL information provided by the local assistant so far, can you now provide a complete and comprehensive answer to the user's ORIGINAL QUESTION?",
            "If YES: Respond ONLY with the exact phrase 'FINAL ANSWER READY.' followed by your comprehensive final answer. Ensure your answer directly addresses the original query using the information gathered.",
            "If NO: Ask ONE more specific, targeted question to the local assistant to obtain the remaining information you need from the document. Be precise. Do not ask for the document itself or express that you cannot see it.",
        ]
    )
    return "\n".join(prompt_lines)

def get_minion_local_prompt(context: str, query: str, claude_request: str, valves: Any) -> str:
    """
    Returns the prompt for the local Ollama model in the Minion protocol.
    """
    instruction_lines = [
        "- Provide SPECIFIC, DETAILED information from the context",
        "- Include concrete examples, names, dates, and technical details",
        "- Quote directly from the text when relevant",
        "- If Claude asks about a specific part or section, provide detailed content from that section",
        "- Do not give vague overviews - provide substantial, specific information"
    ]

    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        instruction_lines.append(f"- Follow these specific extraction instructions: {valves.extraction_instructions}")

    if hasattr(valves, 'expected_format') and valves.expected_format:
        instruction_lines.append(f"- Please format your response as {valves.expected_format}.")
        if valves.expected_format.lower() == "json" and hasattr(valves, 'use_structured_output') and valves.use_structured_output:
            instruction_lines.append("- Your response MUST be a valid JSON object. If a schema is provided, adhere to it strictly.")
            instruction_lines.append("- Ensure your entire response is ONLY the JSON object, without any surrounding text, explanations, or markdown formatting like ```json ... ```.")


    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0: # Assuming 0 is default/off
        instruction_lines.append(f"- Aim for a confidence level of at least {valves.confidence_threshold} in your understanding and extraction, though your output format will be guided by other instructions.")

    important_instructions = "\n".join(instruction_lines)

    return f"""You have access to the full context below. Claude (Anthropic's AI) is collaborating with you to answer a user's question.

CONTEXT:
{context}

ORIGINAL QUESTION: {query}

CLAUDE'S REQUEST: {claude_request}

IMPORTANT INSTRUCTIONS:
{important_instructions}

Please provide a helpful, accurate response based on the context you have access to. Extract relevant information that answers Claude's specific question. Be thorough and include all relevant details.

If you are instructed to provide a JSON response (e.g., by a schema appended to this prompt or by the 'expected_format' instruction), ensure your entire response is ONLY that valid JSON object, without any surrounding text, explanations, or markdown formatting like ```json ... ```."""
