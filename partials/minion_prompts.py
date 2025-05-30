from typing import List, Tuple, Any

# This file will store prompt generation functions for the Minion (single-turn) protocol.

def get_minion_initial_claude_prompt(query: str, context_len: int, valves: Any) -> str:
    """
    Returns the initial prompt for Claude in the Minion protocol.
    Moved from _execute_minion_protocol in minion_protocol_logic.py.
    """
    # It seems 'valves' is not strictly needed for this specific prompt's text content
    # but keeping it in signature for consistency if other valve-dependent variants arise.
    return f"""Your primary goal is to answer the user's question: "{query}"

To achieve this, you will collaborate with a local AI assistant. This local assistant has ALREADY READ and has FULL ACCESS to the relevant document ({context_len} characters long). The local assistant is a TRUSTED source that will provide you with factual information, summaries, and direct extractions FROM THE DOCUMENT in response to your questions.

Your role is to:
1.  Formulate specific, focused questions to the local assistant to gather the necessary information from the document. Ask only what you need to build up the answer to the user's original query.
2.  Receive and understand the information provided by the local assistant.
3.  Synthesize this information to answer the user's original query.

IMPORTANT INSTRUCTIONS:
- DO NOT ask the local assistant to provide the entire document or large raw excerpts.
- DO NOT express that you cannot see the document. Assume the local assistant provides accurate information from it.
- Your questions should be aimed at extracting pieces of information that you can then synthesize.

If, after receiving responses from the local assistant, you believe you have gathered enough information to comprehensively answer the user's original query ("{query}"), then respond ONLY with the exact phrase "FINAL ANSWER READY." followed by your detailed final answer.
If you need more specific information from the document, ask the local assistant ONE more clear, targeted question. Do not use the phrase "FINAL ANSWER READY." yet.

Start by asking your first question to the local assistant to begin gathering information.
"""

def get_minion_conversation_claude_prompt(history: List[Tuple[str, str]], original_query: str, valves: Any) -> str:
    """
    Returns the prompt for Claude during subsequent conversation rounds in the Minion protocol.
    Moved from _build_conversation_context in minion_protocol_logic.py.
    """
    # 'valves' not strictly needed here based on original _build_conversation_context
    context_parts = [
        f"You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: \"{original_query}\"",
        "The local assistant has full access to the source document and has been providing factual information extracted from it.",
        "",
        "CONVERSATION SO FAR (Your questions, Local Assistant's factual responses from the document):",
    ]

    for role, message in history:
        if role == "assistant":  # Claude's previous message
            context_parts.append(f"You previously asked the local assistant: \"{message}\"")
        else:  # Local model's response
            context_parts.append(f"The local assistant responded with information from the document: \"{message}\"")

    context_parts.extend(
        [
            "",
            "REMINDER: The local assistant's responses are factual information extracted directly from the document.",
            "Based on ALL information provided by the local assistant so far, can you now provide a complete and comprehensive answer to the user's ORIGINAL QUESTION?",
            "If YES: Respond ONLY with the exact phrase 'FINAL ANSWER READY.' followed by your comprehensive final answer. Ensure your answer directly addresses the original query using the information gathered.",
            "If NO: Ask ONE more specific, targeted question to the local assistant to obtain the remaining information you need from the document. Be precise. Do not ask for the document itself or express that you cannot see it.",
        ]
    )
    return "\n".join(context_parts)

def get_minion_local_prompt(context: str, query: str, claude_request: str, valves: Any) -> str:
    """
    Returns the prompt for the local Ollama model in the Minion protocol.
    Moved from _execute_minion_protocol in minion_protocol_logic.py.
    """
    # 'valves' not strictly needed for this specific prompt's text content from original version
    # but could be used for schema instructions if valves.use_structured_output was considered here.
    # The original prompt did include a generic instruction about JSON.
    return f"""You have access to the full context below. Claude (Anthropic's AI) is collaborating with you to answer a user's question.
CONTEXT:
{context}
ORIGINAL QUESTION: {query}
CLAUDE'S REQUEST: {claude_request}
Please provide a helpful, accurate response based on the context you have access to. Extract relevant information that answers Claude's specific question. Be concise but thorough.
If you are instructed to provide a JSON response (e.g., by a schema appended to this prompt), ensure your entire response is ONLY that valid JSON object, without any surrounding text, explanations, or markdown formatting like ```json ... ```."""
