# Partials File: partials/minion_prompts.py
from typing import List, Tuple, Any

# This file will store prompt generation functions for the Minion (single-turn) protocol.

def get_minion_initial_claude_prompt(query: str, context_len: int, valves: Any) -> str:
    """
    Returns the initial prompt for Claude in the Minion protocol.
    Moved from _execute_minion_protocol in minion_protocol_logic.py.
    """
    # Escape any quotes in the query to prevent f-string issues
    escaped_query = query.replace('"', '\\"').replace("'", "\\'")
    
    return f'''Your primary goal is to answer the user's question: "{escaped_query}"

To achieve this, you will collaborate with a local AI assistant. This local assistant has ALREADY READ and has FULL ACCESS to the relevant document ({context_len} characters long). The local assistant is a TRUSTED source that will provide you with factual information, summaries, and direct extractions FROM THE DOCUMENT in response to your questions.

Your role is to:
1.  Formulate specific, focused questions to the local assistant to gather the necessary information from the document. Ask only what you need to build up the answer to the user's original query.
2.  Receive and understand the information provided by the local assistant.
3.  Synthesize this information to answer the user's original query.

IMPORTANT INSTRUCTIONS:
- DO NOT ask the local assistant to provide the entire document or large raw excerpts.
- DO NOT express that you cannot see the document. Assume the local assistant provides accurate information from it.
- Your questions should be aimed at extracting pieces of information that you can then synthesize.

If, after receiving responses from the local assistant, you believe you have gathered enough information to comprehensively answer the user's original query ("{escaped_query}"), then respond ONLY with the exact phrase "FINAL ANSWER READY." followed by your detailed final answer.
If you need more specific information from the document, ask the local assistant ONE more clear, targeted question. Do not use the phrase "FINAL ANSWER READY." yet.

Start by asking your first question to the local assistant to begin gathering information.
'''

def get_minion_conversation_claude_prompt(history: List[Tuple[str, str]], original_query: str, valves: Any) -> str:
    """
    Returns the prompt for Claude during subsequent conversation rounds in the Minion protocol.
    Moved from _build_conversation_context in minion_protocol_logic.py.
    """
    # Escape the original query
    escaped_query = original_query.replace('"', '\\"').replace("'", "\\'")
    
    context_parts = [
        f'You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user\'s ORIGINAL QUESTION: "{escaped_query}"',
        "The local assistant has full access to the source document and has been providing factual information extracted from it.",
        "",
        "CONVERSATION SO FAR (Your questions, Local Assistant's factual responses from the document):",
    ]

    for role, message in history:
        if role == "assistant":  # Claude's previous message
            context_parts.append(f'You previously asked the local assistant: "{message}"')
        else:  # Local model's response
            context_parts.append(f'The local assistant responded with information from the document: "{message}"')

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
    # query is the original user query.
    # context is the document chunk.
    # claude_request (the parameter) is the specific question from the remote model to the local model.

    return f"""You are an AI assistant. You have access to the following DOCUMENT:
<document>
{context}
</document>

The remote model (another AI) is asking you a specific question about this document. The remote model's question is:
<remote_model_question>
{claude_request}
</remote_model_question>

Your task is to answer the remote model's question based *only* on the DOCUMENT provided.

Your response MUST be a single JSON object. Do not include any text, explanations, or markdown formatting (like ```json ... ```) outside of this JSON object.

The JSON object must have the following keys:
- "explanation": A concise statement of your reasoning or how you concluded your answer.
- "citation": A direct snippet of the text from the DOCUMENT that supports your answer. If no supporting text is found in the DOCUMENT, this field must be null.
- "answer": The extracted answer to the remote model's question. If the answer cannot be determined from the DOCUMENT, this field must be null.

IMPORTANT: If you cannot confidently determine the information from the DOCUMENT to answer the remote model's question, ALL THREE fields ("explanation", "citation", "answer") in the JSON object must be null.

Provide only the JSON object in your response.
"""