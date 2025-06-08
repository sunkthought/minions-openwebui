# Partials File: partials/minion_prompts.py
from typing import List, Tuple, Any

# This file will store prompt generation functions for the Minion (single-turn) protocol.

def get_minion_initial_claude_prompt(query: str, context_len: int, valves: Any) -> str:
    """
    Returns the initial prompt for Claude in the Minion protocol.
    Enhanced with better question generation guidance.
    """
    # Escape any quotes in the query to prevent f-string issues
    escaped_query = query.replace('"', '\\"').replace("'", "\\'")
    
    return f'''You are a research coordinator working with a knowledgeable local assistant who has access to specific documents.

Your task: Gather information to answer the user's query by asking strategic questions.

USER'S QUERY: "{escaped_query}"

The local assistant has FULL ACCESS to the relevant document ({context_len} characters long) and will provide factual information extracted from it.

Guidelines for effective questions:
1. Ask ONE specific, focused question at a time
2. Build upon previous answers to go deeper
3. Avoid broad questions like "What does the document say?" 
4. Good: "What are the specific budget allocations for Q2?"
   Poor: "Tell me about the budget"
5. Track what you've learned to avoid redundancy

When to conclude:
- Start your response with "I now have sufficient information" when ready to provide the final answer
- You have {valves.max_rounds} rounds maximum to gather information

QUESTION STRATEGY TIPS:
- For factual queries: Ask for specific data points, dates, numbers, or names
- For analytical queries: Ask about relationships, comparisons, or patterns
- For summary queries: Ask about key themes, main points, or conclusions
- For procedural queries: Ask about steps, sequences, or requirements

Remember: The assistant can only see the document, not your conversation history.

If you have gathered enough information to answer "{escaped_query}", respond with "FINAL ANSWER READY." followed by your comprehensive answer.

Otherwise, ask your first strategic question to the local assistant.'''

def get_minion_conversation_claude_prompt(history: List[Tuple[str, str]], original_query: str, valves: Any) -> str:
    """
    Returns the prompt for Claude during subsequent conversation rounds in the Minion protocol.
    Enhanced with better guidance for follow-up questions.
    """
    # Escape the original query
    escaped_query = original_query.replace('"', '\\"').replace("'", "\\'")
    
    current_round = len(history) // 2 + 1
    rounds_remaining = valves.max_rounds - current_round
    
    context_parts = [
        f'You are continuing to gather information to answer: "{escaped_query}"',
        f"Round {current_round} of {valves.max_rounds}",
        "",
        "INFORMATION GATHERED SO FAR:",
    ]

    for i, (role, message) in enumerate(history):
        if role == "assistant":  # Claude's previous message
            context_parts.append(f'\nQ{i//2 + 1}: {message}')
        else:  # Local model's response
            # Extract key information if structured
            if isinstance(message, str) and message.startswith('{'):
                context_parts.append(f'A{i//2 + 1}: {message}')
            else:
                context_parts.append(f'A{i//2 + 1}: {message}')

    context_parts.extend(
        [
            "",
            "DECISION POINT:",
            "Evaluate if you have sufficient information to answer the original question comprehensively.",
            "",
            "✅ If YES: Start with 'FINAL ANSWER READY.' then provide your complete answer",
            f"❓ If NO: Ask ONE more strategic question (you have {rounds_remaining} rounds left)",
            "",
            "TIPS FOR YOUR NEXT QUESTION:",
            "- What specific gaps remain in your understanding?",
            "- Can you drill deeper into any mentioned topics?",
            "- Are there related aspects you haven't explored?",
            "- Would examples or specific details strengthen your answer?",
            "",
            "Remember: Each question should build on what you've learned, not repeat previous inquiries.",
        ]
    )
    return "\n".join(context_parts)

def get_minion_local_prompt(context: str, query: str, claude_request: str, valves: Any) -> str:
    """
    Returns the prompt for the local Ollama model in the Minion protocol.
    Updated to support structured output with LocalAssistantResponse model.
    """
    # query is the original user query.
    # context is the document chunk.
    # claude_request (the parameter) is the specific question from the remote model to the local model.

    base_prompt = f"""You are a helpful assistant with access to the following document:
<document>
{context}
</document>

The remote model (another AI) is asking you a specific question about this document. The remote model's question is:
<remote_model_question>
{claude_request}
</remote_model_question>

Instructions:
1. Answer the specific question asked based ONLY on the document
2. Cite relevant passages or sections when possible
3. If multiple pieces of information are relevant, organize them as key points
4. If information is not in the document, clearly state "This information is not found in the provided document"
5. Indicate your confidence level (HIGH/MEDIUM/LOW) based on how directly the document addresses the question"""

    if valves.use_structured_output:
        structured_output_instructions = """

Respond ONLY with a JSON object in this exact format:
{
    "answer": "Your detailed answer here",
    "confidence": "HIGH/MEDIUM/LOW",
    "key_points": ["point 1", "point 2"] or null,
    "citations": ["relevant quote 1", "relevant quote 2"] or null
}

Do not include any text, explanations, or markdown formatting (like ```json ... ```) outside of this JSON object."""
        return base_prompt + structured_output_instructions
    else:
        return base_prompt + "\n\nProvide a clear, detailed answer to the question."