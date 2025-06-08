# Partials File: partials/minion_prompts.py
from typing import List, Tuple, Any, Optional

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
    Enhanced with better guidance for structured, useful responses.
    """
    # query is the original user query.
    # context is the document chunk.
    # claude_request (the parameter) is the specific question from the remote model to the local model.

    base_prompt = f"""You are a document analysis assistant with exclusive access to the following document:

<document>
{context}
</document>

A research coordinator needs specific information from this document to answer: "{query}"

Their current question is:
<question>
{claude_request}
</question>

RESPONSE GUIDELINES:

1. ACCURACY: Base your answer ONLY on information found in the document above
   
2. CITATIONS: When possible, include direct quotes or specific references:
   - Good: "According to section 3.2, 'the budget increased by 15%'"
   - Good: "The document states on page 4 that..."
   - Poor: "The document mentions something about budgets"

3. ORGANIZATION: For complex answers, structure your response:
   - Use bullet points or numbered lists for multiple items
   - Separate distinct pieces of information clearly
   - Highlight key findings at the beginning

4. CONFIDENCE LEVELS:
   - HIGH: Information directly answers the question with explicit statements
   - MEDIUM: Information partially addresses the question or requires some inference
   - LOW: Information is tangentially related or requires significant interpretation

5. HANDLING MISSING INFORMATION:
   - If not found: "This specific information is not available in the document"
   - If partially found: "The document provides partial information: [explain what's available]"
   - Suggest related info: "While X is not mentioned, the document does discuss Y which may be relevant"

Remember: The coordinator cannot see the document and relies entirely on your accurate extraction."""

    if valves.use_structured_output:
        structured_output_instructions = """

RESPONSE FORMAT:
Respond ONLY with a JSON object in this exact format:
{
    "answer": "Your detailed answer addressing the specific question",
    "confidence": "HIGH/MEDIUM/LOW",
    "key_points": ["Main finding 1", "Main finding 2", "..."] or null,
    "citations": ["Exact quote from document", "Another relevant quote", "..."] or null
}

JSON Guidelines:
- answer: Comprehensive response to the question (required)
- confidence: Your assessment based on criteria above (required)
- key_points: List main findings if multiple important points exist (optional)
- citations: Direct quotes that support your answer (optional but recommended)

IMPORTANT: Output ONLY the JSON object. No additional text, no markdown formatting."""
        return base_prompt + structured_output_instructions
    else:
        non_structured_instructions = """

Format your response clearly with:
- Main answer first
- Supporting details or quotes
- Confidence level (HIGH/MEDIUM/LOW) at the end
- Note if any information is not found in the document"""
        return base_prompt + non_structured_instructions

def get_minion_initial_claude_prompt_with_state(query: str, context_len: int, valves: Any, conversation_state: Optional[Any] = None) -> str:
    """
    Enhanced version of initial prompt that includes conversation state if available.
    """
    base_prompt = get_minion_initial_claude_prompt(query, context_len, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        if state_summary:
            base_prompt = base_prompt.replace(
                "Otherwise, ask your first strategic question to the local assistant.",
                f"""
CONVERSATION STATE CONTEXT:
{state_summary}

Based on this context, ask your first strategic question to the local assistant."""
            )
    
    return base_prompt

def get_minion_conversation_claude_prompt_with_state(history: List[Tuple[str, str]], original_query: str, valves: Any, conversation_state: Optional[Any] = None) -> str:
    """
    Enhanced version of conversation prompt that includes conversation state if available.
    """
    base_prompt = get_minion_conversation_claude_prompt(history, original_query, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        
        # Insert state summary before decision point
        state_section = f"""
CURRENT CONVERSATION STATE:
{state_summary}

TOPICS COVERED: {', '.join(conversation_state.topics_covered) if conversation_state.topics_covered else 'None yet'}
KEY FINDINGS COUNT: {len(conversation_state.key_findings)}
INFORMATION GAPS: {len(conversation_state.information_gaps)}
"""
        
        base_prompt = base_prompt.replace(
            "DECISION POINT:",
            state_section + "\nDECISION POINT:"
        )
    
    return base_prompt