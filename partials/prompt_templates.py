# Partials File: partials/prompt_templates.py
from typing import Dict, Any, List, Tuple, Optional


class PromptTemplates:
    """Centralized prompt templates for both Minion and MinionS protocols."""
    
    # =======================
    # MINION PROTOCOL TEMPLATES
    # =======================
    
    MINION_INITIAL_CLAUDE = '''You are a research coordinator working with a knowledgeable local assistant who has access to specific documents.

Your task: Gather information to answer the user's query by asking strategic questions.

USER'S QUERY: "{query}"

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
- You have {max_rounds} rounds maximum to gather information

QUESTION STRATEGY TIPS:
- For factual queries: Ask for specific data points, dates, numbers, or names
- For analytical queries: Ask about relationships, comparisons, or patterns
- For summary queries: Ask about key themes, main points, or conclusions
- For procedural queries: Ask about steps, sequences, or requirements

Remember: The assistant can only see the document, not your conversation history.

If you have gathered enough information to answer "{query}", respond with "FINAL ANSWER READY." followed by your comprehensive answer.

Otherwise, ask your first strategic question to the local assistant.'''

    MINION_CONVERSATION_CLAUDE = '''You are continuing to gather information to answer: "{original_query}"

Round {current_round} of {max_rounds}

INFORMATION GATHERED SO FAR:
{conversation_history}

DECISION POINT:
Evaluate if you have sufficient information to answer the original question comprehensively.

✅ If YES: Start with 'FINAL ANSWER READY.' then provide your complete answer
❓ If NO: Ask ONE more strategic question (you have {rounds_remaining} rounds left)

TIPS FOR YOUR NEXT QUESTION:
- What specific gaps remain in your understanding?
- Can you drill deeper into any mentioned topics?
- Are there related aspects you haven't explored?
- Would examples or specific details strengthen your answer?

Remember: Each question should build on what you've learned, not repeat previous inquiries.'''

    MINION_LOCAL_ASSISTANT_BASE = '''You are a document analysis assistant with exclusive access to the following document:

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

Remember: The coordinator cannot see the document and relies entirely on your accurate extraction.'''

    MINION_LOCAL_STRUCTURED_OUTPUT = '''

RESPONSE FORMAT:
Respond ONLY with a JSON object in this exact format:
{{
    "answer": "Your detailed answer addressing the specific question",
    "confidence": "HIGH/MEDIUM/LOW",
    "key_points": ["Main finding 1", "Main finding 2", "..."] or null,
    "citations": ["Exact quote from document", "Another relevant quote", "..."] or null
}}

JSON Guidelines:
- answer: Comprehensive response to the question (required)
- confidence: Your assessment based on criteria above (required)
- key_points: List main findings if multiple important points exist (optional)
- citations: Direct quotes that support your answer (optional but recommended)

IMPORTANT: Output ONLY the JSON object. No additional text, no markdown formatting.'''

    MINION_LOCAL_NON_STRUCTURED = '''

Format your response clearly with:
- Main answer first
- Supporting details or quotes
- Confidence level (HIGH/MEDIUM/LOW) at the end
- Note if any information is not found in the document'''

    MINION_INITIAL_CLAUDE_WITH_STATE = '''CONVERSATION STATE CONTEXT:
{state_summary}

Based on this context, ask your first strategic question to the local assistant.'''

    MINION_CONVERSATION_CLAUDE_WITH_STATE = '''
CURRENT CONVERSATION STATE:
{state_summary}

TOPICS COVERED: {topics_covered}
KEY FINDINGS COUNT: {key_findings_count}
INFORMATION GAPS: {information_gaps_count}

DECISION POINT:'''

    MINION_CONVERSATION_WITH_PHASE = '''CURRENT CONVERSATION PHASE:
{phase_guidance}

DECISION POINT:'''

    MINION_DEDUPLICATION_WARNING = '''
PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT):
{previous_questions}

⚠️ IMPORTANT: Avoid asking questions that are semantically similar to the above. Each new question should explore genuinely new information.

Remember: Each question should build on what you've learned, not repeat previous inquiries.'''

    # =======================
    # MINIONS PROTOCOL TEMPLATES
    # =======================
    
    MINIONS_SYNTHESIS_CLAUDE = '''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary}

{synthesis_guidelines}

If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.
Final Answer:'''

    MINIONS_SYNTHESIS_GUIDELINES = '''
SYNTHESIS GUIDELINES:
{guidelines}'''

    MINIONS_LOCAL_TASK_BASE = '''Text to analyze (Chunk {chunk_idx}/{total_chunks} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}

{task_specific_instructions}'''

    MINIONS_TASK_SPECIFIC_INSTRUCTIONS = '''
TASK-SPECIFIC INSTRUCTIONS:
{instructions}'''

    MINIONS_STRUCTURED_OUTPUT_FORMAT = '''

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any text before or after the JSON.

Required JSON structure:
{{
    "explanation": "Brief explanation of your findings for this task",
    "citation": "Direct quote from the text if applicable to this task, or null",
    "answer": "Your complete answer to the task as a SINGLE STRING",
    "confidence": "HIGH, MEDIUM, or LOW"
}}'''

    MINIONS_STRUCTURED_OUTPUT_RULES = '''
CRITICAL RULES FOR JSON OUTPUT:
1. Output ONLY the JSON object - no markdown formatting, no explanatory text, no code blocks
2. The "answer" field MUST be a plain text string, NOT an object or array
3. If listing multiple items, format as a single string (e.g., "Item 1: Description. Item 2: Description.")
4. Use proper JSON escaping for quotes within strings (\\" for quotes inside string values)
5. If information is not found, set "answer" to null and "confidence" to "LOW"
6. The "confidence" field must be exactly one of: "HIGH", "MEDIUM", or "LOW"
7. All string values must be properly quoted and escaped'''

    MINIONS_FORMAT_SPECIFIC_INSTRUCTION = '''5. Format the content WITHIN the "answer" field as {format}. For example, if "bullet points", the "answer" string should look like "- Point 1\\n- Point 2".'''

    MINIONS_JSON_FORMAT_INSTRUCTION = '''5. The overall response is already JSON. Ensure the content of the 'answer' field is a simple string, not further JSON encoded, unless the task specifically asks for a JSON string as the answer.'''

    MINIONS_STRUCTURED_OUTPUT_EXAMPLES = '''

EXAMPLES OF CORRECT JSON OUTPUT:

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
}}'''

    MINIONS_BULLET_POINTS_EXAMPLE = '''

Example with bullet points in answer field:
{{
    "explanation": "Found multiple implementation steps",
    "citation": "The implementation plan consists of three phases...",
    "answer": "- Phase 1: Initial setup and configuration\\n- Phase 2: Testing and validation\\n- Phase 3: Full deployment",
    "confidence": "MEDIUM"
}}'''

    MINIONS_INCORRECT_OUTPUT_EXAMPLES = '''

EXAMPLES OF INCORRECT OUTPUT (DO NOT DO THIS):

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
{{"answer": "some value"}}'''

    MINIONS_NON_STRUCTURED_OUTPUT = '''

Provide a brief, specific answer based ONLY on the text provided above.
{format_instruction}
If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word "NONE".'''

    # =======================
    # SHARED TEMPLATES
    # =======================
    
    STRUCTURED_OUTPUT_INSTRUCTION = '''Respond in JSON format:
{schema}'''

    ERROR_CONTEXT = '''An error occurred: {error}
Please provide a helpful response despite this limitation.'''

    CHUNK_ANALYSIS_HEADER = '''You are analyzing chunk {chunk_num} of {total_chunks} from a larger document to answer: "{query}"

Here is the chunk content:

<document_chunk>
{context}
</document_chunk>

This chunk contains a portion of the document. Your task is to:
1. Ask ONE focused question to extract the most relevant information from this chunk
2. Based on the response, either ask ONE follow-up question OR provide a final answer for this chunk

Be concise and focused since this is one chunk of a larger analysis.

Ask your first question about this chunk:'''

    SYNTHESIS_PROMPT = '''Based on the local assistant's response: "{local_answer}"

Provide a brief summary of what this chunk (#{chunk_num} of {total_chunks}) contributes to answering: "{query}"

Keep it concise since this is just one part of a larger document analysis.'''

    MULTI_CHUNK_SYNTHESIS = '''You have analyzed a document in {chunk_count} chunks to answer: "{query}"

Here are the results from each chunk:

{chunk_results}

Based on all the information gathered from these chunks, provide a comprehensive final answer to the user's original question: "{query}"

Your response should:
1. Synthesize the key information from all chunks
2. Address the specific question asked
3. Be well-organized and coherent
4. Include the most important findings and insights

Provide your final comprehensive answer:'''

    FALLBACK_SYNTHESIS = '''Based on the analysis of {chunk_count} document chunks, here are the key findings for: "{query}"

{chunk_summaries}

Note: This synthesis was generated locally due to a temporary API connectivity issue. The individual chunk analyses above contain the detailed information extracted from the document.'''

    @classmethod
    def get_template(cls, template_name: str, **kwargs) -> str:
        """Get a template by name and format with provided kwargs."""
        template = getattr(cls, template_name, None)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        return template.format(**kwargs)
    
    @classmethod 
    def get_minion_initial_claude_prompt(cls, query: str, context_len: int, max_rounds: int) -> str:
        """Get the initial Claude prompt for Minion protocol."""
        return cls.get_template(
            'MINION_INITIAL_CLAUDE',
            query=query.replace('"', '\\"').replace("'", "\\'"),
            context_len=context_len,
            max_rounds=max_rounds
        )
    
    @classmethod
    def get_minion_conversation_claude_prompt(cls, 
                                            original_query: str, 
                                            current_round: int, 
                                            max_rounds: int,
                                            conversation_history: str) -> str:
        """Get the conversation Claude prompt for Minion protocol."""
        return cls.get_template(
            'MINION_CONVERSATION_CLAUDE',
            original_query=original_query.replace('"', '\\"').replace("'", "\\'"),
            current_round=current_round,
            max_rounds=max_rounds,
            conversation_history=conversation_history,
            rounds_remaining=max_rounds - current_round
        )
    
    @classmethod
    def get_minion_local_prompt(cls, 
                              context: str, 
                              query: str, 
                              claude_request: str, 
                              use_structured_output: bool = False) -> str:
        """Get the local assistant prompt for Minion protocol."""
        base_prompt = cls.get_template(
            'MINION_LOCAL_ASSISTANT_BASE',
            context=context,
            query=query,
            claude_request=claude_request
        )
        
        if use_structured_output:
            return base_prompt + cls.MINION_LOCAL_STRUCTURED_OUTPUT
        else:
            return base_prompt + cls.MINION_LOCAL_NON_STRUCTURED
    
    @classmethod
    def get_minions_synthesis_claude_prompt(cls, query: str, synthesis_input_summary: str, guidelines: List[str] = None) -> str:
        """Get the synthesis prompt for MinionS protocol."""
        synthesis_guidelines = ""
        if guidelines:
            formatted_guidelines = "\n".join([f"- {guideline}" for guideline in guidelines])
            synthesis_guidelines = cls.get_template('MINIONS_SYNTHESIS_GUIDELINES', guidelines=formatted_guidelines)
        
        return cls.get_template(
            'MINIONS_SYNTHESIS_CLAUDE',
            query=query,
            synthesis_input_summary=synthesis_input_summary or "No specific information was extracted by local models.",
            synthesis_guidelines=synthesis_guidelines
        )
    
    @classmethod
    def get_minions_local_task_prompt(cls,
                                    chunk: str,
                                    task: str,
                                    chunk_idx: int,
                                    total_chunks: int,
                                    use_structured_output: bool = False,
                                    task_instructions: List[str] = None,
                                    expected_format: str = None) -> str:
        """Get the local task prompt for MinionS protocol."""
        
        # Build task-specific instructions
        task_specific_instructions = ""
        if task_instructions:
            formatted_instructions = "\n".join([f"- {instruction}" for instruction in task_instructions])
            task_specific_instructions = cls.get_template('MINIONS_TASK_SPECIFIC_INSTRUCTIONS', instructions=formatted_instructions)
        
        base_prompt = cls.get_template(
            'MINIONS_LOCAL_TASK_BASE',
            chunk=chunk,
            task=task,
            chunk_idx=chunk_idx + 1,
            total_chunks=total_chunks,
            task_specific_instructions=task_specific_instructions
        )
        
        if use_structured_output:
            structured_parts = [
                cls.MINIONS_STRUCTURED_OUTPUT_FORMAT,
                cls.MINIONS_STRUCTURED_OUTPUT_RULES
            ]
            
            # Add format-specific instruction
            if expected_format and expected_format.lower() != "json":
                if expected_format.lower() == "bullet points":
                    structured_parts.append(cls.get_template('MINIONS_FORMAT_SPECIFIC_INSTRUCTION', format=expected_format.upper()))
                else:
                    structured_parts.append(cls.get_template('MINIONS_FORMAT_SPECIFIC_INSTRUCTION', format=expected_format.upper()))
            elif expected_format and expected_format.lower() == "json":
                structured_parts.append(cls.MINIONS_JSON_FORMAT_INSTRUCTION)
            
            # Add examples
            structured_parts.append(cls.MINIONS_STRUCTURED_OUTPUT_EXAMPLES)
            
            if expected_format and expected_format.lower() == "bullet points":
                structured_parts.append(cls.MINIONS_BULLET_POINTS_EXAMPLE)
            
            structured_parts.append(cls.MINIONS_INCORRECT_OUTPUT_EXAMPLES)
            
            return base_prompt + "".join(structured_parts)
        else:
            format_instruction = ""
            if expected_format and expected_format.lower() != "text":
                format_instruction = f"Format your entire response as {expected_format.upper()}."
            
            return base_prompt + cls.get_template('MINIONS_NON_STRUCTURED_OUTPUT', format_instruction=format_instruction)
    
    @classmethod
    def build_conversation_history(cls, history: List[Tuple[str, str]]) -> str:
        """Build formatted conversation history for prompts."""
        context_parts = []
        
        for i, (role, message) in enumerate(history):
            if role == "assistant":  # Claude's previous message
                context_parts.append(f'\nQ{i//2 + 1}: {message}')
            else:  # Local model's response
                context_parts.append(f'A{i//2 + 1}: {message}')
        
        return "".join(context_parts)
    
    @classmethod
    def enhance_prompt_with_state(cls, base_prompt: str, state_summary: str, topics_covered: List[str] = None, key_findings_count: int = 0, information_gaps_count: int = 0) -> str:
        """Enhance a prompt with conversation state information."""
        state_context = cls.get_template(
            'MINION_CONVERSATION_CLAUDE_WITH_STATE',
            state_summary=state_summary,
            topics_covered=', '.join(topics_covered) if topics_covered else 'None yet',
            key_findings_count=key_findings_count,
            information_gaps_count=information_gaps_count
        )
        
        return base_prompt.replace("DECISION POINT:", state_context + "\nDECISION POINT:")
    
    @classmethod
    def enhance_prompt_with_deduplication(cls, base_prompt: str, previous_questions: List[str]) -> str:
        """Enhance a prompt with question deduplication guidance."""
        formatted_questions = "\n".join([f"{i}. {q}" for i, q in enumerate(previous_questions, 1)])
        deduplication_section = cls.get_template('MINION_DEDUPLICATION_WARNING', previous_questions=formatted_questions)
        
        return base_prompt.replace(
            "Remember: Each question should build on what you've learned, not repeat previous inquiries.",
            deduplication_section
        )
    
    @classmethod
    def enhance_prompt_with_phase_guidance(cls, base_prompt: str, phase_guidance: str) -> str:
        """Enhance a prompt with phase guidance."""
        phase_context = cls.get_template('MINION_CONVERSATION_WITH_PHASE', phase_guidance=phase_guidance)
        
        return base_prompt.replace("DECISION POINT:", phase_context)
    
    @classmethod
    def get_available_templates(cls) -> List[str]:
        """Get a list of all available template names."""
        return [attr for attr in dir(cls) if not attr.startswith('_') and attr.isupper() and isinstance(getattr(cls, attr), str)]