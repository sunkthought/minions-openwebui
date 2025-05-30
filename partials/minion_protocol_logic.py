import asyncio
import json
from typing import List, Optional, Dict, Any, Tuple, Callable

# All code is self-contained, no external imports needed

def build_minion_conversation_context(
    history: List[Tuple[str, str]], 
    original_query: str
) -> str:
    """Builds the prompt context for the remote model based on conversation history."""
    context_parts = [
        f"You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: \"{original_query}\"",
        "The local assistant has full access to the source document and has been providing factual information extracted from it.",
        "",
        "CONVERSATION SO FAR (Your questions, Local Assistant's factual responses from the document):",
    ]

    for role, message in history:
        if role == "assistant": # Remote model's previous message
            context_parts.append(f"You previously asked the local assistant: \"{message}\"")
        else: # Local model's response
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

def is_minion_final_answer(response: str) -> bool:
    """Checks if the response from the remote model contains the final answer marker."""
    return "FINAL ANSWER READY." in response

def parse_minion_local_response(
    response_text: str, 
    valves: Any, # For debug_mode and use_structured_output flags
    is_structured: bool = False,
    response_model: Optional[Any] = None # e.g., LocalAssistantResponse from minion_models
) -> Dict[str, Any]:
    """
    Parses the local model's response, supporting both text and structured (JSON) formats.
    `response_model` should be the Pydantic model to validate against if structured.
    """
    if is_structured and valves.use_structured_output and response_model:
        try:
            parsed_json = json.loads(response_text)
            # validated_model = response_model(**parsed_json) # This is how it would work with the actual model
            # model_dict = validated_model.dict()
            # For now, just returning the parsed_json as dict if it's a dict
            if isinstance(parsed_json, dict):
                 model_dict = parsed_json
                 model_dict['parse_error'] = None
                 return model_dict
            else: # Not a dict, treat as parsing failure for structured
                 raise ValueError("Parsed JSON is not a dictionary.")

        except Exception as e:
            if valves.debug_mode:
                # In a real app, use logging instead of print
                print(f"DEBUG: Failed to parse structured output in Minion: {e}. Response was: {response_text[:500]}")
            return {"answer": response_text, "confidence": "LOW", "key_points": None, "citations": None, "parse_error": str(e)}
    
    # Fallback for non-structured processing or when use_structured_output is False
    return {"answer": response_text, "confidence": "MEDIUM", "key_points": None, "citations": None, "parse_error": None}

async def execute_minion_protocol(
    valves: Any,
    query: str,
    context: str,
    # These would be imported or passed if this function calls them:
    call_claude_func: Callable, # Placeholder for common_api_calls.call_claude
    call_ollama_func: Callable, # Placeholder for common_api_calls.call_ollama
    local_assistant_response_model: Any, # Placeholder for minion_models.LocalAssistantResponse
    calculate_minion_token_savings_func: Callable[..., Dict[str, Any]] # Added new parameter
) -> str:
    """
    Executes the Minion conversational protocol.
    """
    conversation_log = []
    debug_log = []
    conversation_history: List[Tuple[str, str]] = []
    actual_final_answer = "No final answer was explicitly provided by the remote model."
    claude_declared_final = False

    overall_start_time = 0
    if valves.debug_mode:
        overall_start_time = asyncio.get_event_loop().time()
        debug_log.append(f"🔍 **Debug Info (Minion Protocol v0.2.0):**")
        debug_log.append(f"  - Query: {query[:100]}...")
        debug_log.append(f"  - Context length: {len(context)} chars")
        debug_log.append(f"  - Max rounds: {valves.max_rounds}")
        debug_log.append(f"  - Remote model: {valves.remote_model}")
        debug_log.append(f"  - Local model: {valves.local_model}")
        debug_log.append(f"  - Timeouts: Claude={valves.timeout_claude}s, Local={valves.timeout_local}s")
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**\n")

    initial_claude_prompt = f"""Your primary goal is to answer the user's question: "{query}"

To achieve this, you will collaborate with a local AI assistant. This local assistant has ALREADY READ and has FULL ACCESS to the relevant document ({len(context)} characters long). The local assistant is a TRUSTED source that will provide you with factual information, summaries, and direct extractions FROM THE DOCUMENT in response to your questions.

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

    for round_num in range(valves.max_rounds):
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {round_num + 1}/{valves.max_rounds}... (Debug Mode)**")
        
        if valves.show_conversation:
            conversation_log.append(f"### 🔄 Round {round_num + 1}")

        claude_prompt_for_this_round = initial_claude_prompt if round_num == 0 else build_minion_conversation_context(conversation_history, query)
        
        claude_response = ""
        try:
            if valves.debug_mode: start_time_claude = asyncio.get_event_loop().time()
            claude_response = await call_claude_func(valves, claude_prompt_for_this_round)
            if valves.debug_mode:
                end_time_claude = asyncio.get_event_loop().time()
                time_taken_claude = end_time_claude - start_time_claude
                debug_log.append(f"  ⏱️ Remote model call in round {round_num + 1} took {time_taken_claude:.2f}s. (Debug Mode)")
        except Exception as e:
            error_message = f"❌ Error calling remote model in round {round_num + 1}: {e}"
            conversation_log.append(error_message)
            if valves.debug_mode: debug_log.append(f"  {error_message} (Debug Mode)")
            actual_final_answer = "Minion protocol failed due to remote model API error."
            break 

        conversation_history.append(("assistant", claude_response))
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Remote Model ({valves.remote_model}):**")
            conversation_log.append(f"{claude_response}\n")

        if is_minion_final_answer(claude_response):
            actual_final_answer = claude_response.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_declared_final = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **Remote model indicates FINAL ANSWER READY.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 Remote model declared FINAL ANSWER READY in round {round_num + 1}. (Debug Mode)")
            break

        local_prompt = f"""You have access to the full context below. The remote model ({valves.remote_model}) is collaborating with you to answer a user's question.
CONTEXT:
{context}
ORIGINAL USER QUESTION: {query}
REMOTE MODEL'S REQUEST TO YOU: {claude_response}
Please provide a helpful, accurate response based ONLY on the CONTEXT provided above. Extract relevant information that answers the remote model's specific request. Be concise but thorough.
If you are instructed to provide a JSON response (e.g., by a schema appended to this prompt), ensure your entire response is ONLY that valid JSON object, without any surrounding text, explanations, or markdown formatting like ```json ... ```."""
        
        local_response_str = ""
        try:
            if valves.debug_mode: start_time_ollama = asyncio.get_event_loop().time()
            local_response_str = await call_ollama_func(
                valves,
                local_prompt,
                use_json=True, 
                schema=local_assistant_response_model 
            )
            local_response_data = parse_minion_local_response(
                local_response_str,
                valves,
                is_structured=True,
                response_model=local_assistant_response_model
            )
            if valves.debug_mode:
                end_time_ollama = asyncio.get_event_loop().time()
                time_taken_ollama = end_time_ollama - start_time_ollama
                debug_log.append(f"  ⏱️ Local model call in round {round_num + 1} took {time_taken_ollama:.2f}s. (Debug Mode)")
        except Exception as e:
            error_message = f"❌ Error calling local model in round {round_num + 1}: {e}"
            conversation_log.append(error_message)
            if valves.debug_mode: debug_log.append(f"  {error_message} (Debug Mode)")
            actual_final_answer = "Minion protocol failed due to local model API error."
            break 

        response_for_claude = local_response_data.get("answer", "Error: Could not extract answer from local model.")
        if valves.use_structured_output and local_response_data.get("parse_error") and valves.debug_mode:
            response_for_claude += f" (Local model response parse error: {local_response_data['parse_error']})"
        elif not local_response_data.get("answer") and not local_response_data.get("parse_error"):
             response_for_claude = "Local model provided no answer."

        conversation_history.append(("user", response_for_claude)) # 'user' here refers to the local model acting as user to Claude
        if valves.show_conversation:
            conversation_log.append(f"**💻 Local Model ({valves.local_model}):**")
            if valves.use_structured_output and local_response_data.get("parse_error") is None:
                conversation_log.append(f"```json\n{json.dumps(local_response_data, indent=2)}\n```")
            elif valves.use_structured_output and local_response_data.get("parse_error"):
                conversation_log.append(f"Attempted structured output, but failed. Raw response:\n{local_response_data.get('answer', 'Error displaying local response.')}")
                conversation_log.append(f"(Parse Error: {local_response_data['parse_error']})")
            else:
                conversation_log.append(f"{local_response_data.get('answer', 'Error displaying local response.')}")
            conversation_log.append("\n")

        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {round_num + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**\n")
    
    if not claude_declared_final and conversation_history:
        last_claude_msg_tuple = next((msg for msg in reversed(conversation_history) if msg[0] == "assistant"), None)
        last_claude_msg = last_claude_msg_tuple[1] if last_claude_msg_tuple else "No suitable final message from remote model found."
        actual_final_answer = f"Max rounds reached. Remote model's last message was: \"{last_claude_msg}\""
        if valves.show_conversation:
            conversation_log.append(f"⚠️ Max rounds reached. Using remote model's last message as the result.\n")

    if valves.debug_mode:
        total_execution_time = asyncio.get_event_loop().time() - overall_start_time
        debug_log.append(f"**⏱️ Total Minion protocol execution time: {total_execution_time:.2f}s. (Debug Mode)**")

    output_parts = []
    if valves.show_conversation:
        output_parts.append("## 🗣️ Collaboration Conversation")
        output_parts.extend(conversation_log)
        output_parts.append("---")
    if valves.debug_mode:
        output_parts.append("### 🔍 Debug Log")
        output_parts.extend(debug_log)
        output_parts.append("---")

    output_parts.append(f"## 🎯 Final Answer")
    output_parts.append(actual_final_answer)

    stats = calculate_minion_token_savings_func(conversation_history, context, query) # Use the passed function
    output_parts.append(f"\n## 📊 Efficiency Stats")
    output_parts.append(f"- **Protocol:** Minion (conversational)")
    output_parts.append(f"- **Remote model:** {valves.remote_model}")
    output_parts.append(f"- **Local model:** {valves.local_model}")
    output_parts.append(f"- **Conversation rounds:** {len(conversation_history) // 2}") # Each round has assistant and user message
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    output_parts.append(f"")
    output_parts.append(f"## 💰 Token Savings Analysis ({valves.remote_model})") 
    output_parts.append(f"- **Traditional approach:** ~{stats.get('traditional_tokens', 0):,} tokens")
    output_parts.append(f"- **Minion approach:** ~{stats.get('minion_tokens', 0):,} tokens")
    output_parts.append(f"- **💰 Token Savings:** ~{stats.get('percentage_savings', 0.0):.1f}%")
    
    return "\n".join(output_parts)