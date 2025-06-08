# Partials File: partials/minion_protocol_logic.py
import asyncio
import json
from typing import List, Dict, Any, Tuple, Callable

def _calculate_token_savings(conversation_history: List[Tuple[str, str]], context: str, query: str) -> dict:
    """Calculate token savings for the Minion protocol"""
    chars_per_token = 3.5
    
    # Traditional approach: entire context + query sent to remote model
    traditional_tokens = int((len(context) + len(query)) / chars_per_token)
    
    # Minion approach: only conversation messages sent to remote model
    conversation_content = " ".join(
        [msg[1] for msg in conversation_history if msg[0] == "assistant"]
    )
    minion_tokens = int(len(conversation_content) / chars_per_token)
    
    # Calculate savings
    token_savings = traditional_tokens - minion_tokens
    percentage_savings = (
        (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    )
    
    return {
        'traditional_tokens': traditional_tokens,
        'minion_tokens': minion_tokens,
        'token_savings': token_savings,
        'percentage_savings': percentage_savings
    }

def _is_final_answer(response: str) -> bool:
    """Check if response contains the specific final answer marker."""
    return "FINAL ANSWER READY." in response

def detect_completion(response: str) -> bool:
    """Check if remote model indicates it has sufficient information"""
    completion_phrases = [
        "i now have sufficient information",
        "i can now answer",
        "based on the information gathered",
        "i have enough information",
        "with this information, i can provide",
        "i can now provide a comprehensive answer",
        "based on what the local assistant has told me"
    ]
    response_lower = response.lower()
    
    # Check for explicit final answer marker first
    if "FINAL ANSWER READY." in response:
        return True
    
    # Check for completion phrases
    return any(phrase in response_lower for phrase in completion_phrases)

def _parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, LocalAssistantResponseModel: Any) -> Dict:
    """Parse local model response, supporting both text and structured formats."""
    confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
    default_numeric_confidence = 0.3  # Corresponds to LOW
    
    if is_structured and use_structured_output:
        # Clean up common formatting issues
        cleaned_response = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[7:-3].strip()
        elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
        
        # Try to extract JSON from response with explanatory text
        if not cleaned_response.startswith("{"):
            # Look for JSON object in the response
            json_start = cleaned_response.find("{")
            json_end = cleaned_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_response = cleaned_response[json_start:json_end+1]
        
        try:
            parsed_json = json.loads(cleaned_response)
            
            # Handle missing confidence field with default
            if 'confidence' not in parsed_json:
                parsed_json['confidence'] = 'LOW'
            
            # Ensure required fields have defaults if missing
            if 'answer' not in parsed_json:
                parsed_json['answer'] = None
            if 'key_points' not in parsed_json:
                parsed_json['key_points'] = None
            if 'citations' not in parsed_json:
                parsed_json['citations'] = None
            
            validated_model = LocalAssistantResponseModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            
            # Add numeric confidence for consistency
            text_confidence = model_dict.get('confidence', 'LOW').upper()
            model_dict['numeric_confidence'] = confidence_map.get(text_confidence, default_numeric_confidence)
            
            return model_dict
            
        except json.JSONDecodeError as e:
            if debug_mode:
                print(f"DEBUG: JSON decode error in Minion: {e}. Cleaned response was: {cleaned_response[:500]}")
            
            # Try regex fallback to extract key information
            import re
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
            confidence_match = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', response, re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1)
                confidence = confidence_match.group(1).upper() if confidence_match else "LOW"
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "numeric_confidence": confidence_map.get(confidence, default_numeric_confidence),
                    "key_points": None,
                    "citations": None,
                    "parse_error": f"JSON parse error (recovered): {str(e)}"
                }
            
            # Complete fallback
            return {
                "answer": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence,
                "key_points": None,
                "citations": None,
                "parse_error": str(e)
            }
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in Minion: {e}. Response was: {response[:500]}")
            return {
                "answer": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence,
                "key_points": None, 
                "citations": None, 
                "parse_error": str(e)
            }
    
    # Fallback for non-structured processing
    return {
        "answer": response, 
        "confidence": "MEDIUM", 
        "numeric_confidence": confidence_map.get("MEDIUM", 0.6),
        "key_points": None, 
        "citations": None, 
        "parse_error": None
    }

async def _execute_minion_protocol(
    valves: Any,
    query: str,
    context: str,
    call_claude_func: Callable,
    call_ollama_func: Callable,
    LocalAssistantResponseModel: Any
) -> str:
    """Execute the Minion protocol"""
    conversation_log = []
    debug_log = []
    conversation_history = []
    actual_final_answer = "No final answer was explicitly provided by the remote model."
    claude_declared_final = False
    
    # Initialize metrics tracking
    overall_start_time = asyncio.get_event_loop().time()
    metrics = {
        'confidence_scores': [],
        'confidence_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
        'rounds_completed': 0,
        'completion_via_detection': False,
        'estimated_tokens': 0,
        'chunk_size_used': valves.chunk_size,
        'context_size': len(context)
    }

    if valves.debug_mode:
        debug_log.append(f"🔍 **Debug Info (Minion v0.3.6):**")
        debug_log.append(f"  - Query: {query[:100]}...")
        debug_log.append(f"  - Context length: {len(context)} chars")
        debug_log.append(f"  - Max rounds: {valves.max_rounds}")
        debug_log.append(f"  - Remote model: {valves.remote_model}")
        debug_log.append(f"  - Local model: {valves.local_model}")
        debug_log.append(f"  - Timeouts: Remote={valves.timeout_claude}s, Local={valves.timeout_local}s")
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**\n")

    for round_num in range(valves.max_rounds):
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {round_num + 1}/{valves.max_rounds}... (Debug Mode)**")
        
        if valves.show_conversation:
            conversation_log.append(f"### 🔄 Round {round_num + 1}")

        claude_prompt_for_this_round = ""
        if round_num == 0:
            claude_prompt_for_this_round = get_minion_initial_claude_prompt(query, len(context), valves)
        else:
            # Check if this is the last round and force a final answer
            is_last_round = (round_num == valves.max_rounds - 1)
            if is_last_round:
                # Override with a prompt that forces a final answer
                claude_prompt_for_this_round = f"""You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: "{query}"

The local assistant has full access to the source document and has been providing factual information extracted from it.

CONVERSATION SO FAR:
"""
                for role, message in conversation_history:
                    if role == "assistant":
                        claude_prompt_for_this_round += f"\nYou previously asked: \"{message}\""
                    else:
                        claude_prompt_for_this_round += f"\nLocal assistant responded: \"{message}\""
                
                claude_prompt_for_this_round += f"""

THIS IS YOUR FINAL OPPORTUNITY TO ANSWER. You have gathered sufficient information through {round_num} rounds of questions.

Based on ALL the information provided by the local assistant, you MUST now provide a comprehensive answer to the user's original question: "{query}"

Respond with "FINAL ANSWER READY." followed by your synthesized answer. Do NOT ask any more questions."""
            else:
                claude_prompt_for_this_round = get_minion_conversation_claude_prompt(
                    conversation_history, query, valves
                )
        
        claude_response = ""
        try:
            if valves.debug_mode: 
                start_time_claude = asyncio.get_event_loop().time()
            claude_response = await call_claude_func(valves, claude_prompt_for_this_round)
            if valves.debug_mode:
                end_time_claude = asyncio.get_event_loop().time()
                time_taken_claude = end_time_claude - start_time_claude
                debug_log.append(f"  ⏱️ Remote model call in round {round_num + 1} took {time_taken_claude:.2f}s. (Debug Mode)")
        except Exception as e:
            error_message = f"❌ Error calling the remote model in round {round_num + 1}: {e}"
            conversation_log.append(error_message)
            if valves.debug_mode: 
                debug_log.append(f"  {error_message} (Debug Mode)")
            actual_final_answer = "Minion protocol failed due to remote API error."
            break

        conversation_history.append(("assistant", claude_response))
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Remote Model ({valves.remote_model}):**")
            conversation_log.append(f"{claude_response}\n")

        # Check for explicit final answer or completion indicators
        if _is_final_answer(claude_response):
            actual_final_answer = claude_response.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_declared_final = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **The remote model indicates FINAL ANSWER READY.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 The remote model declared FINAL ANSWER READY in round {round_num + 1}. (Debug Mode)")
            break
        elif valves.enable_completion_detection and detect_completion(claude_response) and round_num > 0:
            # Remote model indicates it has sufficient information
            actual_final_answer = claude_response
            claude_declared_final = True
            metrics['completion_via_detection'] = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **The remote model indicates it has sufficient information to answer.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 Completion detected: Remote model has sufficient information in round {round_num + 1}. (Debug Mode)")
            break

        # Skip local model call if this was the last round and the remote model provided final answer
        if round_num == valves.max_rounds - 1:
            continue

        local_prompt = get_minion_local_prompt(context, query, claude_response, valves)
        
        local_response_str = ""
        try:
            if valves.debug_mode: 
                start_time_ollama = asyncio.get_event_loop().time()
            local_response_str = await call_ollama_func(
                valves,
                local_prompt,
                use_json=True,
                schema=LocalAssistantResponseModel
            )
            local_response_data = _parse_local_response(
                local_response_str,
                is_structured=True,
                use_structured_output=valves.use_structured_output,
                debug_mode=valves.debug_mode,
                LocalAssistantResponseModel=LocalAssistantResponseModel
            )
            
            # Track metrics from local response
            if 'numeric_confidence' in local_response_data:
                metrics['confidence_scores'].append(local_response_data['numeric_confidence'])
            
            confidence_level = local_response_data.get('confidence', 'MEDIUM').upper()
            if confidence_level in metrics['confidence_distribution']:
                metrics['confidence_distribution'][confidence_level] += 1
            
            if valves.debug_mode:
                end_time_ollama = asyncio.get_event_loop().time()
                time_taken_ollama = end_time_ollama - start_time_ollama
                debug_log.append(f"  ⏱️ Local LLM call in round {round_num + 1} took {time_taken_ollama:.2f}s. (Debug Mode)")
        except Exception as e:
            error_message = f"❌ Error calling Local LLM in round {round_num + 1}: {e}"
            conversation_log.append(error_message)
            if valves.debug_mode: 
                debug_log.append(f"  {error_message} (Debug Mode)")
            actual_final_answer = "Minion protocol failed due to Local LLM API error."
            break

        response_for_claude = local_response_data.get("answer", "Error: Could not extract answer from local LLM.")
        if valves.use_structured_output and local_response_data.get("parse_error") and valves.debug_mode:
            response_for_claude += f" (Local LLM response parse error: {local_response_data['parse_error']})"
        elif not local_response_data.get("answer") and not local_response_data.get("parse_error"):
            response_for_claude = "Local LLM provided no answer."

        conversation_history.append(("user", response_for_claude))
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

        # Update round count
        metrics['rounds_completed'] = round_num + 1
        
        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {round_num + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**\n")
    
    if not claude_declared_final and conversation_history:
        # This shouldn't happen with the fix above, but keep as fallback
        last_remote_msg = conversation_history[-1][1] if conversation_history[-1][0] == "assistant" else (conversation_history[-2][1] if len(conversation_history) > 1 and conversation_history[-2][0] == "assistant" else "No suitable final message from the remote model found.")
        actual_final_answer = f"Protocol ended without explicit final answer. The remote model's last response was: \"{last_remote_msg}\""
        if valves.show_conversation:
            conversation_log.append(f"⚠️ Protocol ended without the remote model providing a final answer.\n")

    # Calculate final metrics
    total_execution_time = asyncio.get_event_loop().time() - overall_start_time
    avg_confidence = sum(metrics['confidence_scores']) / len(metrics['confidence_scores']) if metrics['confidence_scores'] else 0.0
    
    # Estimate tokens (rough approximation)
    for role, msg in conversation_history:
        metrics['estimated_tokens'] += len(msg) // 4  # Rough token estimate
    
    if valves.debug_mode:
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

    stats = _calculate_token_savings(conversation_history, context, query)
    output_parts.append(f"\n## 📊 Efficiency Stats")
    output_parts.append(f"- **Protocol:** Minion (conversational)")
    output_parts.append(f"- **Remote model:** {valves.remote_model}")
    output_parts.append(f"- **Local model:** {valves.local_model}")
    output_parts.append(f"- **Conversation rounds:** {len(conversation_history) // 2}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    output_parts.append(f"")
    output_parts.append(f"## 💰 Token Savings Analysis ({valves.remote_model})")
    output_parts.append(f"- **Traditional approach:** ~{stats['traditional_tokens']:,} tokens")
    output_parts.append(f"- **Minion approach:** ~{stats['minion_tokens']:,} tokens")
    output_parts.append(f"- **💰 Token Savings:** ~{stats['percentage_savings']:.1f}%")
    
    # Add conversation metrics
    output_parts.append(f"\n## 📈 Conversation Metrics")
    output_parts.append(f"- **Rounds used:** {metrics['rounds_completed']} of {valves.max_rounds}")
    output_parts.append(f"- **Questions asked:** {metrics['rounds_completed']}")
    output_parts.append(f"- **Average confidence:** {avg_confidence:.2f} ({['LOW', 'MEDIUM', 'HIGH'][int(avg_confidence * 2.99)]})")
    output_parts.append(f"- **Confidence distribution:**")
    for level, count in metrics['confidence_distribution'].items():
        if count > 0:
            output_parts.append(f"  - {level}: {count} response(s)")
    output_parts.append(f"- **Completion method:** {'Early completion detected' if metrics['completion_via_detection'] else 'Reached max rounds or explicit completion'}")
    output_parts.append(f"- **Total duration:** {total_execution_time*1000:.0f}ms")
    output_parts.append(f"- **Estimated tokens:** ~{metrics['estimated_tokens']:,}")
    output_parts.append(f"- **Chunk processing:** {metrics['context_size']:,} chars (max chunk size: {metrics['chunk_size_used']:,})")
    
    return "\n".join(output_parts)