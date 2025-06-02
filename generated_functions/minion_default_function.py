"""
title: Minion Protocol Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.3.5
description: Basic Minion protocol - conversational collaboration between local and cloud models
required_open_webui_version: 0.5.0
license: MIT License
"""


# Partials File: partials/common_imports.py
import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore

# Partials File: partials/minion_models.py
from typing import List, Optional
from pydantic import BaseModel, Field

class LocalAssistantResponse(BaseModel):
    """
    Structured response format for the local assistant in the Minion (conversational) protocol.
    This model defines the expected output structure when the local model processes
    a request from the remote model.
    """
    answer: str = Field(description="The main response to the question posed by the remote model.")
    confidence: str = Field(description="Confidence level of the answer (e.g., HIGH, MEDIUM, LOW).")
    key_points: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of key points extracted from the context related to the answer."
    )
    citations: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of direct quotes or citations from the context supporting the answer."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields during parsing
        # Consider adding an example for documentation if this model is complex:
        # schema_extra = {
        #     "example": {
        #         "answer": "The document states that the project was completed in Q4.",
        #         "confidence": "HIGH",
        #         "key_points": ["Project completion Q4"],
        #         "citations": ["The final report confirms project completion in Q4."]
        #     }
        # }


# Partials File: partials/minion_valves.py
from pydantic import BaseModel, Field

class MinionValves(BaseModel):
    """
    Configuration settings (valves) specifically for the Minion (conversational) pipe.
    These settings control the behavior of the Minion protocol, including API keys,
    model selections, timeouts, operational parameters, extraction instructions,
    expected output format, and confidence threshold.
    """
    # Essential configuration only
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for the remote model (e.g., Claude)"
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model identifier (e.g., for Anthropic: claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    local_model: str = Field(
        default="llama3.2", description="Local Ollama model name"
    )
    max_rounds: int = Field(
        default=2, 
        description="Maximum conversation rounds between remote and local models."
    )
    show_conversation: bool = Field(
        default=True,
        description="Show full conversation between local and remote models in the output.",
    )
    timeout_local: int = Field(
        default=60, 
        description="Timeout for local model calls in seconds. Local model processes full context."
    )
    timeout_claude: int = Field(
        default=60, description="Timeout for remote model API calls in seconds."
    )
    max_tokens_claude: int = Field(
        default=4000, description="Maximum tokens for remote model's responses."
    )
    ollama_num_predict: int = Field(
        default=1000, 
        description="num_predict for Ollama generation (max output tokens for local model)."
    )
    use_structured_output: bool = Field(
        default=False, 
        description="Enable JSON structured output for local model responses (requires local model support)."
    )
    debug_mode: bool = Field(
        default=False, description="Show additional technical details and verbose logs."
    )
    extraction_instructions: str = Field(
        default="", title="Extraction Instructions", description="Specific instructions for the LLM on what to extract or how to process the information."
    )
    expected_format: str = Field(
        default="text", title="Expected Output Format", description="Desired format for the LLM's output (e.g., 'text', 'JSON', 'bullet points')."
    )
    confidence_threshold: float = Field(
        default=0.7, title="Confidence Threshold", description="Minimum confidence level for the LLM's response (0.0-1.0). Primarily a suggestion to the LLM.", ge=0, le=1
    )

    # The following class is part of the Pydantic configuration and is standard.
    # It ensures that extra fields passed to the model are ignored rather than causing an error.
    class Config:
        extra = "ignore"


# Partials File: partials/common_api_calls.py
import aiohttp
import json
from typing import Optional
from pydantic import BaseModel

async def call_claude(
    valves: BaseModel,  # Or a more specific type if Valves is shareable
    prompt: str
) -> str:
    """Call Anthropic Claude API"""
    headers = {
        "x-api-key": valves.anthropic_api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": valves.remote_model,
        "max_tokens": valves.max_tokens_claude,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=valves.timeout_claude
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Anthropic API error: {response.status} - {error_text}"
                )
            result = await response.json()
            if result.get("content") and isinstance(result["content"], list) and len(result["content"]) > 0 and result["content"][0].get("text"):
                return result["content"][0]["text"]
            else:
                # Consider logging instead of print for shared code
                if hasattr(valves, 'debug_mode') and valves.debug_mode:
                    print(f"Unexpected Claude API response format: {result}") 
                raise Exception("Unexpected response format from Anthropic API or empty content.")

async def call_ollama(
    valves: BaseModel,  # Or a more specific type
    prompt: str,
    use_json: bool = False,
    schema: Optional[BaseModel] = None
) -> str:
    """Call Ollama API"""
    payload = {
        "model": valves.local_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": valves.ollama_num_predict},
    }

    if use_json and hasattr(valves, 'use_structured_output') and valves.use_structured_output and schema:
        payload["format"] = "json"
        # Pydantic v1 used schema.schema_json(), v2 uses schema_json = model_json_schema(MyModel) then json.dumps(schema_json)
        # Assuming schema object has a .schema_json() method for simplicity here, may need adjustment
        try:
            schema_for_prompt = schema.schema_json() # For Pydantic v1
        except AttributeError: # Basic fallback for Pydantic v2 or other schema objects
             # This part might need refinement based on actual schema object type if not Pydantic v1 BaseModel
            import inspect
            if hasattr(schema, 'model_json_schema'): # Pydantic v2
                 schema_for_prompt = json.dumps(schema.model_json_schema())
            elif inspect.isclass(schema) and issubclass(schema, BaseModel): # Pydantic v1/v2 class
                 schema_for_prompt = json.dumps(schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema.schema())
            else: # Fallback, might not be perfect
                 schema_for_prompt = "{}" 
                 if hasattr(valves, 'debug_mode') and valves.debug_mode:
                      print("Warning: Could not automatically generate schema_json for prompt.")

        schema_prompt_addition = f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_for_prompt}"
        payload["prompt"] = prompt + schema_prompt_addition
    elif "format" in payload:
        del payload["format"]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{valves.ollama_base_url}/api/generate", json=payload, timeout=valves.timeout_local
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Ollama API error: {response.status} - {error_text}"
                )
            result = await response.json()
            if "response" in result:
                return result["response"].strip()
            else:
                if hasattr(valves, 'debug_mode') and valves.debug_mode:
                    print(f"Unexpected Ollama API response format: {result}")
                raise Exception("Unexpected response format from Ollama API or no response field.")


# Partials File: partials/common_context_utils.py
from typing import List, Dict, Any # Added for type hints used in functions

def extract_context_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract context from conversation history"""
    context_parts = []

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Assume messages longer than 200 chars contain context/documents
            if len(content) > 200:
                context_parts.append(content)

    return "\n\n".join(context_parts)

async def extract_file_content(valves, file_info: Dict[str, Any]) -> str:
    """Extract text content from a single file using Open WebUI's file API"""
    try:
        file_id = file_info.get("id")
        file_name = file_info.get("name", "unknown")

        if not file_id:
            return f"[Could not get file ID for {file_name}]"

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[DEBUG] File ID: {file_id}, Name: {file_name}, Info: {str(file_info)}]"

        # If the file info contains content directly, use it
        if "content" in file_info:
            return file_info["content"]
        
        file_type = file_info.get("type", "unknown")
        file_size = file_info.get("size", "unknown")
        
        return f"[File detected: {file_name} (Type: {file_type}, Size: {file_size})\nNote: File content extraction needs to be configured or content is not directly available in provided file_info]"

    except Exception as e:
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[Error extracting file content: {str(e)}]"
        return f"[Error extracting file content]"

async def extract_context_from_files(valves, files: List[Dict[str, Any]]) -> str:
    """Extract text content from uploaded files using Open WebUI's file system"""
    try:
        if not files:
            return ""

        files_content = []

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            files_content.append(f"[DEBUG] Found {len(files)} uploaded files")

        for file_info in files:
            if isinstance(file_info, dict):
                content = await extract_file_content(valves, file_info)
                if content:
                    if content.startswith("[") and content.endswith("]"):
                        if hasattr(valves, 'debug_mode') and valves.debug_mode:
                            files_content.append(content)
                    else:
                        file_name = file_info.get("name", "unknown_file")
                        files_content.append(f"=== FILE: {file_name} ===\n{content}")
                        
        return "\n\n".join(files_content) if files_content else ""

    except Exception as e:
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[File extraction error: {str(e)}]"
        return ""

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

def _parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, LocalAssistantResponseModel: Any) -> Dict: # Added LocalAssistantResponseModel
    """Parse local model response, supporting both text and structured formats."""
    if is_structured and use_structured_output:
        try:
            parsed_json = json.loads(response)
            validated_model = LocalAssistantResponseModel(**parsed_json) # Use LocalAssistantResponseModel
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            return model_dict
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in Minion: {e}. Response was: {response[:500]}")
            return {"answer": response, "confidence": "LOW", "key_points": None, "citations": None, "parse_error": str(e)}
    
    # Fallback for non-structured processing
    return {"answer": response, "confidence": "MEDIUM", "key_points": None, "citations": None, "parse_error": None}

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

    overall_start_time = 0
    if valves.debug_mode:
        overall_start_time = asyncio.get_event_loop().time()
        debug_log.append(f"🔍 **Debug Info (Minion v0.2.0):**")
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

        if _is_final_answer(claude_response):
            actual_final_answer = claude_response.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_declared_final = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **The remote model indicates FINAL ANSWER READY.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 The remote model declared FINAL ANSWER READY in round {round_num + 1}. (Debug Mode)")
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

        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {round_num + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**\n")
    
    if not claude_declared_final and conversation_history:
        # This shouldn't happen with the fix above, but keep as fallback
        last_remote_msg = conversation_history[-1][1] if conversation_history[-1][0] == "assistant" else (conversation_history[-2][1] if len(conversation_history) > 1 and conversation_history[-2][0] == "assistant" else "No suitable final message from the remote model found.")
        actual_final_answer = f"Protocol ended without explicit final answer. The remote model's last response was: \"{last_remote_msg}\""
        if valves.show_conversation:
            conversation_log.append(f"⚠️ Protocol ended without the remote model providing a final answer.\n")

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
    
    return "\n".join(output_parts)

# Partials File: partials/minion_pipe_method.py
import asyncio
from typing import Any, List, Callable, Dict
from fastapi import Request


async def _call_claude_directly(valves: Any, query: str, call_claude_func: Callable) -> str: # Renamed for clarity
    """Fallback to direct Claude call when no context is available"""
    return await call_claude_func(valves, f"Please answer this question: {query}")

async def minion_pipe(
    pipe_self: Any,
    body: Dict[str, Any], # Typed body
    __user__: Dict[str, Any], # Typed __user__
    __request__: Request,
    __files__: List[Dict[str, Any]] = [], # Typed __files__
    __pipe_id__: str = "minion-claude",
) -> str:
    """Execute the Minion protocol with Claude"""
    try:
        # Validate configuration
        if not pipe_self.valves.anthropic_api_key: # Add ollama key check if necessary
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

        # Extract user message and context
        messages: List[Dict[str, Any]] = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."

        user_query: str = messages[-1]["content"]

        # Extract context from messages AND uploaded files
        context_from_messages: str = extract_context_from_messages(messages[:-1])
        context_from_files: str = await extract_context_from_files(pipe_self.valves, __files__)

        # Combine all context sources
        all_context_parts: List[str] = []
        if context_from_messages:
            all_context_parts.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files:
            all_context_parts.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")

        context: str = "\n\n".join(all_context_parts) if all_context_parts else ""

        if not context:
            # Pass the imported call_claude to _call_claude_directly
            direct_response = await _call_claude_directly(pipe_self.valves, user_query, call_claude_func=call_claude)
            return (
                "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                + direct_response
            )

        # Execute the Minion protocol, passing the imported call_claude, call_ollama, and LocalAssistantResponse
        # The _execute_minion_protocol itself expects these as arguments.
        result: str = await _execute_minion_protocol(
            valves=pipe_self.valves, 
            query=user_query, 
            context=context, 
            call_claude_func=call_claude,  # Pass imported function
            call_ollama_func=call_ollama,  # Pass imported function
            LocalAssistantResponseModel=LocalAssistantResponse # Pass imported class
        )
        return result

    except Exception as e:
        import traceback # Keep import here as it's conditional
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"


class Pipe:
    class Valves(MinionValves):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Minion v0.3.5 (Conversational)"

    def pipes(self):
        """Define the available models"""
        return [
            {
                "id": "minion-claude",
                "name": f" ({self.valves.local_model} + {self.valves.remote_model})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "minion-claude",
    ) -> str:
        """Execute the Minion protocol with Claude"""
        return await minion_pipe(self, body, __user__, __request__, __files__, __pipe_id__)