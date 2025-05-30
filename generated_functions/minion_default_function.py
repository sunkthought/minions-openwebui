"""
title: Minion Protocol (Conversational)
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.2.1
description: Conversational Minion protocol for collaboration between local and remote models.
required_open_webui_version: 0.5.0
license: MIT License
"""


import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore

from typing import List, Optional
from pydantic import BaseModel, Field

class LocalAssistantResponse(BaseModel):
    """
    Structured response format for the local assistant in the Minion (conversational) protocol.
    This model defines the expected output structure when the local model processes
    a request from the remote (Claude) model.
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


from pydantic import BaseModel, Field

class MinionValves(BaseModel):
    """
    Configuration settings (valves) specifically for the Minion (conversational) pipe.
    These settings control the behavior of the Minion protocol, including API keys,
    model selections, timeouts, and operational parameters.
    """
    # Essential configuration only
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for Claude"
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Claude model (claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality)",
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
        default=60, description="Timeout for Claude API calls in seconds."
    )
    max_tokens_claude: int = Field(
        default=4000, description="Maximum tokens for Claude's responses."
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

    # The following class is part of the Pydantic configuration and is standard.
    # It ensures that extra fields passed to the model are ignored rather than causing an error.
    class Config:
        extra = "ignore"


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

from typing import List, Dict, Any, Tuple

def calculate_minion_token_savings(
    conversation_history: List[Tuple[str, str]], 
    context: str, 
    query: str,
    chars_per_token: float = 3.5  # Average characters per token, can be adjusted
) -> Dict[str, Any]:
    """
    Calculates token savings for the Minion (conversational) protocol.

    Args:
        conversation_history: A list of tuples, where each tuple is (role, message_content).
                              'assistant' role typically refers to the remote model.
        context: The full context string that would have been sent in a traditional approach.
        query: The user's query string.
        chars_per_token: An estimated average number of characters per token.

    Returns:
        A dictionary containing:
            'traditional_tokens': Estimated tokens if context+query were sent directly.
            'minion_tokens': Estimated tokens used by the remote model in the Minion protocol.
            'token_savings': Difference between traditional and Minion tokens.
            'percentage_savings': Percentage of tokens saved.
    """
    # Calculate tokens for the traditional approach (sending full context + query)
    traditional_tokens = int((len(context) + len(query)) / chars_per_token)
    
    # Calculate tokens for the Minion approach
    # This typically counts tokens from messages involving the remote model (e.g., Claude)
    # In this specific Minion protocol, 'assistant' messages in history are Claude's.
    minion_protocol_remote_model_tokens = 0
    for role, message_content in conversation_history:
        if role == "assistant":  # Messages from/to the remote model
            minion_protocol_remote_model_tokens += int(len(message_content) / chars_per_token)

    # The 'minion_tokens' are those specifically attributed to the remote model's involvement
    minion_tokens = minion_protocol_remote_model_tokens
    
    token_savings = traditional_tokens - minion_tokens
    percentage_savings = (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    
    return {
        'traditional_tokens': traditional_tokens,
        'minion_tokens': minion_tokens,
        'token_savings': token_savings,
        'percentage_savings': percentage_savings
    }


import asyncio
import json

def build_minion_conversation_context(history: List[Tuple[str, str]], original_query: str) -> str:
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

    context_parts.extend([
        "",
        "REMINDER: The local assistant's responses are factual information extracted directly from the document.",
        "Based on ALL information provided by the local assistant so far, can you now provide a complete and comprehensive answer to the user's ORIGINAL QUESTION?",
        "If YES: Respond ONLY with the exact phrase 'FINAL ANSWER READY.' followed by your comprehensive final answer. Ensure your answer directly addresses the original query using the information gathered.",
        "If NO: Ask ONE more specific, targeted question to the local assistant to obtain the remaining information you need from the document. Be precise. Do not ask for the document itself or express that you cannot see it.",
    ])
    return "\n".join(context_parts)

def is_minion_final_answer(response: str) -> bool:
    """Checks if the response from the remote model contains the final answer marker."""
    return "FINAL ANSWER READY." in response

def parse_minion_local_response(
    response_text: str, 
    valves, 
    is_structured: bool = False,
    response_model: Optional = None
) -> Dict[str, Any]:
    """
    Parses the local model's response, supporting both text and structured (JSON) formats.
    """
    if is_structured and valves.use_structured_output and response_model:
        try:
            parsed_json = json.loads(response_text)
            if isinstance(parsed_json, dict):
                 model_dict = parsed_json
                 model_dict['parse_error'] = None
                 return model_dict
            else:
                 raise ValueError("Parsed JSON is not a dictionary.")
        except Exception as e:
            if valves.debug_mode:
                print(f"DEBUG: Failed to parse structured output in Minion: {e}. Response was: {response_text[:500]}")
            return {"answer": response_text, "confidence": "LOW", "key_points": None, "citations": None, "parse_error": str(e)}
    
    return {"answer": response_text, "confidence": "MEDIUM", "key_points": None, "citations": None, "parse_error": None}

async def execute_minion_protocol(
    valves,
    query: str,
    context: str,
    call_claude_func,
    call_ollama_func,
    local_assistant_response_model,
    calculate_minion_token_savings_func
) -> str:
    """Executes the Minion conversational protocol."""
    conversation_log = []
    debug_log = []
    conversation_history: List[Tuple[str, str]] = []
    actual_final_answer = "No final answer was explicitly provided by the remote model."
    claude_declared_final = False

    overall_start_time = 0
    if valves.debug_mode:
        overall_start_time = asyncio.get_event_loop().time()
        debug_log.append(f"🔍 **Debug Info (Minion Protocol v0.3.0):**")
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

    stats = calculate_minion_token_savings_func(conversation_history, context, query)
    output_parts.append(f"\n## 📊 Efficiency Stats")
    output_parts.append(f"- **Protocol:** Minion (conversational)")
    output_parts.append(f"- **Remote model:** {valves.remote_model}")
    output_parts.append(f"- **Local model:** {valves.local_model}")
    output_parts.append(f"- **Conversation rounds:** {len(conversation_history) // 2}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    output_parts.append(f"")
    output_parts.append(f"## 💰 Token Savings Analysis ({valves.remote_model})") 
    output_parts.append(f"- **Traditional approach:** ~{stats.get('traditional_tokens', 0):,} tokens")
    output_parts.append(f"- **Minion approach:** ~{stats.get('minion_tokens', 0):,} tokens")
    output_parts.append(f"- **💰 Token Savings:** ~{stats.get('percentage_savings', 0.0):.1f}%")
    
    return "\n".join(output_parts)

import traceback

async def _call_claude_directly_helper(pipe_self, query: str) -> str:
    """
    Fallback to direct Claude call when no context is available.
    This is a helper function for minion_pipe.
    """
    return await pipe_self.call_claude_api(pipe_self.valves, f"Please answer this question: {query}")

async def minion_pipe(
    self, # Instance of the specific Pipe class
    body: Dict[str, Any],
    __user__: Dict[str, Any],
    __request__, # fastapi.Request
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = ""
) -> str:
    """
    Executes the Minion (conversational) protocol.
    This function is intended to be the 'pipe' method of a class that has 'valves'
    and access to helper functions for context extraction and protocol execution.
    """
    try:
        # Validate configuration
        if not self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

        messages = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."

        user_query = messages[-1]["content"]

        # Extract context using helper functions expected to be on 'self'
        context_from_messages = self.extract_context_from_messages(messages[:-1])
        context_from_files = await self.extract_context_from_files(self.valves, __files__)

        all_context = []
        if context_from_messages:
            all_context.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files:
            all_context.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")
        context = "\n\n".join(all_context) if all_context else ""

        if not context:
            return (
                "ℹ️ **Note:** No significant context detected. Using standard remote model response.\n\n"
                + await _call_claude_directly_helper(self, user_query)
            )

        # Execute the Minion protocol using the main logic function
        result = await self.execute_minion_protocol(
            self.valves, 
            user_query, 
            context,
            self.call_claude_api,
            self.call_ollama_api,
            self.local_assistant_response_model,
            self.calculate_minion_token_savings_func
        )
        return result

    except Exception as e:
        error_details = traceback.format_exc() if (hasattr(self.valves, 'debug_mode') and self.valves.debug_mode) else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"


class Pipe:
    class Valves(MinionValves):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Minion v0.2.1"

    def pipes(self):
        """Define the available models"""
        return [
            {
                "id": "minion-claude-generated",
                "name": f" ({self.valves.local_model} + {self.valves.remote_model})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "minion-claude-generated",
    ) -> str:
        """Execute the Minion protocol with Claude"""
        
        # Set up references to functions needed by the pipe method
        self.extract_context_from_messages = extract_context_from_messages
        self.extract_context_from_files = extract_context_from_files
        self.call_claude_api = call_claude
        self.call_ollama_api = call_ollama
        self.execute_minion_protocol = execute_minion_protocol
        
        # Protocol-specific dependencies
        self.call_claude_api = call_claude
        self.call_ollama_api = call_ollama
        self.local_assistant_response_model = LocalAssistantResponse
        self.calculate_minion_token_savings_func = calculate_minion_token_savings
        
        # Call the main pipe method
        return await minion_pipe(
            self,
            body,
            __user__,
            __request__,
            __files__,
            __pipe_id__
        )