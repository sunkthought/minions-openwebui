"""
title: MinionS Protocol (Task Decomposition)
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.2.1
description: MinionS protocol for task decomposition and parallel processing.
required_open_webui_version: 0.5.0
license: MIT License
"""


# Dynamically Generated Header End

# --- Start of content from: common_imports.py ---
import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore
# --- End of content from: common_imports.py ---

# --- Start of content from: minions_models.py ---
class TaskResult(BaseModel):
    """
    Structured response format for individual task execution by a local model
    within the MinionS (multi-task, multi-round) protocol.
    This model defines the expected output structure when a local model completes
    a sub-task defined by the remote model.
    """
    explanation: str = Field(
        description="Brief explanation of the findings or the process taken to answer the task."
    )
    citation: Optional[str] = Field(
        default=None, 
        description="Direct quote from the analyzed text (chunk) that supports the answer. Should be None if no specific citation is applicable."
    )
    answer: Optional[str] = Field(
        default=None, 
        description="The specific information extracted or the answer to the sub-task. Should be None if the information is not found in the provided text chunk."
    )
    confidence: str = Field(
        default="LOW", 
        description="Confidence level in the provided answer/explanation (e.g., HIGH, MEDIUM, LOW)."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields during parsing
        # schema_extra = {
        #     "example": {
        #         "explanation": "The document mentions a budget of $500,000 for Phase 1.",
        #         "citation": "Page 5, Paragraph 2: 'Phase 1 of the project has been allocated a budget of $500,000.'",
        #         "answer": "$500,000",
        #         "confidence": "HIGH"
        #     }
        # }

# --- End of content from: minions_models.py ---

# --- Start of content from: minions_valves.py ---
class MinionsValves(BaseModel):
    """
    Configuration settings (valves) specifically for the MinionS (multi-task, multi-round) pipe.
    These settings control the behavior of the MinionS protocol, including API keys,
    model selections, timeouts, task decomposition parameters, and operational parameters.
    """
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for Claude."
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model (e.g., Claude) for task decomposition and synthesis. claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL for local model execution."
    )
    local_model: str = Field(
        default="llama3.2", description="Local Ollama model name for task execution."
    )
    max_rounds: int = Field(
        default=2, description="Maximum task decomposition rounds. Each round involves remote model decomposing tasks and local models executing them."
    )
    max_tasks_per_round: int = Field(
        default=3, description="Maximum number of new sub-tasks the remote model can create in a single decomposition round."
    )
    chunk_size: int = Field(
        default=5000, description="Maximum chunk size in characters for context fed to local models during task execution."
    )
    max_chunks: int = Field( # This was present in minions-fn-claude.py but not in minion-fn-claude.py
        default=2, description="Maximum number of document chunks to process per task by the local model. Helps manage processing load."
    )
    show_conversation: bool = Field(
        default=True,
        description="Show full task decomposition, local model execution details, and synthesis steps in the output.",
    )
    timeout_local: int = Field(
        default=30,
        description="Timeout in seconds for each local model call (per chunk, per task).",
    )
    debug_mode: bool = Field(
        default=False, description="Enable additional technical details and verbose logs for debugging."
    )
    max_round_timeout_failure_threshold_percent: int = Field(
        default=50, 
        description="If this percentage of local model calls (chunk executions) in a round time out, a warning is issued. This suggests results for that round might be incomplete."
    )
    max_tokens_claude: int = Field( # Renamed from max_tokens_claude for consistency if used generally
        default=2000, description="Maximum tokens for remote model (Claude) API calls during decomposition and synthesis."
    )
    timeout_claude: int = Field( # Renamed from timeout_claude for consistency
        default=60, description="Timeout in seconds for remote model (Claude) API calls."
    )
    ollama_num_predict: int = Field(
        default=1000, description="Maximum tokens (num_predict) for local Ollama model responses during task execution."
    )
    use_structured_output: bool = Field(
        default=False, 
        description="Enable JSON structured output for local model responses (requires local model to support JSON mode and the TaskResult schema)."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields passed to the model
        # an_example = MinionsValves().dict() # For schema generation

# --- End of content from: minions_valves.py ---

# --- Start of content from: common_api_calls.py ---
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

# --- End of content from: common_api_calls.py ---

# --- Start of content from: common_context_utils.py ---
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
# --- End of content from: common_context_utils.py ---

# --- Start of content from: minions_token_savings.py ---
ValvesType = Any 

def calculate_minions_token_savings(
    # valves: ValvesType, # Not strictly needed by current logic but kept for consistency if future versions use it
    decomposition_prompts: List[str], 
    synthesis_prompts: List[str],
    all_results_summary_for_claude: str, 
    final_response_claude: str, 
    context_length: int, 
    query_length: int,
    chars_per_token: float = 3.5  # Average characters per token
) -> Dict[str, Any]:
    """
    Calculates token savings for the MinionS protocol by comparing estimated tokens
    for a traditional single remote model call versus the actual remote model calls
    made during the MinionS protocol (decomposition and synthesis phases).

    Args:
        decomposition_prompts: List of prompts sent for task decomposition.
        synthesis_prompts: List of prompts sent for final synthesis.
        all_results_summary_for_claude: The summary of all local model findings sent for synthesis.
        final_response_claude: The final response from the remote model after synthesis.
        context_length: Length of the original full context.
        query_length: Length of the original user query.
        chars_per_token: Estimated average characters per token.

    Returns:
        A dictionary containing token usage and savings statistics for remote model calls.
    """
    # Tokens for traditional approach (full context + query to remote model)
    traditional_tokens_claude = int((context_length + query_length) / chars_per_token)
    
    # Tokens for MinionS approach (sum of all actual calls to remote model)
    minions_tokens_claude = 0
    
    # Add tokens from decomposition prompts
    for prompt_content in decomposition_prompts:
        minions_tokens_claude += int(len(prompt_content) / chars_per_token)
    
    # Add tokens from synthesis prompts
    for prompt_content in synthesis_prompts:
        minions_tokens_claude += int(len(prompt_content) / chars_per_token)
        
    # Add tokens from the summary of results sent during synthesis
    minions_tokens_claude += int(len(all_results_summary_for_claude) / chars_per_token)
    
    # Add tokens from the final response generated by the remote model
    minions_tokens_claude += int(len(final_response_claude) / chars_per_token)
    
    token_savings_claude = traditional_tokens_claude - minions_tokens_claude
    percentage_savings_claude = (token_savings_claude / traditional_tokens_claude * 100) if traditional_tokens_claude > 0 else 0
    
    return {
        'traditional_tokens_claude': traditional_tokens_claude,
        'minions_tokens_claude': minions_tokens_claude,
        'token_savings_claude': token_savings_claude,
        'percentage_savings_claude': percentage_savings_claude,
        'total_decomposition_rounds': len(decomposition_prompts) # Informational
    }

# --- End of content from: minions_token_savings.py ---

# --- Start of content from: minions_protocol_logic.py ---
def parse_minions_tasks(valves, claude_response: str) -> List[str]:
    """Parses tasks from the remote model's (Claude) response for MinionS protocol."""
    lines = claude_response.split("\n")
    tasks = []
    for line in lines:
        line = line.strip()
        if line.startswith(tuple(f"{i}." for i in range(1, 10))) or \
           line.startswith(tuple(f"{i})" for i in range(1, 10))) or \
           line.startswith(("- ", "* ", "+ ")):
            task_content = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            if len(task_content) > 10:  # Keep simple task filter
                tasks.append(task_content)
    return tasks[:valves.max_tasks_per_round]

def create_minions_chunks(valves, context: str) -> List[str]:
    """Creates chunks from the context string based on chunk_size valve."""
    if not context:
        return []
    # Ensure chunk_size is at least 1, and not larger than the context itself if context is very small.
    actual_chunk_size = max(1, min(valves.chunk_size, len(context))) 
    chunks = [
        context[i : i + actual_chunk_size] 
        for i in range(0, len(context), actual_chunk_size)
    ]
    # Limit number of chunks if max_chunks is set and positive
    if hasattr(valves, 'max_chunks') and valves.max_chunks > 0:
        chunks = chunks[:valves.max_chunks]
    return chunks

def parse_minions_local_response(
    response_text: str, 
    valves, 
    is_structured: bool = False,
    task_result_model: Optional = None
) -> Dict[str, Any]:
    """
    Parses the local model's response for MinionS, supporting both text and structured (JSON) formats.
    Includes logic to determine if the response signifies "no relevant information found".
    """
    if is_structured and valves.use_structured_output and task_result_model:
        try:
            parsed_json = json.loads(response_text)
            if isinstance(parsed_json, dict):
                model_dict = parsed_json
                model_dict['parse_error'] = None
                # Crucial for MinionS: Check if the structured response indicates "not found"
                if model_dict.get('answer') is None and model_dict.get('explanation') is not None:
                    model_dict['_is_none_equivalent'] = True
                else:
                    model_dict['_is_none_equivalent'] = False
                return model_dict
            else:
                raise ValueError("Parsed JSON is not a dictionary")
        except Exception as e:
            if valves.debug_mode:
                print(f"DEBUG: Failed to parse structured output in MinionS: {e}. Response was: {response_text[:500]}")
            is_none_equivalent_fallback = response_text.strip().upper() == "NONE"
            return {"answer": response_text, "explanation": response_text, "confidence": "LOW", "citation": None, "parse_error": str(e), "_is_none_equivalent": is_none_equivalent_fallback}
    
    is_none_equivalent_text = response_text.strip().upper() == "NONE"
    return {"answer": response_text, "explanation": response_text, "confidence": "MEDIUM", "citation": None, "parse_error": None, "_is_none_equivalent": is_none_equivalent_text}

async def execute_minions_tasks_on_chunks(
    valves,
    tasks: List[str], 
    chunks: List[str], 
    conversation_log: List[str],
    current_round: int,
    call_ollama_func,
    task_result_model,
    parse_local_response_func
) -> Dict[str, Any]:
    """Executes sub-tasks on document chunks using the local model."""
    overall_task_results = []
    total_chunk_processing_attempts = 0
    total_chunk_processing_timeouts = 0

    for task_idx, task in enumerate(tasks):
        if valves.show_conversation:
            conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
        
        results_for_this_task_from_chunks = []
        chunk_timeout_count_for_task = 0
        num_relevant_chunks_found = 0

        for chunk_idx, chunk in enumerate(chunks):
            total_chunk_processing_attempts += 1
            
            local_prompt_text = f'''Text to analyze (Chunk {chunk_idx + 1}/{len(chunks)} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}'''

            if valves.use_structured_output:
                local_prompt_text += f"\n\nProvide your answer ONLY as a valid JSON object matching the specified schema. If no relevant information is found in THIS SPECIFIC TEXT, ensure the 'answer' field in your JSON response is explicitly set to null (or None)."
            else:
                local_prompt_text += "\n\nProvide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\"."
            
            start_time_ollama_local = 0
            if valves.debug_mode:
                if valves.show_conversation:
                    conversation_log.append(f"   🔄 Task {task_idx + 1} - Trying chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} chars)... (Debug Mode)")
                start_time_ollama_local = asyncio.get_event_loop().time()

            try:
                response_str = await asyncio.wait_for(
                    call_ollama_func(
                        valves,
                        local_prompt_text,
                        use_json=True, 
                        schema=task_result_model 
                    ),
                    timeout=valves.timeout_local,
                )
                response_data = parse_local_response_func(
                    response_str,
                    valves,
                    is_structured=True,
                    task_result_model=task_result_model
                )

                if valves.debug_mode:
                    end_time_ollama_local = asyncio.get_event_loop().time()
                    time_taken_ollama_local = end_time_ollama_local - start_time_ollama_local
                    status_msg = "Parse Error" if response_data.get("parse_error") else ("No relevant info" if response_data.get('_is_none_equivalent') else "Relevant info found")
                    details_msg = f"Error: {response_data['parse_error']}, Raw: {response_data.get('answer', '')[:70]}..." if response_data.get("parse_error") else \
                                  (f"Response indicates no info. Confidence: {response_data.get('confidence', 'N/A')}" if response_data.get('_is_none_equivalent') else \
                                   f"Answer: {response_data.get('answer', '')[:70]}..., Confidence: {response_data.get('confidence', 'N/A')}")
                    if valves.show_conversation:
                         conversation_log.append(f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} processed by local model in {time_taken_ollama_local:.2f}s. Status: {status_msg}. Details: {details_msg} (Debug Mode)")

                if not response_data.get('_is_none_equivalent'):
                    extracted_info = response_data.get('answer') or response_data.get('explanation', 'Could not extract details.')
                    results_for_this_task_from_chunks.append(f"[Chunk {chunk_idx+1}]: {extracted_info}")
                    num_relevant_chunks_found += 1
            
            except asyncio.TimeoutError:
                total_chunk_processing_timeouts += 1
                chunk_timeout_count_for_task +=1
                if valves.show_conversation:
                    conversation_log.append(f"   ⏰ Task {task_idx + 1} - Chunk {chunk_idx + 1} timed out after {valves.timeout_local}s")
                if valves.debug_mode:
                    end_time_ollama_local = asyncio.get_event_loop().time()
                    time_taken_ollama_local = end_time_ollama_local - start_time_ollama_local
                    if valves.show_conversation:
                         conversation_log.append(f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} TIMEOUT after {time_taken_ollama_local:.2f}s. (Debug Mode)")
            except Exception as e:
                if valves.show_conversation:
                    conversation_log.append(f"   ❌ Task {task_idx + 1} - Chunk {chunk_idx + 1} error: {str(e)}")
        
        if results_for_this_task_from_chunks:
            aggregated_result_for_task = "\n".join(results_for_this_task_from_chunks)
            overall_task_results.append({"task": task, "result": aggregated_result_for_task, "status": "success"})
            if valves.show_conversation:
                conversation_log.append(f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). First result snippet: {results_for_this_task_from_chunks[0][:100]}...")
        elif chunk_timeout_count_for_task == len(chunks) and len(chunks) > 0:
            overall_task_results.append({"task": task, "result": f"Timeout on all {len(chunks)} chunks", "status": "timeout_all_chunks"})
            if valves.show_conversation:
                conversation_log.append(f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out.")
        else:
            overall_task_results.append({"task": task, "result": "Information not found or not extracted from any relevant chunk", "status": "not_found"})
            if valves.show_conversation:
                conversation_log.append(f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found/extracted in any chunk.")
    
    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_chunk_processing_attempts,
        "total_chunk_processing_timeouts": total_chunk_processing_timeouts
    }

async def execute_minions_protocol(
    valves,
    query: str, 
    context: str,
    call_claude_func,
    call_ollama_func,
    task_result_model,
    parse_tasks_func,
    create_chunks_func,
    execute_tasks_on_chunks_func,
    parse_local_response_func,
    calculate_token_savings_func
) -> str:
    """Executes the MinionS (multi-task, multi-round) protocol."""
    conversation_log: List[str] = []
    debug_log: List[str] = []
    scratchpad_content = ""
    all_round_results_aggregated: List[Dict[str, Any]] = []
    decomposition_prompts_history: List[str] = []
    synthesis_prompts_history: List[str] = []
    final_answer = "No answer could be synthesized."
    claude_provided_final_answer = False
    total_tasks_executed_local = 0
    total_chunk_processing_timeouts_accumulated = 0

    overall_start_time = 0.0
    if valves.debug_mode:
        overall_start_time = asyncio.get_event_loop().time()
        debug_log.append(f"🔍 **Debug Info (MinionS Protocol v0.3.0):**\n- Query: {query[:100]}...\n- Context length: {len(context)} chars")
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**")

    chunks = create_chunks_func(valves, context)
    if not chunks and context:
        return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size/max_chunks."
    
    if not context:
        if valves.show_conversation: conversation_log.append("ℹ️ No context to process with MinionS. Attempting direct call to remote model.")
        try:
            final_answer = await call_claude_func(valves, f"Please answer this question: {query}")
            return f"## 🎯 Final Answer (Direct)\n{final_answer}"
        except Exception as e:
            return f"❌ **Error in direct remote model call:** {str(e)}"

    for current_round in range(valves.max_rounds):
        if valves.debug_mode: 
            debug_log.append(f"**⚙️ Starting Round {current_round + 1}/{valves.max_rounds}... (Debug Mode)**")
        if valves.show_conversation:
            conversation_log.append(f"### 🎯 Round {current_round + 1}/{valves.max_rounds} - Task Decomposition Phase")
        
        decomposition_prompt = f'''You are a supervisor LLM in a multi-round process. Your goal is to answer: "{query}"
Context has been split into {len(chunks)} chunks. A local LLM will process these chunks for each task you define.
Scratchpad (previous findings): {scratchpad_content if scratchpad_content else "Nothing yet."}

Based on the scratchpad and the original query, identify up to {valves.max_tasks_per_round} specific, simple tasks for the local assistant.
If the information in the scratchpad is sufficient to answer the query, respond ONLY with the exact phrase "FINAL ANSWER READY." followed by the comprehensive answer.
Otherwise, list the new tasks clearly. Ensure tasks are actionable. Avoid redundant tasks.
Format tasks as a simple list (e.g., 1. Task A, 2. Task B).'''
        decomposition_prompts_history.append(decomposition_prompt)
        
        claude_response_text = ""
        try:
            if valves.debug_mode: start_time_claude_decomp = asyncio.get_event_loop().time()
            claude_response_text = await call_claude_func(valves, decomposition_prompt)
            if valves.debug_mode:
                debug_log.append(f"⏱️ Remote model call (Decomposition Round {current_round+1}) took {asyncio.get_event_loop().time() - start_time_claude_decomp:.2f}s. (Debug Mode)")
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Remote Model (Decomposition - Round {current_round + 1}):**\n{claude_response_text}\n")
        except Exception as e:
            if valves.show_conversation: conversation_log.append(f"❌ Error calling remote model for decomposition in round {current_round + 1}: {e}")
            break 

        if "FINAL ANSWER READY." in claude_response_text:
            final_answer = claude_response_text.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_provided_final_answer = True
            if valves.show_conversation: conversation_log.append(f"**🤖 Remote model indicates final answer is ready in round {current_round + 1}.**")
            scratchpad_content += f"\n\n**Round {current_round + 1}:** Remote model provided final answer."
            break 

        tasks = parse_tasks_func(valves, claude_response_text)
        if valves.debug_mode:
            debug_log.append(f"   Identified {len(tasks)} tasks for Round {current_round + 1}. (Debug Mode)")

        if not tasks:
            if valves.show_conversation: conversation_log.append(f"**🤖 Remote model provided no new tasks in round {current_round + 1}. Proceeding to final synthesis.**")
            break
        
        total_tasks_executed_local += len(tasks)
        if valves.show_conversation:
            conversation_log.append(f"### ⚡ Round {current_round + 1} - Parallel Execution Phase (Processing {len(chunks)} chunks for {len(tasks)} tasks)")
        
        execution_details = await execute_tasks_on_chunks_func(
            valves, tasks, chunks, conversation_log, current_round + 1, 
            call_ollama_func, task_result_model, parse_local_response_func
        )
        current_round_task_results = execution_details["results"]

        if execution_details["total_chunk_processing_attempts"] > 0:
            timeout_percentage = (execution_details["total_chunk_processing_timeouts"] / execution_details["total_chunk_processing_attempts"]) * 100
            log_msg = f"**📈 Round {current_round + 1} Local Model Timeout Stats:** {execution_details['total_chunk_processing_timeouts']}/{execution_details['total_chunk_processing_attempts']} chunk calls timed out ({timeout_percentage:.1f}%)."
            if valves.show_conversation: conversation_log.append(log_msg)
            if valves.debug_mode: debug_log.append(log_msg)
            if timeout_percentage >= valves.max_round_timeout_failure_threshold_percent:
                warn_msg = f"⚠️ **Warning:** Round {current_round + 1} exceeded local model timeout threshold. Results may be incomplete."
                if valves.show_conversation: conversation_log.append(warn_msg)
                if valves.debug_mode: debug_log.append(warn_msg)
                scratchpad_content += f"\n\n**Note from Round {current_round + 1}:** High local model timeout rate ({timeout_percentage:.1f}%)."

        round_summary_parts = [f"- {'✅' if r['status'] == 'success' else '❓'} Task: {r['task']}, Result: {r['result'][:100]}..." for r in current_round_task_results]
        if round_summary_parts: scratchpad_content += f"\n\n**Results from Round {current_round + 1}:**\n" + "\n".join(round_summary_parts)
        
        all_round_results_aggregated.extend(current_round_task_results)
        total_chunk_processing_timeouts_accumulated += execution_details["total_chunk_processing_timeouts"]

        if valves.debug_mode: debug_log.append(f"**🏁 Completed Round {current_round + 1}. Cumulative time: {asyncio.get_event_loop().time() - overall_start_time:.2f}s. (Debug Mode)**")
        if current_round == valves.max_rounds - 1 and valves.show_conversation:
            conversation_log.append(f"**🏁 Reached max rounds ({valves.max_rounds}). Proceeding to final synthesis.**")

    if not claude_provided_final_answer:
        if valves.show_conversation: conversation_log.append("\n### 🔄 Final Synthesis Phase")
        successful_results = [r for r in all_round_results_aggregated if r['status'] == 'success']
        if not successful_results:
            final_answer = "No information was successfully gathered by local models across the rounds."
            if valves.show_conversation: conversation_log.append(f"**🤖 Remote Model (Synthesis):** {final_answer}")
        else:
            synthesis_input_summary = "\n".join([f"- Task: {r['task']}\n  Result: {r['result']}" for r in successful_results])
            synthesis_prompt = f'''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary}

If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.
Final Answer:'''
            synthesis_prompts_history.append(synthesis_prompt)
            try:
                if valves.debug_mode: start_time_synth = asyncio.get_event_loop().time()
                final_answer = await call_claude_func(valves, synthesis_prompt)
                if valves.debug_mode: debug_log.append(f"⏱️ Remote model call (Final Synthesis) took {asyncio.get_event_loop().time() - start_time_synth:.2f}s. (Debug Mode)")
                if valves.show_conversation: conversation_log.append(f"**🤖 Remote Model (Final Synthesis):**\n{final_answer}")
            except Exception as e:
                if valves.show_conversation: conversation_log.append(f"❌ Error during final synthesis: {e}")
                final_answer = "Error during final synthesis. Raw findings might be in conversation log."
    
    output_parts = []
    if valves.show_conversation:
        output_parts.append("## 🗣️ MinionS Collaboration (Multi-Round)")
        output_parts.extend(conversation_log)
        output_parts.append("---")
    if valves.debug_mode:
        output_parts.append("### 🔍 Debug Log")
        output_parts.extend(debug_log)
        output_parts.append("---")
    output_parts.append(f"## 🎯 Final Answer")
    output_parts.append(final_answer)

    stats = calculate_token_savings_func(
        decomposition_prompts_history, 
        synthesis_prompts_history,
        scratchpad_content, 
        final_answer, 
        len(context), 
        len(query)
    )
    
    successful_local_tasks = len([r for r in all_round_results_aggregated if r['status'] == 'success'])
    tasks_all_chunks_timed_out = len([r for r in all_round_results_aggregated if r['status'] == 'timeout_all_chunks'])

    output_parts.extend([
        f"\n## 📊 MinionS Efficiency Stats (v0.3.0)",
        f"- **Protocol:** MinionS (Multi-Round)",
        f"- **Rounds executed:** {stats.get('total_decomposition_rounds', 0)}/{valves.max_rounds}",
        f"- **Total tasks for local model:** {total_tasks_executed_local}",
        f"- **Successful tasks (local):** {successful_local_tasks}",
        f"- **Tasks where all chunks timed out (local):** {tasks_all_chunks_timed_out}",
        f"- **Total individual chunk processing timeouts (local):** {total_chunk_processing_timeouts_accumulated}",
        f"- **Context size:** {len(context):,} characters",
        f"\n## 💰 Token Savings Analysis (Remote Model: {valves.remote_model})",
        f"- **Traditional single call (est.):** ~{stats.get('traditional_tokens_claude', 0):,} tokens",
        f"- **MinionS multi-round (Remote Model only):** ~{stats.get('minions_tokens_claude', 0):,} tokens",
        f"- **💰 Est. Remote Model Token savings:** ~{stats.get('percentage_savings_claude', 0.0):.1f}%"
    ])
            
    return "\n".join(output_parts)
# --- End of content from: minions_protocol_logic.py ---

# --- Start of content from: minions_pipe_method.py ---
async def _call_claude_directly_minions_helper(pipe_self, query: str) -> str:
    """
    Fallback to direct Claude call when no context is available (MinionS version).
    """
    return await pipe_self.call_claude_api(pipe_self.valves, f"Please answer this question: {query}")

async def minions_pipe(
    self, # Instance of the specific MinionS Pipe class
    body: Dict[str, Any],
    __user__: Dict[str, Any], 
    __request__, # fastapi.Request
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "" 
) -> str:
    """
    Executes the MinionS (multi-task, multi-round) protocol.
    This function is intended to be the 'pipe' method of a class that has 'valves'
    and access to helper functions for context extraction, protocol execution, etc.
    """
    try:
        if not self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

        messages = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."
        user_query = messages[-1]["content"]

        # Context extraction using helpers from common_context_utils.py
        context_from_messages = self.extract_context_from_messages(messages[:-1])
        context_from_files = await self.extract_context_from_files(self.valves, __files__)

        all_context_parts = []
        if context_from_messages: 
            all_context_parts.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files: 
            all_context_parts.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")
        context = "\n\n".join(all_context_parts) if all_context_parts else ""

        if not context:
            return (
                "ℹ️ **Note:** No significant context detected. Using standard remote model response.\n\n"
                + await _call_claude_directly_minions_helper(self, user_query)
            )

        # Execute the MinionS protocol using the main logic function
        result = await self.execute_minions_protocol(
            self.valves,
            user_query,
            context,
            self.call_claude_api, 
            self.call_ollama_api, 
            self.task_result_model,
            self.parse_tasks_func,
            self.create_chunks_func,
            self.execute_tasks_on_chunks_func,
            self.parse_local_response_func,
            self.calculate_token_savings_func
        )
        return result

    except Exception as e:
        error_details = traceback.format_exc() if (hasattr(self.valves, 'debug_mode') and self.valves.debug_mode) else str(e)
        return f"❌ **Error in MinionS protocol:** {error_details}"
# --- End of content from: minions_pipe_method.py ---


# --- Final Function Class Definition for Open WebUI ---
class Function:
    class Valves(MinionsValves):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, Any]]:
        local_model_name = getattr(self.valves, 'local_model', 'local_model')
        remote_model_name = getattr(self.valves, 'remote_model', 'remote_model')
        return [{
            "id": "minions-claude-generated",
            "name": "MinionS v0.2.1 (" + local_model_name + " + " + remote_model_name + ")",
        }]

    async def pipe(self, body: Dict[str, Any], __user__: Dict[str, Any], __request__: Any, __files__: List[Dict[str, Any]] = [], __pipe_id__: str = ""):
        # Assign common utilities (available globally due to concatenation)
        self.extract_context_from_messages = extract_context_from_messages 
        self.extract_context_from_files = extract_context_from_files
        self.call_claude_api = call_claude 
        self.call_ollama_api = call_ollama
        
        # Assign the main protocol execution function
        self.execute_minions_protocol = execute_minions_protocol
        
        # Assign dependencies for the main protocol execution function
        self.call_claude_api = call_claude
        self.call_ollama_api = call_ollama
        self.task_result_model = TaskResult
        self.calculate_token_savings_func = calculate_minions_token_savings
        self.parse_tasks_func = parse_minions_tasks
        self.create_chunks_func = create_minions_chunks
        self.execute_tasks_on_chunks_func = execute_minions_tasks_on_chunks
        self.parse_local_response_func = parse_minions_local_response
        self.parse_tasks_func = parse_minions_tasks
        self.create_chunks_func = create_minions_chunks
        self.execute_tasks_on_chunks_func = execute_minions_tasks_on_chunks
        self.parse_local_response_func = parse_minions_local_response
        
        # Call the main standalone pipe method
        return await minions_pipe(
            self, 
            body, 
            __user__, 
            __request__, 
            __files__, 
            __pipe_id__ if __pipe_id__ else "minions-claude-generated"
        )

# --- End of Final Function Class Definition ---