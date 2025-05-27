"""
title: Minion Protocol Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.2.0
description: Basic Minion protocol - conversational collaboration between local and cloud models
required_open_webui_version: 0.5.0
license: MIT License
"""

import asyncio
import aiohttp
import json # ADDED for schema serialization in prompt
from typing import List, Optional, Dict # MODIFIED (added Dict)
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore

class LocalAssistantResponse(BaseModel):
    """Structured response format for local assistant in Minion protocol"""
    answer: str = Field(description="The main response to the question")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW confidence in the answer")
    key_points: Optional[List[str]] = Field(default=None, description="Key points extracted from context")
    citations: Optional[List[str]] = Field(default=None, description="Direct quotes supporting the answer")

class Pipe:
    class Valves(BaseModel):
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
        max_rounds: int = Field(default=2, description="Maximum conversation rounds")
        show_conversation: bool = Field(
            default=True,
            description="Show full conversation between local and remote models",
        )
        timeout_local: int = Field(
            default=60, 
            description="Timeout for local model calls in seconds (local model processes full context)."
        )
        timeout_claude: int = Field(
            default=60, description="Timeout for Claude API calls in seconds."
        )
        max_tokens_claude: int = Field(
            default=4000, description="Maximum tokens for Claude's responses."
        )
        ollama_num_predict: int = Field(
            default=1000, description="num_predict for Ollama generation (max output tokens for local model)."
        )
        use_structured_output: bool = Field(
            default=False, 
            description="Use JSON structured output for local model responses (requires local model support)"
        )
        debug_mode: bool = Field(
            default=False, description="Show additional technical details"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Minion v0.2.0"

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

        try:
            # Validate configuration
            if not self.valves.anthropic_api_key:
                return "❌ **Error:** Please configure your Anthropic API key in the function settings."

            # Extract user message and context
            messages = body.get("messages", [])
            if not messages:
                return "❌ **Error:** No messages provided."

            user_query = messages[-1]["content"]

            # Extract context from messages AND uploaded files
            context_from_messages = self._extract_context_from_messages(messages[:-1])
            context_from_files = await self._extract_context_from_files(__files__)

            # Combine all context sources
            all_context = []
            if context_from_messages:
                all_context.append(
                    f"=== CONVERSATION CONTEXT ===\n{context_from_messages}"
                )
            if context_from_files:
                all_context.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")

            context = "\n\n".join(all_context) if all_context else ""

            if not context:
                return (
                    "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                    + await self._call_claude_directly(user_query)
                )

            # Execute the Minion protocol
            result = await self._execute_minion_protocol(user_query, context)
            return result

        except Exception as e:
            import traceback
            error_details = traceback.format_exc() if self.valves.debug_mode else str(e)
            return f"❌ **Error in Minion protocol:** {error_details}"

    async def _execute_minion_protocol(self, query: str, context: str) -> str:
        conversation_log = []
        debug_log = []
        conversation_history = [] # For Claude's context
        actual_final_answer = "No final answer was explicitly provided by Claude." 
        claude_declared_final = False

        overall_start_time = 0
        if self.valves.debug_mode:
            overall_start_time = asyncio.get_event_loop().time()
            debug_log.append(f"🔍 **Debug Info (Minion v0.2.0):**")
            debug_log.append(f"  - Query: {query[:100]}...")
            debug_log.append(f"  - Context length: {len(context)} chars")
            debug_log.append(f"  - Max rounds: {self.valves.max_rounds}")
            debug_log.append(f"  - Remote model: {self.valves.remote_model}")
            debug_log.append(f"  - Local model: {self.valves.local_model}")
            debug_log.append(f"  - Timeouts: Claude={self.valves.timeout_claude}s, Local={self.valves.timeout_local}s")
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

        for round_num in range(self.valves.max_rounds):
            if self.valves.debug_mode:
                debug_log.append(f"**⚙️ Starting Round {round_num + 1}/{self.valves.max_rounds}... (Debug Mode)**")
            
            if self.valves.show_conversation:
                conversation_log.append(f"### 🔄 Round {round_num + 1}")

            claude_prompt_for_this_round = ""
            if round_num == 0:
                claude_prompt_for_this_round = initial_claude_prompt
            else:
                claude_prompt_for_this_round = self._build_conversation_context(
                    conversation_history, query
                )
            
            claude_response = ""
            try:
                if self.valves.debug_mode: start_time_claude = asyncio.get_event_loop().time()
                claude_response = await self._call_claude(claude_prompt_for_this_round)
                if self.valves.debug_mode:
                    end_time_claude = asyncio.get_event_loop().time()
                    time_taken_claude = end_time_claude - start_time_claude
                    debug_log.append(f"  ⏱️ Claude call in round {round_num + 1} took {time_taken_claude:.2f}s. (Debug Mode)")
            except Exception as e:
                error_message = f"❌ Error calling Claude in round {round_num + 1}: {e}"
                conversation_log.append(error_message)
                if self.valves.debug_mode: debug_log.append(f"  {error_message} (Debug Mode)")
                actual_final_answer = "Minion protocol failed due to Claude API error."
                break 

            conversation_history.append(("assistant", claude_response))
            if self.valves.show_conversation:
                conversation_log.append(f"**🤖 Claude ({self.valves.remote_model}):**")
                conversation_log.append(f"{claude_response}\n")

            if self._is_final_answer(claude_response):
                actual_final_answer = claude_response.split("FINAL ANSWER READY.", 1)[1].strip()
                claude_declared_final = True
                if self.valves.show_conversation:
                    conversation_log.append(f"✅ **Claude indicates FINAL ANSWER READY.**\n")
                if self.valves.debug_mode:
                    debug_log.append(f"  🏁 Claude declared FINAL ANSWER READY in round {round_num + 1}. (Debug Mode)")
                break

            local_prompt = f"""You have access to the full context below. Claude (Anthropic's AI) is collaborating with you to answer a user's question.
CONTEXT:
{context}
ORIGINAL QUESTION: {query}
CLAUDE'S REQUEST: {claude_response}
Please provide a helpful, accurate response based on the context you have access to. Extract relevant information that answers Claude's specific question. Be concise but thorough.
If you are instructed to provide a JSON response (e.g., by a schema appended to this prompt), ensure your entire response is ONLY that valid JSON object, without any surrounding text, explanations, or markdown formatting like ```json ... ```."""
        # The schema itself will be appended by _call_ollama if structured output is enabled and a schema is provided.
            
            local_response_str = "" # Renamed from local_response
            try:
                if self.valves.debug_mode: start_time_ollama = asyncio.get_event_loop().time()
                # MODIFIED: Call _call_ollama with new parameters
                local_response_str = await self._call_ollama(
                    local_prompt,
                    use_json=True, # Enable JSON mode in Ollama if valve is on
                    schema=LocalAssistantResponse # Provide the schema model
                )
                local_response_data = self._parse_local_response(
                    local_response_str,
                    is_structured=True # Indicate that structured parsing should be attempted
                )
                if self.valves.debug_mode:
                    end_time_ollama = asyncio.get_event_loop().time()
                    time_taken_ollama = end_time_ollama - start_time_ollama
                    debug_log.append(f"  ⏱️ Local LLM call in round {round_num + 1} took {time_taken_ollama:.2f}s. (Debug Mode)")
            except Exception as e:
                error_message = f"❌ Error calling Local LLM in round {round_num + 1}: {e}"
                conversation_log.append(error_message)
                if self.valves.debug_mode: debug_log.append(f"  {error_message} (Debug Mode)")
                actual_final_answer = "Minion protocol failed due to Local LLM API error."
                break 

            response_for_claude = local_response_data.get("answer", "Error: Could not extract answer from local LLM.")
            if self.valves.use_structured_output and local_response_data.get("parse_error") and self.valves.debug_mode:
                response_for_claude += f" (Local LLM response parse error: {local_response_data['parse_error']})"
            elif not local_response_data.get("answer") and not local_response_data.get("parse_error"): # handles cases where answer might be missing from dict
                 response_for_claude = "Local LLM provided no answer."

            conversation_history.append(("user", response_for_claude))
            if self.valves.show_conversation:
                conversation_log.append(f"**💻 Local Model ({self.valves.local_model}):**")
                if self.valves.use_structured_output and local_response_data.get("parse_error") is None:
                    # Successfully parsed structured output
                    # Ensure json is imported for dumps
                    conversation_log.append(f"```json\n{json.dumps(local_response_data, indent=2)}\n```")
                elif self.valves.use_structured_output and local_response_data.get("parse_error"):
                    # Attempted structured output but failed
                    conversation_log.append(f"Attempted structured output, but failed. Raw response:\n{local_response_data.get('answer', 'Error displaying local response.')}")
                    conversation_log.append(f"(Parse Error: {local_response_data['parse_error']})")
                else:
                    # Plain text output (use_structured_output is False)
                    conversation_log.append(f"{local_response_data.get('answer', 'Error displaying local response.')}")
                conversation_log.append("\n") # Original newline

            if self.valves.debug_mode:
                current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
                debug_log.append(f"**🏁 Completed Round {round_num + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**\n")
        
        if not claude_declared_final and conversation_history:
             # If loop finished without "FINAL ANSWER READY.", take Claude's last response as provisional.
            last_claude_msg = conversation_history[-1][1] if conversation_history[-1][0] == "assistant" else (conversation_history[-2][1] if len(conversation_history) > 1 and conversation_history[-2][0] == "assistant" else "No suitable final message from Claude found.")
            actual_final_answer = f"Max rounds reached. Claude's last message was: \"{last_claude_msg}\""
            if self.valves.show_conversation:
                conversation_log.append(f"⚠️ Max rounds reached. Using Claude's last message as the result.\n")

        if self.valves.debug_mode:
            total_execution_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**⏱️ Total Minion protocol execution time: {total_execution_time:.2f}s. (Debug Mode)**")

        output_parts = []
        if self.valves.show_conversation:
            output_parts.append("## 🗣️ Collaboration Conversation")
            output_parts.extend(conversation_log)
            output_parts.append("---")
        if self.valves.debug_mode:
            output_parts.append("### 🔍 Debug Log") # Make it a sub-header
            output_parts.extend(debug_log)
            output_parts.append("---")

        output_parts.append(f"## 🎯 Final Answer") # Always use this header now
        output_parts.append(actual_final_answer)

        stats = self._calculate_token_savings(conversation_history, context, query)
        output_parts.append(f"\n## 📊 Efficiency Stats")
        output_parts.append(f"- **Protocol:** Minion (conversational)")
        output_parts.append(f"- **Remote model:** {self.valves.remote_model}")
        output_parts.append(f"- **Local model:** {self.valves.local_model}")
        output_parts.append(
            f"- **Conversation rounds:** {len(conversation_history) // 2}"
        )
        output_parts.append(f"- **Context size:** {len(context):,} characters")
        output_parts.append(f"")
        output_parts.append(
            f"## 💰 Token Savings Analysis ({self.valves.remote_model})"
        )
        output_parts.append(
            f"- **Traditional approach:** ~{stats['traditional_tokens']:,} tokens"
        )
        output_parts.append(f"- **Minion approach:** ~{stats['minion_tokens']:,} tokens")
        output_parts.append(f"- **💰 Token Savings:** ~{stats['percentage_savings']:.1f}%")
        
        return "\n".join(output_parts)

    def _calculate_token_savings(self, conversation_history: List, context: str, query: str) -> dict:
        """Calculate token savings for the Minion protocol"""
        chars_per_token = 3.5
                
        # Traditional approach: entire context + query sent to Claude
        traditional_tokens = int((len(context) + len(query)) / chars_per_token)
        
        # Minion approach: only conversation messages sent to Claude
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

    async def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        headers = {
            "x-api-key": self.valves.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.valves.remote_model,
            "max_tokens": self.valves.max_tokens_claude, # Using the valve
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=self.valves.timeout_claude
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
                    if self.valves.debug_mode: print(f"Unexpected Claude API response format: {result}") 
                    raise Exception("Unexpected response format from Anthropic API or empty content.")

    async def _call_ollama(self, prompt: str, use_json: bool = False, schema: Optional[BaseModel] = None) -> str:
        """Call Ollama API"""
        # Existing payload setup
        payload = {
            "model": self.valves.local_model,
            "prompt": prompt, # Original prompt
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": self.valves.ollama_num_predict},
        }

        if use_json and self.valves.use_structured_output and schema:
            payload["format"] = "json"
            # Add schema to prompt for models that support it
            # Ensure json is imported at the top of the file
            schema_for_prompt = json.dumps(schema.model_json_schema(indent=2)) # Pydantic v2
            schema_prompt_addition = f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_for_prompt}"
            payload["prompt"] = prompt + schema_prompt_addition # Append schema to original prompt
        elif "format" in payload: # Clean up if not using JSON format
            del payload["format"]

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.valves.ollama_base_url}/api/generate", json=payload, timeout=self.valves.timeout_local
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
                    if self.valves.debug_mode: print(f"Unexpected Ollama API response format: {result}")
                    raise Exception("Unexpected response format from Ollama API or no response field.")

    async def _call_claude_directly(self, query: str) -> str:
        """Fallback to direct Claude call when no context is available"""
        return await self._call_claude(f"Please answer this question: {query}")

    def _extract_context_from_messages(self, messages: List[dict]) -> str:
        """Extract context from conversation history"""
        context_parts = []

        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Assume messages longer than 200 chars contain context/documents
                if len(content) > 200:
                    context_parts.append(content)

        return "\n\n".join(context_parts)

    async def _extract_context_from_files(self, files: List[dict]) -> str:
        """Extract text content from uploaded files using Open WebUI's file system"""
        try:
            if not files:
                return ""

            files_content = []

            if self.valves.debug_mode:
                files_content.append(f"[DEBUG] Found {len(files)} uploaded files")

            # Process each uploaded file
            for file_info in files:
                if isinstance(file_info, dict):
                    file_content = await self._extract_file_content(file_info)
                    if file_content:
                        file_name = file_info.get("name", "unknown_file")
                        files_content.append(
                            f"=== FILE: {file_name} ===\n{file_content}"
                        )

            return "\n\n".join(files_content) if files_content else ""

        except Exception as e:
            if self.valves.debug_mode:
                return f"[File extraction error: {str(e)}]"
            return ""

    async def _extract_file_content(self, file_info: dict) -> str:
        """Extract text content from a single file using Open WebUI's file API"""
        try:
            # Open WebUI files have an 'id' field that we can use to fetch content
            file_id = file_info.get("id")
            file_name = file_info.get("name", "unknown")

            if not file_id:
                return f"[Could not get file ID for {file_name}]"

            if self.valves.debug_mode:
                return f"[DEBUG] File ID: {file_id}, Name: {file_name}, Info: {str(file_info)}]"

            # Try to get file content via Open WebUI's internal API
            # Note: This might need adjustment based on Open WebUI's internal structure

            # For now, return file info for debugging
            file_type = file_info.get("type", "unknown")
            file_size = file_info.get("size", "unknown")

            # If the file info contains content directly, use it
            if "content" in file_info:
                return file_info["content"]

            # Otherwise, indicate that we detected the file but need content access
            return f"[File detected: {file_name} (Type: {file_type}, Size: {file_size})\nNote: File content extraction needs to be configured - enable debug mode to see file structure]"

        except Exception as e:
            return f"[Error extracting file content: {str(e)}]"

    def _build_conversation_context(
        self, history: List[tuple], original_query: str
    ) -> str:
        context_parts = [
            f"You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: \"{original_query}\"",
            "The local assistant has full access to the source document and has been providing factual information extracted from it.",
            "",
            "CONVERSATION SO FAR (Your questions, Local Assistant's factual responses from the document):",
        ]

        for role, message in history:
            if role == "assistant": # Claude's previous message
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

    def _is_final_answer(self, response: str) -> bool:
        """Check if response contains the specific final answer marker."""
        return "FINAL ANSWER READY." in response

    def _parse_local_response(self, response: str, is_structured: bool = False) -> Dict:
        """Parse local model response, supporting both text and structured formats."""
        if is_structured and self.valves.use_structured_output:
            try:
                # Ensure json is imported at the top of the file for json.loads
                parsed_json = json.loads(response)
                validated_model = LocalAssistantResponse(**parsed_json)
                # Add a 'parse_error': None field for consistency
                model_dict = validated_model.model_dump() # Pydantic v2
                model_dict['parse_error'] = None
                return model_dict
            except Exception as e:
                if self.valves.debug_mode:
                    print(f"DEBUG: Failed to parse structured output in Minion: {e}. Response was: {response[:500]}")
                # Fallback for parsing failure
                return {"answer": response, "confidence": "LOW", "key_points": None, "citations": None, "parse_error": str(e)}
        
        # Fallback for non-structured processing or when use_structured_output is False
        return {"answer": response, "confidence": "MEDIUM", "key_points": None, "citations": None, "parse_error": None}
    # take a bow
