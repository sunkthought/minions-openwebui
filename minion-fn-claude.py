"""
title: Minion Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.1.0
description: Basic Minion protocol - conversational collaboration between local and cloud models
required_open_webui_version: 0.5.0
license: MIT License
"""

import asyncio
import json
import os
import aiohttp
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore

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
            default=45,
            description="Timeout for local model calls in seconds (increase if model is slow)",
        )
        debug_mode: bool = Field(
            default=False, description="Show additional technical details"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Minion v0.1.0"

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
        """Execute the basic Minion protocol with Claude"""

        conversation_log = []
        debug_log = []

        if self.valves.debug_mode:
            debug_info = f"🔍 **Debug Info:**\n"
            debug_info += f"- Query: {query[:100]}...\n"
            debug_info += f"- Context length: {len(context)} chars\n"
            debug_info += f"- Max rounds: {self.valves.max_rounds}\n"
            debug_info += f"- Remote model: {self.valves.remote_model}\n\n"
            debug_log.append(debug_info)

        # Initial prompt for Claude
        claude_prompt = f"""You are collaborating with a local AI assistant that has access to full context/documents, but you don't have direct access to them.

Your goal: Answer this question: "{query}"

The local assistant can see this context (you cannot): [CONTEXT: {len(context)} characters of text including uploaded documents]

Please ask the local assistant specific, focused questions to gather the information you need. Be direct and precise in your requests. Ask only what you need to answer the original question."""

        conversation_history = []

        for round_num in range(self.valves.max_rounds):
            conversation_log.append(f"### 🔄 Round {round_num + 1}")

            # Claude asks question or provides answer
            if round_num == 0:
                claude_response = await self._call_claude(claude_prompt)
            else:
                # Build conversation context for Claude
                conv_context = self._build_conversation_context(
                    conversation_history, query
                )
                claude_response = await self._call_claude(conv_context)

            conversation_history.append(("assistant", claude_response))

            # Show Claude's message
            conversation_log.append(f"**🤖 Claude ({self.valves.remote_model}):**")
            conversation_log.append(f"{claude_response}\n")

            # Check if Claude provided a final answer
            if self._is_final_answer(claude_response):
                conversation_log.append(
                    "✅ **Claude provided final answer - collaboration complete**\n"
                )
                break

            # Local model responds with context access
            local_prompt = f"""You have access to the full context below. Claude (Anthropic's AI) is collaborating with you to answer a user's question.

CONTEXT:
{context}

ORIGINAL QUESTION: {query}

CLAUDE'S REQUEST: {claude_response}

Please provide a helpful, accurate response based on the context you have access to. Extract relevant information that answers Claude's specific question. Be concise but thorough."""

            local_response = await self._call_ollama(local_prompt)
            conversation_history.append(("user", local_response))

            # Show local model's response
            conversation_log.append(f"**💻 Local Model ({self.valves.local_model}):**")
            conversation_log.append(f"{local_response}\n")

        # Format final response
        final_answer = (
            conversation_history[-1][1]
            if conversation_history
            else "No response generated."
        )

        # Build output
        output_parts = []

        # Show the conversation if requested
        if self.valves.show_conversation:
            output_parts.append("## 🗣️ Collaboration Conversation")
            output_parts.extend(conversation_log)
            output_parts.append("---")

        # Show debug info if requested
        if self.valves.debug_mode:
            output_parts.extend(debug_log)

        # Extract just the final answer for clean presentation
        if self._is_final_answer(final_answer):
            output_parts.append(f"## 🎯 Final Answer")
            output_parts.append(final_answer)
        else:
            output_parts.append(f"## 🎯 Result")
            output_parts.append(final_answer)

        # Add protocol stats with token savings
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
            "max_tokens": 5000,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Anthropic API error: {response.status} - {error_text}"
                    )

                result = await response.json()

                # Extract content from Anthropic's response format
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0]["text"]
                else:
                    raise Exception("Unexpected response format from Anthropic API")

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        payload = {
            "model": self.valves.local_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1000},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.valves.ollama_base_url}/api/generate", json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Ollama API error: {response.status} - {error_text}"
                    )

                result = await response.json()
                return result["response"]

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
        """Build context for Claude based on conversation history"""
        context_parts = [
            f"ORIGINAL QUESTION: {original_query}",
            "",
            "CONVERSATION SO FAR:",
        ]

        for role, message in history:
            if role == "assistant":
                context_parts.append(f"You previously asked: {message}")
            else:
                context_parts.append(f"Local assistant responded: {message}")

        context_parts.extend(
            [
                "",
                "Based on this conversation, do you now have enough information to provide a complete answer to the original question?",
                "If YES: Provide your final answer.",
                "If NO: Ask ONE more specific question to get the missing information.",
            ]
        )

        return "\n".join(context_parts)

    def _is_final_answer(self, response: str) -> bool:
        """Check if response contains a final answer"""
        final_indicators = [
            "final answer",
            "in conclusion",
            "therefore",
            "based on this information",
            "the answer is",
            "to summarize",
            "in summary",
            "final response",
            "based on the information provided",
            "here's my answer",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in final_indicators)
    # take a bow 
