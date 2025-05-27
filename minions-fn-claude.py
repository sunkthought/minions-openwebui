"""
title: MinionS Protocol Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.1.0
description: MinionS protocol - task decomposition and parallel processing between local and cloud models
required_open_webui_version: 0.5.0
"""

import asyncio
import aiohttp
from typing import Dict, List
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
        max_rounds: int = Field(default=2, description="Maximum task decomposition rounds")
        max_tasks_per_round: int = Field(
            default=3, description="Maximum number of tasks to create per round"
        )
        chunk_size: int = Field(
            default=5000, description="Maximum chunk size in characters"
        )
        max_chunks: int = Field(
            default=2, description="Maximum number of document chunks to process"
        )
        show_conversation: bool = Field(
            default=True,
            description="Show full task decomposition and execution details",
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
        self.name = "MinionS v0.1.0"

    def pipes(self):
        """Define the available models"""
        return [
            {
                "id": "minions-claude",
                "name": f" ({self.valves.local_model} + {self.valves.remote_model})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "minions-claude",
    ) -> str:
        """Execute the MinionS protocol with Claude"""

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

            # Execute the MinionS protocol
            result = await self._execute_minions_protocol(user_query, context)
            return result

        except Exception as e:
            import traceback
            error_details = traceback.format_exc() if self.valves.debug_mode else str(e)
            return f"❌ **Error in MinionS protocol:** {error_details}"

    async def _execute_minions_protocol(self, query: str, context: str) -> str:
        """Execute the advanced MinionS protocol with task decomposition"""

        conversation_log = []
        debug_log = []

        try:
            if self.valves.debug_mode:
                debug_info = f"🔍 **Debug Info (MinionS):**\n"
                debug_info += f"- Query: {query[:100]}...\n"
                debug_info += f"- Context length: {len(context)} chars\n"
                debug_info += f"- Protocol: MinionS (task decomposition)\n\n"
                debug_log.append(debug_info)

            conversation_log.append("### 🎯 Task Decomposition Phase")

            # Step 1: Task decomposition by Claude
            decomposition_prompt = f"""You are the acting supervisor in an agentic workflow. You are collaborating with a local AI assistant that has access to full context/documents, but you don't have direct access to them.

Your goal: Answer this question: "{query}"

The local assistant can see this context (you cannot): [CONTEXT: {len(context)} characters of text including uploaded documents]

Please ask the local assistant specific, focused questions to gather the information you need. Be direct and precise in your requests. Ask only what you need to answer the original question.
            
Break down this question into no more than {self.valves.max_tasks_per_round} simple, specific tasks. If tasks must happen **in sequence**, do **not** include them all in this round; move to a subsequent round to handle later steps.

Create simple tasks that can be answered to help answer the User's question. Format as a simple list:
1. [First specific task]
2. [Second specific task] 
3. [Third specific task]

Keep tasks simple and focused on extracting specific information to help answer the larger query."""

            try:
                claude_response = await asyncio.wait_for(
                    self._call_claude(decomposition_prompt), timeout=30.0
                )
                conversation_log.append(f"**🤖 Claude (Task Decomposition):**")
                conversation_log.append(f"{claude_response}\n")
            except asyncio.TimeoutError:
                return "❌ **Error:** Task decomposition timed out. Try using the 'minion' protocol instead."

            # Parse tasks
            tasks = self._parse_tasks(claude_response)

            if self.valves.debug_mode:
                debug_log.append(f"**Parsed tasks:** {tasks}")

            conversation_log.append("### ⚡ Parallel Execution Phase")

            # Step 2: Execute tasks efficiently
            chunks = self._create_chunks(context)

            if self.valves.debug_mode:
                debug_log.append(
                    f"**Processing:** {len(chunks)} chunks of ~{self.valves.chunk_size} chars each"
                )
                debug_log.append(
                    f"**Timeout setting:** {self.valves.timeout_local} seconds per chunk"
                )

            # Execute tasks in parallel across chunks
            task_results = await self._execute_tasks_on_chunks(tasks, chunks, conversation_log)

            # Check if we had too many timeouts
            timeout_count = sum(1 for r in task_results if r["result"] == "Timeout")
            if timeout_count >= len(tasks) * 2:
                conversation_log.append(
                    "\n⚠️ **Too many local model timeouts detected. Consider increasing timeout or using simpler models.**"
                )

            conversation_log.append("\n### 🔄 Synthesis Phase")

            # Step 3: Synthesis
            results_summary = "\n".join(
                [f"- {r['task']}: {r['result']}" for r in task_results]
            )

            synthesis_prompt = f"""Combine these task results into a complete answer for: {query}

RESULTS:
{results_summary}

Provide a clear, comprehensive answer based on the information found:"""

            try:
                final_response = await asyncio.wait_for(
                    self._call_claude(synthesis_prompt), timeout=30.0
                )
                conversation_log.append(f"**🤖 Claude (Synthesis):**")
                conversation_log.append(f"{final_response}")
            except asyncio.TimeoutError:
                final_response = (
                    "Analysis completed, but synthesis timed out. Here are the key findings:\n\n"
                    + results_summary
                )

            # Build output
            output_parts = []

            if self.valves.show_conversation:
                output_parts.append("## 🗣️ MinionS Collaboration (Task Decomposition)")
                output_parts.extend(conversation_log)
                output_parts.append("---")

            if self.valves.debug_mode:
                output_parts.extend(debug_log)

            output_parts.append(f"## 🎯 Final Answer")
            output_parts.append(final_response)

            # Calculate stats
            stats = self._calculate_token_savings_minions(
                decomposition_prompt, synthesis_prompt, results_summary, 
                final_response, context, query
            )
            
            total_tasks = len(tasks)
            successful_tasks = len(
                [r for r in task_results if "not found" not in r["result"].lower() and r["result"] != "Timeout"]
            )

            output_parts.append(f"\n## 📊 MinionS Efficiency Stats")
            output_parts.append(
                f"- **Protocol:** MinionS (task decomposition + parallel processing)"
            )
            output_parts.append(
                f"- **Tasks completed:** {successful_tasks}/{total_tasks}"
            )
            output_parts.append(f"- **Chunks processed:** {len(chunks)}")
            output_parts.append(f"- **Timeouts encountered:** {timeout_count}")
            output_parts.append(
                f"- **Local model timeout setting:** {self.valves.timeout_local}s"
            )
            output_parts.append(f"- **Context size:** {len(context):,} characters")
            output_parts.append(f"")
            output_parts.append(
                f"## 💰 Token Savings Analysis ({self.valves.remote_model})"
            )
            output_parts.append(
                f"- **Traditional approach:** ~{stats['traditional_tokens']:,} tokens"
            )
            output_parts.append(f"- **MinionS approach:** ~{stats['minions_tokens']:,} tokens")
            output_parts.append(f"- **💰 Token savings:** ~{stats['percentage_savings']:.1f}%")
            
            return "\n".join(output_parts)

        except Exception as e:
            error_msg = f"❌ **MinionS Protocol Error:** {str(e)}\n\n"
            error_msg += "**Consider using the basic Minion protocol instead.**\n"
            return error_msg

    def _parse_tasks(self, claude_response: str) -> List[str]:
        """Parse tasks from Claude's decomposition response"""
        lines = claude_response.split("\n")
        tasks = []
        for line in lines:
            line = line.strip()
            if any(
                line.startswith(prefix)
                for prefix in ["1.", "2.", "3.", "4.", "-", "*"]
            ):
                task = (
                    line.split(".", 1)[-1].strip()
                    if "." in line
                    else line[1:].strip()
                )
                if len(task) > 10:
                    tasks.append(task)

        # Fallback if parsing failed
        if len(tasks) == 0:
            tasks = [
                "Extract key financial figures or metrics",
                "Identify main points or conclusions",
                "Find important names, dates, or specific details",
            ]

        # Limit to max tasks
        return tasks[:self.valves.max_tasks_per_round]

    def _create_chunks(self, context: str) -> List[str]:
        """Create document chunks for parallel processing"""
        chunk_size = min(len(context) // 2, self.valves.chunk_size)
        chunks = [
            context[i : i + chunk_size] 
            for i in range(0, len(context), chunk_size)
        ]
        return chunks[:self.valves.max_chunks]

    async def _execute_tasks_on_chunks(
        self, tasks: List[str], chunks: List[str], conversation_log: List[str]
    ) -> List[Dict[str, str]]:
        """Execute tasks in parallel across document chunks"""
        task_results = []

        for task_idx, task in enumerate(tasks):
            conversation_log.append(f"**📋 Task {task_idx + 1}:** {task}")

            best_result = None

            # Try chunks in order
            for chunk_idx, chunk in enumerate(chunks):
                # Simplified prompt for faster processing
                local_prompt = f"""Text to analyze:
{chunk[:3000]}

Task: {task}

Provide a brief, specific answer based on this text. If no relevant information, say "NONE"."""

                try:
                    if self.valves.debug_mode:
                        conversation_log.append(
                            f"   🔄 Trying chunk {chunk_idx + 1} (size: {len(chunk)} chars)..."
                        )

                    result = await asyncio.wait_for(
                        self._call_ollama(local_prompt),
                        timeout=self.valves.timeout_local,
                    )

                    if self.valves.debug_mode:
                        conversation_log.append(
                            f"   ✅ Chunk {chunk_idx + 1} completed: {result[:100]}..."
                        )

                    if "NONE" not in result.upper() and len(result.strip()) > 5:
                        best_result = result
                        conversation_log.append(
                            f"**💻 Local Model (chunk {chunk_idx + 1}):** {result[:150]}..."
                        )
                        break  # Found result, move to next task
                    else:
                        conversation_log.append(
                            f"   ℹ️ Chunk {chunk_idx + 1}: No relevant info found"
                        )
                except asyncio.TimeoutError:
                    conversation_log.append(
                        f"   ⏰ Chunk {chunk_idx + 1} timed out after {self.valves.timeout_local}s"
                    )
                    continue
                except Exception as e:
                    conversation_log.append(
                        f"   ❌ Chunk {chunk_idx + 1} error: {str(e)}"
                    )
                    if self.valves.debug_mode:
                        conversation_log.append(f"      Full error: {repr(e)}")
                    continue

            if best_result:
                task_results.append({"task": task, "result": best_result})
            else:
                conversation_log.append(
                    f"**💻 Local Model:** No relevant information found"
                )
                task_results.append(
                    {"task": task, "result": "Information not found"}
                )

        return task_results

    def _calculate_token_savings_minions(
        self, decomposition_prompt: str, synthesis_prompt: str, 
        results_summary: str, final_response: str, context: str, query: str
    ) -> dict:
        """Calculate token savings for the MinionS protocol"""
        chars_per_token = 3.5
        
        # Get actual pricing for the model being used
        model_pricing = self._get_model_pricing(self.valves.remote_model)
        
        # Traditional approach: entire context + query sent to Claude
        traditional_tokens = int((len(context) + len(query)) / chars_per_token)
        
        # MinionS approach: only decomposition + synthesis prompts + results
        decomposition_tokens = int(len(decomposition_prompt) / chars_per_token)
        synthesis_content = synthesis_prompt + final_response + results_summary
        synthesis_tokens = int(len(synthesis_content) / chars_per_token)
        minions_tokens = decomposition_tokens + synthesis_tokens
        
        # Calculate savings
        token_savings = traditional_tokens - minions_tokens
        percentage_savings = (
            (token_savings / traditional_tokens * 100)
            if traditional_tokens > 0
            else 0
        )
        
        return {
            'traditional_tokens': traditional_tokens,
            'minions_tokens': minions_tokens,
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
            "max_tokens": 1000,
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
        
        # Take a bow!
