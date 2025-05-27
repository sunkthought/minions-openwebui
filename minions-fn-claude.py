"""
title: MinionS Protocol Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.2.0
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
            default=30,
            description="Timeout for local model calls per chunk in seconds (DECREASED default to 30s).",
        )
        debug_mode: bool = Field(
            default=False, description="Show additional technical details"
        )
        max_round_timeout_failure_threshold_percent: int = Field(
            default=50, description="If this percentage of local model calls in a round time out, a warning is issued about potentially incomplete results for that round."
        )
        max_tokens_claude: int = Field(
            default=2000, description="Max tokens for Claude API calls"
        )
        timeout_claude: int = Field(
            default=60, description="Timeout for Claude API calls in seconds"
        )
        ollama_num_predict: int = Field(
            default=1000, description="Max tokens for Ollama API calls (num_predict)"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "MinionS v0.2.0 (Multi-Round)"

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
        conversation_log = []
        debug_log = [] # Stays as a separate log for debug specific messages
        scratchpad_content = "" 
        all_round_results_aggregated = [] 
        decomposition_prompts_history = []
        synthesis_prompts_history = []
        final_response = "No answer could be synthesized."
        claude_provided_final_answer = False
        total_tasks_executed_local = 0
        total_chunks_processed_for_stats = 0 
        total_chunk_processing_timeouts_accumulated = 0 # New accumulator
        synthesis_input_summary = "" 

        overall_start_time = asyncio.get_event_loop().time()
        if self.valves.debug_mode:
            # debug_log.append instead of conversation_log for debug-specific timing/flow messages
            debug_log.append(f"🔍 **Debug Info (MinionS v0.2.0):**\n- Query: {query[:100]}...\n- Context length: {len(context)} chars")
            debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**")


        chunks = self._create_chunks(context) 
        if not chunks and context: 
             return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size."
        if not chunks and not context: 
            conversation_log.append("ℹ️ No context or chunks to process with MinionS. Attempting direct call.")
            start_time_claude = 0
            if self.valves.debug_mode: start_time_claude = asyncio.get_event_loop().time()
            try:
                final_response = await self._call_claude_directly(query)
                if self.valves.debug_mode:
                    end_time_claude = asyncio.get_event_loop().time()
                    time_taken_claude = end_time_claude - start_time_claude
                    debug_log.append(f"⏱️ Claude direct call took {time_taken_claude:.2f}s. (Debug Mode)")
                output_parts = []
                if self.valves.show_conversation:
                    output_parts.append("## 🗣️ MinionS Collaboration (Direct Call)")
                    output_parts.extend(conversation_log) # conversation_log for user-facing notes
                    output_parts.append("---")
                if self.valves.debug_mode: # Add debug log if enabled
                    output_parts.append("### 🔍 Debug Log")
                    output_parts.extend(debug_log)
                    output_parts.append("---")
                output_parts.append(f"## 🎯 Final Answer (Direct)\n{final_response}")
                return "\n".join(output_parts)
            except Exception as e:
                return f"❌ **Error in direct Claude call:** {str(e)}"

        total_chunks_processed_for_stats = len(chunks)

        for current_round in range(self.valves.max_rounds):
            if self.valves.debug_mode: 
                debug_log.append(f"**⚙️ Starting Round {current_round + 1}/{self.valves.max_rounds}... (Debug Mode)**")
            
            # This is user-facing, so it goes to conversation_log if show_conversation is True
            if self.valves.show_conversation:
                 conversation_log.append(f"### 🎯 Round {current_round + 1}/{self.valves.max_rounds} - Task Decomposition Phase")
            
            decomposition_prompt = f'''You are a supervisor LLM in a multi-round process. Your goal is to answer: "{query}"
Context has been split into {len(chunks)} chunks. A local LLM will process these chunks for each task you define.
Scratchpad (previous findings): {scratchpad_content if scratchpad_content else "Nothing yet."}

Based on the scratchpad and the original query, identify up to {self.valves.max_tasks_per_round} specific, simple tasks for the local assistant.
If the information in the scratchpad is sufficient to answer the query, respond ONLY with the exact phrase "FINAL ANSWER READY." followed by the comprehensive answer.
Otherwise, list the new tasks clearly. Ensure tasks are actionable. Avoid redundant tasks.
Format tasks as a simple list (e.g., 1. Task A, 2. Task B).'''
            decomposition_prompts_history.append(decomposition_prompt)
            
            start_time_claude_decomp = 0
            if self.valves.debug_mode: start_time_claude_decomp = asyncio.get_event_loop().time()
            try:
                claude_response = await self._call_claude(decomposition_prompt)
                if self.valves.debug_mode:
                    end_time_claude_decomp = asyncio.get_event_loop().time()
                    time_taken_claude_decomp = end_time_claude_decomp - start_time_claude_decomp
                    debug_log.append(f"⏱️ Claude call (Decomposition Round {current_round+1}) took {time_taken_claude_decomp:.2f}s. (Debug Mode)")
                if self.valves.show_conversation:
                    conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round + 1}):**\n{claude_response}\n")
            except Exception as e:
                conversation_log.append(f"❌ Error calling Claude for decomposition in round {current_round + 1}: {e}")
                break 

            if "FINAL ANSWER READY." in claude_response:
                final_response = claude_response.split("FINAL ANSWER READY.", 1)[1].strip()
                claude_provided_final_answer = True
                if self.valves.show_conversation:
                    conversation_log.append(f"**🤖 Claude indicates final answer is ready in round {current_round + 1}.**")
                scratchpad_content += f"\n\n**Round {current_round + 1}:** Claude provided final answer."
                break 

            tasks = self._parse_tasks(claude_response)
            if self.valves.debug_mode:
                debug_log.append(f"   Identified {len(tasks)} tasks for Round {current_round + 1}. (Debug Mode)")
                for task_idx, task_item in enumerate(tasks):
                   debug_log.append(f"    Task {task_idx+1} (Round {current_round+1}): {task_item[:100]}... (Debug Mode)")

            if not tasks:
                if self.valves.show_conversation:
                    conversation_log.append(f"**🤖 Claude provided no new tasks in round {current_round + 1}. Proceeding to final synthesis.**")
                break
            
            total_tasks_executed_local += len(tasks)
            
            if self.valves.show_conversation:
                 conversation_log.append(f"### ⚡ Round {current_round + 1} - Parallel Execution Phase (Processing {len(chunks)} chunks for {len(tasks)} tasks)")
            
            execution_details = await self._execute_tasks_on_chunks(tasks, chunks, conversation_log if self.valves.show_conversation else debug_log, current_round + 1)
            current_round_task_results = execution_details["results"]
            round_chunk_attempts = execution_details["total_chunk_processing_attempts"]
            round_chunk_timeouts = execution_details["total_chunk_processing_timeouts"]

            if round_chunk_attempts > 0:
                timeout_percentage_this_round = (round_chunk_timeouts / round_chunk_attempts) * 100
                # This is an operational stat, so goes to conversation_log for visibility if show_conversation is true
                # but also to debug_log if debug_mode is true for consistent debug tracing.
                log_msg_timeout_stat = f"**📈 Round {current_round + 1} Local LLM Timeout Stats:** {round_chunk_timeouts}/{round_chunk_attempts} chunk calls timed out ({timeout_percentage_this_round:.1f}%)."
                if self.valves.show_conversation: conversation_log.append(log_msg_timeout_stat)
                if self.valves.debug_mode: debug_log.append(log_msg_timeout_stat)


                if timeout_percentage_this_round >= self.valves.max_round_timeout_failure_threshold_percent:
                    warning_msg = f"⚠️ **Warning:** Round {current_round + 1} exceeded local LLM timeout threshold of {self.valves.max_round_timeout_failure_threshold_percent}%. Results from this round may be incomplete or unreliable."
                    if self.valves.show_conversation: conversation_log.append(warning_msg)
                    if self.valves.debug_mode: debug_log.append(warning_msg) # Ensure critical warnings are in debug too
                    scratchpad_content += f"\n\n**Note from Round {current_round + 1}:** High percentage of local model timeouts ({timeout_percentage_this_round:.1f}%) occurred, results for this round may be partial."
            
            round_summary_for_scratchpad_parts = []
            for r_val in current_round_task_results: 
                status_icon = "✅" if r_val['status'] == 'success' else ("⏰" if 'timeout' in r_val['status'] else "❓")
                summary_text = f"- {status_icon} Task: {r_val['task']}, Result: {r_val['result'][:200]}..." if r_val['status'] == 'success' else f"- {status_icon} Task: {r_val['task']}, Status: {r_val['result']}"
                round_summary_for_scratchpad_parts.append(summary_text)
            
            if round_summary_for_scratchpad_parts:
                scratchpad_content += f"\n\n**Results from Round {current_round + 1}:**\n" + "\n".join(round_summary_for_scratchpad_parts)
            
            all_round_results_aggregated.extend(current_round_task_results) 
            total_chunk_processing_timeouts_accumulated += round_chunk_timeouts # Accumulate timeouts

            if self.valves.debug_mode:
                current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
                debug_log.append(f"**🏁 Completed Round {current_round + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**")

            if current_round == self.valves.max_rounds - 1: 
                 if self.valves.show_conversation:
                     conversation_log.append(f"**🏁 Reached max rounds ({self.valves.max_rounds}). Proceeding to final synthesis.**")

        if not claude_provided_final_answer:
            if self.valves.show_conversation: conversation_log.append("\n### 🔄 Final Synthesis Phase")
            if not all_round_results_aggregated:
                final_response = "No information was gathered from the document by local models across the rounds."
                if self.valves.show_conversation: conversation_log.append(f"**🤖 Claude (Synthesis):** {final_response}")
            else:
                synthesis_input_summary = "\n".join([f"- Task: {r['task']}\n  Result: {r['result']}" for r in all_round_results_aggregated if r['status'] == 'success'])
                if not synthesis_input_summary: 
                    synthesis_input_summary = "No definitive information was found by local models. The original query was: " + query
                
                synthesis_prompt = f'''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary if synthesis_input_summary else "No specific information was extracted by local models."}

If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.
Final Answer:'''
                synthesis_prompts_history.append(synthesis_prompt)
                
                start_time_claude_synth = 0
                if self.valves.debug_mode: start_time_claude_synth = asyncio.get_event_loop().time()
                try:
                    final_response = await self._call_claude(synthesis_prompt)
                    if self.valves.debug_mode:
                        end_time_claude_synth = asyncio.get_event_loop().time()
                        time_taken_claude_synth = end_time_claude_synth - start_time_claude_synth
                        debug_log.append(f"⏱️ Claude call (Final Synthesis) took {time_taken_claude_synth:.2f}s. (Debug Mode)")
                    if self.valves.show_conversation:
                        conversation_log.append(f"**🤖 Claude (Final Synthesis):**\n{final_response}")
                except Exception as e:
                    if self.valves.show_conversation: conversation_log.append(f"❌ Error during final synthesis: {e}")
                    final_response = "Error during final synthesis. Raw findings might be available in conversation log."
        
        output_parts = []
        if self.valves.show_conversation:
            output_parts.append("## 🗣️ MinionS Collaboration (Multi-Round)")
            output_parts.extend(conversation_log)
            output_parts.append("---")
        if self.valves.debug_mode:
            output_parts.append("### 🔍 Debug Log")
            output_parts.extend(debug_log)
            output_parts.append("---")
        output_parts.append(f"## 🎯 Final Answer")
        output_parts.append(final_response)

        summary_for_stats = synthesis_input_summary if not claude_provided_final_answer else scratchpad_content

        stats = self._calculate_token_savings_minions(
            decomposition_prompts_history, synthesis_prompts_history,
            summary_for_stats, final_response, 
            len(context), len(query), total_chunks_processed_for_stats, total_tasks_executed_local
        )
        
        total_successful_tasks = len([r for r in all_round_results_aggregated if r['status'] == 'success'])
        # total_timeout_tasks counts tasks where ALL chunks timed out. 
        # total_chunk_processing_timeouts_accumulated counts individual chunk timeouts.
        tasks_with_any_timeout = len([r for r in all_round_results_aggregated if r['status'] == 'timeout_all_chunks'])


        output_parts.append(f"\n## 📊 MinionS Efficiency Stats (v0.2.0)")
        output_parts.append(f"- **Protocol:** MinionS (Multi-Round)")
        output_parts.append(f"- **Rounds executed:** {stats['total_rounds']}/{self.valves.max_rounds}")
        output_parts.append(f"- **Total tasks for local LLM:** {stats['total_tasks_executed_local']}")
        output_parts.append(f"- **Successful tasks (local):** {total_successful_tasks}")
        output_parts.append(f"- **Tasks where all chunks timed out (local):** {tasks_with_any_timeout}")
        output_parts.append(f"- **Total individual chunk processing timeouts (local):** {total_chunk_processing_timeouts_accumulated}")
        output_parts.append(f"- **Chunks processed per task (local):** {stats['total_chunks_processed_local'] if stats['total_tasks_executed_local'] > 0 else 0}") 
        output_parts.append(f"- **Context size:** {len(context):,} characters")
        output_parts.append(f"\n## 💰 Token Savings Analysis (Claude: {self.valves.remote_model})")
        output_parts.append(f"- **Traditional single call (est.):** ~{stats['traditional_tokens_claude']:,} tokens")
        output_parts.append(f"- **MinionS multi-round (Claude only):** ~{stats['minions_tokens_claude']:,} tokens")
        output_parts.append(f"- **💰 Est. Claude Token savings:** ~{stats['percentage_savings_claude']:.1f}%")
            
        return "\n".join(output_parts)

    def _parse_tasks(self, claude_response: str) -> List[str]:
        lines = claude_response.split("\n")
        tasks = []
        for line in lines:
            line = line.strip()
            # More robust parsing for numbered or bulleted lists
            if line.startswith(tuple(f"{i}." for i in range(1, 10))) or \
               line.startswith(tuple(f"{i})" for i in range(1, 10))) or \
               line.startswith(("- ", "* ", "+ ")):
                task = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
                if len(task) > 10: # Keep simple task filter
                    tasks.append(task)
        return tasks[:self.valves.max_tasks_per_round]

    def _create_chunks(self, context: str) -> List[str]:
        if not context: return []
        actual_chunk_size = max(1, min(self.valves.chunk_size, len(context)))
        chunks = [
            context[i : i + actual_chunk_size] 
            for i in range(0, len(context), actual_chunk_size)
        ]
        return chunks

    async def _execute_tasks_on_chunks(
        self, tasks: List[str], chunks: List[str], conversation_log: List[str], current_round: int
    ) -> Dict:
        overall_task_results = []
        total_attempts_this_call = 0
        total_timeouts_this_call = 0

        for task_idx, task in enumerate(tasks):
            conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
            results_for_this_task_from_chunks = []
            chunk_timeout_count_for_task = 0
            num_relevant_chunks_found = 0

            for chunk_idx, chunk in enumerate(chunks):
                total_attempts_this_call += 1
                local_prompt = f'''Text to analyze (Chunk {chunk_idx + 1}/{len(chunks)} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}

Provide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word "NONE".'''
                
                start_time_ollama = 0
                if self.valves.debug_mode:
                    conversation_log.append(
                        f"   🔄 Task {task_idx + 1} - Trying chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} chars)... (Debug Mode)"
                    )
                    start_time_ollama = asyncio.get_event_loop().time()

                try:
                    result = await asyncio.wait_for(
                        self._call_ollama(local_prompt),
                        timeout=self.valves.timeout_local,
                    )
                    if self.valves.debug_mode:
                        end_time_ollama = asyncio.get_event_loop().time()
                        time_taken_ollama = end_time_ollama - start_time_ollama
                        conversation_log.append(
                             f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} processed by local LLM in {time_taken_ollama:.2f}s. Status: {'Relevant info found' if 'NONE' not in result.strip().upper() else 'No relevant info'}. Result: {result[:70]}... (Debug Mode)"
                        )

                    if not (result.strip().upper() == "NONE" or len(result.strip()) < 5):
                        results_for_this_task_from_chunks.append(f"[Chunk {chunk_idx+1}]: {result}")
                        num_relevant_chunks_found += 1
                        # Reduced verbosity for non-debug success, already logged in debug
                except asyncio.TimeoutError:
                    total_timeouts_this_call += 1
                    chunk_timeout_count_for_task +=1
                    conversation_log.append(
                        f"   ⏰ Task {task_idx + 1} - Chunk {chunk_idx + 1} timed out after {self.valves.timeout_local}s"
                    )
                    if self.valves.debug_mode: #Also log time for timeouts in debug
                        end_time_ollama = asyncio.get_event_loop().time()
                        time_taken_ollama = end_time_ollama - start_time_ollama
                        conversation_log.append(
                             f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} TIMEOUT after {time_taken_ollama:.2f}s. (Debug Mode)"
                        )
                except Exception as e:
                    conversation_log.append(
                        f"   ❌ Task {task_idx + 1} - Chunk {chunk_idx + 1} error: {str(e)}"
                    )
            
            if results_for_this_task_from_chunks:
                aggregated_result_for_task = "\n".join(results_for_this_task_from_chunks)
                overall_task_results.append({"task": task, "result": aggregated_result_for_task, "status": "success"})
                conversation_log.append(
                    f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). First result snippet: {results_for_this_task_from_chunks[0][:100]}..."
                )
            elif chunk_timeout_count_for_task > 0 and chunk_timeout_count_for_task == len(chunks):
                 overall_task_results.append({"task": task, "result": f"Timeout on all {len(chunks)} chunks", "status": "timeout_all_chunks"})
                 conversation_log.append(
                    f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out."
                )
            else: # No results, and not all chunks timed out (some might have errored, some returned NONE)
                overall_task_results.append(
                    {"task": task, "result": "Information not found in any relevant chunk", "status": "not_found"}
                )
                conversation_log.append(
                    f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found in any chunk."
                )
        
        return {
            "results": overall_task_results,
            "total_chunk_processing_attempts": total_attempts_this_call,
            "total_chunk_processing_timeouts": total_timeouts_this_call
        }

    def _calculate_token_savings_minions(
        self, decomposition_prompts: List[str], synthesis_prompts: List[str],
        all_results_summary_for_claude: str, final_response_claude: str, 
        context_length: int, query_length: int, total_chunks_processed_local: int,
        total_tasks_executed_local: int
    ) -> dict:
        chars_per_token = 3.5 
        traditional_tokens = int((context_length + query_length) / chars_per_token)
        minions_tokens_claude = 0
        for p in decomposition_prompts:
            minions_tokens_claude += int(len(p) / chars_per_token)
        for p in synthesis_prompts:
            minions_tokens_claude += int(len(p) / chars_per_token)
        minions_tokens_claude += int(len(all_results_summary_for_claude) / chars_per_token)
        minions_tokens_claude += int(len(final_response_claude) / chars_per_token)
        token_savings = traditional_tokens - minions_tokens_claude
        percentage_savings = (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
        
        return {
            'traditional_tokens_claude': traditional_tokens,
            'minions_tokens_claude': minions_tokens_claude,
            'token_savings_claude': token_savings,
            'percentage_savings_claude': percentage_savings,
            'total_rounds': len(decomposition_prompts),
            'total_chunks_processed_local': total_chunks_processed_local,
            'total_tasks_executed_local': total_tasks_executed_local,
        }
    
    async def _call_claude(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.valves.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.valves.remote_model,
            "max_tokens": self.valves.max_tokens_claude, 
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=self.valves.timeout_claude
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error_text}")
                result = await response.json()
                if result.get("content") and isinstance(result["content"], list) and len(result["content"]) > 0 and result["content"][0].get("text"):
                    return result["content"][0]["text"]
                else:
                    if self.valves.debug_mode: print(f"Unexpected Claude API response format: {result}")
                    raise Exception("Unexpected response format from Anthropic API or empty content.")

    async def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.valves.local_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": self.valves.ollama_num_predict},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.valves.ollama_base_url}/api/generate", json=payload, timeout=self.valves.timeout_local
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
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
        
        # Take a bow!
