"""
title: MinionS Protocol Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.2.1
description: MinionS protocol - task decomposition and parallel processing between local and cloud models
required_open_webui_version: 0.5.0
"""

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional # Added Any, Optional
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore
import logging # Added logging
from .partials.minions_models import JobManifest # Added JobManifest
from .partials.minions_prompts import get_minions_code_generation_claude_prompt # Added code-gen prompt

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
        self.logger = logging.getLogger(__name__) # Added logger

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
        actual_initial_claude_prompt_text: str = "" # Stores the text of the first prompt to Claude

        try:
            if self.valves.debug_mode:
                debug_info = f"🔍 **Debug Info (MinionS):**\n"
                debug_info += f"- Query: {query[:100]}...\n"
                debug_info += f"- Context length: {len(context)} chars\n"
                debug_info += f"- Protocol: MinionS (task decomposition)\n\n"
                debug_log.append(debug_info)

            conversation_log.append("### 🎯 Task Decomposition Phase")
            job_manifests: List[JobManifest] = []

            if self.valves.enable_code_decomposition:
                self.logger.info("Using code-based task decomposition.")
                conversation_log.append("🤖 Using Code-Based Task Decomposition")
                context_summary = context[:500] + "..." if len(context) > 500 else context

                # This is the actual first prompt text in this branch
                actual_initial_claude_prompt_text = get_minions_code_generation_claude_prompt(
                    query=query,
                    context_summary=context_summary,
                    valves=self.valves,
                    job_manifest_model_name="JobManifest"
                )

                try:
                    generated_code = await asyncio.wait_for(
                        self._call_claude(actual_initial_claude_prompt_text), timeout=45.0 # Increased timeout for code gen
                    )
                    conversation_log.append(f"**🤖 Claude (Code Generation):**\n```python\n{generated_code}\n```\n")
                    if self.valves.debug_mode:
                        self.logger.debug(f"Raw generated code:\n{generated_code}")
                        debug_log.append(f"**Raw Generated Code:**\n```python\n{generated_code}\n```")

                    job_manifests = self._execute_generated_code(generated_code)
                    self.logger.info(f"Successfully generated {len(job_manifests)} tasks via code.")
                    conversation_log.append(f"✅ Successfully generated {len(job_manifests)} tasks via code.")

                except asyncio.TimeoutError:
                    self.logger.error("Code generation timed out.")
                    conversation_log.append("❌ **Error:** Code generation for tasks timed out.")
                    # Potentially fall back to natural language or return error
                    return "❌ **Error:** Code generation for task decomposition timed out."
                except ValueError as e: # Errors from _execute_generated_code
                    self.logger.error(f"Error executing generated code: {e}")
                    conversation_log.append(f"❌ **Error:** Could not execute generated task code: {e}")
                    # Potentially fall back or return error
                    return f"❌ **Error:** Failed to process generated task code: {e}"
                except Exception as e:
                    self.logger.error(f"Unexpected error during code-based decomposition: {e}")
                    conversation_log.append(f"❌ **Error:** Unexpected error in code-based decomposition: {e}")
                    return f"❌ **Error:** Unexpected error during code-based task decomposition: {e}"

            else:
                self.logger.info("Using natural language task decomposition.")
                conversation_log.append("🗣️ Using Natural Language Task Decomposition")
                # Step 1: Task decomposition by Claude (Natural Language)
                # This is the actual first prompt text in this branch
                actual_initial_claude_prompt_text = f"""You are the acting supervisor in an agentic workflow. You are collaborating with a local AI assistant that has access to full context/documents, but you don't have direct access to them.

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
                        self._call_claude(actual_initial_claude_prompt_text), timeout=30.0
                    )
                    conversation_log.append(f"**🤖 Claude (Task Decomposition):**\n{claude_response}\n")
                except asyncio.TimeoutError:
                    self.logger.error("Natural language task decomposition timed out.")
                    return "❌ **Error:** Task decomposition timed out. Try using the 'minion' protocol instead."

                parsed_task_strings = self._parse_tasks(claude_response)
                job_manifests = []
                for i, task_desc_str in enumerate(parsed_task_strings):
                    job_manifests.append(JobManifest(task_id=f"task_{i+1}", task_description=task_desc_str, chunk_id=None, advice=None))
                self.logger.info(f"Successfully generated {len(job_manifests)} tasks via natural language.")
                conversation_log.append(f"✅ Successfully generated {len(job_manifests)} tasks via natural language.")

            if not job_manifests:
                self.logger.warning("No tasks were generated from decomposition.")
                conversation_log.append("⚠️ No tasks were generated. Synthesis will be based on the original query and context directly.")
                # Fallback: if no tasks, synthesis will effectively just use the full context.
                # Or, return a message indicating no tasks could be derived.
                # For now, proceed, and synthesis will be basic.

            if self.valves.debug_mode:
                task_descriptions_for_log = [jm.task_description for jm in job_manifests]
                debug_log.append(f"**Parsed/Generated JobManifests (descriptions):** {task_descriptions_for_log}")

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
            # TODO: _execute_tasks_on_chunks will need to be updated to accept List[JobManifest]
            task_results = await self._execute_tasks_on_chunks(job_manifests, chunks, conversation_log)

            # Check if we had too many timeouts
            timeout_count = sum(1 for r in task_results if r["result"] == "Timeout") # This might need adjustment based on task_results structure
            if timeout_count >= len(job_manifests) * 2: # Adjusted from len(tasks)
                conversation_log.append(
                    "\n⚠️ **Too many local model timeouts detected. Consider increasing timeout or using simpler models.**"
                )

            conversation_log.append("\n### 🔄 Synthesis Phase")

            # Step 3: Synthesis
            results_summary_parts = []
            if not task_results:
                results_summary_parts.append("No tasks were executed, or no results were produced.")
            else:
                for r_idx, r_item in enumerate(task_results):
                    task_id = r_item.get('task_id', f'N/A_{r_idx}')
                    description = r_item.get('task_description', 'No description provided.')
                    result_text = r_item.get('result', 'No result returned.')

                    description_summary = description[:100] + "..." if len(description) > 100 else description
                    # Ensure result_text is a string before slicing
                    result_text_str = str(result_text)
                    result_summary = result_text_str[:300] + "..." if len(result_text_str) > 300 else result_text_str

                    results_summary_parts.append(f"Task ID: {task_id}\nDescription: {description_summary}\nResult: {result_summary}\n---")
            results_summary = "\n".join(results_summary_parts)

            if self.valves.debug_mode:
                self.logger.debug(f"Results summary for synthesis:\n{results_summary}")


            synthesis_prompt = f"""Based on the following information gathered from various tasks, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{results_summary if results_summary else "No specific information was extracted or tasks executed."}

If the gathered information is insufficient or no tasks were performed, explain what's missing or state that the answer cannot be provided based on the available results.
Final Answer:"""
            # The duplicated simpler synthesis_prompt that was here has been removed.

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
                actual_initial_claude_prompt_text, # This now holds the correct first prompt
                synthesis_prompt, results_summary,
                final_response, context, query
            )
            
            total_tasks = len(job_manifests) # Adjusted from len(tasks)
            successful_tasks = len(
                [r for r in task_results if "not found" not in r["result"].lower() and r["result"] != "Timeout"] # This might need adjustment
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
        self, tasks: List[JobManifest], chunks: List[str], conversation_log: List[str]
    ) -> List[Dict[str, str]]:
        """Execute tasks defined by JobManifest objects in parallel across document chunks."""
        task_results = []
        total_chunks_in_document = len(chunks)

        for task_idx, manifest in enumerate(tasks):
            conversation_log.append(f"**📋 Task {task_idx + 1} (ID: {manifest.task_id}):** {manifest.task_description}")
            self.logger.info(f"Executing task ID {manifest.task_id}: {manifest.task_description}")

            best_result_for_task = "Information not found" # Default if no chunk yields a better answer
            processed_at_least_one_chunk = False

            selected_chunks_with_indices: List[tuple[int, str]] = []
            if manifest.chunk_id is not None and 0 <= manifest.chunk_id < total_chunks_in_document:
                selected_chunks_with_indices = [(manifest.chunk_id, chunks[manifest.chunk_id])]
                if self.valves.debug_mode:
                    self.logger.debug(f"Task {manifest.task_id} targeting specific chunk_id: {manifest.chunk_id}.")
                conversation_log.append(f"   ℹ️ Task targets specific chunk: {manifest.chunk_id + 1}")
            else:
                if manifest.chunk_id is not None: # Invalid chunk_id
                    self.logger.warning(f"Task {manifest.task_id} had invalid chunk_id: {manifest.chunk_id}. Applying to all chunks.")
                    conversation_log.append(f"   ⚠️ Task had invalid chunk_id {manifest.chunk_id}, applying to all chunks.")
                selected_chunks_with_indices = list(enumerate(chunks))
                if self.valves.debug_mode:
                    self.logger.debug(f"Task {manifest.task_id} applying to all {len(selected_chunks_with_indices)} chunks.")
                # conversation_log.append(f"   ℹ️ Task will be applied to all {len(selected_chunks_with_indices)} chunks.")


            for original_chunk_idx, chunk_content in selected_chunks_with_indices:
                processed_at_least_one_chunk = True

                prompt_intro = f"Text to analyze (Chunk {original_chunk_idx + 1}/{total_chunks_in_document} of document):"
                if manifest.chunk_id is not None: # This means it was a targeted chunk_id
                    prompt_intro = f"Text to analyze (Specifically targeted Chunk {original_chunk_idx + 1}/{total_chunks_in_document} of document based on task assignment):"

                # Use self.valves.chunk_size for ensuring prompt doesn't get too big if chunk_content is massive
                # However, chunks are already created with self.valves.chunk_size. If this is for Ollama context window, it's different.
                # For now, assuming chunk_content is already appropriately sized.
                # Using [:self.valves.chunk_size] might truncate an already sized chunk if not careful.
                # The original code used [:3000], let's stick to a defined limit for safety.
                # Consider making this limit a valve if necessary.
                max_chars_for_prompt = 4000 # Example internal limit for local model prompt text portion

                local_prompt = f'''{prompt_intro}
---BEGIN TEXT---
{chunk_content[:max_chars_for_prompt]}
---END TEXT---

Task: {manifest.task_description}
'''
                if manifest.advice:
                    local_prompt += f"\nHint: {manifest.advice}"

                local_prompt += "\n\nProvide a brief, specific answer based ONLY on the text provided above. If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\"."

                try:
                    if self.valves.debug_mode:
                        self.logger.debug(f"   Task {manifest.task_id} trying chunk {original_chunk_idx + 1} (size: {len(chunk_content)} chars).")
                        conversation_log.append(f"   🔄 Task '{manifest.task_description[:30].ellipsis() if len(manifest.task_description)>30 else manifest.task_description}...' on chunk {original_chunk_idx + 1}...")

                    ollama_response = await asyncio.wait_for(
                        self._call_ollama(local_prompt), timeout=self.valves.timeout_local
                    )

                    if self.valves.debug_mode:
                        self.logger.debug(f"   Task {manifest.task_id} chunk {original_chunk_idx + 1} completed. Response: {ollama_response[:100]}...")
                        conversation_log.append(f"   ✅ Chunk {original_chunk_idx + 1} response: {ollama_response[:100].ellipsis() if len(ollama_response)>100 else ollama_response}...")

                    # Check if response indicates information was found
                    if "NONE" not in ollama_response.upper() and len(ollama_response.strip()) > 2: # Added len check
                        best_result_for_task = ollama_response
                        self.logger.info(f"   Task {manifest.task_id} found relevant info in chunk {original_chunk_idx + 1}.")
                        conversation_log.append(f"**💻 Local Model (Task '{manifest.task_description[:30].ellipsis() if len(manifest.task_description)>30 else manifest.task_description}...', Chunk {original_chunk_idx+1}):** {ollama_response[:100].ellipsis() if len(ollama_response)>100 else ollama_response}...")
                        # If task was for a specific chunk, or if we take the first good answer for global tasks
                        break
                except asyncio.TimeoutError:
                    self.logger.warning(f"   Task {manifest.task_id} on chunk {original_chunk_idx + 1} timed out after {self.valves.timeout_local}s.")
                    conversation_log.append(f"   ⏰ Task '{manifest.task_description[:30].ellipsis() if len(manifest.task_description)>30 else manifest.task_description}...' on chunk {original_chunk_idx + 1} timed out.")
                    if best_result_for_task == "Information not found": # Prioritize timeout message if no info found yet
                         best_result_for_task = "Timeout"
                except Exception as e:
                    self.logger.error(f"   Task {manifest.task_id} on chunk {original_chunk_idx + 1} error: {e}")
                    conversation_log.append(f"   ❌ Task '{manifest.task_description[:30].ellipsis() if len(manifest.task_description)>30 else manifest.task_description}...' on chunk {original_chunk_idx + 1} error: {e}")
                    if best_result_for_task == "Information not found": # Prioritize error message
                        best_result_for_task = f"Error: {e}"

            if not processed_at_least_one_chunk and manifest.chunk_id is not None:
                # This case means a specific chunk_id was requested but was invalid (e.g., out of bounds)
                self.logger.warning(f"Task {manifest.task_id} requested specific chunk {manifest.chunk_id} which was not processed (total chunks: {total_chunks_in_document}).")
                conversation_log.append(f"   ⚠️ Task {manifest.task_id} could not be processed as requested chunk {manifest.chunk_id} is invalid.")
                best_result_for_task = "Invalid chunk requested"


            task_results.append({
                "task_id": manifest.task_id,
                "task_description": manifest.task_description, # Keep for easier summary later
                "result": best_result_for_task
            })

            if best_result_for_task == "Information not found" and processed_at_least_one_chunk:
                 conversation_log.append(f"**💻 Local Model (Task '{manifest.task_description[:30].ellipsis() if len(manifest.task_description)>30 else manifest.task_description}...'):** No relevant information found across applicable chunks.")
            elif best_result_for_task == "Timeout" and processed_at_least_one_chunk : # Check processed_at_least_one_chunk
                 conversation_log.append(f"**💻 Local Model (Task '{manifest.task_description[:30].ellipsis() if len(manifest.task_description)>30 else manifest.task_description}...'):** Timed out without finding information.")


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

    def _execute_generated_code(self, generated_code: str) -> List[JobManifest]:
        """
        Safely executes LLM-generated Python code expected to define a list of JobManifest objects.
        """
        local_scope: Dict[str, Any] = {"JobManifest": JobManifest, "job_manifests": []}

        # Strip markdown code block fences if present
        code_to_execute = generated_code.strip()
        if code_to_execute.startswith("```python"):
            code_to_execute = code_to_execute[9:] # Remove ```python
            if code_to_execute.strip().endswith("```"):
                code_to_execute = code_to_execute.strip()[:-3] # Remove ```
        elif code_to_execute.startswith("```"): # Handle if just ``` not ```python
            code_to_execute = code_to_execute[3:]
            if code_to_execute.strip().endswith("```"):
                code_to_execute = code_to_execute.strip()[:-3]

        code_to_execute = code_to_execute.strip()

        try:
            # Execute the generated code
            exec(code_to_execute, {"JobManifest": JobManifest, "__builtins__": {}}, local_scope)

            result = local_scope.get("job_manifests")

            if isinstance(result, list) and all(isinstance(item, JobManifest) for item in result):
                if not result: # Empty list is valid if LLM decides no tasks needed
                    self.logger.info("Generated code produced an empty list of JobManifests.")
                return result
            elif result is None and code_to_execute.startswith("[") and code_to_execute.endswith("]"):
                # Fallback: try to eval if the code itself is a list literal
                # This is a secondary attempt if 'job_manifests' variable wasn't assigned.
                self.logger.info("No 'job_manifests' variable found, attempting to eval code as list literal.")
                try:
                    # Ensure JobManifest is available in the eval context as well
                    # Restricted builtins for safety.
                    eval_result = eval(code_to_execute, {"JobManifest": JobManifest, "__builtins__": {"list": list, "dict": dict}}, {}) # Provide common builtins
                    if isinstance(eval_result, list) and all(isinstance(item, JobManifest) for item in eval_result):
                        if not eval_result:
                             self.logger.info("Evaluated code produced an empty list of JobManifests.")
                        return eval_result
                    else:
                        self.logger.warning(f"Evaluated code did not produce a List[JobManifest]. Type: {type(eval_result)}")
                        raise ValueError("Generated code, when evaluated, did not produce a list of JobManifest objects.")
                except Exception as e_eval:
                    self.logger.error(f"Error evaluating generated code as a list literal: {e_eval}")
                    # Log the problematic code if it's not too long for privacy/security
                    logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
                    self.logger.debug(f"Problematic code for eval: {logged_code}")
                    raise ValueError(f"Generated code could not be executed to find 'job_manifests' nor evaluated as a list literal. Eval error: {e_eval}") from e_eval
            elif result is not None:
                # 'job_manifests' was assigned, but not List[JobManifest]
                self.logger.warning(f"Generated code assigned 'job_manifests', but it was not List[JobManifest]. Type: {type(result)}")
                raise ValueError(f"Generated code assigned 'job_manifests', but it was not a list of JobManifest objects. Found type: {type(result)}")
            else:
                # 'job_manifests' is None and code doesn't look like a list literal
                self.logger.warning("Generated code did not assign to 'job_manifests' and does not appear to be a direct list literal.")
                raise ValueError("Generated code did not produce a list of JobManifest objects, and 'job_manifests' variable was not found or correctly assigned.")

        except SyntaxError as e:
            self.logger.error(f"Syntax error in generated code: {e}")
            logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
            self.logger.debug(f"Problematic code for syntax error: {logged_code}")
            raise ValueError(f"Syntax error in generated code: {e}") from e
        except NameError as e:
            self.logger.error(f"Name error in generated code: {e}. Ensure JobManifest is correctly used and all necessary variables are defined if code isn't self-contained.")
            logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
            self.logger.debug(f"Problematic code for name error: {logged_code}")
            raise ValueError(f"Name error in generated code: {e}. This might happen if the code tries to use undefined variables or modules other than JobManifest.") from e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during execution of generated code: {e}")
            logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
            self.logger.debug(f"Problematic code for general exception: {logged_code}")
            raise ValueError(f"An unexpected error occurred during execution of generated code: {e}") from e
    
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
