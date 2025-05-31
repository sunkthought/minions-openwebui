import logging
import asyncio
import json
from typing import List, Dict, Any, Optional

# Helper function for truncating text
def _truncate_text(text: str, max_length: int) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

async def minions_pipe_method(
    pipe_instance: Any,
    body: dict,
    __user__: dict,
    __request__: 'Request', # Use string literal if Request might not be resolvable by linter in partial
    __files__: List[dict],
    __pipe_id__: str
    # Removed default for files, added __user__, __request__, __pipe_id__
) -> str:
    """
    Main MinionS protocol logic. This function orchestrates task decomposition,
    execution, and synthesis using various helper functions and models that are
    expected to be available in its scope after code generation and concatenation.
    """
    logger = logging.getLogger(__name__)
    valves = pipe_instance.valves

    # --- Dynamically available components (post-concatenation by generator) ---
    # JobManifest and TaskResult model classes are expected to be globally available.
    # TaskResult = pipe_instance.TaskResult # This line removed/ensured not present.

    # Helper Functions from other partials (assumed global after concatenation):
    # _extract_context_from_messages, _extract_context_from_files, _create_chunks
    # get_minions_code_generation_claude_prompt
    # _execute_generated_code, _parse_tasks_helper
    # execute_tasks_on_chunks, calculate_token_savings

    conversation_log: List[str] = []
    debug_log: List[str] = []
    actual_initial_claude_prompt_text: str = ""


    try:
        if valves.debug_mode: logger.info("MinionS pipe method invoked.")
        if not valves.anthropic_api_key: return "❌ **Error:** Anthropic API key not configured."

        messages = body.get("messages", [])
        if not messages: return "❌ **Error:** No messages provided."
        
        user_query = messages[-1]["content"]
        logger.info(f"User query: {_truncate_text(user_query, 100)}")

        context_from_messages = _extract_context_from_messages(messages[:-1], valves, logger)
        context_from_files = await _extract_context_from_files(files, valves, logger)
        all_context_parts = []
        if context_from_messages: all_context_parts.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files: all_context_parts.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")
        context = "\n\n".join(all_context_parts)

        if not context: return "ℹ️ **Note:** No context provided for MinionS."

        if valves.debug_mode:
            debug_log.append(f"Query: {_truncate_text(user_query,100)}, Context len: {len(context)}")

        conversation_log.append("### 🎯 Task Decomposition")
        job_manifests: List[Any] = []

        if valves.enable_code_decomposition:
            conversation_log.append("🤖 Using Code-Based Decomposition")
            actual_initial_claude_prompt_text = get_minions_code_generation_claude_prompt(
                user_query, _truncate_text(context, 500), valves, "JobManifest"
            )
            try:
                generated_code = await asyncio.wait_for(call_claude(valves, actual_initial_claude_prompt_text), timeout=getattr(valves, 'timeout_claude_codegen', 45.0))
                if valves.debug_mode: debug_log.append(f"Generated Code:\n{_truncate_text(generated_code,300)}...")
                job_manifests = _execute_generated_code(generated_code, JobManifest, logger)
                conversation_log.append(f"✅ Generated {len(job_manifests)} tasks via code.")
            except Exception as e:
                logger.error(f"Code Decomp Error: {e}", exc_info=valves.debug_mode)
                return f"❌ **Error (Code Decomp):** {_truncate_text(str(e),100)}"
        else:
            conversation_log.append("🗣️ Using Natural Language Decomposition")
            actual_initial_claude_prompt_text = f"Query: \"{user_query}\"\nContext len: {len(context)}.\nMax tasks: {valves.max_tasks_per_round}. List tasks."
            try:
                claude_response = await asyncio.wait_for(call_claude(valves, actual_initial_claude_prompt_text), timeout=getattr(valves, 'timeout_claude', 30.0))
                if valves.debug_mode: debug_log.append(f"NL Decomp Response:\n{_truncate_text(claude_response,300)}...")
                # _parse_tasks_helper is assumed global
                parsed_task_strings = _parse_tasks_helper(claude_response, valves.max_tasks_per_round, debug_log, valves)
                job_manifests = [JobManifest(task_id=f"task_{i+1}", task_description=d) for i, d in enumerate(parsed_task_strings)]
                conversation_log.append(f"✅ Generated {len(job_manifests)} tasks via NL.")
            except Exception as e:
                logger.error(f"NL Decomp Error: {e}", exc_info=valves.debug_mode)
                return f"❌ **Error (NL Decomp):** {_truncate_text(str(e),100)}"

        if not job_manifests: conversation_log.append("⚠️ No tasks generated.")

        conversation_log.append("### 📄 Chunking & Execution")
        chunks = _create_chunks(context, valves, logger) # Assumed global
        conversation_log.append(f"{len(chunks)} chunks created.")
        task_results: List[Dict[str, str]] = []
        if job_manifests and chunks:
            # execute_tasks_on_chunks is assumed global, pass global call_ollama and TaskResult
            task_results = await execute_tasks_on_chunks(
                job_manifests, chunks, conversation_log, valves, call_ollama, logger, TaskResult
            )
        elif not job_manifests: conversation_log.append("ℹ️ No tasks to execute.")
        elif not chunks: conversation_log.append("ℹ️ No chunks to process."

        conversation_log.append("\n### 🔄 Synthesis")
        summary_parts = [f"ID: {r.get('task_id')}\nDesc: {_truncate_text(r.get('task_description',''),70)}\nRes: {_truncate_text(r.get('result',''),150)}\n---" for r in task_results]
        results_summary = "\n".join(summary_parts) if summary_parts else "No task results."
        synthesis_prompt = f"Query: \"{user_query}\"\nInfo:\n{results_summary}\nFinal Answer:"
        try:
            final_response = await asyncio.wait_for(call_claude(valves, synthesis_prompt), timeout=getattr(valves, 'timeout_claude_synthesis', 45.0))
        except Exception as e:
            logger.error(f"Synthesis Error: {e}", exc_info=valves.debug_mode)
            final_response = f"❌ **Error (Synthesis):** {_truncate_text(str(e),100)}"

        if valves.show_conversation or valves.debug_mode : # Add synth response to conv log if shown
            conversation_log.append(f"**🤖 Claude (Synthesis):**\n{_truncate_text(final_response, 200)}...")

        output_parts = []
        if valves.show_conversation: output_parts.extend(["## 🗣️ MinionS Log", *conversation_log, "---"])
        if valves.debug_mode: output_parts.extend(["## 🔍 Debug Log", *debug_log, "---"])
        output_parts.append(f"## 🎯 Final Answer\n{final_response}")

        # calculate_token_savings is assumed global
        stats = calculate_token_savings(
            [actual_initial_claude_prompt_text or "N/A"], [synthesis_prompt], results_summary,
            final_response, len(context), len(user_query), len(chunks), len(job_manifests)
        )
        success_keywords = ["information not found", "timeout", "error", "invalid chunk requested"]
        successful_count = sum(1 for r in task_results if r.get("result","").lower() not in success_keywords and r.get("result"))

        output_parts.extend([
            f"\n## 📊 MinionS Stats",
            f"- Tasks: {len(job_manifests)} (Success: {successful_count})",
            f"- Chunks: {len(chunks)}",
            f"- Token Savings (Claude): ~{stats.get('percentage_savings_claude', 0):.0f}%"
        ])
        if valves.debug_mode: output_parts.extend([
            f"- Claude Tokens (Traditional): ~{stats.get('traditional_tokens_claude',0):,}",
            f"- Claude Tokens (MinionS): ~{stats.get('minions_tokens_claude',0):,}"
        ])
        return "\n\n".join(output_parts)
    except Exception as e:
        logger.critical(f"MinionS pipe critical error: {e}", exc_info=valves.debug_mode) # Log full exc_info if debug
        return f"❌ **Fatal MinionS Error:** {_truncate_text(str(e),100)}"
# Note: Placeholders for global context utilities (_extract_context_from_messages,
# _extract_context_from_files, _create_chunks) are removed from here.
# They are expected to be defined in other partial files (e.g., common_context_utils.py)
# and made available in the global scope by the generator script.
