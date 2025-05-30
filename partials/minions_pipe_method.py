import traceback

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