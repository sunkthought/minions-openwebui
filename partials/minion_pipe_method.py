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