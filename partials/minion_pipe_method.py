async def _call_claude_directly(valves: Any, query: str, call_claude: Callable) -> str:
    """Fallback to direct Claude call when no context is available"""
    return await call_claude(valves, f"Please answer this question: {query}")

async def minion_pipe(
    pipe_self: Any,
    body: dict,
    __user__: dict,
    __request__: Request,
    __files__: List[dict] = [],
    __pipe_id__: str = "minion-claude",
) -> str:
    """Execute the Minion protocol with Claude"""
    try:
        # Validate configuration
        if not pipe_self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

        # Extract user message and context
        messages = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."

        user_query = messages[-1]["content"]

        # Extract context from messages AND uploaded files
        context_from_messages = extract_context_from_messages(messages[:-1])
        context_from_files = await extract_context_from_files(pipe_self.valves, __files__)

        # Combine all context sources
        all_context = []
        if context_from_messages:
            all_context.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files:
            all_context.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")

        context = "\n\n".join(all_context) if all_context else ""

        if not context:
            return (
                "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                + await _call_claude_directly(pipe_self.valves, user_query, call_claude)
            )

        # Execute the Minion protocol
        result = await _execute_minion_protocol(
            pipe_self.valves, user_query, context, call_claude, call_ollama, LocalAssistantResponse
        )
        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"