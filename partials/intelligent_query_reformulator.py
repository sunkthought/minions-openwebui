import json
from typing import Any, Callable, List, Optional, Dict

# Forward declaration for QueryMetadata if it's not directly importable
# from .query_analyzer import QueryMetadata
# For concatenation, QueryMetadata will be globally available if query_analyzer.py is listed before this file.
# We will assume QueryMetadata is a Dict for now, or define a placeholder if necessary.
# Placeholder to be refined if direct type hinting from QueryAnalyzer is problematic in concatenated file.
QueryMetadata = Dict[str, Any]


class IntelligentQueryReformulator:
    def __init__(self, debug_mode: bool = False, valves: Optional[Any] = None, call_claude_func: Optional[Callable] = None):
        self.debug_mode = debug_mode
        self.valves = valves
        self.call_claude_func = call_claude_func
        self._log_debug("IntelligentQueryReformulator initialized.")

    def _log_debug(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] IntelligentQueryReformulator: {message}")

    async def reformulate(self, query: str, query_metadata: QueryMetadata, conversation_history: Optional[List[str]] = None) -> str:
        self._log_debug(f"Starting intelligent reformulation for query: '{query}'")
        if not self.call_claude_func or not self.valves:
            self._log_debug("Claude call function or valves not configured. Skipping reformulation.")
            return query

        # Strategy selection logic (heuristic-based for now)
        reformulation_strategy_description = "Analyze the user query and apply relevant reformulation strategies. Focus on:"
        strategies_applied = []

        ambiguity_score = query_metadata.get('ambiguity_score', 0.0)
        decomposability_score = query_metadata.get('decomposability_score', 0.0) # Assuming this is present
        query_type = query_metadata.get('query_type', {}).get('value', 'UNKNOWN') # Access .value if QueryType is an Enum

        # Access new valves for thresholds
        min_ambiguity_threshold = getattr(self.valves, 'min_ambiguity_for_intelligent_reformulation', 0.5)
        min_decomposability_threshold = getattr(self.valves, 'min_decomposability_score_for_intelligent_reformulation', 0.5) # Note: this valve name was 'min_decomposability_score_for_intelligent_reformulation' in plan

        trigger_reformulation = False

        if ambiguity_score > min_ambiguity_threshold:
            reformulation_strategy_description += "\n- Specificity Enhancement: Make the query more specific, clear, and less ambiguous. Resolve vague terms."
            strategies_applied.append("Specificity Enhancement")
            trigger_reformulation = True
            self._log_debug(f"High ambiguity detected ({ambiguity_score:.2f} > {min_ambiguity_threshold:.2f}). Prioritizing Specificity Enhancement.")

        if decomposability_score < min_decomposability_threshold: # Lower score means harder to decompose
            reformulation_strategy_description += "\n- Hierarchical Breakdown & Task Alignment: If complex, break it into a primary question and suggest potential sub-questions. Reformulate for clear, actionable steps suitable for a multi-agent system."
            strategies_applied.append("Hierarchical Breakdown & Task Alignment")
            trigger_reformulation = True
            self._log_debug(f"Low decomposability detected ({decomposability_score:.2f} < {min_decomposability_threshold:.2f}). Prioritizing Hierarchical Breakdown & Task Alignment.")
        
        if query_type in ["ANALYSIS_REQUEST", "COMPARISON", "UNKNOWN"] and not trigger_reformulation: # Apply to certain types even if scores are not extreme
             reformulation_strategy_description += "\n- Scope Optimization: Adjust the query's scope. If too broad, narrow it. If too narrow for the implied goal, suggest broadening."
             strategies_applied.append("Scope Optimization")
             trigger_reformulation = True
             self._log_debug(f"Query type '{query_type}' suggests Scope Optimization.")


        if not trigger_reformulation:
            self._log_debug(f"Query does not meet heuristic criteria for intelligent reformulation (Ambiguity: {ambiguity_score:.2f}, Decomposability: {decomposability_score:.2f}). Skipping.")
            return query
        
        self._log_debug(f"Selected strategies: {', '.join(strategies_applied)}")

        # Construct the prompt for Claude
        # Few-shot examples should be embedded here
        prompt_parts = [
            "You are an expert query reformulator. Your goal is to refine user queries to make them clearer, more specific, and better structured for a multi-agent question-answering system (MinionS).",
            "The system first analyzes the query, and the following metadata is available:",
            f"Original Query: "{query}"",
            f"Query Metadata: {json.dumps(query_metadata, indent=2)}",
        ]

        if conversation_history:
            history_str = "\n".join([f"- {msg}" for msg in conversation_history[-3:]]) # Last 3 turns
            prompt_parts.append(f"Recent Conversation History (for context):\n{history_str}")
        
        prompt_parts.append(f"\nBased on this, please reformulate the 'Original Query'. {reformulation_strategy_description}")
        prompt_parts.append("\nConsider the following examples of good reformulations:")

        # Few-shot examples:
        prompt_parts.append("""
        ---
        Example 1: Specificity Enhancement
        Original Query: "Tell me about the economy."
        Query Metadata: {"ambiguity_score": 0.8, "query_type": {"value": "ANALYSIS_REQUEST"}, "scope": {"value": "BROAD"}}
        Reformulated Query: "What are the key indicators of the current US economic status, including inflation, unemployment, and GDP growth?"

        Example 2: Hierarchical Breakdown & Task Alignment
        Original Query: "Analyze the impact of AI on healthcare and suggest investment opportunities."
        Query Metadata: {"decomposability_score": 0.3, "query_type": {"value": "ANALYSIS_REQUEST"}, "detected_patterns": [{"type": "MULTI_PART", "text": "AI on healthcare AND investment opportunities"}]}
        Reformulated Query: "What is the primary impact of AI on the healthcare sector, focusing on diagnostics, treatment, and patient care? (Follow up: Based on this impact, what are promising investment opportunities in AI-driven healthcare companies?)"

        Example 3: Scope Optimization (Narrowing)
        Original Query: "Everything about renewable energy."
        Query Metadata: {"ambiguity_score": 0.6, "query_type": {"value": "ANALYSIS_REQUEST"}, "scope": {"value": "COMPREHENSIVE"}}
        Reformulated Query: "What are the recent advancements and future prospects of solar and wind energy technologies?"
        
        Example 4: Task Alignment (Implicit Breakdown for MinionS)
        Original Query: "Should I invest in stocks or bonds now given the market volatility?"
        Query Metadata: {"ambiguity_score": 0.4, "decomposability_score": 0.6, "query_type": {"value": "COMPARISON"}}
        Reformulated Query: "Compare the current risk-reward profile of investing in the S&P 500 index versus 10-year US Treasury bonds, considering recent market volatility."
        ---
        """)

        prompt_parts.append("\nProvide ONLY the reformulated query as a single line of text. If the original query is already optimal and needs no changes based on the strategies, return the original query verbatim.")
        prompt_parts.append("\nReformulated Query:")

        final_prompt = "\n".join(prompt_parts)
        self._log_debug(f"Prompt for Claude:\n{final_prompt}")

        try:
            # Use the specific model from valves if available
            model_to_use = getattr(self.valves, 'intelligent_reformulation_model', None)
            if not model_to_use: # Fallback to general remote_model if specific one not set
                 model_to_use = getattr(self.valves, 'remote_model', 'claude-3-5-haiku-20241022') # Default from plan if all else fails

            self._log_debug(f"Using model for intelligent reformulation: {model_to_use}")
            
            # Create a temporary valve-like object for the call_claude_func if it expects one
            # This assumes call_claude_func might look at valves.remote_model, valves.max_tokens_claude etc.
            # We can pass the full self.valves or a modified one.
            # For now, let's assume call_claude_func can take a simple prompt and model name.
            # If it requires the full 'valves' object, we need to ensure it's correctly passed.
            # The call_claude function in common_api_calls.py takes (valves, prompt, model_override=None)
            
            reformulated_query = await self.call_claude_func(
                self.valves, # Pass the main valves object
                final_prompt,
                model_override=model_to_use # Pass the specific model for this call
            )

            reformulated_query = reformulated_query.strip()

            if not reformulated_query or reformulated_query.lower() == query.lower():
                self._log_debug("Claude returned an empty response or the same query. No changes applied.")
                return query
            
            # Basic check to avoid Claude's meta-comments like "Okay, here's the reformulated query:"
            if reformulated_query.startswith("Reformulated Query:") or reformulated_query.startswith("Okay, here's the reformulated query:"):
                reformulated_query = reformulated_query.split(":", 1)[-1].strip()
            
            # Further check if Claude just says "Original query:"
            if reformulated_query.startswith("Original Query:") or reformulated_query.lower().startswith("the original query is already optimal"):
                 self._log_debug("Claude indicated no changes needed or returned the original query marker.")
                 return query


            self._log_debug(f"Intelligently reformulated query: '{reformulated_query}'")
            return reformulated_query

        except Exception as e:
            self._log_debug(f"Error during Claude call for intelligent reformulation: {e}")
            return query

if __name__ == '__main__':
    # Example usage for testing (requires async environment to run)
    class MockValves:
        def __init__(self):
            self.debug_mode = True
            self.min_ambiguity_for_intelligent_reformulation = 0.5
            self.min_decomposability_score_for_intelligent_reformulation = 0.5 # Corrected valve name
            self.intelligent_reformulation_model = "claude-3-opus-20240229" # Example model
            self.remote_model = "claude-3-haiku-20240307" # Fallback
            self.max_tokens_claude = 2000 # For call_claude

    async def mock_call_claude(valves, prompt, model_override=None):
        print(f"--- MOCK CLAUDE CALL (Model: {model_override or valves.remote_model}) ---")
        # print(f"Prompt: {prompt}")
        # Simulate Claude's response based on prompt keywords
        if "Tell me about the economy." in prompt:
            return "What are the key indicators of the current US economic status, including inflation, unemployment, and GDP growth?"
        if "Analyze the impact of AI on healthcare and suggest investment opportunities." in prompt:
            return "What is the primary impact of AI on the healthcare sector, focusing on diagnostics, treatment, and patient care? (Follow up: Based on this impact, what are promising investment opportunities in AI-driven healthcare companies?)"
        if "Everything about renewable energy." in prompt:
            return "What are the recent advancements and future prospects of solar and wind energy technologies?"
        if "already optimal" in prompt: # Test case for no change
            return query_to_test_no_change
        return "Mocked reformulated query based on analysis."

    async def main_test():
        valves = MockValves()
        reformulator = IntelligentQueryReformulator(debug_mode=True, valves=valves, call_claude_func=mock_call_claude)

        global query_to_test_no_change # To make it accessible in mock_call_claude
        
        test_queries_metadata = [
            {
                "query": "Tell me about the economy.",
                "metadata": {"original_query": "Tell me about the economy.", "ambiguity_score": 0.8, "decomposability_score": 0.7, "query_type": {"value": "ANALYSIS_REQUEST"}, "scope": {"value": "BROAD"}},
                "history": ["User: What's new today?", "Assistant: Markets are up."]
            },
            {
                "query": "Analyze the impact of AI on healthcare and suggest investment opportunities.",
                "metadata": {"original_query": "Analyze the impact of AI on healthcare and suggest investment opportunities.", "ambiguity_score": 0.4, "decomposability_score": 0.3, "query_type": {"value": "ANALYSIS_REQUEST"}, "detected_patterns": [{"type": "MULTI_PART", "text": "AI on healthcare AND investment opportunities"}]},
                "history": None
            },
            {
                "query": "Everything about renewable energy.",
                "metadata": {"original_query": "Everything about renewable energy.", "ambiguity_score": 0.6, "decomposability_score": 0.6, "query_type": {"value": "ANALYSIS_REQUEST"}, "scope": {"value": "COMPREHENSIVE"}},
                "history": None
            },
            {
                "query": "What is the current price of Bitcoin?", # Should skip reformulation by default thresholds
                "metadata": {"original_query": "What is the current price of Bitcoin?", "ambiguity_score": 0.1, "decomposability_score": 0.9, "query_type": {"value": "QUESTION"}, "scope": {"value": "SPECIFIC"}},
                "history": None
            },
        ]
        
        query_to_test_no_change = "Is the sky blue?"
        test_queries_metadata.append(
             {
                "query": query_to_test_no_change, 
                "metadata": {"original_query": query_to_test_no_change, "ambiguity_score": 0.9, "decomposability_score": 0.9, "query_type": {"value": "QUESTION"}, "scope": {"value": "SPECIFIC"}}, # High ambiguity to trigger, but prompt asks Claude to return original if optimal
                "history": None
            }
        )


        for item in test_queries_metadata:
            print(f"\n--- Testing Query: '{item['query']}' ---")
            reformulated = await reformulator.reformulate(item["query"], item["metadata"], conversation_history=item.get("history"))
            print(f"Original: {item['query']}")
            print(f"Reformulated: {reformulated}")

    # import asyncio # Uncomment to run test
    # asyncio.run(main_test()) # Uncomment to run test
