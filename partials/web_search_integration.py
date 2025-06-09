# Partials File: partials/web_search_integration.py

import re
from typing import List, Dict, Optional, Any
import asyncio
import json
from .error_handling import MinionError

class WebSearchIntegration:
    """
    Handles web search integration for MinionS tasks using Open WebUI's search tool format.
    Enables task decomposition to include web search when analyzing queries that require
    both document analysis and external information.
    """
    
    def __init__(self, valves, debug_mode: bool = False):
        self.valves = valves
        self.debug_mode = debug_mode
        self.search_results_cache = {}
    
    def is_web_search_enabled(self) -> bool:
        """Check if web search is enabled via valves."""
        return getattr(self.valves, 'enable_web_search', False)
    
    def requires_web_search(self, task_description: str, query: str) -> bool:
        """
        Analyze if a task requires web search based on task description and original query.
        
        Args:
            task_description: The specific task to be executed
            query: The original user query
            
        Returns:
            bool: True if web search is needed
        """
        web_search_indicators = [
            "current", "latest", "recent", "today", "now", "2024", "2025",
            "news", "update", "compare with", "versus", "vs", 
            "market price", "stock", "weather", "status",
            "search online", "web search", "internet",
            "fact check", "verify", "confirm"
        ]
        
        combined_text = f"{task_description} {query}".lower()
        return any(indicator in combined_text for indicator in web_search_indicators)
    
    def determine_task_type(self, task_description: str, query: str, has_documents: bool) -> str:
        """
        Determine the type of task based on requirements.
        
        Args:
            task_description: The specific task to be executed
            query: The original user query
            has_documents: Whether documents are available for analysis
            
        Returns:
            str: Task type - "document_analysis", "web_search", or "hybrid"
        """
        needs_web_search = self.requires_web_search(task_description, query)
        
        if has_documents and needs_web_search:
            return "hybrid"
        elif needs_web_search:
            return "web_search"
        else:
            return "document_analysis"
    
    def generate_search_query(self, task_description: str, original_query: str) -> str:
        """
        Generate an optimized search query for the given task.
        
        Args:
            task_description: The specific task requiring web search
            original_query: The original user query for context
            
        Returns:
            str: Optimized search query
        """
        # Extract key terms and concepts
        key_terms = []
        
        # Remove common stop words but keep important query words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        # Combine task and query, extract meaningful terms
        combined = f"{task_description} {original_query}"
        words = re.findall(r'\b\w+\b', combined.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        # Build search query (limit to most relevant terms)
        search_query = " ".join(unique_terms[:8])  # Limit to 8 terms
        
        if self.debug_mode:
            print(f"[WebSearch] Generated search query: '{search_query}' from task: '{task_description}'")
        
        return search_query
    
    async def execute_web_search(self, search_query: str) -> Dict[str, Any]:
        """
        Execute web search using Open WebUI's search tool format.
        
        Args:
            search_query: The search query to execute
            
        Returns:
            Dict containing search results with citations
        """
        # Check cache first
        if search_query in self.search_results_cache:
            if self.debug_mode:
                print(f"[WebSearch] Using cached results for: '{search_query}'")
            return self.search_results_cache[search_query]
        
        try:
            # Generate the search tool call using Open WebUI format
            search_tool_call = f'''__TOOL_CALL__
{{"name": "web_search", "parameters": {{"query": "{search_query}"}}}}
__TOOL_CALL__'''
            
            if self.debug_mode:
                print(f"[WebSearch] Executing search: {search_tool_call}")
            
            # Note: In a real implementation, this would be handled by Open WebUI's pipeline
            # For now, we return a structured placeholder that matches expected format
            search_results = {
                "query": search_query,
                "tool_call": search_tool_call,
                "results": [],
                "citations": [],
                "status": "pending_tool_execution"
            }
            
            # Cache the results
            self.search_results_cache[search_query] = search_results
            
            return search_results
            
        except Exception as e:
            error_msg = f"Web search failed for query '{search_query}': {str(e)}"
            if self.debug_mode:
                print(f"[WebSearch] {error_msg}")
            raise MinionError(error_msg)
    
    def parse_search_results(self, raw_results: str) -> List[Dict[str, Any]]:
        """
        Parse search results returned by Open WebUI's search tool.
        
        Args:
            raw_results: Raw search results from the tool
            
        Returns:
            List of parsed search result dictionaries
        """
        try:
            # Try to parse as JSON first
            if raw_results.strip().startswith('{') or raw_results.strip().startswith('['):
                parsed = json.loads(raw_results)
                if isinstance(parsed, dict) and 'results' in parsed:
                    return parsed['results']
                elif isinstance(parsed, list):
                    return parsed
                else:
                    return [parsed]
            
            # Fallback: parse structured text format
            results = []
            current_result = {}
            
            lines = raw_results.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Title:'):
                    if current_result:
                        results.append(current_result)
                    current_result = {'title': line[6:].strip()}
                elif line.startswith('URL:'):
                    current_result['url'] = line[4:].strip()
                elif line.startswith('Snippet:'):
                    current_result['snippet'] = line[8:].strip()
                elif line and 'title' in current_result and 'snippet' not in current_result:
                    current_result['snippet'] = line
            
            if current_result:
                results.append(current_result)
            
            return results
            
        except Exception as e:
            if self.debug_mode:
                print(f"[WebSearch] Failed to parse search results: {str(e)}")
            return [{"title": "Search Results", "snippet": raw_results, "url": ""}]
    
    def format_search_context(self, search_results: List[Dict[str, Any]], task_description: str) -> str:
        """
        Format search results into context for task execution.
        
        Args:
            search_results: Parsed search results
            task_description: The task that required web search
            
        Returns:
            str: Formatted context string
        """
        if not search_results:
            return f"No web search results found for task: {task_description}"
        
        context = f"Web search results for task '{task_description}':\n\n"
        
        for i, result in enumerate(search_results[:5], 1):  # Limit to top 5 results
            title = result.get('title', 'Untitled')
            snippet = result.get('snippet', 'No description available')
            url = result.get('url', '')
            
            context += f"{i}. {title}\n"
            context += f"   Source: {url}\n"
            context += f"   Content: {snippet}\n\n"
        
        return context
    
    def create_web_search_citation(self, result: Dict[str, Any], relevant_text: str) -> str:
        """
        Create a properly formatted citation for web search results.
        
        Args:
            result: Search result dictionary
            relevant_text: The relevant text that was cited
            
        Returns:
            str: Formatted citation
        """
        title = result.get('title', 'Web Search Result')
        url = result.get('url', '')
        
        if url:
            return f"Web source: {title} ({url}): \"{relevant_text}\""
        else:
            return f"Web source: {title}: \"{relevant_text}\""
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about web search usage.
        
        Returns:
            Dict with search statistics
        """
        return {
            "total_searches": len(self.search_results_cache),
            "cached_queries": list(self.search_results_cache.keys()),
            "web_search_enabled": self.is_web_search_enabled()
        }