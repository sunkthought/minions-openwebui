# Partials File: partials/unified_web_search_handler.py
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from .streaming_mixins import StreamingPatterns


class UnifiedWebSearchHandler:
    """
    Handles web search integration for both Minion and MinionS protocols.
    Provides unified web search handling with streaming updates.
    """
    
    def __init__(self, 
                web_search: Any,
                tool_bridge: Any,
                citation_manager: Any,
                streaming_manager: Any):
        self.web_search = web_search
        self.tool_bridge = tool_bridge
        self.citation_manager = citation_manager
        self.streaming_manager = streaming_manager
        self.search_cache = {}
        self.search_history = []
    
    async def handle_search_if_needed(self,
                                    query: str,
                                    context: str = "",
                                    task_description: str = None,
                                    protocol_type: str = "minion") -> Dict[str, Any]:
        """
        Unified web search handling with streaming updates.
        
        Args:
            query: User query
            context: Document context (if any)
            task_description: Specific task description (for MinionS)
            protocol_type: "minion" or "minions" for protocol-specific behavior
            
        Returns:
            Dict with search_performed, results, and enriched_context
        """
        # Initialize result structure
        search_result = {
            "search_performed": False,
            "results": [],
            "enriched_context": context,
            "search_queries": [],
            "search_metadata": {}
        }
        
        # Check if web search is enabled
        if not self.web_search.is_web_search_enabled():
            return search_result
        
        # Determine if search is needed
        needs_search = await self._determine_search_necessity(
            query, context, task_description, protocol_type
        )
        
        if not needs_search:
            return search_result
        
        # Execute search with streaming progress
        return await self._execute_search_with_progress(
            query, context, task_description, protocol_type
        )
    
    async def _determine_search_necessity(self,
                                        query: str,
                                        context: str,
                                        task_description: str,
                                        protocol_type: str) -> bool:
        """Determine if web search is necessary for the given inputs."""
        
        # Protocol-specific search determination
        if protocol_type == "minions" and task_description:
            # For MinionS, check if the specific task requires web search
            return self.web_search.requires_web_search(task_description, query)
        else:
            # For Minion, check if the query needs current information
            return (
                self.web_search.requires_web_search("", query) or
                (not context and self._query_needs_current_info(query))
            )
    
    def _query_needs_current_info(self, query: str) -> bool:
        """Determine if query requires current information."""
        current_info_indicators = [
            "current", "latest", "today", "recent", "now",
            "2024", "2025", "news", "update", "trending",
            "happening", "this year", "this month"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in current_info_indicators)
    
    async def _execute_search_with_progress(self,
                                          query: str,
                                          context: str,
                                          task_description: str,
                                          protocol_type: str) -> Dict[str, Any]:
        """Execute web search with streaming progress updates."""
        
        search_start_time = time.time()
        
        # Use streaming patterns for the search operation
        async def search_operation():
            return await self._perform_search_operation(
                query, context, task_description, protocol_type
            )
        
        try:
            # Execute search with retry logic and progress tracking
            search_result = await StreamingPatterns.stream_with_retry(
                search_operation,
                self.streaming_manager,
                max_attempts=2,
                phase_name="web_search",
                backoff_factor=1.5
            )
            
            # Log search performance
            search_duration = time.time() - search_start_time
            search_result["search_metadata"]["duration"] = search_duration
            search_result["search_metadata"]["timestamp"] = search_start_time
            
            # Add to search history
            self.search_history.append({
                "query": query,
                "task_description": task_description,
                "protocol_type": protocol_type,
                "duration": search_duration,
                "results_count": len(search_result.get("results", []))
            })
            
            return search_result
            
        except Exception as e:
            # Handle search failure gracefully
            if self.streaming_manager and self.streaming_manager.is_streaming_enabled():
                await self.streaming_manager.stream_granular_update(
                    "web_search",
                    "search_failed",
                    1.0,
                    f"Web search failed: {str(e)}"
                )
            
            return {
                "search_performed": False,
                "results": [],
                "enriched_context": context,
                "search_queries": [],
                "search_metadata": {"error": str(e)},
                "error": f"Web search failed: {str(e)}"
            }
    
    async def _perform_search_operation(self,
                                      query: str,
                                      context: str,
                                      task_description: str,
                                      protocol_type: str) -> Dict[str, Any]:
        """Perform the actual search operation with detailed progress."""
        
        # Step 1: Generate search queries
        if self.streaming_manager and self.streaming_manager.is_streaming_enabled():
            await self.streaming_manager.stream_granular_update(
                "web_search",
                "generating_queries",
                0.1,
                "Generating optimized search queries"
            )
        
        search_queries = await self._generate_search_queries(
            query, task_description, protocol_type
        )
        
        # Step 2: Execute searches
        if self.streaming_manager and self.streaming_manager.is_streaming_enabled():
            await self.streaming_manager.stream_granular_update(
                "web_search",
                "executing_searches",
                0.3,
                f"Executing {len(search_queries)} search queries"
            )
        
        all_results = []
        for i, search_query in enumerate(search_queries):
            # Check cache first
            cache_key = self._generate_cache_key(search_query)
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    all_results.extend(cached_result["results"])
                    continue
            
            # Perform search
            search_results = await self._execute_single_search(search_query)
            all_results.extend(search_results)
            
            # Cache results
            self.search_cache[cache_key] = {
                "results": search_results,
                "timestamp": time.time(),
                "query": search_query
            }
            
            # Update progress
            progress = 0.3 + (0.4 * (i + 1) / len(search_queries))
            if self.streaming_manager and self.streaming_manager.is_streaming_enabled():
                await self.streaming_manager.stream_granular_update(
                    "web_search",
                    f"search_{i+1}_completed",
                    progress,
                    f"Completed search {i+1}/{len(search_queries)}: {len(search_results)} results"
                )
        
        # Step 3: Process and rank results
        if self.streaming_manager and self.streaming_manager.is_streaming_enabled():
            await self.streaming_manager.stream_granular_update(
                "web_search",
                "processing_results",
                0.8,
                f"Processing and ranking {len(all_results)} search results"
            )
        
        processed_results = await self._process_search_results(
            all_results, query, task_description
        )
        
        # Step 4: Generate enriched context
        if self.streaming_manager and self.streaming_manager.is_streaming_enabled():
            await self.streaming_manager.stream_granular_update(
                "web_search",
                "enriching_context",
                0.9,
                "Enriching context with search results"
            )
        
        enriched_context = await self._generate_enriched_context(
            context, processed_results, query, task_description
        )
        
        # Final completion
        if self.streaming_manager and self.streaming_manager.is_streaming_enabled():
            await self.streaming_manager.stream_granular_update(
                "web_search",
                "search_completed",
                1.0,
                f"Web search completed: {len(processed_results)} relevant results found"
            )
        
        return {
            "search_performed": True,
            "results": processed_results,
            "enriched_context": enriched_context,
            "search_queries": search_queries,
            "search_metadata": {
                "total_results": len(all_results),
                "processed_results": len(processed_results),
                "queries_executed": len(search_queries)
            }
        }
    
    async def _generate_search_queries(self,
                                     query: str,
                                     task_description: str,
                                     protocol_type: str) -> List[str]:
        """Generate optimized search queries based on the input."""
        search_queries = []
        
        if protocol_type == "minions" and task_description:
            # For MinionS, generate queries based on the specific task
            primary_query = self.web_search.generate_search_query(task_description, query)
            search_queries.append(primary_query)
            
            # Generate additional queries for comprehensive coverage
            additional_queries = await self._generate_additional_task_queries(
                task_description, query
            )
            search_queries.extend(additional_queries)
        else:
            # For Minion, generate queries based on the main query
            primary_query = self.web_search.generate_search_query("", query)
            search_queries.append(primary_query)
            
            # Generate related queries
            related_queries = await self._generate_related_queries(query)
            search_queries.extend(related_queries)
        
        # Remove duplicates and limit to reasonable number
        unique_queries = list(dict.fromkeys(search_queries))  # Preserves order
        return unique_queries[:3]  # Limit to 3 queries maximum
    
    async def _generate_additional_task_queries(self,
                                              task_description: str,
                                              query: str) -> List[str]:
        """Generate additional search queries for task-specific searches."""
        additional_queries = []
        
        # Extract key concepts from task description
        task_lower = task_description.lower()
        
        # Add time-based queries for current information requests
        if any(word in task_lower for word in ["current", "latest", "recent", "today"]):
            time_query = f"{query} 2024 latest"
            additional_queries.append(time_query)
        
        # Add comparative queries
        if any(word in task_lower for word in ["compare", "versus", "vs", "difference"]):
            comparative_query = f"{query} comparison analysis"
            additional_queries.append(comparative_query)
        
        return additional_queries
    
    async def _generate_related_queries(self, query: str) -> List[str]:
        """Generate related queries for comprehensive search coverage."""
        related_queries = []
        
        # Add specific variation queries
        query_lower = query.lower()
        
        # Add news/update queries for current events
        if any(word in query_lower for word in ["news", "update", "current", "latest"]):
            news_query = f"{query} news update"
            related_queries.append(news_query)
        
        return related_queries
    
    async def _execute_single_search(self, search_query: str) -> List[Dict[str, Any]]:
        """Execute a single search query using the tool bridge."""
        try:
            # Use tool bridge to execute web search
            search_result = await self.tool_bridge.execute_web_search(search_query)
            return search_result.get("results", [])
        except Exception as e:
            # Log error but don't fail the entire operation
            if hasattr(self.web_search, 'debug_mode') and self.web_search.debug_mode:
                print(f"DEBUG [UnifiedWebSearchHandler]: Search failed for '{search_query}': {str(e)}")
            return []
    
    async def _process_search_results(self,
                                    results: List[Dict[str, Any]],
                                    query: str,
                                    task_description: str) -> List[Dict[str, Any]]:
        """Process and rank search results by relevance."""
        if not results:
            return []
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Score results based on relevance
        scored_results = []
        for result in unique_results:
            score = self._calculate_relevance_score(result, query, task_description)
            scored_results.append((score, result))
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results[:10]]  # Top 10 results
    
    def _calculate_relevance_score(self,
                                 result: Dict[str, Any],
                                 query: str,
                                 task_description: str) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("url", "").lower()
        
        query_terms = query.lower().split()
        task_terms = task_description.lower().split() if task_description else []
        
        # Score based on query term matches
        for term in query_terms:
            if term in title:
                score += 2.0  # Title matches are important
            if term in snippet:
                score += 1.0  # Snippet matches
            if term in url:
                score += 0.5  # URL matches
        
        # Score based on task term matches (for MinionS)
        for term in task_terms:
            if term in title:
                score += 1.5
            if term in snippet:
                score += 0.75
        
        # Bonus for recent results (if timestamp available)
        if "date" in result or "timestamp" in result:
            score += 0.5
        
        # Penalty for very short snippets
        if len(snippet) < 50:
            score -= 0.5
        
        return max(0.0, score)  # Ensure non-negative score
    
    async def _generate_enriched_context(self,
                                       original_context: str,
                                       search_results: List[Dict[str, Any]],
                                       query: str,
                                       task_description: str) -> str:
        """Generate enriched context by combining original context with search results."""
        if not search_results:
            return original_context
        
        # Build web search context section
        web_context_parts = ["=== WEB SEARCH RESULTS ==="]
        
        for i, result in enumerate(search_results[:5]):  # Top 5 results
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No snippet available")
            url = result.get("url", "")
            
            result_section = f"""
Result {i+1}: {title}
URL: {url}
Summary: {snippet}
"""
            web_context_parts.append(result_section.strip())
        
        web_context = "\n\n".join(web_context_parts)
        
        # Combine with original context
        if original_context:
            return f"{original_context}\n\n{web_context}"
        else:
            return web_context
    
    def _generate_cache_key(self, search_query: str) -> str:
        """Generate cache key for search query."""
        return f"search_{hash(search_query.lower())}"
    
    def _is_cache_valid(self, cached_result: Dict[str, Any], max_age_hours: float = 1.0) -> bool:
        """Check if cached search result is still valid."""
        cache_timestamp = cached_result.get("timestamp", 0)
        current_time = time.time()
        age_hours = (current_time - cache_timestamp) / 3600
        
        return age_hours < max_age_hours
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get the search history for debugging/analytics."""
        return self.search_history.copy()
    
    def clear_search_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.search_cache)
        valid_entries = sum(1 for entry in self.search_cache.values() if self._is_cache_valid(entry))
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "cache_hit_rate": valid_entries / total_entries if total_entries > 0 else 0.0
        }