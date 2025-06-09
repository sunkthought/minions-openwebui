# Partials File: partials/citation_manager.py

import re
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urlparse

class CitationManager:
    """
    Advanced Citation System for MinionS using Open WebUI's inline citation format.
    Manages citations from both document sources and web search results,
    ensuring proper formatting and traceability.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.citation_registry = {}
        self.citation_counter = 0
        self.source_mapping = {}
    
    def create_citation_id(self, source_type: str, source_id: str) -> str:
        """
        Create a unique citation ID for tracking.
        
        Args:
            source_type: Type of source ('document', 'web', 'rag')
            source_id: Unique identifier for the source
            
        Returns:
            str: Unique citation ID
        """
        self.citation_counter += 1
        citation_id = f"{source_type}_{self.citation_counter}"
        
        self.source_mapping[citation_id] = {
            "type": source_type,
            "source_id": source_id,
            "created_at": None  # Could add timestamp if needed
        }
        
        return citation_id
    
    def register_document_citation(self, document_name: str, chunk_info: Dict[str, Any], 
                                 cited_text: str, relevance_score: float = None) -> str:
        """
        Register a citation from a document source.
        
        Args:
            document_name: Name of the source document
            chunk_info: Information about the chunk (page, section, etc.)
            cited_text: The actual text being cited
            relevance_score: Optional relevance score from RAG
            
        Returns:
            str: Citation ID for reference
        """
        citation_id = self.create_citation_id("document", document_name)
        
        self.citation_registry[citation_id] = {
            "type": "document",
            "document_name": document_name,
            "chunk_info": chunk_info,
            "cited_text": cited_text,
            "relevance_score": relevance_score,
            "formatted_citation": self._format_document_citation(
                document_name, chunk_info, cited_text, relevance_score
            )
        }
        
        if self.debug_mode:
            print(f"[Citation] Registered document citation {citation_id}: {document_name}")
        
        return citation_id
    
    def register_web_citation(self, search_result: Dict[str, Any], cited_text: str) -> str:
        """
        Register a citation from a web search result.
        
        Args:
            search_result: Web search result with title, url, snippet
            cited_text: The actual text being cited
            
        Returns:
            str: Citation ID for reference
        """
        url = search_result.get('url', '')
        title = search_result.get('title', 'Web Source')
        
        citation_id = self.create_citation_id("web", url or title)
        
        self.citation_registry[citation_id] = {
            "type": "web",
            "title": title,
            "url": url,
            "search_result": search_result,
            "cited_text": cited_text,
            "formatted_citation": self._format_web_citation(search_result, cited_text)
        }
        
        if self.debug_mode:
            print(f"[Citation] Registered web citation {citation_id}: {title}")
        
        return citation_id
    
    def register_rag_citation(self, rag_chunk: Dict[str, Any], cited_text: str) -> str:
        """
        Register a citation from RAG-retrieved content.
        
        Args:
            rag_chunk: RAG chunk with metadata and relevance score
            cited_text: The actual text being cited
            
        Returns:
            str: Citation ID for reference
        """
        doc_name = rag_chunk.get('document_name', 'RAG Source')
        chunk_id = rag_chunk.get('chunk_id', 'unknown_chunk')
        
        citation_id = self.create_citation_id("rag", chunk_id)
        
        self.citation_registry[citation_id] = {
            "type": "rag",
            "document_name": doc_name,
            "chunk_id": chunk_id,
            "rag_chunk": rag_chunk,
            "cited_text": cited_text,
            "relevance_score": rag_chunk.get('relevance_score'),
            "formatted_citation": self._format_rag_citation(rag_chunk, cited_text)
        }
        
        if self.debug_mode:
            print(f"[Citation] Registered RAG citation {citation_id}: {doc_name}")
        
        return citation_id
    
    def _format_document_citation(self, document_name: str, chunk_info: Dict[str, Any], 
                                cited_text: str, relevance_score: float = None) -> str:
        """Format a document citation according to Open WebUI standards."""
        citation_parts = [f"Document: {document_name}"]
        
        # Add page information if available
        if chunk_info.get('page'):
            citation_parts.append(f"Page {chunk_info['page']}")
        
        # Add section information if available
        if chunk_info.get('section'):
            citation_parts.append(f"Section: {chunk_info['section']}")
        
        # Add relevance score if available (from RAG)
        if relevance_score is not None:
            citation_parts.append(f"Relevance: {relevance_score:.2f}")
        
        location = ", ".join(citation_parts)
        return f"{location}: \"{self._truncate_citation_text(cited_text)}\""
    
    def _format_web_citation(self, search_result: Dict[str, Any], cited_text: str) -> str:
        """Format a web citation according to Open WebUI standards."""
        title = search_result.get('title', 'Web Source')
        url = search_result.get('url', '')
        
        # Extract domain for cleaner display
        domain = ""
        if url:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
            except:
                domain = url[:50] + "..." if len(url) > 50 else url
        
        if domain:
            return f"Web: {title} ({domain}): \"{self._truncate_citation_text(cited_text)}\""
        else:
            return f"Web: {title}: \"{self._truncate_citation_text(cited_text)}\""
    
    def _format_rag_citation(self, rag_chunk: Dict[str, Any], cited_text: str) -> str:
        """Format a RAG citation according to Open WebUI standards."""
        doc_name = rag_chunk.get('document_name', 'RAG Source')
        chunk_id = rag_chunk.get('chunk_id', 'unknown')
        relevance = rag_chunk.get('relevance_score', 0.0)
        
        metadata = rag_chunk.get('metadata', {})
        location_parts = [f"Document: {doc_name}"]
        
        if metadata.get('page'):
            location_parts.append(f"Page {metadata['page']}")
        if metadata.get('section'):
            location_parts.append(f"Section: {metadata['section']}")
        
        location_parts.append(f"Chunk: {chunk_id}")
        location_parts.append(f"Relevance: {relevance:.2f}")
        
        location = ", ".join(location_parts)
        return f"{location}: \"{self._truncate_citation_text(cited_text)}\""
    
    def _truncate_citation_text(self, text: str, max_length: int = 100) -> str:
        """Truncate citation text to a reasonable length."""
        if len(text) <= max_length:
            return text
        
        # Find a good break point near the limit
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.7:  # If we can break at 70% of max length
            return text[:last_space] + "..."
        else:
            return text[:max_length] + "..."
    
    def format_inline_citation(self, citation_id: str, cited_text: str) -> str:
        """
        Format text with inline citation using Open WebUI's citation tags.
        
        Args:
            citation_id: ID of the registered citation
            cited_text: Text to be cited
            
        Returns:
            str: Text formatted with inline citation tags
        """
        if citation_id not in self.citation_registry:
            if self.debug_mode:
                print(f"[Citation] Warning: Citation ID {citation_id} not found in registry")
            return cited_text
        
        citation_info = self.citation_registry[citation_id]
        formatted_citation = citation_info['formatted_citation']
        
        # Use Open WebUI's citation format: <cite>text</cite>
        return f'<cite title="{formatted_citation}">{cited_text}</cite>'
    
    def add_citation_to_text(self, text: str, citation_id: str, 
                           cited_portion: str = None) -> str:
        """
        Add citation to specific portion of text.
        
        Args:
            text: The full text
            citation_id: ID of the citation to add
            cited_portion: Specific portion to cite (if None, cites whole text)
            
        Returns:
            str: Text with citation added
        """
        if cited_portion is None:
            return self.format_inline_citation(citation_id, text)
        
        if cited_portion in text:
            cited_text = self.format_inline_citation(citation_id, cited_portion)
            return text.replace(cited_portion, cited_text, 1)  # Replace only first occurrence
        else:
            if self.debug_mode:
                print(f"[Citation] Warning: Cited portion not found in text")
            return text
    
    def extract_citations_from_task_result(self, task_result: Dict[str, Any], 
                                         context_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and register citations from a task result.
        
        Args:
            task_result: Task result containing answer and citation fields
            context_sources: List of source contexts (documents, web results, RAG chunks)
            
        Returns:
            Dict: Enhanced task result with properly formatted citations
        """
        enhanced_result = task_result.copy()
        
        # Get the citation text from the task result
        raw_citation = task_result.get('citation', '')
        answer_text = task_result.get('answer', '')
        
        if not raw_citation:
            return enhanced_result
        
        # Try to match citation with source contexts
        best_match = self._find_best_citation_match(raw_citation, context_sources)
        
        if best_match:
            source_type = best_match['type']
            cited_text = raw_citation
            
            if source_type == 'document':
                citation_id = self.register_document_citation(
                    best_match['document_name'],
                    best_match.get('chunk_info', {}),
                    cited_text
                )
            elif source_type == 'web':
                citation_id = self.register_web_citation(
                    best_match['search_result'],
                    cited_text
                )
            elif source_type == 'rag':
                citation_id = self.register_rag_citation(
                    best_match['rag_chunk'],
                    cited_text
                )
            else:
                citation_id = None
            
            # Format the answer with inline citations
            if citation_id and answer_text:
                enhanced_result['answer'] = self.add_citation_to_text(
                    answer_text, citation_id, cited_text
                )
                enhanced_result['citation_id'] = citation_id
        
        return enhanced_result
    
    def _find_best_citation_match(self, citation_text: str, 
                                context_sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the best matching source for a citation text.
        
        Args:
            citation_text: The citation text to match
            context_sources: Available source contexts
            
        Returns:
            Dict: Best matching source or None
        """
        best_match = None
        best_score = 0.0
        
        for source in context_sources:
            content = source.get('content', '')
            score = self._calculate_text_similarity(citation_text, content)
            
            if score > best_score and score > 0.3:  # Minimum similarity threshold
                best_score = score
                best_match = source
        
        return best_match
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def generate_citation_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all citations used.
        
        Returns:
            Dict: Citation summary with counts and sources
        """
        summary = {
            "total_citations": len(self.citation_registry),
            "by_type": {"document": 0, "web": 0, "rag": 0},
            "sources": []
        }
        
        for citation_id, citation_info in self.citation_registry.items():
            citation_type = citation_info['type']
            summary["by_type"][citation_type] += 1
            
            if citation_type == "document":
                source_name = citation_info['document_name']
            elif citation_type == "web":
                source_name = citation_info['title']
            elif citation_type == "rag":
                source_name = citation_info['document_name']
            else:
                source_name = "Unknown"
            
            summary["sources"].append({
                "id": citation_id,
                "type": citation_type,
                "source": source_name,
                "citation": citation_info['formatted_citation']
            })
        
        return summary
    
    def clear_citations(self) -> None:
        """Clear all citation data for a new session."""
        self.citation_registry.clear()
        self.source_mapping.clear()
        self.citation_counter = 0
        
        if self.debug_mode:
            print("[Citation] Citation registry cleared")