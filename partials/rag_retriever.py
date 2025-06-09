# Partials File: partials/rag_retriever.py

import re
from typing import List, Dict, Optional, Any, Tuple
from .error_handling import MinionError

class RAGRetriever:
    """
    Native RAG Pipeline Integration for MinionS using Open WebUI's RAG infrastructure.
    Provides intelligent retrieval instead of naive chunking by leveraging Open WebUI's
    document reference syntax and retrieval mechanisms.
    """
    
    def __init__(self, valves, debug_mode: bool = False):
        self.valves = valves
        self.debug_mode = debug_mode
        self.document_registry = DocumentRegistry()
        self.retrieved_chunks_cache = {}
    
    def is_native_rag_enabled(self) -> bool:
        """Check if native RAG is enabled via valves."""
        return getattr(self.valves, 'use_native_rag', True)
    
    def get_rag_top_k(self) -> int:
        """Get the top-k setting for RAG retrieval."""
        return getattr(self.valves, 'rag_top_k', 5)
    
    def get_relevance_threshold(self) -> float:
        """Get the relevance threshold for RAG retrieval."""
        return getattr(self.valves, 'rag_relevance_threshold', 0.7)
    
    def detect_document_references(self, query: str, tasks: List[str] = None) -> List[str]:
        """
        Detect document references using the '#' syntax in queries and tasks.
        
        Args:
            query: The original user query
            tasks: List of task descriptions (optional)
            
        Returns:
            List of document IDs/names referenced
        """
        document_refs = []
        
        # Pattern to match #document_name or #"document name with spaces"
        pattern = r'#(?:"([^"]+)"|(\S+))'
        
        # Check main query
        matches = re.finditer(pattern, query)
        for match in matches:
            doc_ref = match.group(1) if match.group(1) else match.group(2)
            if doc_ref not in document_refs:
                document_refs.append(doc_ref)
        
        # Check task descriptions if provided
        if tasks:
            for task in tasks:
                matches = re.finditer(pattern, task)
                for match in matches:
                    doc_ref = match.group(1) if match.group(1) else match.group(2)
                    if doc_ref not in document_refs:
                        document_refs.append(doc_ref)
        
        if self.debug_mode and document_refs:
            print(f"[RAG] Detected document references: {document_refs}")
        
        return document_refs
    
    def retrieve_relevant_chunks(self, task_description: str, document_refs: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a task using RAG pipeline.
        
        Args:
            task_description: The specific task requiring information
            document_refs: Optional list of specific documents to search
            
        Returns:
            List of relevant chunks with metadata and relevance scores
        """
        cache_key = f"{task_description}|{document_refs}"
        if cache_key in self.retrieved_chunks_cache:
            if self.debug_mode:
                print(f"[RAG] Using cached retrieval for task: {task_description[:50]}...")
            return self.retrieved_chunks_cache[cache_key]
        
        try:
            retrieved_chunks = []
            
            if self.is_native_rag_enabled() and document_refs:
                # Use native RAG with document references
                for doc_ref in document_refs:
                    chunks = self._retrieve_from_document(task_description, doc_ref)
                    retrieved_chunks.extend(chunks)
            else:
                # Fallback to general retrieval (assuming all available documents)
                retrieved_chunks = self._retrieve_general(task_description)
            
            # Filter by relevance threshold
            threshold = self.get_relevance_threshold()
            filtered_chunks = [
                chunk for chunk in retrieved_chunks 
                if chunk.get('relevance_score', 0.0) >= threshold
            ]
            
            # Sort by relevance and limit to top-k
            top_k = self.get_rag_top_k()
            filtered_chunks.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            final_chunks = filtered_chunks[:top_k]
            
            # Cache the results
            self.retrieved_chunks_cache[cache_key] = final_chunks
            
            if self.debug_mode:
                print(f"[RAG] Retrieved {len(final_chunks)} relevant chunks for task")
                for i, chunk in enumerate(final_chunks[:3]):  # Show top 3
                    score = chunk.get('relevance_score', 0.0)
                    preview = chunk.get('content', '')[:100]
                    print(f"[RAG]   {i+1}. Score: {score:.3f}, Preview: {preview}...")
            
            return final_chunks
            
        except Exception as e:
            error_msg = f"RAG retrieval failed for task '{task_description}': {str(e)}"
            if self.debug_mode:
                print(f"[RAG] {error_msg}")
            # Return empty list to allow fallback to naive chunking
            return []
    
    def _retrieve_from_document(self, task_description: str, doc_ref: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific document using RAG.
        
        Args:
            task_description: The task requiring information
            doc_ref: Document reference ID/name
            
        Returns:
            List of retrieved chunks with metadata
        """
        # In a real implementation, this would interface with Open WebUI's RAG system
        # For now, we simulate the expected structure
        
        # Simulate RAG retrieval result structure
        simulated_chunks = [
            {
                "content": f"Simulated RAG content from {doc_ref} for task: {task_description}",
                "document_id": doc_ref,
                "document_name": doc_ref,
                "chunk_id": f"{doc_ref}_chunk_1",
                "relevance_score": 0.85,
                "start_position": 0,
                "end_position": 100,
                "metadata": {
                    "page": 1,
                    "section": "Introduction",
                    "document_type": "pdf"
                }
            }
        ]
        
        return simulated_chunks
    
    def _retrieve_general(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Perform general retrieval across all available documents.
        
        Args:
            task_description: The task requiring information
            
        Returns:
            List of retrieved chunks with metadata
        """
        # Simulate general RAG retrieval
        simulated_chunks = [
            {
                "content": f"General RAG content for task: {task_description}",
                "document_id": "general_doc",
                "document_name": "Available Documents",
                "chunk_id": "general_chunk_1",
                "relevance_score": 0.75,
                "start_position": 0,
                "end_position": 100,
                "metadata": {
                    "source": "multi_document",
                    "retrieval_type": "general"
                }
            }
        ]
        
        return simulated_chunks
    
    def format_rag_context(self, retrieved_chunks: List[Dict[str, Any]], task_description: str) -> str:
        """
        Format retrieved RAG chunks into context for task execution.
        
        Args:
            retrieved_chunks: List of retrieved chunks with metadata
            task_description: The task requiring this context
            
        Returns:
            str: Formatted context string with relevance scores
        """
        if not retrieved_chunks:
            return f"No relevant information found via RAG for task: {task_description}"
        
        context = f"Relevant information retrieved for task '{task_description}':\n\n"
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            content = chunk.get('content', 'No content available')
            score = chunk.get('relevance_score', 0.0)
            doc_name = chunk.get('document_name', 'Unknown Document')
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            
            context += f"[Relevance: {score:.2f}] Document: {doc_name} (ID: {chunk_id})\n"
            context += f"{content}\n\n"
        
        return context
    
    def create_rag_citation(self, chunk: Dict[str, Any], relevant_text: str) -> str:
        """
        Create a properly formatted citation for RAG-retrieved content.
        
        Args:
            chunk: Retrieved chunk with metadata
            relevant_text: The specific text being cited
            
        Returns:
            str: Formatted citation with document and chunk information
        """
        doc_name = chunk.get('document_name', 'Unknown Document')
        chunk_id = chunk.get('chunk_id', 'Unknown Chunk')
        relevance = chunk.get('relevance_score', 0.0)
        
        metadata = chunk.get('metadata', {})
        page = metadata.get('page', '')
        section = metadata.get('section', '')
        
        citation_parts = [f"Document: {doc_name}"]
        
        if page:
            citation_parts.append(f"Page {page}")
        if section:
            citation_parts.append(f"Section: {section}")
        
        citation_parts.append(f"Chunk ID: {chunk_id}")
        citation_parts.append(f"Relevance: {relevance:.2f}")
        
        location = ", ".join(citation_parts)
        return f"{location}: \"{relevant_text}\""
    
    def should_fallback_to_naive_chunking(self, retrieved_chunks: List[Dict[str, Any]], 
                                         document_content: str) -> bool:
        """
        Determine if we should fallback to naive chunking.
        
        Args:
            retrieved_chunks: Results from RAG retrieval
            document_content: Original document content
            
        Returns:
            bool: True if should fallback to naive chunking
        """
        if not self.is_native_rag_enabled():
            return True
        
        if not retrieved_chunks:
            if self.debug_mode:
                print("[RAG] No chunks retrieved, falling back to naive chunking")
            return True
        
        # Check if retrieved content seems insufficient
        total_retrieved_chars = sum(len(chunk.get('content', '')) for chunk in retrieved_chunks)
        if total_retrieved_chars < 500:  # Less than 500 characters retrieved
            if self.debug_mode:
                print(f"[RAG] Retrieved content too small ({total_retrieved_chars} chars), falling back")
            return True
        
        return False
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about RAG retrieval usage.
        
        Returns:
            Dict with retrieval statistics
        """
        return {
            "native_rag_enabled": self.is_native_rag_enabled(),
            "top_k_setting": self.get_rag_top_k(),
            "relevance_threshold": self.get_relevance_threshold(),
            "total_retrievals": len(self.retrieved_chunks_cache),
            "document_registry_size": len(self.document_registry.documents)
        }


class DocumentRegistry:
    """
    Registry to track available documents and their metadata for multi-document support.
    """
    
    def __init__(self):
        self.documents = {}
        self.cross_references = {}
    
    def register_document(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """
        Register a document with its metadata.
        
        Args:
            doc_id: Unique document identifier
            metadata: Document metadata (name, type, size, upload_date, etc.)
        """
        self.documents[doc_id] = {
            "id": doc_id,
            "name": metadata.get("name", doc_id),
            "type": metadata.get("type", "unknown"),
            "size": metadata.get("size", 0),
            "upload_date": metadata.get("upload_date"),
            "chunk_count": metadata.get("chunk_count", 0),
            "last_accessed": None
        }
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document."""
        return self.documents.get(doc_id)
    
    def list_available_documents(self) -> List[Dict[str, Any]]:
        """Get list of all available documents."""
        return list(self.documents.values())
    
    def find_related_documents(self, doc_id: str) -> List[str]:
        """Find documents related to the given document ID."""
        return self.cross_references.get(doc_id, [])
    
    def add_cross_reference(self, doc_id1: str, doc_id2: str) -> None:
        """Add a cross-reference between two documents."""
        if doc_id1 not in self.cross_references:
            self.cross_references[doc_id1] = []
        if doc_id2 not in self.cross_references:
            self.cross_references[doc_id2] = []
        
        if doc_id2 not in self.cross_references[doc_id1]:
            self.cross_references[doc_id1].append(doc_id2)
        if doc_id1 not in self.cross_references[doc_id2]:
            self.cross_references[doc_id2].append(doc_id1)