# Partials File: partials/minion_models.py
from typing import List, Optional, Dict, Set, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio

class LocalAssistantResponse(BaseModel):
    """
    Structured response format for the local assistant in the Minion (conversational) protocol.
    This model defines the expected output structure when the local model processes
    a request from the remote model.
    """
    answer: str = Field(description="The main response to the question posed by the remote model.")
    confidence: str = Field(description="Confidence level of the answer (e.g., HIGH, MEDIUM, LOW).")
    key_points: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of key points extracted from the context related to the answer."
    )
    citations: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of direct quotes or citations from the context supporting the answer."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields during parsing
        # Consider adding an example for documentation if this model is complex:
        # schema_extra = {
        #     "example": {
        #         "answer": "The document states that the project was completed in Q4.",
        #         "confidence": "HIGH",
        #         "key_points": ["Project completion Q4"],
        #         "citations": ["The final report confirms project completion in Q4."]
        #     }
        # }

class ConversationMetrics(BaseModel):
    """
    Metrics tracking for Minion protocol conversations.
    Captures performance data for analysis and optimization.
    """
    rounds_used: int = Field(description="Number of Q&A rounds completed in the conversation")
    questions_asked: int = Field(description="Total number of questions asked by the remote model")
    avg_answer_confidence: float = Field(
        description="Average confidence score across all local model responses (0.0-1.0)"
    )
    total_tokens_used: int = Field(
        default=0,
        description="Estimated total tokens used across all API calls"
    )
    conversation_duration_ms: float = Field(
        description="Total conversation duration in milliseconds"
    )
    completion_detected: bool = Field(
        description="Whether the conversation ended via completion detection vs max rounds"
    )
    unique_topics_explored: int = Field(
        default=0,
        description="Count of distinct topics/themes in questions (optional)"
    )
    confidence_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of confidence levels (HIGH/MEDIUM/LOW counts)"
    )
    chunks_processed: int = Field(
        default=1,
        description="Number of document chunks processed in this conversation"
    )
    chunk_size_used: int = Field(
        default=0,
        description="The chunk size setting used for this conversation"
    )
    
    class Config:
        extra = "ignore"

class ConversationState(BaseModel):
    """Tracks the evolving state of a Minion conversation"""
    qa_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    topics_covered: Set[str] = Field(default_factory=set)
    key_findings: Dict[str, str] = Field(default_factory=dict)
    information_gaps: List[str] = Field(default_factory=list)
    current_phase: str = Field(default="exploration")
    knowledge_graph: Dict[str, List[str]] = Field(default_factory=dict)  # topic -> related facts
    phase_transitions: List[Dict[str, str]] = Field(default_factory=list)  # Track phase transitions
    
    def add_qa_pair(self, question: str, answer: str, confidence: str, key_points: List[str] = None):
        """Add a Q&A pair and update derived state"""
        self.qa_pairs.append({
            "round": len(self.qa_pairs) + 1,
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "key_points": key_points or [],
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        })
        
    def get_state_summary(self) -> str:
        """Generate a summary of current conversation state for the remote model"""
        summary = f"Conversation Phase: {self.current_phase}\n"
        summary += f"Questions Asked: {len(self.qa_pairs)}\n"
        
        if self.key_findings:
            summary += "\nKey Findings:\n"
            for topic, finding in self.key_findings.items():
                summary += f"- {topic}: {finding}\n"
                
        if self.information_gaps:
            summary += "\nInformation Gaps:\n"
            for gap in self.information_gaps:
                summary += f"- {gap}\n"
                
        return summary

class QuestionDeduplicator:
    """Handles detection and prevention of duplicate questions in conversations"""
    def __init__(self, similarity_threshold: float = 0.8):
        self.asked_questions: List[str] = []
        self.question_embeddings: List[str] = []  # Simplified: store normalized forms
        self.similarity_threshold = similarity_threshold
        
    def is_duplicate(self, question: str) -> Tuple[bool, Optional[str]]:
        """Check if question is semantically similar to a previous question"""
        # Normalize the question
        normalized = self._normalize_question(question)
        
        # Check for semantic similarity (simplified approach)
        for idx, prev_normalized in enumerate(self.question_embeddings):
            # Safety check to prevent IndexError
            if idx < len(self.asked_questions):
                if self._calculate_similarity(normalized, prev_normalized) > self.similarity_threshold:
                    return True, self.asked_questions[idx]
                
        return False, None
        
    def add_question(self, question: str):
        """Add question to the deduplication store"""
        self.asked_questions.append(question)
        self.question_embeddings.append(self._normalize_question(question))
        
    def _normalize_question(self, question: str) -> str:
        """Simple normalization - in production, use embeddings"""
        # Remove common words, lowercase, extract key terms
        stop_words = {"what", "is", "the", "are", "how", "does", "can", "you", "tell", "me", "about", 
                      "please", "could", "would", "explain", "describe", "provide", "give", "any",
                      "there", "which", "when", "where", "who", "why", "do", "have", "has", "been",
                      "was", "were", "will", "be", "being", "a", "an", "and", "or", "but", "in",
                      "on", "at", "to", "for", "of", "with", "by", "from", "as", "this", "that"}
        
        # Extract words and filter
        words = question.lower().replace("?", "").replace(".", "").replace(",", "").split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(sorted(key_words))
        
    def _calculate_similarity(self, q1: str, q2: str) -> float:
        """Calculate similarity between normalized questions using Jaccard similarity"""
        words1 = set(q1.split())
        words2 = set(q2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
        
    def get_all_questions(self) -> List[str]:
        """Return all previously asked questions"""
        return self.asked_questions.copy()

class ConversationPhase(str, Enum):
    """Phases of a structured conversation"""
    EXPLORATION = "exploration"
    DEEP_DIVE = "deep_dive"
    GAP_FILLING = "gap_filling"
    SYNTHESIS = "synthesis"

class ConversationFlowController:
    """Controls the flow of conversation through different phases"""
    def __init__(self):
        self.current_phase = ConversationPhase.EXPLORATION
        self.phase_question_count = {phase: 0 for phase in ConversationPhase}
        self.phase_transitions = {
            ConversationPhase.EXPLORATION: ConversationPhase.DEEP_DIVE,
            ConversationPhase.DEEP_DIVE: ConversationPhase.GAP_FILLING,
            ConversationPhase.GAP_FILLING: ConversationPhase.SYNTHESIS,
            ConversationPhase.SYNTHESIS: ConversationPhase.SYNTHESIS
        }
        
    def should_transition(self, state: ConversationState, valves: Any = None) -> bool:
        """Determine if conversation should move to next phase"""
        current_count = self.phase_question_count[self.current_phase]
        
        if self.current_phase == ConversationPhase.EXPLORATION:
            # Move on after configured questions or when main topics identified
            max_questions = valves.max_exploration_questions if valves else 3
            return current_count >= max_questions or (current_count >= 2 and len(state.topics_covered) >= 3)
            
        elif self.current_phase == ConversationPhase.DEEP_DIVE:
            # Move on after exploring key topics in detail
            max_questions = valves.max_deep_dive_questions if valves else 4
            return current_count >= max_questions or len(state.key_findings) >= 5
            
        elif self.current_phase == ConversationPhase.GAP_FILLING:
            # Move to synthesis when gaps are addressed
            max_questions = valves.max_gap_filling_questions if valves else 2
            return current_count >= max_questions or len(state.information_gaps) == 0
            
        return False
        
    def get_phase_guidance(self) -> str:
        """Get prompting guidance for current phase"""
        guidance = {
            ConversationPhase.EXPLORATION: 
                "You are in the EXPLORATION phase. Ask broad questions to understand the document's main topics and structure.",
            ConversationPhase.DEEP_DIVE:
                "You are in the DEEP DIVE phase. Focus on specific topics that are most relevant to the user's query.",
            ConversationPhase.GAP_FILLING:
                "You are in the GAP FILLING phase. Address specific information gaps identified in previous rounds.",
            ConversationPhase.SYNTHESIS:
                "You are in the SYNTHESIS phase. You should now have enough information. Prepare your final answer."
        }
        return guidance.get(self.current_phase, "")
        
    def transition_to_next_phase(self):
        """Move to the next conversation phase"""
        self.current_phase = self.phase_transitions[self.current_phase]
        self.phase_question_count[self.current_phase] = 0
        
    def increment_question_count(self):
        """Increment the question count for the current phase"""
        self.phase_question_count[self.current_phase] += 1
        
    def get_phase_status(self) -> Dict[str, Any]:
        """Get current phase status for debugging"""
        return {
            "current_phase": self.current_phase.value,
            "questions_in_phase": self.phase_question_count[self.current_phase],
            "total_questions_by_phase": {
                phase.value: count for phase, count in self.phase_question_count.items()
            }
        }

class AnswerValidator:
    """Validates answer quality and generates clarification requests"""
    
    @staticmethod
    def validate_answer(answer: str, confidence: str, question: str) -> Dict[str, Any]:
        """Validate answer quality and completeness"""
        issues = []
        
        # Check for non-answers
        non_answer_phrases = [
            "i don't know",
            "not sure",
            "unclear",
            "cannot determine",
            "no information",
            "not available",
            "not found",
            "not mentioned",
            "not specified"
        ]
        
        answer_lower = answer.lower()
        
        # Check if answer is too vague for low confidence
        if len(answer.split()) < 10 and confidence == "LOW":
            issues.append("Answer seems too brief for a low-confidence response")
            
        # Check for non-answers without clear indication
        non_answer_found = any(phrase in answer_lower for phrase in non_answer_phrases)
        if non_answer_found and "not found" not in answer_lower and "not available" not in answer_lower:
            issues.append("Answer indicates uncertainty without clearly stating if information is missing")
            
        # Check if answer addresses the question keywords
        question_keywords = set(question.lower().split()) - {
            "what", "how", "when", "where", "who", "why", "is", "are", "the", "does", "can", "could", "would", "should"
        }
        answer_keywords = set(answer_lower.split())
        
        if question_keywords:
            keyword_overlap = len(question_keywords.intersection(answer_keywords)) / len(question_keywords)
            if keyword_overlap < 0.2:
                issues.append("Answer may not directly address the question asked")
        
        # Check for extremely short answers to complex questions
        if len(question.split()) > 10 and len(answer.split()) < 5:
            issues.append("Answer seems too brief for a complex question")
            
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "needs_clarification": len(issues) > 0 and confidence != "HIGH",
            "severity": "high" if len(issues) >= 2 else "low" if len(issues) == 1 else "none"
        }
        
    @staticmethod
    def generate_clarification_request(validation_result: Dict[str, Any], original_question: str, answer: str) -> str:
        """Generate a clarification request based on validation issues"""
        if not validation_result["issues"]:
            return ""
            
        clarification = "I need clarification on your previous answer. "
        
        for issue in validation_result["issues"]:
            if "too brief" in issue:
                clarification += "Could you provide more detail or explanation? "
            elif "uncertainty" in issue:
                clarification += "Is this information not available in the document, or is it unclear? Please be explicit about what information is missing. "
            elif "not directly address" in issue:
                clarification += f"Let me rephrase the question: {original_question} "
            elif "complex question" in issue:
                clarification += "This seems like a complex topic that might need a more comprehensive answer. "
                
        # Add specific guidance based on the answer content
        if len(answer.split()) < 5:
            clarification += "If the information isn't in the document, please say so explicitly. If it is available, please provide the specific details."
        else:
            clarification += "Please provide more specific information from the document to fully address my question."
                
        return clarification.strip()

# --- v0.3.9 Open WebUI Integration Models for Minion Protocol ---

class ConversationType(Enum):
    """Types of conversational interactions for v0.3.9 Open WebUI integrations"""
    DOCUMENT_CONVERSATION = "document_conversation"
    WEB_ENHANCED_CONVERSATION = "web_enhanced_conversation"
    MULTI_DOCUMENT_CONVERSATION = "multi_document_conversation"
    HYBRID_CONVERSATION = "hybrid_conversation"

class WebSearchResult(BaseModel):
    """Result from web search integration in conversational context"""
    query: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    relevance_score: float = 0.0
    source_domain: str = ""
    conversation_round: int = 0  # Which round this was used in
    
    class Config:
        extra = "ignore"

class RAGChunk(BaseModel):
    """RAG retrieved chunk with metadata for conversational use"""
    content: str
    document_id: str
    document_name: str
    chunk_id: str
    relevance_score: float
    start_position: int = 0
    end_position: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    conversation_round: int = 0  # Which round this was retrieved for
    
    class Config:
        extra = "ignore"

class Citation(BaseModel):
    """Citation with Open WebUI inline format support for conversations"""
    citation_id: str
    source_type: str  # "document", "web", "rag"
    cited_text: str
    formatted_citation: str
    relevance_score: Optional[float] = None
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    conversation_round: int = 0  # Which round this citation was created in
    
    class Config:
        extra = "ignore"

class EnhancedLocalAssistantResponse(BaseModel):
    """Enhanced local assistant response with v0.3.9 features"""
    answer: str = Field(description="The main response to the question posed by the remote model.")
    confidence: str = Field(description="Confidence level of the answer (e.g., HIGH, MEDIUM, LOW).")
    key_points: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of key points extracted from the context related to the answer."
    )
    citations: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of direct quotes or citations from the context supporting the answer."
    )
    
    # v0.3.9 enhancements
    conversation_type: ConversationType = Field(default=ConversationType.DOCUMENT_CONVERSATION)
    enhanced_citations: List[Citation] = Field(default_factory=list)
    web_search_results: List[WebSearchResult] = Field(default_factory=list)
    rag_chunks_used: List[RAGChunk] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "ignore"

class StreamingConversationUpdate(BaseModel):
    """Streaming update message for conversational flow"""
    update_type: str  # phase, round_progress, search, citation, final_answer
    message: str
    round_number: int = 0
    conversation_phase: str = ""
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "ignore"

class DocumentReference(BaseModel):
    """Document reference for multi-document conversations"""
    document_id: str
    document_name: str
    document_type: str = "unknown"
    size_bytes: int = 0
    chunk_count: int = 0
    upload_date: Optional[str] = None
    last_accessed: Optional[str] = None
    conversation_relevance: float = 0.0  # How relevant to current conversation
    
    class Config:
        extra = "ignore"

class ConversationKnowledgeBase(BaseModel):
    """Knowledge base context for multi-document conversations"""
    available_documents: List[DocumentReference] = Field(default_factory=list)
    referenced_documents: List[str] = Field(default_factory=list)
    cross_document_relationships: Dict[str, List[str]] = Field(default_factory=dict)
    conversation_focus_documents: List[str] = Field(default_factory=list)  # Docs most relevant to current conversation
    
    class Config:
        extra = "ignore"

class EnhancedConversationMetrics(BaseModel):
    """Enhanced metrics for v0.3.9 conversational execution"""
    rounds_used: int = Field(description="Number of Q&A rounds completed in the conversation")
    questions_asked: int = Field(description="Total number of questions asked by the remote model")
    avg_answer_confidence: float = Field(
        description="Average confidence score across all local model responses (0.0-1.0)"
    )
    total_tokens_used: int = Field(
        default=0,
        description="Estimated total tokens used across all API calls"
    )
    conversation_duration_ms: float = Field(
        description="Total conversation duration in milliseconds"
    )
    completion_detected: bool = Field(
        description="Whether the conversation ended via completion detection vs max rounds"
    )
    
    # v0.3.9 enhancements
    web_searches_performed: int = 0
    rag_retrievals_performed: int = 0
    citations_generated: int = 0
    documents_accessed: int = 0
    streaming_updates_sent: int = 0
    conversation_type: ConversationType = ConversationType.DOCUMENT_CONVERSATION
    
    class Config:
        extra = "ignore"
