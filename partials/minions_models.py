# Partials File: partials/minions_models.py
from typing import Optional, Dict, List # Add Dict
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass

class TaskResult(BaseModel):
    """
    Structured response format for individual task execution by a local model
    within the MinionS (multi-task, multi-round) protocol.
    This model defines the expected output structure when a local model completes
    a sub-task defined by the remote model.
    """
    explanation: str = Field(
        description="Brief explanation of the findings or the process taken to answer the task."
    )
    citation: Optional[str] = Field(
        default=None, 
        description="Direct quote from the analyzed text (chunk) that supports the answer. Should be None if no specific citation is applicable."
    )
    answer: Optional[str] = Field(
        default=None, 
        description="The specific information extracted or the answer to the sub-task. Should be None if the information is not found in the provided text chunk."
    )
    confidence: str = Field(
        default="LOW", 
        description="Confidence level in the provided answer/explanation (e.g., HIGH, MEDIUM, LOW)."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields during parsing
        # schema_extra = {
        #     "example": {
        #         "explanation": "The document mentions a budget of $500,000 for Phase 1.",
        #         "citation": "Page 5, Paragraph 2: 'Phase 1 of the project has been allocated a budget of $500,000.'",
        #         "answer": "$500,000",
        #         "confidence": "HIGH"
        #     }
        # }

class RoundMetrics(BaseModel):
    """
    Model to store various metrics collected during a single round of the MinionS protocol.
    """
    round_number: int
    tasks_executed: int
    task_success_count: int
    task_failure_count: int
    avg_chunk_processing_time_ms: float
    total_unique_findings_count: int = 0 # Defaulting as per plan for Iteration 1
    execution_time_ms: float
    success_rate: float # Calculated as task_success_count / tasks_executed

    # New fields for Iteration 2
    avg_confidence_score: float = 0.0 # Default to 0.0
    confidence_distribution: Dict[str, int] = Field(default_factory=lambda: {"HIGH": 0, "MEDIUM": 0, "LOW": 0})
    confidence_trend: str = "N/A" # Default to N/A

    # New fields for Iteration 3
    new_findings_count_this_round: int = 0
    duplicate_findings_count_this_round: int = 0
    redundancy_percentage_this_round: float = 0.0
    # cross_round_similarity_score: float = 0.0 # Deferred

    # New fields for Iteration 4
    sufficiency_score: float = Field(default=0.0, description="Overall information sufficiency score (0-1).")
    information_components: Dict[str, bool] = Field(default_factory=dict, description="Status of identified information components from the query.")
    component_coverage_percentage: float = Field(default=0.0, description="Percentage of information components addressed (0-1).")

    # New fields for Iteration 5 (Convergence Detection)
    information_gain_rate: float = Field(default=0.0, description="Rate of new information gained in this round, typically based on the count of new findings.")
    novel_findings_percentage_this_round: float = Field(default=0.0, description="Percentage of findings in this round that are new compared to all findings from this round (new + duplicate).")
    task_failure_rate_trend: str = Field(default="N/A", description="Trend of task failures (e.g., 'increasing', 'decreasing', 'stable') compared to the previous round.")
    convergence_detected_this_round: bool = Field(default=False, description="Flag indicating if convergence criteria were met based on this round's analysis.")
    predicted_value_of_next_round: str = Field(default="N/A", description="Qualitative prediction of the potential value of executing another round (e.g., 'low', 'medium', 'high').")

    class Config:
        extra = "ignore"

# New models for v0.3.8 scaling strategies and adaptive round control

class ScalingStrategy(Enum):
    """Scaling strategies from the MinionS paper"""
    NONE = "none"
    REPEATED_SAMPLING = "repeated_sampling"  # Run tasks multiple times
    FINER_DECOMPOSITION = "finer_decomposition"  # Break into smaller subtasks
    CONTEXT_CHUNKING = "context_chunking"  # Use smaller, overlapping chunks

@dataclass
class RoundAnalysis:
    """Analysis of round results for adaptive round control"""
    information_gain: float  # 0.0-1.0, comparing new vs previous info
    average_confidence: float  # 0.0-1.0, based on HIGH/MEDIUM/LOW
    coverage_ratio: float  # 0.0-1.0, how much of query is addressed
    should_continue: bool
    reason: str

class RepeatedSamplingResult(BaseModel):
    """Result from repeated sampling strategy"""
    original_result: TaskResult
    sample_results: List[TaskResult]
    aggregated_result: TaskResult
    confidence_boost: float = 0.0
    consistency_score: float = 0.0  # How consistent were the samples
    
    class Config:
        extra = "ignore"

class DecomposedTask(BaseModel):
    """A task that has been further decomposed"""
    original_task: str
    subtasks: List[str]
    subtask_results: List[TaskResult] = []
    synthesized_result: Optional[TaskResult] = None
    
    class Config:
        extra = "ignore"

class ChunkingStrategy(BaseModel):
    """Configuration for context chunking strategy"""
    chunk_size: int
    overlap_ratio: float  # 0.0-0.5
    chunks_created: int = 0
    overlap_chars: int = 0
    
    class Config:
        extra = "ignore"

# --- v0.3.9 Open WebUI Integration Models ---

class TaskType(Enum):
    """Types of tasks for v0.3.9 Open WebUI integrations"""
    DOCUMENT_ANALYSIS = "document_analysis"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"
    SYNTHESIS = "synthesis"

class WebSearchResult(BaseModel):
    """Result from web search integration"""
    query: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    relevance_score: float = 0.0
    source_domain: str = ""
    
    class Config:
        extra = "ignore"

class RAGChunk(BaseModel):
    """RAG retrieved chunk with metadata"""
    content: str
    document_id: str
    document_name: str
    chunk_id: str
    relevance_score: float
    start_position: int = 0
    end_position: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "ignore"

class Citation(BaseModel):
    """Citation with Open WebUI inline format support"""
    citation_id: str
    source_type: str  # "document", "web", "rag"
    cited_text: str
    formatted_citation: str
    relevance_score: Optional[float] = None
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "ignore"

class EnhancedTaskResult(BaseModel):
    """Enhanced task result with v0.3.9 features"""
    explanation: str = Field(
        description="Brief explanation of the findings or the process taken to answer the task."
    )
    citation: Optional[str] = Field(
        default=None, 
        description="Direct quote from the analyzed text (chunk) that supports the answer."
    )
    answer: Optional[str] = Field(
        default=None, 
        description="The specific information extracted or the answer to the sub-task."
    )
    confidence: str = Field(
        default="LOW", 
        description="Confidence level in the provided answer/explanation (e.g., HIGH, MEDIUM, LOW)."
    )
    
    # v0.3.9 enhancements
    task_type: TaskType = Field(default=TaskType.DOCUMENT_ANALYSIS)
    citations: List[Citation] = Field(default_factory=list)
    web_search_results: List[WebSearchResult] = Field(default_factory=list)
    rag_chunks_used: List[RAGChunk] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "ignore"

class TaskVisualization(BaseModel):
    """Task visualization data for Mermaid diagrams"""
    task_id: str
    description: str
    task_type: TaskType
    status: str  # pending, running, completed, failed
    document_refs: List[str] = Field(default_factory=list)
    web_query: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    class Config:
        extra = "ignore"

class StreamingUpdate(BaseModel):
    """Streaming update message"""
    update_type: str  # phase, task_progress, search, error, metrics
    message: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "ignore"

class DocumentReference(BaseModel):
    """Document reference for multi-document support"""
    document_id: str
    document_name: str
    document_type: str = "unknown"
    size_bytes: int = 0
    chunk_count: int = 0
    upload_date: Optional[str] = None
    last_accessed: Optional[str] = None
    
    class Config:
        extra = "ignore"

class KnowledgeBaseContext(BaseModel):
    """Context for multi-document knowledge base operations"""
    available_documents: List[DocumentReference] = Field(default_factory=list)
    referenced_documents: List[str] = Field(default_factory=list)
    cross_document_relationships: Dict[str, List[str]] = Field(default_factory=dict)
    
    class Config:
        extra = "ignore"

class PipelineMetrics(BaseModel):
    """Enhanced metrics for v0.3.9 pipeline execution"""
    execution_time_ms: float = 0.0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    web_searches_performed: int = 0
    rag_retrievals_performed: int = 0
    citations_generated: int = 0
    documents_processed: int = 0
    streaming_updates_sent: int = 0
    tokens_saved_vs_naive: int = 0
    
    class Config:
        extra = "ignore"
