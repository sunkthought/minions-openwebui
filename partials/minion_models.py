# Partials File: partials/minion_models.py
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

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
