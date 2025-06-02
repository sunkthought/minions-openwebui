# Partials File: partials/minion_models.py
from typing import List, Optional
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
