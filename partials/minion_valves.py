# Partials File: partials/minion_valves.py
from typing import Dict
from pydantic import BaseModel, Field

class MinionValves(BaseModel):
    """
    Configuration settings (valves) specifically for the Minion (conversational) pipe.
    These settings control the behavior of the Minion protocol, including API keys,
    model selections, timeouts, operational parameters, extraction instructions,
    expected output format, and confidence threshold.
    """
    # Essential configuration only
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for the remote model (e.g., Claude)"
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model identifier (e.g., for Anthropic: claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    local_model: str = Field(
        default="llama3.2", description="Local Ollama model name"
    )
    max_rounds: int = Field(
        default=2, 
        description="Maximum conversation rounds between remote and local models."
    )
    show_conversation: bool = Field(
        default=True,
        description="Show full conversation between local and remote models in the output.",
    )
    timeout_local: int = Field(
        default=60, 
        description="Timeout for local model calls in seconds. Local model processes full context."
    )
    timeout_claude: int = Field(
        default=60, description="Timeout for remote model API calls in seconds."
    )
    max_tokens_claude: int = Field(
        default=4000, description="Maximum tokens for remote model's responses."
    )
    ollama_num_predict: int = Field(
        default=1000, 
        description="num_predict for Ollama generation (max output tokens for local model)."
    )
    chunk_size: int = Field(
        default=5000, 
        description="Maximum chunk size in characters for context fed to local models during conversation."
    )
    max_chunks: int = Field(
        default=2, 
        description="Maximum number of document chunks to process. Helps manage processing load for large documents."
    )
    
    # Custom local model parameters
    local_model_context_length: int = Field(
        default=4096,
        description="Context window size for the local model. Set this based on your local model's capabilities."
    )
    local_model_temperature: float = Field(
        default=0.7,
        description="Temperature for local model generation (0.0-2.0). Lower values make output more focused and deterministic.",
        ge=0.0,
        le=2.0
    )
    local_model_top_k: int = Field(
        default=40,
        description="Top-k sampling for local model. Limits vocabulary to top k tokens. Set to 0 to disable.",
        ge=0
    )
    chunk_size: int = Field(
        default=5000, 
        description="Maximum chunk size in characters for context fed to local models during conversation."
    )
    max_chunks: int = Field(
        default=2, 
        description="Maximum number of document chunks to process. Helps manage processing load for large documents."
    )
    use_structured_output: bool = Field(
        default=True, 
        description="Enable JSON structured output for local model responses (requires local model support)."
    )
    enable_completion_detection: bool = Field(
        default=True,
        description="Enable detection of when the remote model has gathered sufficient information without explicit 'FINAL ANSWER READY' marker."
    )
    enable_completion_detection: bool = Field(
        default=True,
        description="Enable detection of when the remote model has gathered sufficient information without explicit 'FINAL ANSWER READY' marker."
    )
    debug_mode: bool = Field(
        default=False, description="Show additional technical details and verbose logs."
    )
    extraction_instructions: str = Field(
        default="", title="Extraction Instructions", description="Specific instructions for the LLM on what to extract or how to process the information."
    )
    expected_format: str = Field(
        default="text", title="Expected Output Format", description="Desired format for the LLM's output (e.g., 'text', 'JSON', 'bullet points')."
    )
    confidence_threshold: float = Field(
        default=0.7, title="Confidence Threshold", description="Minimum confidence level for the LLM's response (0.0-1.0). Primarily a suggestion to the LLM.", ge=0, le=1
    )
    
    # Conversation State Tracking (v0.3.6b)
    track_conversation_state: bool = Field(
        default=True,
        description="Enable comprehensive conversation state tracking for better context awareness"
    )
    
    # Question Deduplication (v0.3.6b)
    enable_deduplication: bool = Field(
        default=True,
        description="Prevent duplicate questions by detecting semantic similarity"
    )
    deduplication_threshold: float = Field(
        default=0.8,
        description="Similarity threshold for question deduplication (0-1). Higher = stricter matching",
        ge=0.0,
        le=1.0
    )
    
    # Conversation Flow Control (v0.3.6b)
    enable_flow_control: bool = Field(
        default=True,
        description="Enable phased conversation flow (exploration → deep dive → gap filling → synthesis)"
    )
    questions_per_phase: Dict[str, int] = Field(
        default={"exploration": 3, "deep_dive": 4, "gap_filling": 2, "synthesis": 1},
        description="Maximum questions per conversation phase"
    )

    # The following class is part of the Pydantic configuration and is standard.
    # It ensures that extra fields passed to the model are ignored rather than causing an error.
    class Config:
        extra = "ignore"
