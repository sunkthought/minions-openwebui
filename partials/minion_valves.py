# Partials File: partials/minion_valves.py
from pydantic import BaseModel, Field

class MinionValves(BaseModel):
    """
    Configuration settings (valves) specifically for the Minion (conversational) pipe.
    These settings control the behavior of the Minion protocol, including API keys,
    model selections, timeouts, operational parameters, extraction instructions,
    expected output format, and confidence threshold.
    """
    # Essential configuration only
    supervisor_provider: str = Field(
        default="anthropic", 
        description="Provider for supervisor model: 'anthropic' or 'openai'"
    )
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for the remote model (e.g., Claude)"
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key"
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model identifier (e.g., for Anthropic: claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality; for OpenAI: gpt-4o, gpt-4-turbo, gpt-4)",
    )
    openai_model: str = Field(
        default="gpt-4o", 
        description="OpenAI model to use when supervisor_provider is 'openai'"
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
    use_structured_output: bool = Field(
        default=True, 
        description="Enable JSON structured output for local model responses (requires local model support)."
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
    max_exploration_questions: int = Field(
        default=3,
        description="Maximum questions in exploration phase (broad understanding)",
        ge=1,
        le=10
    )
    max_deep_dive_questions: int = Field(
        default=4,
        description="Maximum questions in deep dive phase (specific topics)",
        ge=1,
        le=10
    )
    max_gap_filling_questions: int = Field(
        default=2,
        description="Maximum questions in gap filling phase (missing information)",
        ge=1,
        le=10
    )
    
    # Answer Validation (v0.3.6b)
    enable_answer_validation: bool = Field(
        default=True,
        description="Enable answer quality validation and clarification requests"
    )
    max_clarification_attempts: int = Field(
        default=1,
        description="Maximum clarification requests per question",
        ge=0,
        le=3
    )

    # --- v0.3.9 Open WebUI Integration Features ---
    
    # Web Search Integration
    enable_web_search: bool = Field(
        default=False,
        title="Enable Web Search",
        description="Enable web search integration for conversational queries that require external information."
    )
    
    # Native RAG Pipeline Integration
    use_native_rag: bool = Field(
        default=True,
        title="Use Native RAG",
        description="Use Open WebUI's RAG infrastructure for intelligent retrieval instead of naive chunking."
    )
    rag_top_k: int = Field(
        default=5,
        title="RAG Top-K Results",
        description="Number of top relevant chunks to retrieve from RAG pipeline.",
        ge=1, le=20
    )
    rag_relevance_threshold: float = Field(
        default=0.7,
        title="RAG Relevance Threshold",
        description="Minimum relevance score for RAG retrieved chunks (0.0-1.0).",
        ge=0.0, le=1.0
    )
    
    # Streaming Response Support
    enable_streaming_responses: bool = Field(
        default=True,
        title="Enable Streaming Responses",
        description="Provide real-time updates during conversational rounds."
    )
    
    # Advanced Citation System
    enable_advanced_citations: bool = Field(
        default=True,
        title="Enable Advanced Citations",
        description="Use Open WebUI's inline citation format for traceable conversational responses."
    )
    citation_max_length: int = Field(
        default=100,
        title="Citation Max Length",
        description="Maximum length for citation text before truncation.",
        ge=50, le=500
    )
    
    # Multi-Document Knowledge Base Support
    enable_multi_document_context: bool = Field(
        default=True,
        title="Multi-Document Context",
        description="Enable conversations across multiple documents in knowledge base."
    )

    # The following class is part of the Pydantic configuration and is standard.
    # It ensures that extra fields passed to the model are ignored rather than causing an error.
    class Config:
        extra = "ignore"
