# Partials File: partials/minions_valves.py
from pydantic import BaseModel, Field

class MinionsValves(BaseModel):
    """
    Configuration settings (valves) specifically for the MinionS (multi-task, multi-round) pipe.
    These settings control the behavior of the MinionS protocol, including API keys,
    model selections, timeouts, task decomposition parameters, operational parameters,
    extraction instructions, expected output format, and confidence threshold.
    """
    supervisor_provider: str = Field(
        default="anthropic", 
        description="Provider for supervisor model: 'anthropic' or 'openai'"
    )
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for the remote model (e.g., Claude)."
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key"
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model (e.g., Claude) for task decomposition and synthesis. claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality; for OpenAI: gpt-4o, gpt-4-turbo, gpt-4.",
    )
    openai_model: str = Field(
        default="gpt-4o", 
        description="OpenAI model to use when supervisor_provider is 'openai'"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL for local model execution."
    )
    local_model: str = Field(
        default="llama3.2", description="Local Ollama model name for task execution."
    )
    max_rounds: int = Field(
        default=2, description="Maximum task decomposition rounds. Each round involves remote model decomposing tasks and local models executing them."
    )
    max_tasks_per_round: int = Field(
        default=3, description="Maximum number of new sub-tasks the remote model can create in a single decomposition round."
    )
    chunk_size: int = Field(
        default=5000, description="Maximum chunk size in characters for context fed to local models during task execution."
    )
    max_chunks: int = Field( # This was present in minions-fn-claude.py but not in minion-fn-claude.py
        default=2, description="Maximum number of document chunks to process per task by the local model. Helps manage processing load."
    )
    show_conversation: bool = Field(
        default=True,
        description="Show full task decomposition, local model execution details, and synthesis steps in the output.",
    )
    timeout_local: int = Field(
        default=30,
        description="Timeout in seconds for each local model call (per chunk, per task).",
    )
    debug_mode: bool = Field(
        default=False, description="Enable additional technical details and verbose logs for debugging."
    )
    max_round_timeout_failure_threshold_percent: int = Field(
        default=50, 
        description="If this percentage of local model calls (chunk executions) in a round time out, a warning is issued. This suggests results for that round might be incomplete."
    )
    max_tokens_claude: int = Field(
        default=2000, description="Maximum tokens for remote model API calls during decomposition and synthesis."
    )
    timeout_claude: int = Field(
        default=60, description="Timeout in seconds for remote model API calls."
    )
    ollama_num_predict: int = Field(
        default=1000, description="Maximum tokens (num_predict) for local Ollama model responses during task execution."
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
        description="Enable JSON structured output for local model responses (requires local model to support JSON mode and the TaskResult schema)."
    )
    structured_output_fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback parsing when structured output fails. If False, parsing errors will be propagated."
    )
    extraction_instructions: str = Field(
        default="", title="Extraction Instructions", description="Specific instructions for the LLM on what to extract or how to process the information for each task."
    )
    expected_format: str = Field(
        default="text", title="Expected Output Format", description="Desired format for the LLM's output for each task (e.g., 'text', 'JSON', 'bullet points'). Note: 'JSON' enables specific structured output fields like 'explanation', 'citation', 'answer'."
    )
    confidence_threshold: float = Field(
        default=0.7, title="Confidence Threshold", description="Minimum confidence level for the LLM's response for each task (0.0-1.0). Primarily a suggestion to the LLM.", ge=0, le=1
    )

    # --- Performance Profile Valve ---
    performance_profile: str = Field(
        default="balanced",
        title="Performance Profile",
        description="Overall performance profile: 'high_quality', 'balanced', 'fastest_results'. Affects base thresholds and max_rounds before other adaptive modifiers.",
        json_schema_extra={"enum": ["high_quality", "balanced", "fastest_results"]}
    )

    # --- Scaling Strategies (v0.3.8) ---
    scaling_strategy: str = Field(
        default="none", 
        description="Scaling strategy: none, repeated_sampling, finer_decomposition, context_chunking"
    )
    repeated_samples: int = Field(
        default=3, 
        description="Number of samples for repeated_sampling strategy"
    )
    fine_decomposition_factor: int = Field(
        default=2, 
        description="Factor for breaking tasks into subtasks"
    )
    chunk_overlap: float = Field(
        default=0.1, 
        description="Overlap ratio for context_chunking (0.0-0.5)",
        ge=0.0,
        le=0.5
    )

    # --- Adaptive Round Control (v0.3.8) ---
    adaptive_rounds: bool = Field(
        default=True, 
        description="Use adaptive round control"
    )
    min_info_gain: float = Field(
        default=0.1, 
        description="Minimum information gain to continue (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    confidence_threshold_adaptive: float = Field(
        default=0.8, 
        description="Confidence threshold to stop early (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    min_rounds: int = Field(
        default=1, 
        description="Minimum rounds before adaptive control",
        ge=1
    )

    # New fields for Iteration 5: Static Early Stopping Rules
    enable_early_stopping: bool = Field(
        default=False,
        title="Enable Early Stopping",
        description="Enable early stopping of rounds based on confidence and query complexity."
    )
    simple_query_confidence_threshold: float = Field(
        default=0.85,
        title="Simple Query Confidence Threshold",
        description="Confidence threshold (0.0-1.0) to stop early for SIMPLE queries.",
        ge=0, le=1
    )
    medium_query_confidence_threshold: float = Field(
        default=0.75,
        title="Medium Query Confidence Threshold",
        description="Confidence threshold (0.0-1.0) to stop early for MEDIUM queries.",
        ge=0, le=1
    )
    min_rounds_before_stopping: int = Field(
        default=1,
        title="Minimum Rounds Before Stopping",
        description="Minimum number of rounds to execute before early stopping can be triggered.",
        ge=1
    )
    # max_rounds (already exists) will be used for COMPLEX queries.

    # New fields for Iteration 5: Convergence Detection Based Early Stopping
    convergence_novelty_threshold: float = Field(
        default=0.10,
        title="Convergence Novelty Threshold",
        description="Minimum percentage of novel findings required per round to consider it non-convergent. E.g., 0.10 means less than 10% new findings might indicate convergence if other criteria met.",
        ge=0, le=1
    )
    convergence_rounds_min_novelty: int = Field(
        default=2,
        title="Convergence Rounds for Minimum Novelty",
        description="Number of consecutive rounds novelty must be below 'convergence_novelty_threshold' to trigger convergence.",
        ge=1
    )
    convergence_sufficiency_threshold: float = Field(
        default=0.7,
        title="Convergence Sufficiency Threshold",
        description="Minimum sufficiency score required for convergence-based early stopping. E.g., 0.7 means 70% sufficiency needed.",
        ge=0, le=1
    )
    # min_rounds_before_convergence_check could be added if distinct from min_rounds_before_stopping
    # For now, ConvergenceDetector uses its own default or relies on min_rounds_before_stopping implicitly
    # if min_rounds_before_convergence_check is not explicitly set in valves.

    # --- v0.3.9 Open WebUI Integration Features ---
    
    # Web Search Integration
    enable_web_search: bool = Field(
        default=False,
        title="Enable Web Search",
        description="Enable web search integration for tasks that require external information."
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
        description="Provide real-time updates during long-running operations."
    )
    
    # Visual Task Decomposition
    show_task_visualization: bool = Field(
        default=True,
        title="Show Task Visualization",
        description="Display task decomposition process using Mermaid diagrams."
    )
    
    # Advanced Citation System
    enable_advanced_citations: bool = Field(
        default=True,
        title="Enable Advanced Citations",
        description="Use Open WebUI's inline citation format for traceable responses."
    )
    citation_max_length: int = Field(
        default=100,
        title="Citation Max Length",
        description="Maximum length for citation text before truncation.",
        ge=50, le=500
    )

    # --- Adaptive Threshold Valves ---
    enable_adaptive_thresholds: bool = Field(
        default=True,
        title="Enable Adaptive Thresholds",
        description="Allow the system to dynamically adjust confidence, sufficiency, and novelty thresholds based on document size, query complexity, and first-round performance."
    )
    doc_size_small_char_limit: int = Field(
        default=5000,
        title="Small Document Character Limit",
        description="Documents with character count below this are considered 'small' for threshold adjustments."
    )
    doc_size_large_char_start: int = Field(
        default=50000,
        title="Large Document Character Start",
        description="Documents with character count above this are considered 'large' for threshold adjustments."
    )
    confidence_modifier_small_doc: float = Field(
        default=0.0,
        title="Confidence Modifier for Small Docs",
        description="Value added to base confidence thresholds if document is small (e.g., -0.05 to be less strict). Applied to general confidence checks if any."
    )
    confidence_modifier_large_doc: float = Field(
        default=0.0,
        title="Confidence Modifier for Large Docs",
        description="Value added to base confidence thresholds if document is large (e.g., +0.05 to be more strict)."
    )
    sufficiency_modifier_simple_query: float = Field(
        default=0.0,
        title="Sufficiency Modifier for Simple Queries",
        description="Value added to base sufficiency thresholds for simple queries (e.g., -0.1 to require less sufficiency)."
    )
    sufficiency_modifier_complex_query: float = Field(
        default=0.0,
        title="Sufficiency Modifier for Complex Queries",
        description="Value added to base sufficiency thresholds for complex queries (e.g., +0.1 to require more)."
    )
    novelty_modifier_simple_query: float = Field(
        default=0.0,
        title="Novelty Threshold Modifier for Simple Queries",
        description="Value added to the base novelty threshold (making it potentially easier to achieve 'low novelty') for simple queries."
    )
    novelty_modifier_complex_query: float = Field(
        default=0.0,
        title="Novelty Threshold Modifier for Complex Queries",
        description="Value added to the base novelty threshold (making it potentially harder to achieve 'low novelty') for complex queries."
    )
    first_round_high_novelty_threshold: float = Field(
        default=0.75,
        title="First Round High Novelty Threshold (%)",
        description="If first round's novel_findings_percentage_this_round is above this (e.g., 0.75 for 75%), it's considered a high novelty first round.",
        ge=0, le=1
    )
    sufficiency_modifier_high_first_round_novelty: float = Field(
        default=-0.05,
        title="Sufficiency Modifier for High First Round Novelty",
        description="Value added to sufficiency thresholds if first round novelty is high (e.g., -0.05 to relax sufficiency requirement)."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields passed to the model
        # an_example = MinionsValves().dict() # For schema generation
