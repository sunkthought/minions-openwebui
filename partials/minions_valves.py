from pydantic import BaseModel, Field

class MinionsValves(BaseModel):
    """
    Configuration settings (valves) specifically for the MinionS (multi-task, multi-round) pipe.
    These settings control the behavior of the MinionS protocol, including API keys,
    model selections, timeouts, task decomposition parameters, and operational parameters.
    """
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for Claude."
    )
    remote_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Remote model (e.g., Claude) for task decomposition and synthesis. claude-3-5-haiku-20241022 for cost efficiency, claude-3-5-sonnet-20241022 for quality.",
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
    max_tokens_claude: int = Field( # Renamed from max_tokens_claude for consistency if used generally
        default=2000, description="Maximum tokens for remote model (Claude) API calls during decomposition and synthesis."
    )
    timeout_claude: int = Field( # Renamed from timeout_claude for consistency
        default=60, description="Timeout in seconds for remote model (Claude) API calls."
    )
    ollama_num_predict: int = Field(
        default=1000, description="Maximum tokens (num_predict) for local Ollama model responses during task execution."
    )
    use_structured_output: bool = Field(
        default=False, 
        description="Enable JSON structured output for local model responses (requires local model to support JSON mode and the TaskResult schema)."
    )

    class Config:
        extra = "ignore" # Ignore any extra fields passed to the model
        # an_example = MinionsValves().dict() # For schema generation
