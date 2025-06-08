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
    use_structured_output: bool = Field(
        default=True, 
        description="Enable JSON structured output for local model responses (requires local model support)."
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

    # The following class is part of the Pydantic configuration and is standard.
    # It ensures that extra fields passed to the model are ignored rather than causing an error.
    class Config:
        extra = "ignore"
