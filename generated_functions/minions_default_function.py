"""
title: MinionS Protocol Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.3.4b
description: MinionS protocol - task decomposition and parallel processing between local and cloud models
required_open_webui_version: 0.5.0
license: MIT License
"""


# Partials File: partials/common_imports.py
import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any, Tuple, Callable, Awaitable
from pydantic import BaseModel, Field
from fastapi import Request # type: ignore

# Partials File: partials/minions_models.py
from typing import Optional, Dict # Add Dict
from pydantic import BaseModel, Field

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


# Partials File: partials/minions_valves.py
from pydantic import BaseModel, Field

class MinionsValves(BaseModel):
    """
    Configuration settings (valves) specifically for the MinionS (multi-task, multi-round) pipe.
    These settings control the behavior of the MinionS protocol, including API keys,
    model selections, timeouts, task decomposition parameters, operational parameters,
    extraction instructions, expected output format, and confidence threshold.
    """
    anthropic_api_key: str = Field(
        default="", description="Anthropic API key for the remote model (e.g., Claude)."
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
    max_tokens_claude: int = Field(
        default=2000, description="Maximum tokens for remote model API calls during decomposition and synthesis."
    )
    timeout_claude: int = Field(
        default=60, description="Timeout in seconds for remote model API calls."
    )
    ollama_num_predict: int = Field(
        default=1000, description="Maximum tokens (num_predict) for local Ollama model responses during task execution."
    )
    use_structured_output: bool = Field(
        default=False, 
        description="Enable JSON structured output for local model responses (requires local model to support JSON mode and the TaskResult schema)."
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


# Partials File: partials/common_api_calls.py
import aiohttp
import json
from typing import Optional
from pydantic import BaseModel

async def call_claude(
    valves: BaseModel,  # Or a more specific type if Valves is shareable
    prompt: str
) -> str:
    """Call Anthropic Claude API"""
    headers = {
        "x-api-key": valves.anthropic_api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": valves.remote_model,
        "max_tokens": valves.max_tokens_claude,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=valves.timeout_claude
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Anthropic API error: {response.status} - {error_text}"
                )
            result = await response.json()
            if result.get("content") and isinstance(result["content"], list) and len(result["content"]) > 0 and result["content"][0].get("text"):
                return result["content"][0]["text"]
            else:
                # Consider logging instead of print for shared code
                if hasattr(valves, 'debug_mode') and valves.debug_mode:
                    print(f"Unexpected Claude API response format: {result}") 
                raise Exception("Unexpected response format from Anthropic API or empty content.")

async def call_ollama(
    valves: BaseModel,  # Or a more specific type
    prompt: str,
    use_json: bool = False,
    schema: Optional[BaseModel] = None
) -> str:
    """Call Ollama API"""
    payload = {
        "model": valves.local_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": valves.ollama_num_predict},
    }

    if use_json and hasattr(valves, 'use_structured_output') and valves.use_structured_output and schema:
        payload["format"] = "json"
        # Pydantic v1 used schema.schema_json(), v2 uses schema_json = model_json_schema(MyModel) then json.dumps(schema_json)
        # Assuming schema object has a .schema_json() method for simplicity here, may need adjustment
        try:
            schema_for_prompt = schema.schema_json() # For Pydantic v1
        except AttributeError: # Basic fallback for Pydantic v2 or other schema objects
             # This part might need refinement based on actual schema object type if not Pydantic v1 BaseModel
            import inspect
            if hasattr(schema, 'model_json_schema'): # Pydantic v2
                 schema_for_prompt = json.dumps(schema.model_json_schema())
            elif inspect.isclass(schema) and issubclass(schema, BaseModel): # Pydantic v1/v2 class
                 schema_for_prompt = json.dumps(schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema.schema())
            else: # Fallback, might not be perfect
                 schema_for_prompt = "{}" 
                 if hasattr(valves, 'debug_mode') and valves.debug_mode:
                      print("Warning: Could not automatically generate schema_json for prompt.")

        schema_prompt_addition = f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_for_prompt}"
        payload["prompt"] = prompt + schema_prompt_addition
    elif "format" in payload:
        del payload["format"]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{valves.ollama_base_url}/api/generate", json=payload, timeout=valves.timeout_local
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Ollama API error: {response.status} - {error_text}"
                )
            result = await response.json()
            if "response" in result:
                return result["response"].strip()
            else:
                if hasattr(valves, 'debug_mode') and valves.debug_mode:
                    print(f"Unexpected Ollama API response format: {result}")
                raise Exception("Unexpected response format from Ollama API or no response field.")


# Partials File: partials/common_context_utils.py
from typing import List, Dict, Any # Added for type hints used in functions

def extract_context_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract context from conversation history"""
    context_parts = []

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Assume messages longer than 200 chars contain context/documents
            if len(content) > 200:
                context_parts.append(content)

    return "\n\n".join(context_parts)

async def extract_file_content(valves, file_info: Dict[str, Any]) -> str:
    """Extract text content from a single file using Open WebUI's file API"""
    try:
        file_id = file_info.get("id")
        file_name = file_info.get("name", "unknown")

        if not file_id:
            return f"[Could not get file ID for {file_name}]"

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[DEBUG] File ID: {file_id}, Name: {file_name}, Info: {str(file_info)}]"

        # If the file info contains content directly, use it
        if "content" in file_info:
            return file_info["content"]
        
        file_type = file_info.get("type", "unknown")
        file_size = file_info.get("size", "unknown")
        
        return f"[File detected: {file_name} (Type: {file_type}, Size: {file_size})\nNote: File content extraction needs to be configured or content is not directly available in provided file_info]"

    except Exception as e:
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[Error extracting file content: {str(e)}]"
        return f"[Error extracting file content]"

async def extract_context_from_files(valves, files: List[Dict[str, Any]]) -> str:
    """Extract text content from uploaded files using Open WebUI's file system"""
    try:
        if not files:
            return ""

        files_content = []

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            files_content.append(f"[DEBUG] Found {len(files)} uploaded files")

        for file_info in files:
            if isinstance(file_info, dict):
                content = await extract_file_content(valves, file_info)
                if content:
                    if content.startswith("[") and content.endswith("]"):
                        if hasattr(valves, 'debug_mode') and valves.debug_mode:
                            files_content.append(content)
                    else:
                        file_name = file_info.get("name", "unknown_file")
                        files_content.append(f"=== FILE: {file_name} ===\n{content}")
                        
        return "\n\n".join(files_content) if files_content else ""

    except Exception as e:
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[File extraction error: {str(e)}]"
        return ""

# Partials File: partials/common_file_processing.py
from typing import List

def create_chunks(context: str, chunk_size: int, max_chunks: int) -> List[str]:
    """Create chunks from context"""
    if not context:
        return []
    actual_chunk_size = max(1, min(chunk_size, len(context)))
    chunks = [
        context[i : i + actual_chunk_size]
        for i in range(0, len(context), actual_chunk_size)
    ]
    return chunks[:max_chunks] if max_chunks > 0 else chunks


# Partials File: partials/query_analyzer.py
import re
from typing import List, Dict, Any, Literal, TypedDict, Optional, Tuple
from enum import Enum

# Attempt to import spaCy, fall back to basic entity extraction if not available
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    # If using older spaCy, sentencizer might need to be added explicitly if not part of default 'en_core_web_sm'
    if NLP and not NLP.has_pipe("sentencizer") and not NLP.has_pipe("parser"): # Parser often includes sentence boundaries
        try:
            NLP.add_pipe("sentencizer")
            print("QueryAnalyzer: Added sentencizer to spaCy model.")
        except Exception as e:
            print(f"QueryAnalyzer: Warning - could not add sentencizer to spaCy model: {e}. Sentence-based pronoun ambiguity might be affected.")

except ImportError:
    NLP = None
    print("Warning: spaCy not found. Falling back to basic entity extraction for QueryAnalyzer.")

class QueryType(Enum):
    QUESTION = "question"
    COMMAND = "command"
    ANALYSIS_REQUEST = "analysis_request"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"

class ScopeIndicator(Enum):
    SPECIFIC = "specific"
    BROAD = "broad"
    COMPREHENSIVE = "comprehensive"
    UNKNOWN = "unknown"

class Entity(TypedDict):
    text: str
    label: str # e.g., PERSON, ORG, DATE, METRIC, CONCEPT
    start_char: int
    end_char: int

class TemporalReference(TypedDict):
    text: str
    type: str # e.g., YEAR, QUARTER, DATE_RANGE
    start_char: int
    end_char: int

class QueryPattern(TypedDict):
    type: str # e.g., MULTI_PART, NESTED_QUERY, IMPLICIT_QUERY, FOLLOW_UP
    text: str # The part of the query that matches the pattern

class AmbiguityDetail(TypedDict):
    type: str # e.g., PRONOUN, TEMPORAL, SCOPE, COMPARATIVE, ENTITY
    text: str # The ambiguous phrase or term
    suggestion: Optional[str] # Optional suggestion for clarification

class QueryMetadata(TypedDict):
    original_query: str
    query_type: QueryType
    entities: List[Entity]
    temporal_refs: List[TemporalReference]
    action_verbs: List[str]
    scope: ScopeIndicator
    ambiguity_markers: List[str] # Specific words/phrases initially identified
    detected_patterns: List[QueryPattern]
    # Iteration 2 fields
    ambiguity_score: float
    decomposability_score: float # Will be refined later or use previous logic
    detailed_ambiguity_report: List[AmbiguityDetail]


class QueryAnalyzer:
    def __init__(self, query: str, debug_mode: bool = False):
        self.original_query = query
        self.query_lower = query.lower()
        self.debug_mode = debug_mode
        self.entities_cache: Optional[List[Entity]] = None # Cache for entities

        if NLP:
            self.doc = NLP(query)
        else:
            self.doc = None

        # Predefined lists (some might be expanded for ambiguity detection)
        self.action_verbs_keywords = {
            "analyze": ["analyze", "analysis", "examine", "study"],
            "compare": ["compare", "contrast", "difference", "vs", "versus"],
            "summarize": ["summarize", "summary", "recap", "overview", "gist"],
            "extract": ["extract", "find", "get", "list", "show", "identify", "retrieve"],
            "find": ["find", "locate", "search for", "where is"],
            "command": ["generate", "create", "write", "build", "develop", "draft"],
            "question": ["what", "who", "when", "where", "why", "how", "is", "are", "do", "does", "can", "could", "would", "should"]
        }
        self.scope_keywords = {
            ScopeIndicator.SPECIFIC: ["specific", "particular", "only", "just"],
            ScopeIndicator.BROAD: ["broad", "general", "overall"],
            ScopeIndicator.COMPREHENSIVE: ["all", "every", "comprehensive", "full", "complete", "entire"]
        }
        self.ambiguity_marker_keywords = [ # General markers, specific checks will be in ambiguity detectors
            "it", "they", "them", "this", "that", "those", "these",
            "recent", "recently", "current", "previous", "next", "last", # Can be temporal ambiguity
            "some", "any", "few", "several", "many", "much",
            "better", "worse", "more", "less", "larger", "smaller", "higher", "lower" # Can be comparative ambiguity
        ]
        self.pronouns_list = ["it", "they", "them", "this", "that", "those", "these", "he", "she", "him", "her", "itself", "themselves"]
        self.vague_temporal_terms = ["recently", "last time", "current", "soon", "later", "previous", "next time", "earlier"]
        self.vague_scope_terms = ["performance", "results", "data", "issues", "status", "updates", "information", "details", "summary", "overview"]
        self.comparative_terms = ["better", "worse", "more", "less", "larger", "smaller", "higher", "lower", "increased", "decreased", "improved", "reduced"]

        self.multi_part_conjunctions = [" and ", " or ", " but also ", " as well as "]
        self.implicit_query_starters = ["revenue figures", "sales data", "performance metrics"]

    def _log_debug(self, message: str):
        if self.debug_mode:
            print(f"QueryAnalyzer DEBUG: {message}")

    def extract_query_type(self) -> QueryType:
        # (Code from previous version, ensure it's the corrected one)
        self._log_debug(f"Extracting query type for: '{self.original_query}'")
        if any(verb in self.query_lower for verb in self.action_verbs_keywords["compare"]):
            self._log_debug("Query type: COMPARISON (matched 'compare' keywords)")
            return QueryType.COMPARISON

        is_analysis = any(verb in self.query_lower for verb in self.action_verbs_keywords["analyze"])
        is_summarize = any(verb in self.query_lower for verb in self.action_verbs_keywords["summarize"])
        is_extract = any(verb in self.query_lower for verb in self.action_verbs_keywords["extract"])

        if is_analysis or is_summarize or is_extract:
            if is_analysis:
                self._log_debug("Query type: ANALYSIS_REQUEST (matched 'analyze' keywords)")
                return QueryType.ANALYSIS_REQUEST

            is_question_format = any(q_word in self.query_lower for q_word in self.action_verbs_keywords["question"])

            if is_question_format and (is_summarize or is_extract):
                try:
                    first_action_verb_indices = []
                    if is_summarize:
                        for v in self.action_verbs_keywords["summarize"]:
                            if v in self.query_lower: first_action_verb_indices.append(self.query_lower.find(v))
                    if is_extract:
                        for v in self.action_verbs_keywords["extract"]:
                            if v in self.query_lower: first_action_verb_indices.append(self.query_lower.find(v))

                    first_question_word_indices = []
                    for qv in self.action_verbs_keywords["question"]:
                        if qv in self.query_lower: first_question_word_indices.append(self.query_lower.find(qv))

                    if first_question_word_indices and first_action_verb_indices:
                        if min(first_question_word_indices) < min(first_action_verb_indices):
                            self._log_debug("Query type: QUESTION (matched summarize/extract keywords but also question format)")
                            return QueryType.QUESTION
                except (ValueError, IndexError):
                    pass

            self._log_debug("Query type: COMMAND (matched summarize/extract keywords, not clearly a question)")
            return QueryType.COMMAND

        if any(self.query_lower.startswith(verb) for verb in self.action_verbs_keywords["command"]):
            if not any(q_word == self.query_lower.split()[0] for q_word in self.action_verbs_keywords["question"] if self.query_lower.split()):
                 self._log_debug("Query type: COMMAND (starts with command verb)")
                 return QueryType.COMMAND

        if any(self.query_lower.startswith(q_word) for q_word in self.action_verbs_keywords["question"]):
            self._log_debug("Query type: QUESTION (starts with question word)")
            return QueryType.QUESTION
        if "?" in self.original_query:
            self._log_debug("Query type: QUESTION (contains '?')")
            return QueryType.QUESTION

        self._log_debug("No specific query type matched, defaulting to UNKNOWN.")
        return QueryType.UNKNOWN

    def extract_entities(self) -> List[Entity]:
        if self.entities_cache is not None:
            return self.entities_cache

        self._log_debug("Extracting entities...")
        entities: List[Entity] = []
        if self.doc:
            for ent in self.doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                })
            self._log_debug(f"Found {len(entities)} entities using spaCy: {entities}")
        else:
            for match in re.finditer(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", self.original_query):
                 entities.append({
                    "text": match.group(0),
                    "label": "UNKNOWN_CAPITALIZED",
                    "start_char": match.start(),
                    "end_char": match.end()
                })
            self._log_debug(f"Found {len(entities)} entities using basic regex: {entities}")
        self.entities_cache = entities
        return entities

    def extract_temporal_references(self) -> List[TemporalReference]:
        # (Code from previous version)
        self._log_debug("Extracting temporal references...")
        temporal_refs: List[TemporalReference] = []
        year_patterns = [
            (r"(19|20)\d{2}", "YEAR"),
            (r"FY\s*(19|20)\d{2}", "FISCAL_YEAR"),
            (r"Q[1-4]\s*(?:(?:19|20)\d{2})?", "QUARTER_YEAR"),
            (r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*(?:19|20)\d{2})?", "MONTH_DAY_YEAR"),
            (r"(?:this|last|next)\s+(?:year|quarter|month|week)", "RELATIVE_TIME")
        ]
        for pattern, ref_type in year_patterns:
            for match in re.finditer(pattern, self.original_query, re.IGNORECASE):
                temporal_refs.append({
                    "text": match.group(0),
                    "type": ref_type,
                    "start_char": match.start(),
                    "end_char": match.end()
                })
        self._log_debug(f"Found {len(temporal_refs)} temporal references: {temporal_refs}")
        return temporal_refs

    def extract_action_verbs(self) -> List[str]:
        # (Code from previous version)
        self._log_debug("Extracting action verbs...")
        verbs = set()
        if self.doc:
            for token in self.doc:
                if token.pos_ == "VERB":
                    lemma = token.lemma_.lower()
                    for _, keywords in self.action_verbs_keywords.items(): # Check all categories
                        if lemma in keywords or token.lower_ in keywords:
                            verbs.add(lemma)
                            break
        else:
            for _, keywords in self.action_verbs_keywords.items():
                for keyword in keywords:
                    # More robust check for keyword presence
                    if f" {keyword} " in f" {self.query_lower} " or \
                       self.query_lower.startswith(f"{keyword} ") or \
                       self.query_lower.endswith(f" {keyword}"):
                        verbs.add(keyword.split()[0])

        self._log_debug(f"Found action verbs: {list(verbs)}")
        return list(verbs)

    def extract_scope_indicator(self) -> ScopeIndicator:
        # (Code from previous version)
        self._log_debug("Extracting scope indicator...")
        for scope_enum, keywords in self.scope_keywords.items():
            if any(keyword in self.query_lower for keyword in keywords):
                self._log_debug(f"Matched scope: {scope_enum}")
                return scope_enum
        self._log_debug("No specific scope matched, defaulting to UNKNOWN.")
        return ScopeIndicator.UNKNOWN

    def extract_ambiguity_markers(self) -> List[str]:
        # (Code from previous version - this becomes less important as specific detectors are added)
        self._log_debug("Extracting general ambiguity markers (words)...")
        markers = []
        # This method will become less critical as specific detectors are more robust.
        # For now, it can still provide a quick list of potentially ambiguous words.
        for marker_token in self.query_lower.split():
            if marker_token in self.ambiguity_marker_keywords:
                 markers.append(marker_token)
        self._log_debug(f"Found general ambiguity marker words: {markers}")
        return markers

    def detect_patterns(self) -> List[QueryPattern]:
        # (Code from previous version)
        self._log_debug("Detecting query patterns...")
        patterns: List[QueryPattern] = []
        for conj in self.multi_part_conjunctions:
            if conj in self.query_lower:
                parts = self.original_query.lower().split(conj)
                if len(parts) > 1 and "?" in self.original_query :
                     patterns.append({"type": "MULTI_PART", "text": self.original_query})
                     self._log_debug(f"Detected MULTI_PART pattern (general) using '{conj}'")
                     break
        if "where..." in self.query_lower or "that have" in self.query_lower or "which are" in self.query_lower:
            if "find all companies that have revenue > $1M in sectors where" in self.query_lower:
                 patterns.append({"type": "NESTED_QUERY", "text": self.original_query})
                 self._log_debug("Detected NESTED_QUERY pattern (specific example)")
            elif "where" in self.query_lower.split() and self.query_lower.index("where") > 0 :
                 patterns.append({"type": "NESTED_QUERY", "text": self.original_query})
                 self._log_debug("Detected NESTED_QUERY pattern (general 'where')")
        for starter in self.implicit_query_starters:
            if self.query_lower.startswith(starter) and "?" not in self.original_query:
                patterns.append({"type": "IMPLICIT_QUERY", "text": self.original_query})
                self._log_debug(f"Detected IMPLICIT_QUERY pattern with starter '{starter}'")
                break
        follow_up_phrases = ["how about", "what about", "and for", "and then", "tell me more about that"]
        if any(self.query_lower.startswith(phrase) for phrase in follow_up_phrases):
            patterns.append({"type": "FOLLOW_UP", "text": self.original_query})
            self._log_debug("Detected FOLLOW_UP pattern")
        self._log_debug(f"Detected patterns: {patterns}")
        return patterns

    # --- Iteration 2: Ambiguity Detection Methods ---
    def _detect_pronoun_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        if not self.doc: # spaCy not available
            for pronoun in self.pronouns_list:
                if pronoun in self.query_lower.split():
                    report.append({
                        "type": "PRONOUN", "text": pronoun,
                        "suggestion": "Clarify what this pronoun refers to (spaCy not available for context)."
                    })
            return report

        # spaCy is available
        # A simple approach: check if pronoun is too far from a preceding noun/proper noun
        # More advanced: use coreference resolution if available and performant enough
        for sent in self.doc.sents:
            for token in sent:
                if token.lower_ in self.pronouns_list:
                    # Basic check: is there a recent noun phrase (potential antecedent)?
                    # This is a heuristic. Real coreference is complex.
                    has_recent_antecedent = False
                    # Check current sentence before the pronoun
                    for i in range(token.i - 1, sent.start, -1): # Iterate backwards from pronoun in the same sentence
                        prev_token = self.doc[i]
                        if prev_token.pos_ in ["NOUN", "PROPN"]:
                            has_recent_antecedent = True
                            break
                        if i < token.i - 5 : # Stop looking back too far within the sentence
                            break

                    if not has_recent_antecedent:
                         # Could also check previous sentence if this is not the first sentence
                        if sent.start > 0: # Not the first sentence in the query
                            prev_sent = self.doc[sent.start-1].sent # Get previous sentence
                            for prev_token in reversed(list(prev_sent)): # Iterate backwards through previous sentence
                                if prev_token.pos_ in ["NOUN", "PROPN"]:
                                    has_recent_antecedent = True
                                    break
                                if prev_token.i < prev_sent.end - 5: # stop looking too far
                                    break


                        if not has_recent_antecedent:
                            report.append({
                                "type": "PRONOUN", "text": token.text,
                                "suggestion": f"Clarify what '{token.text}' refers to. Consider rephrasing or providing more context."
                            })
        self._log_debug(f"Pronoun ambiguity report: {report}")
        return report

    def _detect_temporal_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        # Get already identified specific temporal refs to avoid double flagging
        specific_temporal_texts = [ref['text'].lower() for ref in self.extract_temporal_references()]

        for term in self.vague_temporal_terms:
            if term in self.query_lower:
                # Check if this vague term is part of an already identified specific reference
                is_part_of_specific = False
                for specific_ref_text in specific_temporal_texts:
                    if term in specific_ref_text: # e.g. "last year" is specific, "last" alone might be vague
                        is_part_of_specific = True
                        break
                if not is_part_of_specific:
                    report.append({
                        "type": "TEMPORAL", "text": term,
                        "suggestion": f"'{term}' is vague. Specify a date, range, or event (e.g., 'current quarter', 'last month')."
                    })
        self._log_debug(f"Temporal ambiguity report: {report}")
        return report

    def _detect_scope_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        # Check if vague scope terms are used without specific entities nearby
        # This is a heuristic. Context is key.
        entities = self.extract_entities()
        entity_texts_lower = [e['text'].lower() for e in entities]

        for term in self.vague_scope_terms:
            if term in self.query_lower:
                # Simple check: is the vague term accompanied by a specific entity?
                # e.g., "sales data" vs "data for product X"
                is_clarified_by_entity = False
                term_index = self.query_lower.find(term)
                if term_index != -1:
                    # Look for entities within a window around the term
                    window_start = max(0, term_index - 20)
                    window_end = min(len(self.query_lower), term_index + len(term) + 20)
                    query_window = self.query_lower[window_start:window_end]
                    if any(e_text in query_window for e_text in entity_texts_lower):
                        is_clarified_by_entity = True

                if not is_clarified_by_entity:
                    report.append({
                        "type": "SCOPE", "text": term,
                        "suggestion": f"'{term}' is vague. Specify for what entity, product, or area (e.g., '{term} for sales team', '{term} of Q3')."
                    })
        self._log_debug(f"Scope ambiguity report: {report}")
        return report

    def _detect_comparative_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        for term in self.comparative_terms:
            if term in self.query_lower.split(): # Check for whole word
                # Simple check: is there a "than" or "to" nearby, or multiple entities being compared?
                # This is a basic heuristic.
                has_baseline_indicator = " than " in self.query_lower or " to " in self.query_lower or " compared to " in self.query_lower
                entities = self.extract_entities() # Ensure entities are extracted

                # Count entities that are not dates or quantities, as those usually aren't the items being compared directly
                relevant_entities_count = len([e for e in entities if e['label'] not in ['DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY', 'CARDINAL', 'ORDINAL']])

                if not has_baseline_indicator and relevant_entities_count < 2:
                    report.append({
                        "type": "COMPARATIVE", "text": term,
                        "suggestion": f"'{term}' implies a comparison. Specify what it's being compared to or provide a baseline."
                    })
        self._log_debug(f"Comparative ambiguity report: {report}")
        return report

    def _detect_entity_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        entities = self.extract_entities()

        # Count occurrences of ORG, PRODUCT, etc.
        org_entities = [e['text'] for e in entities if e['label'] == 'ORG']
        product_entities = [e['text'] for e in entities if e['label'] == 'PRODUCT'] # Assuming PRODUCT label if spaCy provides it

        generic_references = {
            "the company": (org_entities, "ORG"),
            "the project": (product_entities, "PRODUCT"), # Example, adjust label as needed
            "the product": (product_entities, "PRODUCT")
        }

        for generic_term, (entity_list, entity_type_label) in generic_references.items():
            if generic_term in self.query_lower:
                if len(entity_list) > 1:
                    report.append({
                        "type": "ENTITY", "text": generic_term,
                        "suggestion": f"'{generic_term}' is ambiguous. Multiple {entity_type_label}s detected: {', '.join(entity_list)}. Specify which one."
                    })
                elif not entity_list and not any(e['label'] == entity_type_label for e in entities): # No specific entity of this type mentioned
                     report.append({
                        "type": "ENTITY", "text": generic_term,
                        "suggestion": f"'{generic_term}' is used, but no specific {entity_type_label} was identified in the query. Please specify."
                    })

        self._log_debug(f"Entity ambiguity report: {report}")
        return report

    def _calculate_ambiguity_score(self, detailed_report: List[AmbiguityDetail]) -> float:
        if not detailed_report:
            return 0.0

        # Simple scoring: each ambiguity type adds a fixed amount, with a cap.
        # Could be weighted by type later.
        # Max score 1.0
        # Each detected ambiguity adds 0.25 to the score.
        score = len(detailed_report) * 0.25

        self._log_debug(f"Calculated ambiguity score: {min(score, 1.0)} from {len(detailed_report)} ambiguities.")
        return min(score, 1.0) # Cap at 1.0

    def analyze(self) -> QueryMetadata:
        self._log_debug(f"Starting analysis for query: '{self.original_query}' (Iteration 2)")

        # Clear entity cache for fresh analysis if called multiple times (though typically once per instance)
        self.entities_cache = None

        query_type = self.extract_query_type()
        entities = self.extract_entities() # Call it once and cache for other ambiguity detectors
        temporal_refs = self.extract_temporal_references()
        action_verbs = self.extract_action_verbs()
        scope = self.extract_scope_indicator()
        general_ambiguity_markers = self.extract_ambiguity_markers() # Still useful for a quick glance
        detected_patterns = self.detect_patterns()

        # Iteration 2: Detailed Ambiguity Detection
        detailed_ambiguity_report: List[AmbiguityDetail] = []
        detailed_ambiguity_report.extend(self._detect_pronoun_ambiguity())
        detailed_ambiguity_report.extend(self._detect_temporal_ambiguity())
        detailed_ambiguity_report.extend(self._detect_scope_ambiguity())
        detailed_ambiguity_report.extend(self._detect_comparative_ambiguity())
        detailed_ambiguity_report.extend(self._detect_entity_ambiguity())

        ambiguity_score = self._calculate_ambiguity_score(detailed_ambiguity_report)

        # Decomposability score (can be refined, using Iteration 1 logic for now)
        decomposability_score = 0.0
        if query_type == QueryType.COMPARISON or \
           any(p['type'] == 'MULTI_PART' for p in detected_patterns) or \
           any(p['type'] == 'NESTED_QUERY' for p in detected_patterns) or \
           len(action_verbs) > 1 :
            decomposability_score = 0.6
        if ambiguity_score > 0.5: # High ambiguity might make it harder to decompose reliably
            decomposability_score = max(0, decomposability_score - 0.2)


        metadata: QueryMetadata = {
            "original_query": self.original_query,
            "query_type": query_type,
            "entities": entities,
            "temporal_refs": temporal_refs,
            "action_verbs": action_verbs,
            "scope": scope,
            "ambiguity_markers": general_ambiguity_markers, # General list
            "detected_patterns": detected_patterns,
            "ambiguity_score": ambiguity_score,
            "decomposability_score": decomposability_score,
            "detailed_ambiguity_report": detailed_ambiguity_report # New field
        }
        self._log_debug(f"Analysis complete (Iteration 2). Metadata: {metadata}")
        return metadata

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    queries_to_test_iter2 = [
        "What is its status and how does it compare to the previous one?", # Pronoun, Comparative, Temporal
        "Analyze performance data.", # Scope, Temporal (current implied)
        "Is the new system better?", # Comparative
        "Tell me about the company's recent results.", # Entity, Temporal, Scope
        "List all projects for the client.", # Entity (if multiple clients or no client identified)
        "Generate a summary of their findings.", # Pronoun
        "Compare sales in 2023 to 2022 for product X and product Y.", # Low ambiguity
        "What are the current risks?", # Temporal, Scope
        "Find issues related to the new deployment.", # Scope
        "How much did revenue increase for the division?" # Comparative, Entity (if multiple divisions)
    ]

    for q in queries_to_test_iter2:
        print(f"--- Analyzing Query (Iter 2): \"{q}\" ---")
        analyzer = QueryAnalyzer(q, debug_mode=True)
        meta = analyzer.analyze()
        print(f"Query Type: {meta['query_type'].value}")
        # print(f"Entities: {meta['entities']}")
        # print(f"Temporal Refs: {meta['temporal_refs']}")
        # print(f"Action Verbs: {meta['action_verbs']}")
        # print(f"Scope: {meta['scope'].value}")
        # print(f"Ambiguity Markers (general): {meta['ambiguity_markers']}")
        # print(f"Detected Patterns: {meta['detected_patterns']}")
        print(f"Ambiguity Score: {meta['ambiguity_score']:.2f}")
        print(f"Detailed Ambiguity Report: {meta['detailed_ambiguity_report']}")
        print(f"Decomposability Score: {meta['decomposability_score']:.2f}")
        print("---------------------------------------\n")

    # Test with no spaCy again if needed
    # _temp_nlp = NLP; NLP = None
    # analyzer_no_spacy = QueryAnalyzer("What is its profit?", debug_mode=True)
    # meta_no_spacy = analyzer_no_spacy.analyze()
    # print(f"Ambiguity Report (no spaCy): {meta_no_spacy['detailed_ambiguity_report']}")
    # NLP = _temp_nlp


# Partials File: partials/query_expander.py
import re
from typing import List, Dict, Any, Optional

# Attempt to import spaCy for lemmatization, fall back if not available
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except ImportError:
    NLP = None
    print("Warning: spaCy not found. QueryExpander's synonym matching might be less effective.")

class QueryExpander:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

        # Domain-specific contextual phrases/keywords
        # These are just examples; more sophisticated domain detection might be needed.
        self.domain_keywords = {
            "finance": ["revenue", "profit", "loss", "stock", "market", "investment", "ebitda", "financial"],
            "research": ["study", "research", "paper", "findings", "experiment", "analyze", "methodology"],
            "coding": ["code", "function", "class", "module", "debug", "algorithm", "error", "python", "java"],
            "writing": ["draft", "edit", "chapter", "section", "narrative", "style", "tone", "grammar"]
        }

        # Default examples for context injection (can be expanded)
        self.default_doc_type_context_examples = {
            "default": { # General context, less specific
                "prepend": [],
                "append": []
            },
            "research": {
                "prepend": ["According to the research paper, ", "Based on the study, "],
                "append": [" considering the methodologies described.", " in the context of this research."]
            },
            "finance": {
                "prepend": ["Regarding the financial data, ", "From a financial perspective, "],
                "append": [" as per the financial statements.", " considering market conditions."]
            },
            "writing": {
                "prepend": ["In terms of the draft, ", "Considering the writing style, "],
                "append": [" to improve clarity and flow.", " keeping the target audience in mind."]
            },
            "coding": {
                "prepend": ["For the given code, ", "Regarding the algorithm, "],
                "append": [" ensure efficiency and proper error handling.", " following best coding practices."]
            }
        }

        # Templates for query completion
        self.default_completion_templates = {
            # Default domain
            r"^(?:what about|how about|and)\s+(.+)$": r"What is the status of ?", # "what about sales" -> "What is the status of sales?"
            r"^([A-Za-z\s]+)\s+(figures|data|details|information|summary|overview)$": r"What are the  ?", # "sales figures" -> "What are the sales figures?"
            r"^([A-Za-z\s]+)\s+(report|analysis)$": r"Provide a  .", # "market report" -> "Provide a market report."
            r"^(compare|comparison)\s+([\w\s]+)\s+(?:and|vs|to)\s+([\w\s]+)$": r"Compare  with .", # "compare A to B"
            r"^(compare|comparison)\s+([\d]{4})\s+([\d]{4})$": r"Compare the results between  and .", # "Comparison 2023 2024"
            r"^(risk factors|key risks)$": r"What are the main risk factors mentioned?",
            # Finance specific (examples)
            r"^(Q[1-4])\s+(revenue|sales|profit)$": r"What was the  for ?", # "Q3 revenue"
            r"^(((?:[A-Z]{2,5}))\s+stock price)$": r"What is the current stock price for ?", # "MSFT stock price"
            # Research specific (examples)
            r"^(findings for|results of)\s+(.+)$": r"What are the findings for ?",
            # Coding specific (examples)
            r"^(fix|debug)\s+(.+)$": r"How can I fix the error in ?",
             r"^(generate code for|write a function to)\s+(.+)$": r"Generate Python code to ." # Default to Python or make configurable
        }

        # Synonym sets (can be greatly expanded)
        self.default_synonym_sets = {
            "default": {
                "show": ["display", "list", "present"],
                "important": ["critical", "key", "significant", "main"],
                "issue": ["problem", "challenge", "difficulty", "concern"]
            },
            "finance": {
                "profit": ["earnings", "income", "net income", "bottom line"],
                "revenue": ["sales", "turnover", "top line"],
                "company": ["organization", "firm", "business", "corporation"],
                "ebitda": ["earnings before interest, taxes, depreciation and amortization"]
            },
            "research": {
                "study": ["paper", "investigation", "analysis", "examination"],
                "findings": ["results", "conclusions", "discoveries"],
                "correlation": ["relationship", "association", "connection"]
            },
            "coding": {
                "error": ["bug", "defect", "issue", "glitch"],
                "function": ["method", "subroutine", "procedure"],
                "fix": ["debug", "resolve", "patch", "correct"]
            },
            "writing": {
                "summary": ["abstract", "synopsis", "overview", "recapitulation"],
                "improve": ["enhance", "refine", "polish", "strengthen"],
                "clarity": ["lucidity", "clearness", "precision"]
            }
        }

    def _log_debug(self, message: str):
        if self.debug_mode:
            print(f"QueryExpander DEBUG: {message}")

    def _get_lemma(self, word: str) -> str:
        if NLP:
            doc = NLP(word)
            return doc[0].lemma_.lower()
        return word.lower() # Fallback if no spaCy

    def _infer_domain(self, query: str, provided_domain: Optional[str] = None) -> str:
        if provided_domain and provided_domain in self.domain_keywords:
            return provided_domain

        query_lower = query.lower()
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                self._log_debug(f"Inferred domain: {domain}")
                return domain
        self._log_debug("Domain not inferred, using 'default'.")
        return "default"

    def add_document_type_context(self, query: str,
                                  doc_type_examples: Optional[Dict[str, Any]] = None,
                                  domain_hint: Optional[str] = None) -> str:
        self._log_debug(f"Original query for context injection: '{query}'")
        if doc_type_examples is None:
            doc_type_examples = self.default_doc_type_context_examples

        domain = self._infer_domain(query, domain_hint)
        domain_contexts = doc_type_examples.get(domain, doc_type_examples.get("default", {}))

        expanded_query = query
        if domain_contexts:
            if domain_contexts.get("prepend"):
                # Simple: pick the first prepend phrase if multiple exist
                phrase_to_prepend = domain_contexts["prepend"][0]
                if not query.lower().startswith(phrase_to_prepend.lower()): # Avoid double prepending
                    expanded_query = phrase_to_prepend + expanded_query
                    self._log_debug(f"Prepended context for domain '{domain}': '{phrase_to_prepend}'")

            if domain_contexts.get("append"):
                # Simple: pick the first append phrase
                phrase_to_append = domain_contexts["append"][0]
                if not query.lower().endswith(phrase_to_append.lower()): # Avoid double appending
                    expanded_query += phrase_to_append
                    self._log_debug(f"Appended context for domain '{domain}': '{phrase_to_append}'")

        if expanded_query != query:
            self._log_debug(f"Query after context injection: '{expanded_query}'")
        else:
            self._log_debug("No context injected.")
        return expanded_query

    def complete_query(self, query: str,
                       completion_templates: Optional[Dict[str, str]] = None,
                       domain_hint: Optional[str] = None) -> str:
        self._log_debug(f"Original query for completion: '{query}'")
        if completion_templates is None:
            completion_templates = self.default_completion_templates

        # Domain could be used to select a subset of templates in the future
        # current_domain = self._infer_domain(query, domain_hint)

        for pattern, template in completion_templates.items():
            try:
                if re.match(pattern, query, re.IGNORECASE):
                    # Check if the query ALREADY looks like the completed version to avoid re-completing
                    # This is a simple check; might need refinement.
                    # Example: if template is "What is ?" and query is "What is sales?", don't change.
                    # A rough check: if the template's structure (non-group parts) is already in query.
                    temp_template_test = template.replace(r"", ".*").replace(r"", ".*").replace(r"", ".*")
                    if re.match(f"^{temp_template_test}$", query, re.IGNORECASE):
                        self._log_debug(f"Query '{query}' already seems complete for pattern '{pattern}'. Skipping.")
                        continue

                    expanded_query = re.sub(pattern, template, query, flags=re.IGNORECASE)
                    if expanded_query != query:
                        self._log_debug(f"Query completed: '{pattern}' -> '{expanded_query}'")
                        return expanded_query # Return on first match for simplicity
            except re.error as e:
                self._log_debug(f"Regex error for pattern '{pattern}': {e}")
                continue

        self._log_debug("No query completion applied.")
        return query

    def expand_synonyms(self, query: str,
                        synonym_sets: Optional[Dict[str, Any]] = None,
                        domain_hint: Optional[str] = None,
                        max_expansions_per_term: int = 1) -> str:
        self._log_debug(f"Original query for synonym expansion: '{query}'")
        if synonym_sets is None:
            synonym_sets = self.default_synonym_sets

        domain = self._infer_domain(query, domain_hint)

        # Combine default synonyms with domain-specific ones, domain takes precedence
        active_synonyms: Dict[str, List[str]] = {}
        active_synonyms.update(synonym_sets.get("default", {}))
        active_synonyms.update(synonym_sets.get(domain, {})) # Domain specific overrides/adds

        if not active_synonyms:
            self._log_debug("No active synonym sets for expansion.")
            return query

        # Use spaCy for tokenization and lemmatization if available
        # Otherwise, simple word splitting and lowercasing
        words_to_process = []
        if NLP and hasattr(self, 'doc') and self.doc: # Assuming self.doc is populated if NLP is available
            # doc = NLP(query) # Re-process to ensure fresh tokens if query was modified
            words_to_process = [(token.text, token.lemma_.lower()) for token in self.doc]
        else:
            words_to_process = [(word, word.lower()) for word in query.split()]

        expanded_query_parts = []
        original_query_parts = query.split() # For reconstruction if no spaCy
        word_idx_no_spacy = 0

        for original_term, term_lemma in words_to_process:
            found_synonym = False
            if term_lemma in active_synonyms:
                syns = active_synonyms[term_lemma]
                if syns:
                    # Build "term OR synonym1 OR synonym2" string
                    # Limit number of synonyms to avoid overly long queries
                    syns_to_add = syns[:max_expansions_per_term]
                    expansion_str = f"{original_term} OR {' OR '.join(syns_to_add)}"
                    # Basic check to avoid redundant expansion like "profit OR profit"
                    if original_term.lower() in [s.lower() for s in syns_to_add]:
                        expansion_str = original_term # Just use original term

                    expanded_query_parts.append(f"({expansion_str})")
                    self._log_debug(f"Expanded '{original_term}' (lemma: {term_lemma}) to '({expansion_str})' using domain '{domain}'")
                    found_synonym = True

            if not found_synonym:
                expanded_query_parts.append(original_term)

            word_idx_no_spacy +=1

        if not expanded_query_parts: # Should not happen if query has content
            return query

        expanded_query = " ".join(expanded_query_parts)
        if expanded_query != query:
            self._log_debug(f"Query after synonym expansion: '{expanded_query}'")
        else:
            self._log_debug("No synonyms applied.")
        return expanded_query

    def expand(self, query: str,
               apply_context_injection: bool = True,
               apply_completion: bool = True,
               apply_synonyms: bool = True,
               domain_hint: Optional[str] = None) -> str:
        self._log_debug(f"Starting expansion for query: '{query}'")
        current_query = query

        if apply_context_injection:
            current_query = self.add_document_type_context(current_query, domain_hint=domain_hint)

        if apply_completion:
            current_query = self.complete_query(current_query, domain_hint=domain_hint)

        if apply_synonyms:
            # Re-initialize self.doc for synonym expansion if spaCy is used,
            # as query might have changed after completion/context steps.
            if NLP:
                self.doc = NLP(current_query)
            current_query = self.expand_synonyms(current_query, domain_hint=domain_hint)

        self._log_debug(f"Final expanded query: '{current_query}'")
        return current_query

if __name__ == '__main__':
    expander = QueryExpander(debug_mode=True)

    queries_to_test_iter3 = [
        ("Q3 revenue", "finance"),
        ("Comparison 2023 2024", "default"),
        ("risk factors", "finance"),
        ("fix the login function", "coding"),
        ("summarize chapter 3", "writing"),
        ("market analysis", "finance"), # Test context injection + completion
        ("company profit", "finance"), # Test synonym expansion
        ("important research issue", "research") # Test synonym + context
    ]

    for q, domain in queries_to_test_iter3:
        print(f"--- Expanding Query (Iter 3): \"{q}\" (Domain Hint: {domain}) ---")

        # Test individual methods
        print("  Testing Completion:")
        completed = expander.complete_query(q, domain_hint=domain)
        print(f"  Completed: '{completed}'")

        print("  Testing Context Injection:")
        contexted = expander.add_document_type_context(completed, domain_hint=domain)
        print(f"  Context Injected: '{contexted}'")

        print("  Testing Synonym Expansion:")
        # Re-init doc for synonym expansion if spaCy is used
        if NLP: expander.doc = NLP(contexted)
        synonymized = expander.expand_synonyms(contexted, domain_hint=domain)
        print(f"  Synonymized: '{synonymized}'")

        print("  Testing Full Expansion Pipeline:")
        expanded_q = expander.expand(q, domain_hint=domain)
        print(f"  Fully Expanded: '{expanded_q}'")
        print("---------------------------------------\n")

    print("--- Testing query not needing much expansion ---")
    q_no_expand = "What is the detailed revenue breakdown for Q1 2024 for all US regions?"
    expanded_q_no_expand = expander.expand(q_no_expand, domain_hint="finance")
    print(f"  Fully Expanded: '{expanded_q_no_expand}'")
    print("---------------------------------------\n")


# Partials File: partials/minions_prompts.py
from typing import List, Any, Optional

# This file will store prompt generation functions for the MinionS (multi-turn, multi-task) protocol.

def get_minions_synthesis_claude_prompt(query: str, synthesis_input_summary: str, valves: Any) -> str:
    """
    Returns the synthesis prompt for Claude in the MinionS protocol.
    Logic moved from _execute_minions_protocol in minions_pipe_method.py.
    'synthesis_input_summary' is the aggregation of successful task results.
    """
    prompt_lines = [
        f'''Based on all the information gathered across multiple rounds, provide a comprehensive answer to the original query: "{query}"

GATHERED INFORMATION:
{synthesis_input_summary if synthesis_input_summary else "No specific information was extracted by local models."}
'''
    ]

    # Add instructions from valves for synthesis guidance
    synthesis_instructions = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        synthesis_instructions.append(f"When synthesizing the final answer, consider these overall instructions: {valves.extraction_instructions}")
    if hasattr(valves, 'expected_format') and valves.expected_format:
        synthesis_instructions.append(f"Format the final synthesized answer as {valves.expected_format}.")
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0: # Assuming 0 is default/off
        synthesis_instructions.append(f"Aim for an overall confidence level of at least {valves.confidence_threshold} in your synthesized answer.")

    if synthesis_instructions:
        prompt_lines.append("\nSYNTHESIS GUIDELINES:")
        prompt_lines.extend(synthesis_instructions)
        prompt_lines.append("") # Add a newline for separation

    prompt_lines.append("If the gathered information is insufficient, explain what's missing or state that the answer cannot be provided.")
    prompt_lines.append("Final Answer:")
    return "\n".join(prompt_lines)

def get_minions_local_task_prompt(
    chunk: str, 
    task: str, 
    chunk_idx: int, 
    total_chunks: int, 
    valves: Any, 
) -> str:
    """
    Returns the prompt for the local Ollama model for a specific task on a chunk 
    in the MinionS protocol.
    Logic moved from execute_tasks_on_chunks in minions_protocol_logic.py.
    """
    prompt_lines = [
        f'''Text to analyze (Chunk {chunk_idx + 1}/{total_chunks} of document):
---BEGIN TEXT---
{chunk}
---END TEXT---

Task: {task}'''
    ]

    task_specific_instructions = []
    if hasattr(valves, 'extraction_instructions') and valves.extraction_instructions:
        task_specific_instructions.append(f"- Follow these specific extraction instructions: {valves.extraction_instructions}")

    # Confidence threshold is a general guideline for the task.
    if hasattr(valves, 'confidence_threshold') and valves.confidence_threshold > 0:
        task_specific_instructions.append(f"- Aim for a confidence level of at least {valves.confidence_threshold} in your findings for this task.")

    if task_specific_instructions:
        prompt_lines.append("\nTASK-SPECIFIC INSTRUCTIONS:")
        prompt_lines.extend(task_specific_instructions)

    if valves.use_structured_output:
        prompt_lines.append(f'''\n
IMPORTANT: Provide your answer as a valid JSON object with the following structure:
{{
    "explanation": "Brief explanation of your findings for this task",
    "citation": "Direct quote from the text if applicable to this task, or null",
    "answer": "Your complete answer to the task as a SINGLE STRING"
}}''')

        structured_output_rules = [
            "CRITICAL RULES FOR JSON:",
            "1. The \"answer\" field MUST be a plain text string, NOT an object or array.",
            "2. If you need to list multiple items in the \"answer\" field, format them as a single string with clear separators (e.g., \"Item 1: Description. Item 2: Description.\").",
            "3. Do NOT create nested JSON structures within any field.",
            "4. If you cannot confidently determine the information from the provided text to answer the task, ALL THREE fields (\"explanation\", \"citation\", \"answer\") in the JSON object must be null."
        ]
        prompt_lines.extend(structured_output_rules)

        # Rule 5 regarding expected_format needs to be renumbered if it was part of the list,
        # but it's added conditionally after extending structured_output_rules.
        # So, its conditional addition logic remains correct without renumbering.

        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() != "json":
            prompt_lines.append(f"5. Format the content WITHIN the \"answer\" field as {valves.expected_format.upper()}. For example, if \"bullet points\", the \"answer\" string should look like \"- Point 1\\n- Point 2\".")
        elif hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() == "json":
             prompt_lines.append("5. The overall response is already JSON. Ensure the content of the 'answer' field is a simple string, not further JSON encoded, unless the task specifically asks for a JSON string as the answer.")


        prompt_lines.append(f'''
EXAMPLE of CORRECT format:
{{
    "explanation": "Found information about X in the text for the task",
    "citation": "The text states 'X is Y'...",
    "answer": "X is Y according to the document."
}}''')
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() == "bullet points":
            prompt_lines.append(f'''
EXAMPLE for "answer" field formatted as BULLET POINTS:
{{
    "explanation": "Found several points for the task.",
    "citation": "Relevant quote...",
    "answer": "- Point 1: Details about point 1.\\n- Point 2: Details about point 2."
}}''')

        prompt_lines.append(f'''
EXAMPLE of INCORRECT format (DO NOT DO THIS):
{{
    "answer": {{"key": "value"}}  // WRONG - "answer" field must be a string!
}}''')

    else: # Not using structured output
        prompt_lines.append("\n\nProvide a brief, specific answer based ONLY on the text provided above.")
        if hasattr(valves, 'expected_format') and valves.expected_format and valves.expected_format.lower() != "text":
            prompt_lines.append(f"Format your entire response as {valves.expected_format.upper()}.")
        prompt_lines.append("If no relevant information is found in THIS SPECIFIC TEXT, respond with the single word \"NONE\".")
    
    return "\n".join(prompt_lines)

# Partials File: partials/minions_decomposition_logic.py
from typing import List, Callable, Any, Dict, Awaitable, Tuple # Added Dict, Awaitable, Tuple
import asyncio # Added asyncio as call_claude_func is async

# This helper will be based on the current parse_tasks from minions_protocol_logic.py
def _parse_tasks_helper(claude_response: str, max_tasks: int, debug_log: List[str], valves: Any) -> List[str]:
    """
    Parse tasks from Claude's response.
    Enhanced to match the original parse_tasks more closely.
    """
    lines = claude_response.split("\n")
    tasks = []
    for line in lines:
        line = line.strip()
        # More robust parsing for numbered or bulleted lists from original parse_tasks
        if line.startswith(tuple(f"{i}." for i in range(1, 10))) or \
           line.startswith(tuple(f"{i})" for i in range(1, 10))) or \
           line.startswith(("- ", "* ", "+ ")):
            # Attempt to split by the first space after the list marker
            parts = line.split(None, 1)
            task = parts[1].strip() if len(parts) > 1 else ""
            if len(task) > 10:  # Keep simple task filter
                tasks.append(task)
            elif valves.debug_mode and task:
                 debug_log.append(f"   [Debug] Task too short, skipped: '{task}' (Length: {len(task)})")

    if not tasks and valves.debug_mode:
        debug_log.append(f"   [Debug] No tasks parsed from Claude response: {claude_response[:200]}...")
        # Fallback could be added here if necessary, but original parse_tasks didn't have one.

    return tasks[:max_tasks]

async def decompose_task(
    valves: Any,
    call_claude_func: Callable[..., Awaitable[str]],
    query: str,
    scratchpad_content: str,
    num_chunks: int,
    max_tasks_per_round: int,
    current_round: int,
    conversation_log: List[str],
    debug_log: List[str]
) -> Tuple[List[str], str, str]:  # Added third return value for the prompt
    """
    Constructs the decomposition prompt, calls Claude, and parses tasks.
    Returns: (tasks, claude_response, decomposition_prompt)
    """
    # Base decomposition prompt
    base_decomposition_prompt = f'''You are a supervisor LLM in a multi-round process. Your goal is to answer: "{query}"
Context has been split into {num_chunks} chunks. A local LLM will process these chunks for each task you define.
Scratchpad (previous findings): {scratchpad_content if scratchpad_content else "Nothing yet."}

Based on the scratchpad and the original query, identify up to {max_tasks_per_round} specific, simple tasks for the local assistant.
If the information in the scratchpad is sufficient to answer the query, respond ONLY with the exact phrase "FINAL ANSWER READY." followed by the comprehensive answer.
Otherwise, list the new tasks clearly. Ensure tasks are actionable. Avoid redundant tasks.
Format tasks as a simple list (e.g., 1. Task A, 2. Task B).'''

    # Enhance the prompt with task formulation guidance
    decomposition_prompt = _enhance_decomposition_prompt(base_decomposition_prompt, valves)

    if valves.show_conversation:
        conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round}):** Sending prompt:\n```\n{decomposition_prompt}\n```")

    start_time_claude_decomp = 0
    if valves.debug_mode:
        start_time_claude_decomp = asyncio.get_event_loop().time()
        debug_log.append(f"   [Debug] Sending decomposition prompt to Claude (Round {current_round}):\n{decomposition_prompt}")

    try:
        claude_response = await call_claude_func(valves, decomposition_prompt)
        
        if valves.debug_mode:
            end_time_claude_decomp = asyncio.get_event_loop().time()
            time_taken_claude_decomp = end_time_claude_decomp - start_time_claude_decomp
            debug_log.append(f"   ⏱️ Claude call (Decomposition Round {current_round}) took {time_taken_claude_decomp:.2f}s.")
            debug_log.append(f"   [Debug] Claude response (Decomposition Round {current_round}):\n{claude_response}")

        tasks = _parse_tasks_helper(claude_response, max_tasks_per_round, debug_log, valves)
        
        if valves.debug_mode:
            debug_log.append(f"   Identified {len(tasks)} tasks for Round {current_round} from decomposition response.")
            for task_idx, task_item in enumerate(tasks):
                debug_log.append(f"    Task {task_idx+1} (Round {current_round}): {task_item[:100]}...")
        
        return tasks, claude_response, decomposition_prompt  # Return the prompt too

    except Exception as e:
        error_msg = f"❌ Error calling Claude for decomposition in round {current_round}: {e}"
        if valves.show_conversation:
            conversation_log.append(error_msg)
        if valves.debug_mode:
            debug_log.append(f"   {error_msg}")
        return [], f"CLAUDE_ERROR: {error_msg}", ""  # Return empty prompt on error
    
def _enhance_decomposition_prompt(base_prompt: str, valves: Any) -> str:
    """
    Enhances the decomposition prompt with additional guidance to ensure
    tasks are formulated to receive string responses.
    """
    task_formulation_guidance = '''

IMPORTANT TASK FORMULATION RULES:
1. Each task should request information that can be expressed as plain text
2. Avoid tasks that implicitly request structured data (like "Create a table of..." or "List with categories...")
3. Instead of "Extract and categorize X by Y", use "Describe X including information about Y"
4. Tasks should be answerable with narrative text, not data structures

GOOD TASK EXAMPLES:
- "Summarize the key advancements described in each Part, presenting them as a narrative"
- "Describe the AI winters mentioned, including their timeframes and characteristics"
- "Explain the progression of AI development across different periods"

BAD TASK EXAMPLES (avoid these):
- "Create a structured list of Parts with their key points"
- "Extract and categorize advancements by Part number"
- "Build a timeline table of AI winters"

Remember: The local assistant will return text strings, not structured data.'''
    
    return base_prompt + task_formulation_guidance


# Partials File: partials/minions_protocol_logic.py
import asyncio
import json
import hashlib # Import hashlib
from typing import List, Dict, Any, Callable # Removed Optional, Awaitable

# parse_tasks function removed, will be part of minions_decomposition_logic.py

# Removed create_chunks function from here

def parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, TaskResultModel: Any) -> Dict:
    """Parse local model response, supporting both text and structured formats"""
    confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
    default_numeric_confidence = 0.3 # Corresponds to LOW

    if is_structured and use_structured_output:
        try:
            parsed_json = json.loads(response)
            
            # Safety net: if answer is a dict/list, stringify it
            if 'answer' in parsed_json and not isinstance(parsed_json['answer'], (str, type(None))):
                if debug_mode:
                    print(f"DEBUG: Converting non-string answer to string: {type(parsed_json['answer'])}")
                parsed_json['answer'] = json.dumps(parsed_json['answer']) if parsed_json['answer'] else None
            
            validated_model = TaskResultModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            
            text_confidence = model_dict.get('confidence', 'LOW').upper()
            model_dict['numeric_confidence'] = confidence_map.get(text_confidence, default_numeric_confidence)

            # Check if the structured response indicates "not found" via its 'answer' field
            if model_dict.get('answer') is None:
                model_dict['_is_none_equivalent'] = True
            else:
                model_dict['_is_none_equivalent'] = False
            return model_dict
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in MinionS: {e}. Response was: {response[:500]}")
            # Fallback for parsing failure
            is_none_equivalent_fallback = response.strip().upper() == "NONE"
            return {"answer": response, "explanation": response, "confidence": "LOW", "numeric_confidence": default_numeric_confidence, "parse_error": str(e), "_is_none_equivalent": is_none_equivalent_fallback}
    
    # Fallback for non-structured processing
    is_none_equivalent_text = response.strip().upper() == "NONE"
    # Confidence is MEDIUM by default in this path
    return {"answer": response, "explanation": response, "confidence": "MEDIUM", "numeric_confidence": confidence_map['MEDIUM'], "citation": None, "parse_error": None, "_is_none_equivalent": is_none_equivalent_text}

async def execute_tasks_on_chunks(
    tasks: List[str], 
    chunks: List[str], 
    conversation_log: List[str], 
    current_round: int,
    valves: Any,
    call_ollama: Callable,
    TaskResult: Any
) -> Dict:
    """Execute tasks on chunks using local model"""
    overall_task_results = []
    total_attempts_this_call = 0
    total_timeouts_this_call = 0

    # Initialize Metrics
    round_start_time = asyncio.get_event_loop().time()
    tasks_executed_count = 0
    task_success_count = 0
    task_failure_count = 0
    chunk_processing_times = []
    # Initialize Confidence Accumulators
    round_confidence_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    aggregated_task_confidences = []

    for task_idx, task in enumerate(tasks):
        tasks_executed_count += 1 # Track Tasks Executed
        conversation_log.append(f"**📋 Task {task_idx + 1} (Round {current_round}):** {task}")
        results_for_this_task_from_chunks = []
        current_task_chunk_confidences = [] # Initialize for current task
        chunk_timeout_count_for_task = 0
        num_relevant_chunks_found = 0

        for chunk_idx, chunk in enumerate(chunks):
            total_attempts_this_call += 1
            
            # Call the new function for local task prompt
            local_prompt = get_minions_local_task_prompt( # Ensure this function is defined or imported
                chunk=chunk,
                task=task,
                chunk_idx=chunk_idx,
                total_chunks=len(chunks),
                valves=valves
            )
            
            # Track Chunk Processing Time
            chunk_start_time = asyncio.get_event_loop().time()
            # start_time_ollama variable was previously used for debug,
            # let's ensure we use chunk_start_time for metrics consistently.
            # If start_time_ollama is still needed for debug, it can be kept separate.
            # For metrics, we'll use chunk_start_time and chunk_end_time.

            if valves.debug_mode:
                conversation_log.append(
                    f"   🔄 Task {task_idx + 1} - Trying chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} chars)... (Debug Mode)"
                )
                # start_time_ollama = asyncio.get_event_loop().time() # This was for debug, let's use chunk_start_time

            try:
                response_str = await asyncio.wait_for(
                    call_ollama(
                        valves,
                        local_prompt,
                        use_json=True,
                        schema=TaskResult
                    ),
                    timeout=valves.timeout_local,
                )
                chunk_end_time = asyncio.get_event_loop().time()
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)

                response_data = parse_local_response(
                    response_str,
                    is_structured=True,
                    use_structured_output=valves.use_structured_output,
                    debug_mode=valves.debug_mode,
                    TaskResultModel=TaskResult # Pass TaskResult to parse_local_response
                )

                # Collect Confidence per Chunk
                numeric_confidence = response_data.get('numeric_confidence', 0.3) # Default to LOW numeric
                text_confidence = response_data.get('confidence', 'LOW').upper()
                response_data['fingerprint'] = None # Initialize fingerprint

                if not response_data.get('_is_none_equivalent') and not response_data.get('parse_error'):
                    if text_confidence in round_confidence_distribution:
                        round_confidence_distribution[text_confidence] += 1
                    current_task_chunk_confidences.append(numeric_confidence)

                    # Fingerprint Generation Logic
                    answer_text = response_data.get('answer')
                    if answer_text: # Ensure answer_text is not None and not empty
                        normalized_text = answer_text.lower().strip()
                        if normalized_text: # Ensure normalized_text is not empty
                            response_data['fingerprint'] = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
                
                if valves.debug_mode:
                    # end_time_ollama = asyncio.get_event_loop().time() # Already have chunk_end_time
                    time_taken_ollama = (chunk_end_time - chunk_start_time) # Use metric times for debug consistency
                    status_msg = ""
                    details_msg = ""
                    if response_data.get("parse_error"):
                        status_msg = "Parse Error"
                        details_msg = f"Error: {response_data['parse_error']}, Raw: {response_data.get('answer', '')[:70]}..."
                    elif response_data['_is_none_equivalent']:
                        status_msg = "No relevant info"
                        details_msg = f"Response indicates no info found. Confidence: {response_data.get('confidence', 'N/A')}"
                    else:
                        status_msg = "Relevant info found"
                        details_msg = f"Answer: {response_data.get('answer', '')[:70]}..., Confidence: {response_data.get('confidence', 'N/A')}"
        
                    conversation_log.append(
                         f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} processed by local LLM in {time_taken_ollama:.2f}s. Status: {status_msg}. Details: {details_msg} (Debug Mode)"
                    )

                if not response_data.get('_is_none_equivalent'): # Check with .get for safety
                    extracted_info = response_data.get('answer') or response_data.get('explanation', 'Could not extract details.')
                    # Store as dict with fingerprint
                    results_for_this_task_from_chunks.append({
                        "text": f"[Chunk {chunk_idx+1}]: {extracted_info}",
                        "fingerprint": response_data.get('fingerprint')
                    })
                    num_relevant_chunks_found += 1
                    # Note: current_task_chunk_confidences is already appended if valid (based on earlier logic)
                    
            except asyncio.TimeoutError:
                chunk_end_time = asyncio.get_event_loop().time() # Capture time even on timeout
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)
                total_timeouts_this_call += 1
                chunk_timeout_count_for_task += 1
                conversation_log.append(
                    f"   ⏰ Task {task_idx + 1} - Chunk {chunk_idx + 1} timed out after {valves.timeout_local}s"
                )
                if valves.debug_mode:
                    # end_time_ollama = asyncio.get_event_loop().time() # Already have chunk_end_time
                    time_taken_ollama = (chunk_end_time - chunk_start_time) # Use metric times
                    conversation_log.append(
                         f"   ⏱️ Task {task_idx+1}, Chunk {chunk_idx+1} TIMEOUT after {time_taken_ollama:.2f}s. (Debug Mode)"
                    )
            except Exception as e:
                # It's good practice to also record chunk processing time if an unexpected exception occurs
                chunk_end_time = asyncio.get_event_loop().time()
                chunk_processing_times.append((chunk_end_time - chunk_start_time) * 1000)
                conversation_log.append(
                    f"   ❌ Task {task_idx + 1} - Chunk {chunk_idx + 1} error: {str(e)}"
                )
        
        # Track Task Success/Failure and Aggregate Confidence per Task
        if results_for_this_task_from_chunks:
            task_success_count += 1
            avg_task_confidence = sum(current_task_chunk_confidences) / len(current_task_chunk_confidences) if current_task_chunk_confidences else 0.0
            aggregated_task_confidences.append({
                "task": task,
                "avg_numeric_confidence": avg_task_confidence,
                "contributing_successful_chunks": len(current_task_chunk_confidences)
            })
            # Modify overall_task_results for successful tasks
            detailed_results = [{"text": res["text"], "fingerprint": res["fingerprint"]} for res in results_for_this_task_from_chunks if isinstance(res, dict)]
            aggregated_text_result = "\n".join([res["text"] for res in detailed_results])
            overall_task_results.append({
                "task": task,
                "result": aggregated_text_result,
                "status": "success",
                "detailed_findings": detailed_results
            })
            conversation_log.append(
                f"**💻 Local Model (Aggregated for Task {task_idx + 1}, Round {current_round}):** Found info in {num_relevant_chunks_found}/{len(chunks)} chunk(s). Avg Confidence: {avg_task_confidence:.2f}. First result snippet: {detailed_results[0]['text'][:100] if detailed_results else 'N/A'}..."
            )
        elif chunk_timeout_count_for_task > 0 and chunk_timeout_count_for_task == len(chunks):
            task_failure_count += 1 # All chunks timed out
            aggregated_task_confidences.append({"task": task, "avg_numeric_confidence": 0.0, "contributing_successful_chunks": 0})
            overall_task_results.append({
                "task": task,
                "result": f"Timeout on all {len(chunks)} chunks",
                "status": "timeout_all_chunks",
                "detailed_findings": [] # Add empty list for consistency
            })
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** All {len(chunks)} chunks timed out."
            )
        else: # No relevant info found or other errors
            task_failure_count += 1
            aggregated_task_confidences.append({"task": task, "avg_numeric_confidence": 0.0, "contributing_successful_chunks": 0})
            overall_task_results.append({
                "task": task,
                "result": "Information not found in any relevant chunk",
                "status": "not_found",
                "detailed_findings": [] # Add empty list for consistency
            })
            conversation_log.append(
                f"**💻 Local Model (Task {task_idx + 1}, Round {current_round}):** No relevant information found in any chunk."
            )
        # current_task_chunk_confidences is implicitly reset at the start of the task loop

    # Calculate Final Metrics
    round_end_time = asyncio.get_event_loop().time()
    execution_time_ms = (round_end_time - round_start_time) * 1000
    avg_chunk_processing_time_ms = sum(chunk_processing_times) / len(chunk_processing_times) if chunk_processing_times else 0
    success_rate = task_success_count / tasks_executed_count if tasks_executed_count > 0 else 0

    # Prepare Metrics Object (as a dictionary for now, as per instructions)
    round_metrics_data = {
        "round_number": current_round,
        "tasks_executed": tasks_executed_count,
        "task_success_count": task_success_count,
        "task_failure_count": task_failure_count,
        "avg_chunk_processing_time_ms": avg_chunk_processing_time_ms,
        "execution_time_ms": execution_time_ms,
        "success_rate": success_rate,
        # total_unique_findings_count will be handled later, defaulting in the model
    }

    return {
        "results": overall_task_results,
        "total_chunk_processing_attempts": total_attempts_this_call,
        "total_chunk_processing_timeouts": total_timeouts_this_call,
        "round_metrics_data": round_metrics_data,
        "confidence_metrics_data": { # New confidence data
            "task_confidences": aggregated_task_confidences,
            "round_confidence_distribution": round_confidence_distribution
        }
    }

def calculate_token_savings(
    decomposition_prompts: List[str], 
    synthesis_prompts: List[str],
    all_results_summary: str, 
    final_response: str, 
    context_length: int, 
    query_length: int, 
    total_chunks: int,
    total_tasks: int
) -> dict:
    """Calculate token savings for MinionS protocol"""
    chars_per_token = 3.5
    
    # Traditional approach: entire context + query sent to Claude
    traditional_tokens = int((context_length + query_length) / chars_per_token)
    
    # MinionS approach: only prompts and summaries sent to Claude
    minions_tokens = 0
    for p in decomposition_prompts:
        minions_tokens += int(len(p) / chars_per_token)
    for p in synthesis_prompts:
        minions_tokens += int(len(p) / chars_per_token)
    minions_tokens += int(len(all_results_summary) / chars_per_token)
    minions_tokens += int(len(final_response) / chars_per_token)
    
    token_savings = traditional_tokens - minions_tokens
    percentage_savings = (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    
    return {
        'traditional_tokens_claude': traditional_tokens,
        'minions_tokens_claude': minions_tokens,
        'token_savings_claude': token_savings,
        'percentage_savings_claude': percentage_savings,
        'total_rounds': len(decomposition_prompts),
        'total_chunks_processed_local': total_chunks,
        'total_tasks_executed_local': total_tasks,
    }

# Partials File: partials/minion_sufficiency_analyzer.py
import re
from typing import Dict, List, Tuple, Any

class InformationSufficiencyAnalyzer:
    """
    Analyzes text to determine if it sufficiently addresses components of an initial query.
    """
    def __init__(self, query: str, debug_mode: bool = False):
        """
        Initializes the analyzer with the user's query and debug mode.
        """
        self.query = query
        self.debug_mode = debug_mode
        self.components: Dict[str, Dict[str, Any]] = {} # Stores {component_name: {"keywords": [...], "is_addressed": False, "confidence": 0.0}}
        self._identify_components()

    def _identify_components(self):
        """
        Identifies key components or topics from the user's query.
        Uses heuristics like quoted phrases, capitalized words, and generic fallbacks.
        """
        # Basic keyword extraction. This is a simple heuristic and can be expanded.
        # It looks for Nouns, Proper Nouns, and Adjectives, trying to form simple topics.
        # Example: "Compare the budget and timeline for Project Alpha and Project Beta"
        # Might identify: "budget", "timeline", "Project Alpha", "Project Beta"
        # Then forms components like "budget Project Alpha", "timeline Project Alpha", etc.

        # For simplicity in this iteration, we'll use a more direct approach:
        # Look for quoted phrases or capitalized words as potential components.
        # Or, define a few generic components if query is too simple.

        # Let's try to find quoted phrases first
        quoted_phrases = re.findall(r'"([^"]+)"', self.query)
        for phrase in quoted_phrases:
            self.components[phrase] = {"keywords": [kw.lower() for kw in phrase.split()], "is_addressed": False, "confidence": 0.0}

        # If no quoted phrases, look for capitalized words/phrases (potential proper nouns or topics)
        if not self.components:
            common_fillers = [
                "Here", "The", "This", "That", "There", "It", "Who", "What", "When", "Where", "Why", "How",
                "Is", "Are", "Was", "Were", "My", "Your", "His", "Her", "Its", "Our", "Their", "An",
                "As", "At", "But", "By", "For", "From", "In", "Into", "Of", "On", "Or", "Over",
                "So", "Then", "To", "Under", "Up", "With", "I"
            ]
            common_fillers_lower = [f.lower() for f in common_fillers]

            # Regex to find sequences of capitalized words, possibly including 'and', 'or', 'for', 'the'
            potential_topics = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+(?:and|or|for|the|[A-Z][a-zA-Z]*))*\b', self.query)

            topics = []
            for pt in potential_topics:
                is_multi_word = ' ' in pt
                # A single word is a common filler if its lowercased version is in common_fillers_lower
                is_common_filler_single_word = not is_multi_word and pt.lower() in common_fillers_lower
                # A single word is significant if it's an acronym (all upper) or longer than 3 chars
                is_significant_single_word = not is_multi_word and (pt.isupper() or len(pt) > 3)

                if is_multi_word: # Always include multi-word capitalized phrases
                    topics.append(pt)
                elif is_significant_single_word and not is_common_filler_single_word:
                    # Include significant single words only if they are NOT common fillers
                    topics.append(pt)

            if self.debug_mode and potential_topics:
                 print(f"DEBUG [SufficiencyAnalyzer]: Potential capitalized topics found: {potential_topics}")
                 print(f"DEBUG [SufficiencyAnalyzer]: Filtered topics after common word/length check: {topics}")

            for topic in topics:
                # Avoid adding overlapping sub-phrases if a larger phrase is already a component
                is_sub_phrase = False
                for existing_comp in self.components.keys():
                    if topic in existing_comp and topic != existing_comp:
                        is_sub_phrase = True
                        break
                if not is_sub_phrase:
                    self.components[topic] = {"keywords": [kw.lower() for kw in topic.split()], "is_addressed": False, "confidence": 0.0}

        # If still no components (e.g., simple query like "summarize this"), create generic ones.
        if not self.components:
            if "compare" in self.query.lower() or "contrast" in self.query.lower():
                self.components["comparison points"] = {"keywords": ["compare", "contrast", "similarit", "difference"], "is_addressed": False, "confidence": 0.0} # Added common keywords
                self.components["subject 1 details"] = {"keywords": [], "is_addressed": False, "confidence": 0.0} # Placeholder, keyword matching might be hard
                self.components["subject 2 details"] = {"keywords": [], "is_addressed": False, "confidence": 0.0} # Placeholder
            elif "summarize" in self.query.lower() or "overview" in self.query.lower():
                self.components["main points"] = {"keywords": ["summary", "summarize", "overview", "main point", "key aspect"], "is_addressed": False, "confidence": 0.0}
                self.components["details"] = {"keywords": ["detail", "specific", "elaborate"], "is_addressed": False, "confidence": 0.0}
            else: # Default fallback component
                self.components["overall query"] = {"keywords": [kw.lower() for kw in self.query.split()[:5]], "is_addressed": False, "confidence": 0.0} # Use first few words of query

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Identified components: {list(self.components.keys())}")

    def update_components(self, text_to_analyze: str, round_avg_confidence: float):
        """
        Updates the status of query components based on the provided text and confidence.
        Marks components as addressed if their keywords are found in the text.
        """
        # In this version, we'll use round_avg_confidence as a proxy for the confidence
        # of the information that might address a component.
        # A more advanced version could try to link specific task confidences.
        text_lower = text_to_analyze.lower()
        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Updating components based on text (first 100 chars): {text_lower[:100]}...")

        for comp_name, comp_data in self.components.items():
            if not comp_data["is_addressed"]:
                # If keywords are defined, require all keywords for the component to be present.
                # This is a strict rule and might need adjustment (e.g., any keyword, or a percentage).
                if comp_data["keywords"]:
                    all_keywords_present = all(kw in text_lower for kw in comp_data["keywords"])
                    if all_keywords_present:
                        comp_data["is_addressed"] = True
                        comp_data["confidence"] = round_avg_confidence # Use round's average confidence
                        if self.debug_mode:
                            print(f"DEBUG [SufficiencyAnalyzer]: Component '{comp_name}' ADDRESSED by keyword match. Confidence set to {round_avg_confidence:.2f}")
                # If no keywords (e.g. generic components like "subject 1 details"), this logic won't address them.
                # This is a limitation of the current basic keyword approach for generic components.
                # For this iteration, such components might remain unaddressed unless their names/generic keywords appear.

    def calculate_sufficiency_score(self) -> Tuple[float, float, Dict[str, bool]]:
        """
        Calculates the overall sufficiency score based on component coverage and confidence.
        Returns score, coverage percentage, and status of each component.
        """
        if not self.components:
            return 0.0, 0.0, {}

        addressed_components_count = 0
        total_confidence_of_addressed = 0.0
        component_status_for_metrics: Dict[str, bool] = {}

        for comp_name, comp_data in self.components.items():
            component_status_for_metrics[comp_name] = comp_data["is_addressed"]
            if comp_data["is_addressed"]:
                addressed_components_count += 1
                total_confidence_of_addressed += comp_data["confidence"]

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Addressed components: {addressed_components_count}/{len(self.components)}")

        component_coverage_percentage = (addressed_components_count / len(self.components)) if len(self.components) > 0 else 0.0

        avg_confidence_of_addressed = (total_confidence_of_addressed / addressed_components_count) if addressed_components_count > 0 else 0.0

        # Score is a product of coverage and average confidence of what's covered.
        sufficiency_score = component_coverage_percentage * avg_confidence_of_addressed

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Coverage: {component_coverage_percentage:.2f}, Avg Confidence of Addressed: {avg_confidence_of_addressed:.2f}, Sufficiency Score: {sufficiency_score:.2f}")

        return sufficiency_score, component_coverage_percentage, component_status_for_metrics

    def get_analysis_details(self) -> Dict[str, Any]:
        """
        Returns a dictionary with the sufficiency score, coverage, and component status.
        """
        sufficiency_score, component_coverage_percentage, component_status = self.calculate_sufficiency_score()
        return {
            "sufficiency_score": sufficiency_score,
            "component_coverage_percentage": component_coverage_percentage,
            "information_components_status": component_status # Changed key name slightly to avoid conflict if used directly in RoundMetrics
        }


# Partials File: partials/minion_convergence_detector.py
from typing import Optional, List, Dict, Any, Tuple

# Attempt to import RoundMetrics and Valves type hints for clarity,
# but handle potential circular dependency or generation-time issues
# by using 'Any' if direct import is problematic during generation.
try:
    # Assuming valves structure will be available, or use Any
    # from .minions_valves import MinionSValves # This might not exist as a direct importable type
    ValvesType = Any # Placeholder for valve types from pipe_self.valves
except ImportError:
    RoundMetrics = Any
    ValvesType = Any

class ConvergenceDetector:
    def __init__(self, debug_mode: bool = False):
        """Initializes the ConvergenceDetector."""
        self.debug_mode = debug_mode
        if self.debug_mode:
            print("DEBUG [ConvergenceDetector]: Initialized.")

    def calculate_round_convergence_metrics(
        self,
        current_round_metric: RoundMetrics,
        previous_round_metric: Optional[RoundMetrics]
    ) -> Dict[str, Any]:
        """
        Calculates specific convergence-related metrics for the current round.
        These will be used to update the current_round_metric object.
        """
        calculated_metrics = {}

        # 1. Information Gain Rate (using new_findings_count_this_round)
        # Ensure current_round_metric has the attribute, otherwise default to 0. Useful if RoundMetrics is Any.
        calculated_metrics["information_gain_rate"] = float(getattr(current_round_metric, "new_findings_count_this_round", 0))

        # 2. Novel Findings Percentage for this round
        new_findings = getattr(current_round_metric, "new_findings_count_this_round", 0)
        duplicate_findings = getattr(current_round_metric, "duplicate_findings_count_this_round", 0)
        total_findings_this_round = new_findings + duplicate_findings

        if total_findings_this_round > 0:
            calculated_metrics["novel_findings_percentage_this_round"] = new_findings / total_findings_this_round
        else:
            calculated_metrics["novel_findings_percentage_this_round"] = 0.0

        # 3. Task Failure Rate Trend
        trend = "N/A"
        if previous_round_metric:
            current_success_rate = getattr(current_round_metric, "success_rate", 0.0)
            previous_success_rate = getattr(previous_round_metric, "success_rate", 0.0)
            tolerance = 0.05
            if current_success_rate < previous_success_rate - tolerance:
                trend = "increasing_failures"
            elif current_success_rate > previous_success_rate + tolerance:
                trend = "decreasing_failures"
            else:
                trend = "stable_failures"
        calculated_metrics["task_failure_rate_trend"] = trend

        # 4. Predicted Value of Next Round (simple heuristic)
        predicted_value = "medium"
        novelty_current_round = calculated_metrics["novel_findings_percentage_this_round"]

        if novelty_current_round > 0.5:
            predicted_value = "high"
        elif novelty_current_round < 0.1:
            predicted_value = "low"

        if trend == "increasing_failures" and predicted_value == "medium":
            predicted_value = "low"
        elif trend == "decreasing_failures" and predicted_value == "medium":
            predicted_value = "high"

        calculated_metrics["predicted_value_of_next_round"] = predicted_value

        if self.debug_mode:
            round_num_debug = getattr(current_round_metric, "round_number", "Unknown")
            print(f"DEBUG [ConvergenceDetector]: Calculated metrics for round {round_num_debug}: {calculated_metrics}")

        return calculated_metrics

    def check_for_convergence(
        self,
        current_round_metric: RoundMetrics,
        sufficiency_score: float,
        total_rounds_executed: int,
        effective_novelty_to_use: float, # New parameter for dynamic threshold
        effective_sufficiency_to_use: float, # New parameter for dynamic threshold
        valves: ValvesType, # Still needed for min_rounds_before_convergence_check, convergence_rounds_min_novelty
        all_round_metrics: List[RoundMetrics]
    ) -> Tuple[bool, str]:
        """
        Checks if convergence criteria are met using potentially dynamic thresholds.
        """
        min_rounds_for_conv_check = getattr(valves, "min_rounds_before_convergence_check", 2)
        if total_rounds_executed < min_rounds_for_conv_check:
             if self.debug_mode:
                print(f"DEBUG [ConvergenceDetector]: Skipping convergence check for round {getattr(current_round_metric, 'round_number', 'N/A')}, min rounds for convergence check not met ({total_rounds_executed}/{min_rounds_for_conv_check}).")
             return False, ""

        # Convergence criteria:
        # 1. Low novelty for a certain number of consecutive rounds (using effective_novelty_to_use).
        # 2. Sufficiency score is above a threshold (using effective_sufficiency_to_use).

        low_novelty_streak = 0
        required_streak_length = getattr(valves, "convergence_rounds_min_novelty", 2)
        # Use the passed effective_novelty_to_use instead of getattr(valves, "convergence_novelty_threshold", 0.10)

        if len(all_round_metrics) >= required_streak_length:
            is_streak = True
            for i in range(required_streak_length):
                metric_to_check = all_round_metrics[-(i+1)]

                if not hasattr(metric_to_check, 'novel_findings_percentage_this_round') or \
                   getattr(metric_to_check, 'novel_findings_percentage_this_round') >= effective_novelty_to_use:
                    is_streak = False
                    break
            if is_streak:
                low_novelty_streak = required_streak_length

        current_round_num_debug = getattr(current_round_metric, "round_number", "N/A")
        # Use the passed effective_sufficiency_to_use instead of getattr(valves, "convergence_sufficiency_threshold", 0.7)

        if self.debug_mode:
            print(f"DEBUG [ConvergenceDetector]: Checking convergence for round {current_round_num_debug}:")
            print(f"  Using Effective Novelty Threshold: {effective_novelty_to_use:.2f}, Required streak: {required_streak_length}")
            print(f"  Actual low novelty streak achieved for check: {low_novelty_streak} (needs to be >= {required_streak_length})")
            print(f"  Sufficiency score: {sufficiency_score:.2f}, Using Effective Sufficiency Threshold: {effective_sufficiency_to_use:.2f}")

        if low_novelty_streak >= required_streak_length and sufficiency_score >= effective_sufficiency_to_use:
            reason = (
                f"Convergence detected: Novelty < {effective_novelty_to_use*100:.0f}% "
                f"for {low_novelty_streak} round(s) AND "
                f"Sufficiency ({sufficiency_score:.2f}) >= {effective_sufficiency_to_use:.2f}."
            )
            if self.debug_mode:
                print(f"DEBUG [ConvergenceDetector]: {reason}")
            return True, reason

        return False, ""


# Partials File: partials/minions_pipe_method.py
import asyncio
import re # Added re
from enum import Enum # Ensured Enum is present
from typing import Any, List, Callable, Dict
from fastapi import Request

# Removed: from .common_query_utils import QueryComplexityClassifier, QueryComplexity

# --- Content from common_query_utils.py START ---
class QueryComplexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"

class QueryComplexityClassifier:
    """Classifies query complexity based on keywords and length."""
    def __init__(self, debug_mode: bool = False):
        """Initializes the classifier with debug mode."""
        self.debug_mode = debug_mode
        # Keywords indicating complexity
        self.complex_keywords = [
            "analyze", "compare", "contrast", "summarize", "explain in detail",
            "discuss", "critique", "evaluate", "recommend", "predict", "what if",
            "how does", "why does", "implications"
        ]
        self.medium_keywords = [
            "list", "describe", "details of", "tell me about", "what are the"
        ]
        # Question words (simple ones often start fact-based questions)
        self.simple_question_starters = ["what is", "who is", "when was", "where is", "define"]

    def classify_query(self, query: str) -> QueryComplexity:
        """
        Classifies the given query into SIMPLE, MEDIUM, or COMPLEX.
        Uses rules based on keywords and word count.
        """
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Query='{query_lower}', WordCount={word_count}")

        # Rule 1: Complex Keywords
        for keyword in self.complex_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched complex keyword '{keyword}'")
                return QueryComplexity.COMPLEX

        # Rule 2: Word Count for Complex
        if word_count > 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched complex by word count (>25)")
            return QueryComplexity.COMPLEX

        # Rule 3: Word Count for Simple (and simple question starters)
        if word_count < 10:
            is_simple_starter = any(query_lower.startswith(starter) for starter in self.simple_question_starters)
            if is_simple_starter:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched simple by word count (<10) and starter.")
                return QueryComplexity.SIMPLE

        # Rule 4: Medium Keywords
        for keyword in self.medium_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched medium keyword '{keyword}'")
                return QueryComplexity.MEDIUM

        if word_count >= 10 and word_count <= 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched medium by word count (10-25)")
            return QueryComplexity.MEDIUM

        if word_count < 10: # Default for short queries not caught by simple_question_starters
             if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Defaulting short query to MEDIUM (no simple starter)")
             return QueryComplexity.MEDIUM

        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Defaulting to MEDIUM (no other rules matched clearly)")
        return QueryComplexity.MEDIUM
# --- Content from common_query_utils.py END ---


async def _call_claude_directly(valves: Any, query: str, call_claude_func: Callable) -> str:
    """Fallback to direct Claude call when no context is available"""
    return await call_claude_func(valves, f"Please answer this question: {query}")

async def _execute_minions_protocol(
    valves: Any,
    query: str,
    context: str,
    call_claude: Callable,
    call_ollama_func: Callable,
    TaskResultModel: Any
) -> str:
    """Execute the MinionS protocol"""
    conversation_log = []
    debug_log = []
    scratchpad_content = ""
    all_round_results_aggregated = []
    all_round_metrics: List[RoundMetrics] = [] # Initialize Metrics List
    global_unique_fingerprints_seen = set() # Initialize Global Fingerprint Set
    decomposition_prompts_history = []
    synthesis_prompts_history = []
    final_response = "No answer could be synthesized."
    claude_provided_final_answer = False
    total_tasks_executed_local = 0
    total_chunks_processed_for_stats = 0
    total_chunk_processing_timeouts_accumulated = 0
    synthesis_input_summary = ""
    early_stopping_reason_for_output = None # Initialize for storing stopping reason

    overall_start_time = asyncio.get_event_loop().time()

    # User query is passed directly to _execute_minions_protocol
    user_query_original = query # Store original query for logging

    # --- Query Expansion (Iteration 3) ---
    # Determine domain_hint (e.g., from valves or other sources if available, else None)
    # For now, we'll assume no explicit domain_hint is passed here, QueryExpander will infer.
    domain_hint_for_expansion = getattr(valves, "domain_hint", None)
    processed_query = user_query_original

    if getattr(valves, "enable_query_expansion", True): # Default to True if valve not present
        if valves.debug_mode:
            debug_log.append(f"✨ **Starting Query Expansion... (Iteration 3)**")
            debug_log.append(f"   Original User Query: '{user_query_original}'")

        expander = QueryExpander(debug_mode=valves.debug_mode)
        processed_query = expander.expand(
            query=user_query_original,
            apply_context_injection=getattr(valves, "qe_apply_context_injection", True),
            apply_completion=getattr(valves, "qe_apply_completion", True),
            apply_synonyms=getattr(valves, "qe_apply_synonyms", True),
            domain_hint=domain_hint_for_expansion
        )
        if valves.debug_mode:
            if processed_query != user_query_original:
                debug_log.append(f"   Expanded Query: '{processed_query}'")
            else:
                debug_log.append(f"   Query not significantly altered by expansion steps.")
            debug_log.append(f"✨ **Finished Query Expansion.**")
    else:
        if valves.debug_mode:
            debug_log.append(f"ℹ️ Query Expansion is DISABLED via valve 'enable_query_expansion'. Using original query.")

    user_query = processed_query # This is the query to be used by the rest of the protocol

    # --- Query Analysis (Iteration 1 became Iteration 2 with ambiguity) ---
    if valves.debug_mode:
        debug_log.append("🧠 **Starting Query Analysis... (Iteration 2 Features)**") # Updated comment

    # QueryAnalyzer should use the potentially expanded query
    query_analyzer = QueryAnalyzer(query=user_query, debug_mode=valves.debug_mode)
    query_metadata: QueryMetadata = query_analyzer.analyze()

    if valves.debug_mode:
        # Log the original query if it was expanded, for comparison
        if user_query_original != user_query:
             debug_log.append(f"   Original Query (Pre-Expansion): {user_query_original}")
        debug_log.append(f"   Analyzed Query (Post-Expansion): {query_metadata['original_query']}") # This will be the processed_query
        debug_log.append(f"   Query Type: {query_metadata['query_type'].value}")
        debug_log.append(f"   Detected Entities: {query_metadata['entities']}")
        debug_log.append(f"   Temporal References: {query_metadata['temporal_refs']}")
        debug_log.append(f"   Action Verbs: {query_metadata['action_verbs']}")
        debug_log.append(f"   Scope: {query_metadata['scope'].value}")
        debug_log.append(f"   Ambiguity Markers: {query_metadata['ambiguity_markers']}")
        debug_log.append(f"   Detected Patterns: {query_metadata['detected_patterns']}")
        debug_log.append(f"   Ambiguity Score (Iter 2): {query_metadata['ambiguity_score']:.2f}")
        debug_log.append(f"   Detailed Ambiguity Report (Iter 2): {query_metadata['detailed_ambiguity_report']}")
        debug_log.append(f"   Decomposability Score (Iter 2): {query_metadata['decomposability_score']:.2f}")

        # New warning for high ambiguity score
        # Assuming a threshold can be added to valves, e.g., valves.high_ambiguity_threshold
        # For now, hardcode 0.7 as per plan, but ideally this would be configurable.
        high_ambiguity_threshold = getattr(valves, 'high_ambiguity_threshold', 0.7)
        if query_metadata['ambiguity_score'] > high_ambiguity_threshold:
            debug_log.append(f"   ⚠️ WARNING: High ambiguity score ({query_metadata['ambiguity_score']:.2f}) detected. "
                             f"Query may require clarification for optimal processing. Threshold: {high_ambiguity_threshold}")

        debug_log.append("🧠 **Finished Query Analysis.**")
    # --- End Query Analysis ---

    # --- Performance Profile Logic ---
    current_run_max_rounds = valves.max_rounds
    current_run_base_sufficiency_thresh = valves.convergence_sufficiency_threshold
    current_run_base_novelty_thresh = valves.convergence_novelty_threshold
    current_run_simple_query_confidence_thresh = valves.simple_query_confidence_threshold
    current_run_medium_query_confidence_thresh = valves.medium_query_confidence_threshold

    profile_applied_details = [f"🧠 Performance Profile selected: {valves.performance_profile}"]

    if valves.performance_profile == "high_quality":
        current_run_max_rounds = min(valves.max_rounds + 1, 10)
        current_run_base_sufficiency_thresh = min(0.95, valves.convergence_sufficiency_threshold + 0.1)
        current_run_base_novelty_thresh = max(0.03, valves.convergence_novelty_threshold - 0.03)
        current_run_simple_query_confidence_thresh = min(0.95, valves.simple_query_confidence_threshold + 0.1)
        current_run_medium_query_confidence_thresh = min(0.95, valves.medium_query_confidence_threshold + 0.1)
        profile_applied_details.append(f"   - Applied 'high_quality' adjustments: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")
    elif valves.performance_profile == "fastest_results":
        current_run_max_rounds = max(1, valves.max_rounds - 1)
        current_run_base_sufficiency_thresh = max(0.05, valves.convergence_sufficiency_threshold - 0.1)
        current_run_base_novelty_thresh = min(0.95, valves.convergence_novelty_threshold + 0.05)
        current_run_simple_query_confidence_thresh = max(0.05, valves.simple_query_confidence_threshold - 0.1)
        current_run_medium_query_confidence_thresh = max(0.05, valves.medium_query_confidence_threshold - 0.1)
        profile_applied_details.append(f"   - Applied 'fastest_results' adjustments: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")
    else: # balanced
        profile_applied_details.append(f"   - Using 'balanced' profile: MaxRounds={current_run_max_rounds}, BaseSuffThresh={current_run_base_sufficiency_thresh:.2f}, BaseNovThresh={current_run_base_novelty_thresh:.2f}, SimpleConfThresh={current_run_simple_query_confidence_thresh:.2f}, MediumConfThresh={current_run_medium_query_confidence_thresh:.2f}")

    if valves.debug_mode:
        debug_log.extend(profile_applied_details)
        debug_log.append(f"🔍 **Debug Info (MinionS v0.2.0):**\n- Query: {user_query[:100]}...\n- Context length: {len(context)} chars") # Original debug line moved after profile logic
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**")


    # Instantiate Sufficiency Analyzer
    # IMPORTANT: Sufficiency Analyzer should use the *original* user query to determine user's intent for components
    analyzer = InformationSufficiencyAnalyzer(query=user_query_original, debug_mode=valves.debug_mode)
    if valves.debug_mode:
        debug_log.append(f"🧠 Sufficiency Analyzer initialized for *original* query: {user_query_original[:100]}...")
        debug_log.append(f"   Identified components: {list(analyzer.components.keys())}")

    # Instantiate Convergence Detector
    convergence_detector = ConvergenceDetector(debug_mode=valves.debug_mode)
    if valves.debug_mode:
        debug_log.append(f"🧠 Convergence Detector initialized.")

    # Initialize Query Complexity Classifier and Classify Query
    # Query Complexity should likely operate on the *original* query or the analyzed one, depending on desired behavior.
    # For now, let's use the potentially expanded 'user_query' for classification, as it's what downstream tasks see.
    query_classifier = QueryComplexityClassifier(debug_mode=valves.debug_mode)
    query_complexity_level = query_classifier.classify_query(user_query) # Using expanded query

    if valves.debug_mode:
        debug_log.append(f"🧠 Query (post-expansion) classified as: {query_complexity_level.value} (Debug Mode)")
    # Optional: Add to conversation_log if you want user to see it always
    # if valves.show_conversation:
    #     conversation_log.append(f"🧠 Initial query classified as complexity: {query_complexity_level.value}")

    # --- Dynamic Threshold Initialization ---
    doc_size_category = "medium" # Default
    context_len = len(context)
    if context_len < valves.doc_size_small_char_limit:
        doc_size_category = "small"
    elif context_len > valves.doc_size_large_char_start:
        doc_size_category = "large"

    if valves.debug_mode:
        debug_log.append(f"🧠 Document size category: {doc_size_category} (Length: {context_len} chars)")

    # Initialize effective thresholds with base values (now from current_run_... variables)
    effective_sufficiency_threshold = current_run_base_sufficiency_thresh
    effective_novelty_threshold = current_run_base_novelty_thresh
    # Base confidence thresholds for simple/medium queries will use current_run_... variables where they are applied.

    if valves.debug_mode:
        debug_log.append(f"🧠 Initial effective thresholds (after profile adjustments): Sufficiency={effective_sufficiency_threshold:.2f}, Novelty={effective_novelty_threshold:.2f}")
        debug_log.append(f"   Effective Simple Confidence Thresh (base for adaptation)={current_run_simple_query_confidence_thresh:.2f}, Medium Confidence Thresh (base for adaptation)={current_run_medium_query_confidence_thresh:.2f}")

    if valves.enable_adaptive_thresholds:
        if valves.debug_mode:
            debug_log.append(f"🧠 Adaptive thresholds ENABLED. Applying query/doc modifiers...")

        # Apply query complexity modifiers to sufficiency and novelty
        if query_complexity_level == QueryComplexity.SIMPLE:
            effective_sufficiency_threshold += valves.sufficiency_modifier_simple_query
            effective_novelty_threshold += valves.novelty_modifier_simple_query
        elif query_complexity_level == QueryComplexity.COMPLEX:
            effective_sufficiency_threshold += valves.sufficiency_modifier_complex_query
            effective_novelty_threshold += valves.novelty_modifier_complex_query

        # Clamp all thresholds to sensible ranges (e.g., 0.05 to 0.95)
        effective_sufficiency_threshold = max(0.05, min(0.95, effective_sufficiency_threshold))
        effective_novelty_threshold = max(0.05, min(0.95, effective_novelty_threshold))

        if valves.debug_mode:
            debug_log.append(f"   After query complexity mods: Eff.Sufficiency={effective_sufficiency_threshold:.2f}, Eff.Novelty={effective_novelty_threshold:.2f}")

    chunks = create_chunks(context, valves.chunk_size, valves.max_chunks)
    if not chunks and context:
        return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size."
    if not chunks and not context:
        conversation_log.append("ℹ️ No context or chunks to process with MinionS. Attempting direct call.")
        start_time_claude = 0
        if valves.debug_mode: 
            start_time_claude = asyncio.get_event_loop().time()
        try:
            final_response = await _call_claude_directly(valves, query, call_claude_func=call_claude)
            if valves.debug_mode:
                end_time_claude = asyncio.get_event_loop().time()
                time_taken_claude = end_time_claude - start_time_claude
                debug_log.append(f"⏱️ Claude direct call took {time_taken_claude:.2f}s. (Debug Mode)")
            output_parts = []
            if valves.show_conversation:
                output_parts.append("## 🗣️ MinionS Collaboration (Direct Call)")
                output_parts.extend(conversation_log)
                output_parts.append("---")
            if valves.debug_mode:
                output_parts.append("### 🔍 Debug Log")
                output_parts.extend(debug_log)
                output_parts.append("---")
            output_parts.append(f"## 🎯 Final Answer (Direct)\n{final_response}")
            return "\n".join(output_parts)
        except Exception as e:
            return f"❌ **Error in direct Claude call:** {str(e)}"

    total_chunks_processed_for_stats = len(chunks)

    # Initialize effective confidence threshold variables to store them for the performance report
    final_effective_simple_conf_thresh = current_run_simple_query_confidence_thresh
    final_effective_medium_conf_thresh = current_run_medium_query_confidence_thresh

    for current_round in range(current_run_max_rounds): # Use current_run_max_rounds
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {current_round + 1}/{current_run_max_rounds}... (Debug Mode)**") # Use current_run_max_rounds
        
        if valves.show_conversation:
            conversation_log.append(f"### 🎯 Round {current_round + 1}/{current_run_max_rounds} - Task Decomposition Phase") # Use current_run_max_rounds

        # Call the new decompose_task function
        # Note: now returns three values instead of two
        # DECOMPOSITION uses the user_query (potentially expanded)
        tasks, claude_response_for_decomposition, decomposition_prompt = await decompose_task(
            valves=valves,
            call_claude_func=call_claude,
            query=user_query, # Use the processed query
            scratchpad_content=scratchpad_content,
            num_chunks=len(chunks),
            max_tasks_per_round=valves.max_tasks_per_round,
            current_round=current_round + 1,
            conversation_log=conversation_log,
            debug_log=debug_log
        )
        
        # Store the decomposition prompt in history
        if decomposition_prompt:  # Only add if not empty (error case)
            decomposition_prompts_history.append(decomposition_prompt)
        
        # Handle Claude communication errors from decompose_task
        if claude_response_for_decomposition.startswith("CLAUDE_ERROR:"):
            error_message = claude_response_for_decomposition.replace("CLAUDE_ERROR: ", "")
            final_response = f"MinionS protocol failed during task decomposition: {error_message}"
            break

        # Log the raw Claude response if conversation is shown
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round + 1}):**\n{claude_response_for_decomposition}\n")

        # Check for "FINAL ANSWER READY."
        if "FINAL ANSWER READY." in claude_response_for_decomposition:
            final_response = claude_response_for_decomposition.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_provided_final_answer = True
            early_stopping_reason_for_output = "Claude provided FINAL ANSWER READY." # Explicitly set reason
            if valves.show_conversation: # This log already exists
                conversation_log.append(f"**🤖 Claude indicates final answer is ready in round {current_round + 1}.**")
            scratchpad_content += f"\n\n**Round {current_round + 1}:** Claude provided final answer. Stopping." # Added "Stopping."
            break

        if not tasks:
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Claude provided no new tasks in round {current_round + 1}. Proceeding to final synthesis.**")
            break
        
        total_tasks_executed_local += len(tasks)
        
        if valves.show_conversation:
            conversation_log.append(f"### ⚡ Round {current_round + 1} - Parallel Execution Phase (Processing {len(chunks)} chunks for {len(tasks)} tasks)")
        
        execution_details = await execute_tasks_on_chunks(
            tasks, chunks, conversation_log if valves.show_conversation else debug_log, 
            current_round + 1, valves, call_ollama_func, TaskResultModel
        )
        current_round_task_results = execution_details["results"]
        round_chunk_attempts = execution_details["total_chunk_processing_attempts"]
        round_chunk_timeouts = execution_details["total_chunk_processing_timeouts"]

        # Process Metrics After execute_tasks_on_chunks
        raw_metrics_data = execution_details.get("round_metrics_data")

        # Extract and Calculate Confidence Metrics
        confidence_data = execution_details.get("confidence_metrics_data")
        task_confidences = confidence_data.get("task_confidences", []) if confidence_data else []

        round_avg_numeric_confidence = 0.0
        if task_confidences:
            total_confidence_sum = sum(tc['avg_numeric_confidence'] for tc in task_confidences if tc.get('contributing_successful_chunks', 0) > 0)
            num_successful_tasks_with_confidence = sum(1 for tc in task_confidences if tc.get('contributing_successful_chunks', 0) > 0)
            if num_successful_tasks_with_confidence > 0:
                round_avg_numeric_confidence = total_confidence_sum / num_successful_tasks_with_confidence

        round_confidence_distribution = confidence_data.get("round_confidence_distribution", {"HIGH": 0, "MEDIUM": 0, "LOW": 0}) if confidence_data else {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        # Determine Confidence Trend (before creating current RoundMetrics object)
        confidence_trend = "N/A"
        if all_round_metrics: # Check if there are previous rounds' metrics
            previous_round_metric = all_round_metrics[-1]
            previous_avg_confidence = previous_round_metric.avg_confidence_score
            diff = round_avg_numeric_confidence - previous_avg_confidence
            if diff > 0.05:
                confidence_trend = "improving"
            elif diff < -0.05:
                confidence_trend = "declining"
            else:
                confidence_trend = "stable"

        if raw_metrics_data:
            # Process Findings for Redundancy Metrics
            current_round_new_findings = 0
            current_round_duplicate_findings = 0
            # current_round_fingerprints_seen_this_round = set() # Not strictly needed for plan's definition

            task_execution_results = execution_details.get("results", [])
            for task_result in task_execution_results:
                if task_result.get("status") == "success" and task_result.get("detailed_findings"):
                    for finding in task_result["detailed_findings"]:
                        fingerprint = finding.get("fingerprint")
                        if fingerprint:
                            # Check against global set first for duplicates from previous rounds or earlier in this round
                            if fingerprint in global_unique_fingerprints_seen:
                                current_round_duplicate_findings += 1
                            else:
                                current_round_new_findings += 1
                                global_unique_fingerprints_seen.add(fingerprint)
                            # current_round_fingerprints_seen_this_round.add(fingerprint)


            total_findings_this_round = current_round_new_findings + current_round_duplicate_findings
            redundancy_percentage_this_round = (current_round_duplicate_findings / total_findings_this_round) * 100 if total_findings_this_round > 0 else 0.0

            try: # Add a try-except block for robustness when creating RoundMetrics
                round_metric = RoundMetrics(
                    round_number=raw_metrics_data["round_number"],
                    tasks_executed=raw_metrics_data["tasks_executed"],
                    task_success_count=raw_metrics_data["task_success_count"],
                    task_failure_count=raw_metrics_data["task_failure_count"],
                    avg_chunk_processing_time_ms=raw_metrics_data["avg_chunk_processing_time_ms"],
                    # total_unique_findings_count=0,  # Placeholder for Iteration 1 REMOVED
                    execution_time_ms=raw_metrics_data["execution_time_ms"],
                    success_rate=raw_metrics_data["success_rate"],
                    # Add new confidence fields
                    avg_confidence_score=round_avg_numeric_confidence,
                    confidence_distribution=round_confidence_distribution,
                    confidence_trend=confidence_trend,
                    # Add new redundancy fields
                    new_findings_count_this_round=current_round_new_findings,
                    duplicate_findings_count_this_round=current_round_duplicate_findings,
                    redundancy_percentage_this_round=redundancy_percentage_this_round,
                    total_unique_findings_count=len(global_unique_fingerprints_seen)
                    # Sufficiency fields will be added below
                )
                all_round_metrics.append(round_metric) # Add before sufficiency update

                # --> Sufficiency Analysis <--
                metric_to_update = round_metric # This is the one we just added
                if valves.debug_mode:
                    debug_log.append(f"   [Debug] Updating Sufficiency Analyzer with scratchpad content for round {current_round + 1}...")

                analyzer.update_components(text_to_analyze=scratchpad_content, round_avg_confidence=metric_to_update.avg_confidence_score)
                sufficiency_details = analyzer.get_analysis_details()

                metric_to_update.sufficiency_score = sufficiency_details["sufficiency_score"]
                metric_to_update.component_coverage_percentage = sufficiency_details["component_coverage_percentage"]
                metric_to_update.information_components = sufficiency_details["information_components_status"]

                if valves.debug_mode:
                    debug_log.append(f"   [Debug] Sufficiency for round {current_round + 1}: Score={metric_to_update.sufficiency_score:.2f}, Coverage={metric_to_update.component_coverage_percentage:.2f}")
                    debug_log.append(f"   [Debug] Component Status: {metric_to_update.information_components}")

                # --> Convergence Detection Calculations (after sufficiency is updated) <--
                if metric_to_update: # Ensure we have the current round's metric object
                    previous_round_metric_obj = all_round_metrics[-2] if len(all_round_metrics) > 1 else None

                    convergence_calcs = convergence_detector.calculate_round_convergence_metrics(
                        current_round_metric=metric_to_update,
                        previous_round_metric=previous_round_metric_obj
                    )

                    metric_to_update.information_gain_rate = convergence_calcs.get("information_gain_rate", 0.0)
                    metric_to_update.novel_findings_percentage_this_round = convergence_calcs.get("novel_findings_percentage_this_round", 0.0)
                    metric_to_update.task_failure_rate_trend = convergence_calcs.get("task_failure_rate_trend", "N/A")
                    metric_to_update.predicted_value_of_next_round = convergence_calcs.get("predicted_value_of_next_round", "N/A")
                    # convergence_detected_this_round is set by check_for_convergence below

                    if valves.debug_mode:
                        debug_log.append(f"   [Debug] Convergence Detector calculated for round {metric_to_update.round_number}: InfoGain={metric_to_update.information_gain_rate:.0f}, Novelty={metric_to_update.novel_findings_percentage_this_round:.2%}, FailTrend={metric_to_update.task_failure_rate_trend}, NextRoundValue={metric_to_update.predicted_value_of_next_round}")

                # Format and append metrics summary (now includes redundancy, sufficiency, AND convergence calcs)
                component_status_summary = {k: ('Met' if v else 'Not Met') for k,v in metric_to_update.information_components.items()}
                metrics_summary = (
                    f"**📊 Round {metric_to_update.round_number} Metrics:**\n"
                    f"  - Tasks Executed: {metric_to_update.tasks_executed}, Success Rate: {metric_to_update.success_rate:.2%}\n"
                    f"  - Task Counts (S/F): {metric_to_update.task_success_count}/{metric_to_update.task_failure_count}\n"
                    f"  - Findings (New/Dup): {metric_to_update.new_findings_count_this_round}/{metric_to_update.duplicate_findings_count_this_round}, Total Unique: {metric_to_update.total_unique_findings_count}\n"
                    f"  - Redundancy This Round: {metric_to_update.redundancy_percentage_this_round:.1f}%\n"
                    f"  - Avg Confidence: {metric_to_update.avg_confidence_score:.2f} ({metric_to_update.confidence_trend})\n"
                    f"  - Confidence Dist (H/M/L): {metric_to_update.confidence_distribution.get('HIGH',0)}/{metric_to_update.confidence_distribution.get('MEDIUM',0)}/{metric_to_update.confidence_distribution.get('LOW',0)}\n"
                    f"  - Sufficiency Score: {metric_to_update.sufficiency_score:.2f}, Info Coverage: {metric_to_update.component_coverage_percentage:.2%}\n"
                    f"  - Components Status: {component_status_summary}\n"
                    f"  - Info Gain Rate: {metric_to_update.information_gain_rate:.0f}, Novelty This Round: {metric_to_update.novel_findings_percentage_this_round:.1%}\n"
                    f"  - Task Fail Trend: {metric_to_update.task_failure_rate_trend}, Predicted Next Round Value: {metric_to_update.predicted_value_of_next_round}\n"
                    f"  - Converged This Round: {'Yes' if metric_to_update.convergence_detected_this_round else 'No'}\n" # Will be updated by convergence check later
                    f"  - Round Time: {metric_to_update.execution_time_ms:.0f} ms, Avg Chunk Time: {metric_to_update.avg_chunk_processing_time_ms:.0f} ms"
                )
                scratchpad_content += f"\n\n{metrics_summary}"
                if valves.show_conversation:
                    conversation_log.append(metrics_summary)

            except KeyError as e:
                if valves.debug_mode:
                    debug_log.append(f"⚠️ **Metrics Error:** Missing key {e} in round_metrics_data for round {current_round + 1}. Skipping metrics for this round.")
            except Exception as e: # Catch any other validation error from Pydantic
                 if valves.debug_mode:
                    debug_log.append(f"⚠️ **Metrics Error:** Could not create RoundMetrics object for round {current_round + 1} due to {type(e).__name__}: {e}. Skipping metrics for this round.")


        if round_chunk_attempts > 0:
            timeout_percentage_this_round = (round_chunk_timeouts / round_chunk_attempts) * 100
            log_msg_timeout_stat = f"**📈 Round {current_round + 1} Local LLM Timeout Stats:** {round_chunk_timeouts}/{round_chunk_attempts} chunk calls timed out ({timeout_percentage_this_round:.1f}%)."
            if valves.show_conversation: 
                conversation_log.append(log_msg_timeout_stat)
            if valves.debug_mode: 
                debug_log.append(log_msg_timeout_stat)

            if timeout_percentage_this_round >= valves.max_round_timeout_failure_threshold_percent:
                warning_msg = f"⚠️ **Warning:** Round {current_round + 1} exceeded local LLM timeout threshold of {valves.max_round_timeout_failure_threshold_percent}%. Results from this round may be incomplete or unreliable."
                if valves.show_conversation: 
                    conversation_log.append(warning_msg)
                if valves.debug_mode: 
                    debug_log.append(warning_msg)
                scratchpad_content += f"\n\n**Note from Round {current_round + 1}:** High percentage of local model timeouts ({timeout_percentage_this_round:.1f}%) occurred, results for this round may be partial."
        
        round_summary_for_scratchpad_parts = []
        for r_val in current_round_task_results:
            status_icon = "✅" if r_val['status'] == 'success' else ("⏰" if 'timeout' in r_val['status'] else "❓")
            summary_text = f"- {status_icon} Task: {r_val['task']}, Result: {r_val['result'][:200]}..." if r_val['status'] == 'success' else f"- {status_icon} Task: {r_val['task']}, Status: {r_val['result']}"
            round_summary_for_scratchpad_parts.append(summary_text)
        
        if round_summary_for_scratchpad_parts:
            scratchpad_content += f"\n\n**Results from Round {current_round + 1}:**\n" + "\n".join(round_summary_for_scratchpad_parts)
        
        all_round_results_aggregated.extend(current_round_task_results)
        total_chunk_processing_timeouts_accumulated += round_chunk_timeouts

        # Placeholder for Sufficiency-Based Stopping Logic (Debug)
        # This is checked *before* other early stopping conditions like confidence thresholds.
        # The 'metric_to_update' variable should be the most up-to-date version of the current round's metrics.
        # It now includes sufficiency and initial convergence calculation fields.

        # --- First Round Novelty Adjustment (occurs only after round 0 processing) ---
        if current_round == 0 and valves.enable_adaptive_thresholds and 'metric_to_update' in locals() and metric_to_update:
            first_round_novelty_perc = metric_to_update.novel_findings_percentage_this_round
            if first_round_novelty_perc > valves.first_round_high_novelty_threshold:
                original_eff_sufficiency_before_1st_round_adj = effective_sufficiency_threshold
                effective_sufficiency_threshold += valves.sufficiency_modifier_high_first_round_novelty
                effective_sufficiency_threshold = max(0.05, min(0.95, effective_sufficiency_threshold)) # Clamp again
                if valves.debug_mode:
                    debug_log.append(
                        f"🧠 High first round novelty ({first_round_novelty_perc:.2%}) detected. "
                        f"Adjusting effective sufficiency threshold from {original_eff_sufficiency_before_1st_round_adj:.2f} to {effective_sufficiency_threshold:.2f}."
                    )

        if valves.debug_mode and 'metric_to_update' in locals() and metric_to_update:
            # Placeholder Debug for Sufficiency (already exists, using the dynamically adjusted threshold now)
            # This hypothetical threshold is just for this debug log, actual check uses effective_sufficiency_threshold
            debug_hypothetical_sufficiency_thresh_for_log = 0.75
            if metric_to_update.sufficiency_score >= debug_hypothetical_sufficiency_thresh_for_log:
                debug_log.append(
                    f"   [Debug Placeholder] Sufficiency score {metric_to_update.sufficiency_score:.2f} >= "
                    f"{debug_hypothetical_sufficiency_thresh_for_log} (hypothetical debug value). "
                    f"Effective sufficiency for convergence check is {effective_sufficiency_threshold:.2f}."
                )
            else:
                debug_log.append(
                    f"   [Debug Placeholder] Sufficiency score {metric_to_update.sufficiency_score:.2f} < "
                    f"{debug_hypothetical_sufficiency_thresh_for_log} (hypothetical debug value). "
                    f"Effective sufficiency for convergence check is {effective_sufficiency_threshold:.2f}."
                )

        # --> Convergence Check (for early stopping) <--
        # This comes before the original early stopping logic. If convergence is met, we stop.
        if 'metric_to_update' in locals() and metric_to_update and valves.enable_early_stopping:
            converged, convergence_reason = convergence_detector.check_for_convergence(
                current_round_metric=metric_to_update,
                sufficiency_score=metric_to_update.sufficiency_score, # Base sufficiency from analyzer
                total_rounds_executed=current_round + 1,
                effective_novelty_to_use=effective_novelty_threshold, # Pass calculated value
                effective_sufficiency_to_use=effective_sufficiency_threshold, # Pass calculated value
                valves=valves,
                all_round_metrics=all_round_metrics
            )
            if converged:
                metric_to_update.convergence_detected_this_round = True
                # Update the metrics_summary in scratchpad and conversation_log one last time with Converged=Yes
                # This is a bit repetitive but ensures the log reflects the final state that caused the stop.
                component_status_summary = {k: ('Met' if v else 'Not Met') for k,v in metric_to_update.information_components.items()}
                updated_metrics_summary_for_convergence_stop = (
                    f"**📊 Round {metric_to_update.round_number} Metrics (Final Update Before Convergence Stop):**\n"
                    f"  - Tasks Executed: {metric_to_update.tasks_executed}, Success Rate: {metric_to_update.success_rate:.2%}\n"
                    f"  - Task Counts (S/F): {metric_to_update.task_success_count}/{metric_to_update.task_failure_count}\n"
                    f"  - Findings (New/Dup): {metric_to_update.new_findings_count_this_round}/{metric_to_update.duplicate_findings_count_this_round}, Total Unique: {metric_to_update.total_unique_findings_count}\n"
                    f"  - Redundancy This Round: {metric_to_update.redundancy_percentage_this_round:.1f}%\n"
                    f"  - Avg Confidence: {metric_to_update.avg_confidence_score:.2f} ({metric_to_update.confidence_trend})\n"
                    f"  - Confidence Dist (H/M/L): {metric_to_update.confidence_distribution.get('HIGH',0)}/{metric_to_update.confidence_distribution.get('MEDIUM',0)}/{metric_to_update.confidence_distribution.get('LOW',0)}\n"
                    f"  - Sufficiency Score: {metric_to_update.sufficiency_score:.2f}, Info Coverage: {metric_to_update.component_coverage_percentage:.2%}\n"
                    f"  - Components Status: {component_status_summary}\n"
                    f"  - Info Gain Rate: {metric_to_update.information_gain_rate:.0f}, Novelty This Round: {metric_to_update.novel_findings_percentage_this_round:.1%}\n"
                    f"  - Task Fail Trend: {metric_to_update.task_failure_rate_trend}, Predicted Next Round Value: {metric_to_update.predicted_value_of_next_round}\n"
                    f"  - Converged This Round: {'Yes' if metric_to_update.convergence_detected_this_round else 'No'}\n"
                    f"  - Round Time: {metric_to_update.execution_time_ms:.0f} ms, Avg Chunk Time: {metric_to_update.avg_chunk_processing_time_ms:.0f} ms"
                )
                scratchpad_content += f"\n\n{updated_metrics_summary_for_convergence_stop}" # Append the final metrics to scratchpad
                if valves.show_conversation: # Replace the last metrics log with the fully updated one
                    if conversation_log and conversation_log[-1].startswith("**📊 Round"): conversation_log[-1] = updated_metrics_summary_for_convergence_stop
                    else: conversation_log.append(updated_metrics_summary_for_convergence_stop)

                early_stopping_reason_for_output = convergence_reason
                if valves.show_conversation:
                    conversation_log.append(f"**⚠️ Early Stopping Triggered (Convergence):** {convergence_reason}")
                if valves.debug_mode:
                    debug_log.append(f"**⚠️ Early Stopping Triggered (Convergence):** {convergence_reason} (Debug Mode)")
                scratchpad_content += f"\n\n**EARLY STOPPING (Convergence Round {current_round + 1}):** {convergence_reason}"
                if valves.debug_mode:
                     debug_log.append(f"**🏁 Breaking loop due to convergence in Round {current_round + 1}. (Debug Mode)**")
                break # Exit the round loop

        # Original Early Stopping Logic (Confidence-based)
        # This will only be reached if convergence was NOT met and we didn't break above.
        # Note: 'round_metric' is the original metric object from raw_metrics_data,
        # 'metric_to_update' is the same object, but after being updated with sufficiency.
        # So, using 'metric_to_update' here for consistency if we were to integrate sufficiency into this logic.
        # However, the current early stopping is based on avg_confidence_score which is set before sufficiency.
        # For now, we keep `round_metric` for the existing logic as it was originally.
        # If sufficiency were to be a primary driver for early stopping, this would need refactoring.
        if valves.enable_early_stopping and round_metric: # Ensure round_metric exists
            stop_early = False
            stopping_reason = ""

            # Ensure we've met the minimum number of rounds
            if (current_round + 1) >= valves.min_rounds_before_stopping:
                current_avg_confidence = round_metric.avg_confidence_score # This is from the current round's raw metrics

                # Determine effective confidence threshold for current query type
                current_query_type_base_confidence_threshold = 0.0 # This will be set to the current_run_... value
                threshold_name_for_log = "N/A"
                # Store the calculated effective confidence threshold for the performance report
                # Initialize with base, then adapt if needed
                effective_confidence_threshold_for_query_this_check = 0.0


                if query_complexity_level == QueryComplexity.SIMPLE:
                    current_query_type_base_confidence_threshold = current_run_simple_query_confidence_thresh # Use current_run_
                    threshold_name_for_log = "Simple Query Confidence"
                    effective_confidence_threshold_for_query_this_check = current_query_type_base_confidence_threshold
                    if valves.enable_adaptive_thresholds:
                        if doc_size_category == "small":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_small_doc
                        elif doc_size_category == "large":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_large_doc
                        effective_confidence_threshold_for_query_this_check = max(0.05, min(0.95, effective_confidence_threshold_for_query_this_check))
                    final_effective_simple_conf_thresh = effective_confidence_threshold_for_query_this_check # Store for report

                elif query_complexity_level == QueryComplexity.MEDIUM:
                    current_query_type_base_confidence_threshold = current_run_medium_query_confidence_thresh # Use current_run_
                    threshold_name_for_log = "Medium Query Confidence"
                    effective_confidence_threshold_for_query_this_check = current_query_type_base_confidence_threshold
                    if valves.enable_adaptive_thresholds:
                        if doc_size_category == "small":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_small_doc
                        elif doc_size_category == "large":
                            effective_confidence_threshold_for_query_this_check += valves.confidence_modifier_large_doc
                        effective_confidence_threshold_for_query_this_check = max(0.05, min(0.95, effective_confidence_threshold_for_query_this_check))
                    final_effective_medium_conf_thresh = effective_confidence_threshold_for_query_this_check # Store for report

                if valves.debug_mode and query_complexity_level != QueryComplexity.COMPLEX:
                     debug_log.append(f"   [Debug] Adaptive Confidence Check: QueryType={query_complexity_level.value}, BaseThreshForProfile={current_query_type_base_confidence_threshold:.2f}, EffectiveThreshForStopCheck={effective_confidence_threshold_for_query_this_check:.2f} (DocSize: {doc_size_category})")

                if query_complexity_level != QueryComplexity.COMPLEX and current_avg_confidence >= effective_confidence_threshold_for_query_this_check:
                    stop_early = True
                    stopping_reason = (
                        f"{query_complexity_level.value} query confidence ({current_avg_confidence:.2f}) "
                        f"met/exceeded effective threshold ({effective_confidence_threshold_for_query_this_check:.2f}) "
                        f"after round {current_round + 1}."
                    )
                # No specific confidence-based early stopping rule for COMPLEX queries; they run max_rounds or until convergence.

            if stop_early:
                if valves.show_conversation:
                    conversation_log.append(f"**⚠️ Early Stopping Triggered:** {stopping_reason}")
                if valves.debug_mode:
                    debug_log.append(f"**⚠️ Early Stopping Triggered:** {stopping_reason} (Debug Mode)")

                early_stopping_reason_for_output = stopping_reason # Store it for final output
                scratchpad_content += f"\n\n**EARLY STOPPING TRIGGERED (Round {current_round + 1}):** {stopping_reason}"
                # Add a final log message before breaking, as the "Completed Round" message will be skipped
                if valves.debug_mode:
                     debug_log.append(f"**🏁 Breaking loop due to early stopping in Round {current_round + 1}. (Debug Mode)**")
                break # Exit the round loop

        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {current_round + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**")

        if current_round == current_run_max_rounds - 1: # Use current_run_max_rounds
            if valves.show_conversation:
                conversation_log.append(f"**🏁 Reached max rounds ({current_run_max_rounds}). Proceeding to final synthesis.**") # Use current_run_max_rounds

    if not claude_provided_final_answer:
        if valves.show_conversation:
            conversation_log.append("\n### 🔄 Final Synthesis Phase")
        if not all_round_results_aggregated:
            final_response = "No information was gathered from the document by local models across the rounds."
            if valves.show_conversation:
                conversation_log.append(f"**🤖 Claude (Synthesis):** {final_response}")
        else:
            synthesis_input_summary = "\n".join([f"- Task: {r['task']}\n  Result: {r['result']}" for r in all_round_results_aggregated if r['status'] == 'success'])
            if not synthesis_input_summary:
                # Use original query here for context if no results, as it's the user's direct input
                synthesis_input_summary = "No definitive information was found by local models. The original query was: " + user_query_original
            
            # Synthesis prompt should use the user_query (potentially expanded) that tasks were based on,
            # but the original query might also be useful context for Claude if expansion was heavy.
            # For now, stick to user_query for consistency with decomposition.
            synthesis_prompt = get_minions_synthesis_claude_prompt(user_query, synthesis_input_summary, valves)
            synthesis_prompts_history.append(synthesis_prompt)
            
            start_time_claude_synth = 0
            if valves.debug_mode:
                start_time_claude_synth = asyncio.get_event_loop().time()
            try:
                final_response = await call_claude(valves, synthesis_prompt)
                if valves.debug_mode:
                    end_time_claude_synth = asyncio.get_event_loop().time()
                    time_taken_claude_synth = end_time_claude_synth - start_time_claude_synth
                    debug_log.append(f"⏱️ Claude call (Final Synthesis) took {time_taken_claude_synth:.2f}s. (Debug Mode)")
                if valves.show_conversation:
                    conversation_log.append(f"**🤖 Claude (Final Synthesis):**\n{final_response}")
            except Exception as e:
                if valves.show_conversation:
                    conversation_log.append(f"❌ Error during final synthesis: {e}")
                final_response = "Error during final synthesis. Raw findings might be available in conversation log."
    
    output_parts = []
    if valves.show_conversation:
        output_parts.append("## 🗣️ MinionS Collaboration (Multi-Round)")
        output_parts.extend(conversation_log)
        output_parts.append("---")
    if valves.debug_mode:
        output_parts.append("### 🔍 Debug Log")
        output_parts.extend(debug_log)
        output_parts.append("---")
    output_parts.append(f"## 🎯 Final Answer")
    output_parts.append(final_response)

    summary_for_stats = synthesis_input_summary if not claude_provided_final_answer else scratchpad_content

    stats = calculate_token_savings(
        decomposition_prompts_history, synthesis_prompts_history, # Prompts
        summary_for_stats, final_response, # Responses
        len(context), len(user_query_original), # Lengths (use original query length for user perspective)
        total_chunks_processed_for_stats, total_tasks_executed_local # Counts
    )
    
    # Override the total_rounds if needed to show actual rounds executed
    actual_rounds_executed = len(decomposition_prompts_history) # Number of rounds for which decomposition prompts were made
    # If loop broke early, current_round might be less than actual_rounds_executed -1
    # If no decomposition prompts (e.g. direct call, or error before first decomp), then 0.
    # If loop completed, actual_rounds_executed should be current_run_max_rounds (if prompts were made each round)
    # Or, if loop broke, it's the number of rounds that *started* decomposition.
    # A simple way: if all_round_metrics exists, it's len(all_round_metrics)
    if all_round_metrics: # This is a more reliable count of rounds that completed metric generation
        actual_rounds_executed = len(all_round_metrics)
    elif actual_rounds_executed == 0 and current_round >=0: # Fallback if decomp history is empty but loop ran
         actual_rounds_executed = min(current_round + 1, current_run_max_rounds)


    # --- Performance Report Section ---
    performance_report_parts = ["\n## 📝 Performance Report"]
    performance_report_parts.append(f"- **Total rounds executed:** {actual_rounds_executed} / {current_run_max_rounds} (Profile Max)")
    performance_report_parts.append(f"- **Stopping reason:** {early_stopping_reason_for_output if early_stopping_reason_for_output else 'Max rounds reached or no further tasks.'}")

    last_metric = all_round_metrics[-1] if all_round_metrics else None
    if last_metric:
        performance_report_parts.append(f"- **Final Sufficiency Score:** {getattr(last_metric, 'sufficiency_score', 'N/A'):.2f}")
        performance_report_parts.append(f"- **Final Component Coverage:** {getattr(last_metric, 'component_coverage_percentage', 'N/A'):.2%}")
        performance_report_parts.append(f"- **Final Information Components Status:** {str(getattr(last_metric, 'information_components', {}))}")
        performance_report_parts.append(f"- **Final Convergence Detected:** {'Yes' if getattr(last_metric, 'convergence_detected_this_round', False) else 'No'}")
    else:
        performance_report_parts.append("- *No round metrics available for final values.*")

    performance_report_parts.append("- **Effective Thresholds Used (Final Values):**")
    performance_report_parts.append(f"  - Sufficiency (for convergence): {effective_sufficiency_threshold:.2f}")
    performance_report_parts.append(f"  - Novelty (for convergence): {effective_novelty_threshold:.2f}")
    if query_complexity_level == QueryComplexity.SIMPLE:
        performance_report_parts.append(f"  - Confidence (for simple query early stop): {final_effective_simple_conf_thresh:.2f}")
    elif query_complexity_level == QueryComplexity.MEDIUM:
        performance_report_parts.append(f"  - Confidence (for medium query early stop): {final_effective_medium_conf_thresh:.2f}")
    else: # COMPLEX
        performance_report_parts.append(f"  - Confidence (early stopping not applicable for COMPLEX queries based on this threshold type)")
    
    performance_report_parts.append(f"- **Performance Profile Applied:** {valves.performance_profile}")
    performance_report_parts.append(f"- **Adaptive Thresholds Enabled:** {'Yes' if valves.enable_adaptive_thresholds else 'No'}")
    performance_report_parts.append(f"- **Document Size Category:** {doc_size_category}")


    output_parts.extend(performance_report_parts)
    if valves.debug_mode:
        debug_log.append("\n--- Performance Report (Debug Copy) ---")
        debug_log.extend(performance_report_parts)
        debug_log.append("--- End Performance Report (Debug Copy) ---")

    output_parts.append(f"\n## 📊 MinionS Efficiency Stats (v0.2.0)")
    output_parts.append(f"- **Protocol:** MinionS (Multi-Round)")
    output_parts.append(f"- **Query Complexity:** {query_complexity_level.value}")
    output_parts.append(f"- **Rounds executed (Profile Max):** {actual_rounds_executed}/{current_run_max_rounds}") # Use current_run_max_rounds
    output_parts.append(f"- **Total tasks for local LLM:** {stats['total_tasks_executed_local']}")

    # --- Explicitly define variables for the MinionS Efficiency Stats block ---
    # Ensure 'all_round_results_aggregated' is the correct list of task results.
    # And 'valves' and 'debug_log' are assumed to be accessible in this scope for the warning.

    explicit_total_successful_tasks = 0
    explicit_tasks_with_any_timeout = 0 # Renamed for clarity from tasks_with_any_timeout

    if 'all_round_results_aggregated' in locals() and isinstance(all_round_results_aggregated, list):
        explicit_total_successful_tasks = len([
            r for r in all_round_results_aggregated if isinstance(r, dict) and r.get('status') == 'success'
        ])
        explicit_tasks_with_any_timeout = len([
            r for r in all_round_results_aggregated if isinstance(r, dict) and r.get('status') == 'timeout_all_chunks'
        ])
    else:
        # This case should ideally not happen if the protocol ran correctly.
        # Adding a log if debug_mode is on and valves is accessible.
        if 'valves' in locals() and hasattr(valves, 'debug_mode') and valves.debug_mode and 'debug_log' in locals():
            debug_log.append("⚠️ Warning: 'all_round_results_aggregated' not found or not a list when calculating final efficiency stats.")
    # --- End of explicit definitions ---

    output_parts.append(f"- **Successful tasks (local):** {explicit_total_successful_tasks}")
    output_parts.append(f"- **Tasks where all chunks timed out (local):** {explicit_tasks_with_any_timeout}")
    output_parts.append(f"- **Total individual chunk processing timeouts (local):** {total_chunk_processing_timeouts_accumulated}")
    output_parts.append(f"- **Chunks processed per task (local):** {stats['total_chunks_processed_local'] if stats['total_tasks_executed_local'] > 0 else 0}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    if early_stopping_reason_for_output:
        output_parts.append(f"- **Early Stopping Triggered:** {early_stopping_reason_for_output}")
    output_parts.append(f"\n## 💰 Token Savings Analysis (Claude: {valves.remote_model})")
    output_parts.append(f"- **Traditional single call (est.):** ~{stats['traditional_tokens_claude']:,} tokens")
    output_parts.append(f"- **MinionS multi-round (Claude only):** ~{stats['minions_tokens_claude']:,} tokens")
    output_parts.append(f"- **💰 Est. Claude Token savings:** ~{stats['percentage_savings_claude']:.1f}%")
    
    return "\n".join(output_parts)

async def minions_pipe_method(
    pipe_self: Any,
    body: dict,
    __user__: dict,
    __request__: Request,
    __files__: List[Dict[str, Any]] = [],
    __pipe_id__: str = "minions-claude",
) -> str:
    """Execute the MinionS protocol with Claude"""
    try:
        # Validate configuration
        if not pipe_self.valves.anthropic_api_key:
            return "❌ **Error:** Please configure your Anthropic API key (and Ollama settings if applicable) in the function settings."

        # Extract user message and context
        messages: List[Dict[str, Any]] = body.get("messages", [])
        if not messages:
            return "❌ **Error:** No messages provided."

        user_query: str = messages[-1]["content"]

        # Extract context from messages AND uploaded files
        context_from_messages: str = extract_context_from_messages(messages[:-1])
        context_from_files: str = await extract_context_from_files(pipe_self.valves, __files__)

        # Combine all context sources
        all_context_parts: List[str] = []
        if context_from_messages:
            all_context_parts.append(f"=== CONVERSATION CONTEXT ===\n{context_from_messages}")
        if context_from_files:
            all_context_parts.append(f"=== UPLOADED DOCUMENTS ===\n{context_from_files}")

        context: str = "\n\n".join(all_context_parts) if all_context_parts else ""

        # If no context, make a direct call to Claude
        if not context:
            direct_response = await _call_claude_directly(pipe_self.valves, user_query, call_claude_func=call_claude)
            return (
                "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                + direct_response
            )

        # Execute the MinionS protocol with correct parameter names
        result: str = await _execute_minions_protocol(
            pipe_self.valves, 
            user_query, 
            context, 
            call_claude,    # Changed from call_claude_func
            call_ollama,    # Changed from call_ollama_func
            TaskResult      # Changed from TaskResultModel
        )
        return result

    except Exception as e:
        import traceback
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in MinionS protocol:** {error_details}"


class Pipe:
    class Valves(MinionsValves):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "MinionS v0.3.4b (Task Decomposition)"

    def pipes(self):
        """Define the available models"""
        return [
            {
                "id": "minions-claude",
                "name": f" ({self.valves.local_model} + {self.valves.remote_model})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "minions-claude",
    ) -> str:
        """Execute the MinionS protocol with Claude"""
        return await minions_pipe_method(self, body, __user__, __request__, __files__, __pipe_id__)