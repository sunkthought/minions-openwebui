"""
title: Minion Protocol Integration for Open WebUI
author: Wil Everts and the @SunkThought team
author_url: https://github.com/SunkThought/minions-openwebui
original_author: Copyright (c) 2025 Sabri Eyuboglu, Avanika Narayan, Dan Biderman, and the rest of the Minions team (@HazyResearch wrote the original MinionS Protocol paper and code examples on github that spawned this)
original_author_url: https://github.com/HazyResearch/
funding_url: https://github.com/HazyResearch/minions
version: 0.3.6b
description: Basic Minion protocol - conversational collaboration between local and cloud models
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

# Partials File: partials/minion_models.py
from typing import List, Optional, Dict, Set, Any, Tuple
from pydantic import BaseModel, Field
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

    # The following class is part of the Pydantic configuration and is standard.
    # It ensures that extra fields passed to the model are ignored rather than causing an error.
    class Config:
        extra = "ignore"


# Partials File: partials/common_api_calls.py
import aiohttp
import json
from typing import Optional, Dict, Set
from pydantic import BaseModel

# Known models that support JSON/structured output
STRUCTURED_OUTPUT_CAPABLE_MODELS: Set[str] = {
    "llama3.2", "llama3.1", "llama3", "llama2",
    "mistral", "mixtral", "mistral-nemo",
    "qwen2", "qwen2.5", 
    "gemma2", "gemma",
    "phi3", "phi",
    "command-r", "command-r-plus",
    "deepseek-coder", "deepseek-coder-v2",
    "codellama",
    "dolphin-llama3", "dolphin-mixtral",
    "solar", "starling-lm",
    "yi", "zephyr",
    "neural-chat", "openchat"
}

def model_supports_structured_output(model_name: str) -> bool:
    """Check if a model is known to support structured output"""
    if not model_name:
        return False
    
    # Normalize model name for comparison
    model_lower = model_name.lower()
    
    # Check exact matches first
    if model_lower in STRUCTURED_OUTPUT_CAPABLE_MODELS:
        return True
    
    # Check partial matches (for versioned models like llama3.2:1b)
    for known_model in STRUCTURED_OUTPUT_CAPABLE_MODELS:
        if model_lower.startswith(known_model):
            return True
    
    return False

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
    options = {
        "temperature": getattr(valves, 'local_model_temperature', 0.7),
        "num_predict": valves.ollama_num_predict,
        "num_ctx": getattr(valves, 'local_model_context_length', 4096),
    }
    
    # Add top_k only if it's greater than 0
    top_k = getattr(valves, 'local_model_top_k', 40)
    if top_k > 0:
        options["top_k"] = top_k
    
    payload = {
        "model": valves.local_model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    # Check if we should use structured output
    should_use_structured = (
        use_json and 
        hasattr(valves, 'use_structured_output') and 
        valves.use_structured_output and 
        schema and
        model_supports_structured_output(valves.local_model)
    )
    
    if should_use_structured:
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
    elif use_json and hasattr(valves, 'use_structured_output') and valves.use_structured_output:
        # Model doesn't support structured output but it was requested
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            print(f"DEBUG: Model '{valves.local_model}' does not support structured output. Using text-based parsing fallback.")
    
    if "format" in payload and not should_use_structured:
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


# Partials File: partials/minion_prompts.py
from typing import List, Tuple, Any, Optional

# This file will store prompt generation functions for the Minion (single-turn) protocol.

def get_minion_initial_claude_prompt(query: str, context_len: int, valves: Any) -> str:
    """
    Returns the initial prompt for Claude in the Minion protocol.
    Enhanced with better question generation guidance.
    """
    # Escape any quotes in the query to prevent f-string issues
    escaped_query = query.replace('"', '\\"').replace("'", "\\'")
    
    return f'''You are a research coordinator working with a knowledgeable local assistant who has access to specific documents.

Your task: Gather information to answer the user's query by asking strategic questions.

USER'S QUERY: "{escaped_query}"

The local assistant has FULL ACCESS to the relevant document ({context_len} characters long) and will provide factual information extracted from it.

Guidelines for effective questions:
1. Ask ONE specific, focused question at a time
2. Build upon previous answers to go deeper
3. Avoid broad questions like "What does the document say?" 
4. Good: "What are the specific budget allocations for Q2?"
   Poor: "Tell me about the budget"
5. Track what you've learned to avoid redundancy

When to conclude:
- Start your response with "I now have sufficient information" when ready to provide the final answer
- You have {valves.max_rounds} rounds maximum to gather information

QUESTION STRATEGY TIPS:
- For factual queries: Ask for specific data points, dates, numbers, or names
- For analytical queries: Ask about relationships, comparisons, or patterns
- For summary queries: Ask about key themes, main points, or conclusions
- For procedural queries: Ask about steps, sequences, or requirements

Remember: The assistant can only see the document, not your conversation history.

If you have gathered enough information to answer "{escaped_query}", respond with "FINAL ANSWER READY." followed by your comprehensive answer.

Otherwise, ask your first strategic question to the local assistant.'''

def get_minion_conversation_claude_prompt(history: List[Tuple[str, str]], original_query: str, valves: Any) -> str:
    """
    Returns the prompt for Claude during subsequent conversation rounds in the Minion protocol.
    Enhanced with better guidance for follow-up questions.
    """
    # Escape the original query
    escaped_query = original_query.replace('"', '\\"').replace("'", "\\'")
    
    current_round = len(history) // 2 + 1
    rounds_remaining = valves.max_rounds - current_round
    
    context_parts = [
        f'You are continuing to gather information to answer: "{escaped_query}"',
        f"Round {current_round} of {valves.max_rounds}",
        "",
        "INFORMATION GATHERED SO FAR:",
    ]

    for i, (role, message) in enumerate(history):
        if role == "assistant":  # Claude's previous message
            context_parts.append(f'\nQ{i//2 + 1}: {message}')
        else:  # Local model's response
            # Extract key information if structured
            if isinstance(message, str) and message.startswith('{'):
                context_parts.append(f'A{i//2 + 1}: {message}')
            else:
                context_parts.append(f'A{i//2 + 1}: {message}')

    context_parts.extend(
        [
            "",
            "DECISION POINT:",
            "Evaluate if you have sufficient information to answer the original question comprehensively.",
            "",
            "✅ If YES: Start with 'FINAL ANSWER READY.' then provide your complete answer",
            f"❓ If NO: Ask ONE more strategic question (you have {rounds_remaining} rounds left)",
            "",
            "TIPS FOR YOUR NEXT QUESTION:",
            "- What specific gaps remain in your understanding?",
            "- Can you drill deeper into any mentioned topics?",
            "- Are there related aspects you haven't explored?",
            "- Would examples or specific details strengthen your answer?",
            "",
            "Remember: Each question should build on what you've learned, not repeat previous inquiries.",
        ]
    )
    return "\n".join(context_parts)

def get_minion_local_prompt(context: str, query: str, claude_request: str, valves: Any) -> str:
    """
    Returns the prompt for the local Ollama model in the Minion protocol.
    Enhanced with better guidance for structured, useful responses.
    """
    # query is the original user query.
    # context is the document chunk.
    # claude_request (the parameter) is the specific question from the remote model to the local model.

    base_prompt = f"""You are a document analysis assistant with exclusive access to the following document:

<document>
{context}
</document>

A research coordinator needs specific information from this document to answer: "{query}"

Their current question is:
<question>
{claude_request}
</question>

RESPONSE GUIDELINES:

1. ACCURACY: Base your answer ONLY on information found in the document above
   
2. CITATIONS: When possible, include direct quotes or specific references:
   - Good: "According to section 3.2, 'the budget increased by 15%'"
   - Good: "The document states on page 4 that..."
   - Poor: "The document mentions something about budgets"

3. ORGANIZATION: For complex answers, structure your response:
   - Use bullet points or numbered lists for multiple items
   - Separate distinct pieces of information clearly
   - Highlight key findings at the beginning

4. CONFIDENCE LEVELS:
   - HIGH: Information directly answers the question with explicit statements
   - MEDIUM: Information partially addresses the question or requires some inference
   - LOW: Information is tangentially related or requires significant interpretation

5. HANDLING MISSING INFORMATION:
   - If not found: "This specific information is not available in the document"
   - If partially found: "The document provides partial information: [explain what's available]"
   - Suggest related info: "While X is not mentioned, the document does discuss Y which may be relevant"

Remember: The coordinator cannot see the document and relies entirely on your accurate extraction."""

    if valves.use_structured_output:
        structured_output_instructions = """

RESPONSE FORMAT:
Respond ONLY with a JSON object in this exact format:
{
    "answer": "Your detailed answer addressing the specific question",
    "confidence": "HIGH/MEDIUM/LOW",
    "key_points": ["Main finding 1", "Main finding 2", "..."] or null,
    "citations": ["Exact quote from document", "Another relevant quote", "..."] or null
}

JSON Guidelines:
- answer: Comprehensive response to the question (required)
- confidence: Your assessment based on criteria above (required)
- key_points: List main findings if multiple important points exist (optional)
- citations: Direct quotes that support your answer (optional but recommended)

IMPORTANT: Output ONLY the JSON object. No additional text, no markdown formatting."""
        return base_prompt + structured_output_instructions
    else:
        non_structured_instructions = """

Format your response clearly with:
- Main answer first
- Supporting details or quotes
- Confidence level (HIGH/MEDIUM/LOW) at the end
- Note if any information is not found in the document"""
        return base_prompt + non_structured_instructions

def get_minion_initial_claude_prompt_with_state(query: str, context_len: int, valves: Any, conversation_state: Optional[Any] = None) -> str:
    """
    Enhanced version of initial prompt that includes conversation state if available.
    """
    base_prompt = get_minion_initial_claude_prompt(query, context_len, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        if state_summary:
            base_prompt = base_prompt.replace(
                "Otherwise, ask your first strategic question to the local assistant.",
                f"""
CONVERSATION STATE CONTEXT:
{state_summary}

Based on this context, ask your first strategic question to the local assistant."""
            )
    
    return base_prompt

def get_minion_conversation_claude_prompt_with_state(history: List[Tuple[str, str]], original_query: str, valves: Any, conversation_state: Optional[Any] = None, previous_questions: Optional[List[str]] = None) -> str:
    """
    Enhanced version of conversation prompt that includes conversation state if available.
    """
    base_prompt = get_minion_conversation_claude_prompt(history, original_query, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        
        # Insert state summary before decision point
        state_section = f"""
CURRENT CONVERSATION STATE:
{state_summary}

TOPICS COVERED: {', '.join(conversation_state.topics_covered) if conversation_state.topics_covered else 'None yet'}
KEY FINDINGS COUNT: {len(conversation_state.key_findings)}
INFORMATION GAPS: {len(conversation_state.information_gaps)}
"""
        
        base_prompt = base_prompt.replace(
            "DECISION POINT:",
            state_section + "\nDECISION POINT:"
        )
    
    # Add deduplication guidance if previous questions provided
    if previous_questions and valves.enable_deduplication:
        questions_section = "\nPREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT):\n"
        for i, q in enumerate(previous_questions, 1):
            questions_section += f"{i}. {q}\n"
        
        questions_section += "\n⚠️ IMPORTANT: Avoid asking questions that are semantically similar to the above. Each new question should explore genuinely new information.\n"
        
        base_prompt = base_prompt.replace(
            "Remember: Each question should build on what you've learned, not repeat previous inquiries.",
            questions_section + "Remember: Each question should build on what you've learned, not repeat previous inquiries."
        )
    
    return base_prompt

# Partials File: partials/minion_protocol_logic.py
import asyncio
import json
import re
from typing import List, Dict, Any, Tuple, Callable

def _calculate_token_savings(conversation_history: List[Tuple[str, str]], context: str, query: str) -> dict:
    """Calculate token savings for the Minion protocol"""
    chars_per_token = 3.5
    
    # Traditional approach: entire context + query sent to remote model
    traditional_tokens = int((len(context) + len(query)) / chars_per_token)
    
    # Minion approach: only conversation messages sent to remote model
    conversation_content = " ".join(
        [msg[1] for msg in conversation_history if msg[0] == "assistant"]
    )
    minion_tokens = int(len(conversation_content) / chars_per_token)
    
    # Calculate savings
    token_savings = traditional_tokens - minion_tokens
    percentage_savings = (
        (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    )
    
    return {
        'traditional_tokens': traditional_tokens,
        'minion_tokens': minion_tokens,
        'token_savings': token_savings,
        'percentage_savings': percentage_savings
    }

def _is_final_answer(response: str) -> bool:
    """Check if response contains the specific final answer marker."""
    return "FINAL ANSWER READY." in response

def detect_completion(response: str) -> bool:
    """Check if remote model indicates it has sufficient information"""
    completion_phrases = [
        "i now have sufficient information",
        "i can now answer",
        "based on the information gathered",
        "i have enough information",
        "with this information, i can provide",
        "i can now provide a comprehensive answer",
        "based on what the local assistant has told me"
    ]
    response_lower = response.lower()
    
    # Check for explicit final answer marker first
    if "FINAL ANSWER READY." in response:
        return True
    
    # Check for completion phrases
    return any(phrase in response_lower for phrase in completion_phrases)

def _parse_local_response(response: str, is_structured: bool, use_structured_output: bool, debug_mode: bool, LocalAssistantResponseModel: Any) -> Dict:
    """Parse local model response, supporting both text and structured formats."""
    confidence_map = {'HIGH': 0.9, 'MEDIUM': 0.6, 'LOW': 0.3}
    default_numeric_confidence = 0.3  # Corresponds to LOW
    
    if is_structured and use_structured_output:
        # Clean up common formatting issues
        cleaned_response = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[7:-3].strip()
        elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
        
        # Try to extract JSON from response with explanatory text
        if not cleaned_response.startswith("{"):
            # Look for JSON object in the response
            json_start = cleaned_response.find("{")
            json_end = cleaned_response.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_response = cleaned_response[json_start:json_end+1]
        
        try:
            parsed_json = json.loads(cleaned_response)
            
            # Handle missing confidence field with default
            if 'confidence' not in parsed_json:
                parsed_json['confidence'] = 'LOW'
            
            # Ensure required fields have defaults if missing
            if 'answer' not in parsed_json:
                parsed_json['answer'] = None
            if 'key_points' not in parsed_json:
                parsed_json['key_points'] = None
            if 'citations' not in parsed_json:
                parsed_json['citations'] = None
            
            validated_model = LocalAssistantResponseModel(**parsed_json)
            model_dict = validated_model.dict()
            model_dict['parse_error'] = None
            
            # Add numeric confidence for consistency
            text_confidence = model_dict.get('confidence', 'LOW').upper()
            model_dict['numeric_confidence'] = confidence_map.get(text_confidence, default_numeric_confidence)
            
            return model_dict
            
        except json.JSONDecodeError as e:
            if debug_mode:
                print(f"DEBUG: JSON decode error in Minion: {e}. Cleaned response was: {cleaned_response[:500]}")
            
            # Try regex fallback to extract key information
            import re
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
            confidence_match = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', response, re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1)
                confidence = confidence_match.group(1).upper() if confidence_match else "LOW"
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "numeric_confidence": confidence_map.get(confidence, default_numeric_confidence),
                    "key_points": None,
                    "citations": None,
                    "parse_error": f"JSON parse error (recovered): {str(e)}"
                }
            
            # Complete fallback
            return {
                "answer": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence,
                "key_points": None,
                "citations": None,
                "parse_error": str(e)
            }
        except Exception as e:
            if debug_mode:
                print(f"DEBUG: Failed to parse structured output in Minion: {e}. Response was: {response[:500]}")
            return {
                "answer": response, 
                "confidence": "LOW", 
                "numeric_confidence": default_numeric_confidence,
                "key_points": None, 
                "citations": None, 
                "parse_error": str(e)
            }
    
    # Fallback for non-structured processing
    return {
        "answer": response, 
        "confidence": "MEDIUM", 
        "numeric_confidence": confidence_map.get("MEDIUM", 0.6),
        "key_points": None, 
        "citations": None, 
        "parse_error": None
    }

async def _execute_minion_protocol(
    valves: Any,
    query: str,
    context: str,
    call_claude_func: Callable,
    call_ollama_func: Callable,
    LocalAssistantResponseModel: Any,
    ConversationStateModel: Any = None,
    QuestionDeduplicatorModel: Any = None
) -> str:
    """Execute the Minion protocol"""
    conversation_log = []
    debug_log = []
    conversation_history = []
    actual_final_answer = "No final answer was explicitly provided by the remote model."
    claude_declared_final = False
    
    # Initialize conversation state if enabled
    conversation_state = None
    if valves.track_conversation_state and ConversationStateModel:
        conversation_state = ConversationStateModel()
    
    # Initialize question deduplicator if enabled
    deduplicator = None
    if valves.enable_deduplication and QuestionDeduplicatorModel:
        deduplicator = QuestionDeduplicatorModel(
            similarity_threshold=valves.deduplication_threshold
        )
    
    # Initialize metrics tracking
    overall_start_time = asyncio.get_event_loop().time()
    metrics = {
        'confidence_scores': [],
        'confidence_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
        'rounds_completed': 0,
        'completion_via_detection': False,
        'estimated_tokens': 0,
        'chunk_size_used': valves.chunk_size,
        'context_size': len(context)
    }

    if valves.debug_mode:
        debug_log.append(f"🔍 **Debug Info (Minion v0.3.6b):**")
        debug_log.append(f"  - Query: {query[:100]}...")
        debug_log.append(f"  - Context length: {len(context)} chars")
        debug_log.append(f"  - Max rounds: {valves.max_rounds}")
        debug_log.append(f"  - Remote model: {valves.remote_model}")
        debug_log.append(f"  - Local model: {valves.local_model}")
        debug_log.append(f"  - Timeouts: Remote={valves.timeout_claude}s, Local={valves.timeout_local}s")
        debug_log.append(f"**⏱️ Overall process started. (Debug Mode)**\n")

    for round_num in range(valves.max_rounds):
        if valves.debug_mode:
            debug_log.append(f"**⚙️ Starting Round {round_num + 1}/{valves.max_rounds}... (Debug Mode)**")
        
        if valves.show_conversation:
            conversation_log.append(f"### 🔄 Round {round_num + 1}")

        claude_prompt_for_this_round = ""
        if round_num == 0:
            # Use state-aware prompt if state tracking is enabled
            if conversation_state and valves.track_conversation_state:
                claude_prompt_for_this_round = get_minion_initial_claude_prompt_with_state(query, len(context), valves, conversation_state)
            else:
                claude_prompt_for_this_round = get_minion_initial_claude_prompt(query, len(context), valves)
        else:
            # Check if this is the last round and force a final answer
            is_last_round = (round_num == valves.max_rounds - 1)
            if is_last_round:
                # Override with a prompt that forces a final answer
                claude_prompt_for_this_round = f"""You are a supervisor LLM collaborating with a trusted local AI assistant to answer the user's ORIGINAL QUESTION: "{query}"

The local assistant has full access to the source document and has been providing factual information extracted from it.

CONVERSATION SO FAR:
"""
                for role, message in conversation_history:
                    if role == "assistant":
                        claude_prompt_for_this_round += f"\nYou previously asked: \"{message}\""
                    else:
                        claude_prompt_for_this_round += f"\nLocal assistant responded: \"{message}\""
                
                claude_prompt_for_this_round += f"""

THIS IS YOUR FINAL OPPORTUNITY TO ANSWER. You have gathered sufficient information through {round_num} rounds of questions.

Based on ALL the information provided by the local assistant, you MUST now provide a comprehensive answer to the user's original question: "{query}"

Respond with "FINAL ANSWER READY." followed by your synthesized answer. Do NOT ask any more questions."""
            else:
                # Use state-aware prompt if state tracking is enabled
                if conversation_state and valves.track_conversation_state:
                    previous_questions = deduplicator.get_all_questions() if deduplicator else None
                    claude_prompt_for_this_round = get_minion_conversation_claude_prompt_with_state(
                        conversation_history, query, valves, conversation_state, previous_questions
                    )
                else:
                    claude_prompt_for_this_round = get_minion_conversation_claude_prompt(
                        conversation_history, query, valves
                    )
        
        claude_response = ""
        try:
            if valves.debug_mode: 
                start_time_claude = asyncio.get_event_loop().time()
            claude_response = await call_claude_func(valves, claude_prompt_for_this_round)
            if valves.debug_mode:
                end_time_claude = asyncio.get_event_loop().time()
                time_taken_claude = end_time_claude - start_time_claude
                debug_log.append(f"  ⏱️ Remote model call in round {round_num + 1} took {time_taken_claude:.2f}s. (Debug Mode)")
        except Exception as e:
            error_message = f"❌ Error calling the remote model in round {round_num + 1}: {e}"
            conversation_log.append(error_message)
            if valves.debug_mode: 
                debug_log.append(f"  {error_message} (Debug Mode)")
            actual_final_answer = "Minion protocol failed due to remote API error."
            break

        conversation_history.append(("assistant", claude_response))
        if valves.show_conversation:
            conversation_log.append(f"**🤖 Remote Model ({valves.remote_model}):**")
            conversation_log.append(f"{claude_response}\n")

        # Check for explicit final answer or completion indicators
        if _is_final_answer(claude_response):
            actual_final_answer = claude_response.split("FINAL ANSWER READY.", 1)[1].strip()
            claude_declared_final = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **The remote model indicates FINAL ANSWER READY.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 The remote model declared FINAL ANSWER READY in round {round_num + 1}. (Debug Mode)")
            break
        elif valves.enable_completion_detection and detect_completion(claude_response) and round_num > 0:
            # Remote model indicates it has sufficient information
            actual_final_answer = claude_response
            claude_declared_final = True
            metrics['completion_via_detection'] = True
            if valves.show_conversation:
                conversation_log.append(f"✅ **The remote model indicates it has sufficient information to answer.**\n")
            if valves.debug_mode:
                debug_log.append(f"  🏁 Completion detected: Remote model has sufficient information in round {round_num + 1}. (Debug Mode)")
            break

        # Skip local model call if this was the last round and the remote model provided final answer
        if round_num == valves.max_rounds - 1:
            continue

        # Extract question from Claude's response for deduplication check
        question_to_check = claude_response.strip()
        
        # Check for duplicate questions if deduplication is enabled
        if deduplicator and valves.enable_deduplication:
            is_dup, original_question = deduplicator.is_duplicate(question_to_check)
            
            if is_dup:
                # Log the duplicate detection
                if valves.show_conversation:
                    conversation_log.append(f"⚠️ **Duplicate question detected! Similar to: '{original_question[:100]}...'**")
                    conversation_log.append(f"**Requesting a different question...**\n")
                
                if valves.debug_mode:
                    debug_log.append(f"  ⚠️ Duplicate question detected in round {round_num + 1}. (Debug Mode)")
                
                # Create a prompt asking for a different question
                dedup_prompt = f"""The question you just asked is too similar to a previous question: "{original_question}"

Please ask a DIFFERENT question that explores new aspects of the information needed to answer: "{query}"

Focus on areas not yet covered in our conversation."""
                
                # Request a new question
                try:
                    new_claude_response = await call_claude_func(valves, dedup_prompt)
                    claude_response = new_claude_response
                    question_to_check = claude_response.strip()
                    
                    # Update conversation history with the new question
                    conversation_history[-1] = ("assistant", claude_response)
                    
                    if valves.show_conversation:
                        conversation_log.append(f"**🤖 Remote Model (New Question):**")
                        conversation_log.append(f"{claude_response}\n")
                except Exception as e:
                    # If we can't get a new question, continue with the duplicate
                    if valves.debug_mode:
                        debug_log.append(f"  ❌ Failed to get alternative question: {e} (Debug Mode)")
        
        # Add the question to deduplicator after checks
        if deduplicator:
            deduplicator.add_question(question_to_check)

        local_prompt = get_minion_local_prompt(context, query, claude_response, valves)
        
        local_response_str = ""
        try:
            if valves.debug_mode: 
                start_time_ollama = asyncio.get_event_loop().time()
            local_response_str = await call_ollama_func(
                valves,
                local_prompt,
                use_json=True,
                schema=LocalAssistantResponseModel
            )
            local_response_data = _parse_local_response(
                local_response_str,
                is_structured=True,
                use_structured_output=valves.use_structured_output,
                debug_mode=valves.debug_mode,
                LocalAssistantResponseModel=LocalAssistantResponseModel
            )
            
            # Track metrics from local response
            if 'numeric_confidence' in local_response_data:
                metrics['confidence_scores'].append(local_response_data['numeric_confidence'])
            
            confidence_level = local_response_data.get('confidence', 'MEDIUM').upper()
            if confidence_level in metrics['confidence_distribution']:
                metrics['confidence_distribution'][confidence_level] += 1
            
            if valves.debug_mode:
                end_time_ollama = asyncio.get_event_loop().time()
                time_taken_ollama = end_time_ollama - start_time_ollama
                debug_log.append(f"  ⏱️ Local LLM call in round {round_num + 1} took {time_taken_ollama:.2f}s. (Debug Mode)")
        except Exception as e:
            error_message = f"❌ Error calling Local LLM in round {round_num + 1}: {e}"
            conversation_log.append(error_message)
            if valves.debug_mode: 
                debug_log.append(f"  {error_message} (Debug Mode)")
            actual_final_answer = "Minion protocol failed due to Local LLM API error."
            break

        response_for_claude = local_response_data.get("answer", "Error: Could not extract answer from local LLM.")
        if valves.use_structured_output and local_response_data.get("parse_error") and valves.debug_mode:
            response_for_claude += f" (Local LLM response parse error: {local_response_data['parse_error']})"
        elif not local_response_data.get("answer") and not local_response_data.get("parse_error"):
            response_for_claude = "Local LLM provided no answer."

        conversation_history.append(("user", response_for_claude))
        
        # Update conversation state if enabled
        if conversation_state and valves.track_conversation_state:
            # Extract the question from Claude's response
            question = claude_response.strip()
            
            # Add Q&A pair to state
            conversation_state.add_qa_pair(
                question=question,
                answer=response_for_claude,
                confidence=local_response_data.get('confidence', 'MEDIUM'),
                key_points=local_response_data.get('key_points')
            )
            
            # Extract topics from the question (simple keyword extraction)
            keywords = re.findall(r'\b[A-Z][a-z]+\b|\b\w{5,}\b', question)
            for keyword in keywords[:3]:  # Add up to 3 keywords as topics
                conversation_state.topics_covered.add(keyword.lower())
            
            # Update key findings if high confidence answer
            if local_response_data.get('confidence') == 'HIGH' and local_response_data.get('key_points'):
                for idx, point in enumerate(local_response_data['key_points'][:2]):
                    conversation_state.key_findings[f"round_{round_num+1}_finding_{idx+1}"] = point
        
        if valves.show_conversation:
            conversation_log.append(f"**💻 Local Model ({valves.local_model}):**")
            if valves.use_structured_output and local_response_data.get("parse_error") is None:
                conversation_log.append(f"```json\n{json.dumps(local_response_data, indent=2)}\n```")
            elif valves.use_structured_output and local_response_data.get("parse_error"):
                conversation_log.append(f"Attempted structured output, but failed. Raw response:\n{local_response_data.get('answer', 'Error displaying local response.')}")
                conversation_log.append(f"(Parse Error: {local_response_data['parse_error']})")
            else:
                conversation_log.append(f"{local_response_data.get('answer', 'Error displaying local response.')}")
            conversation_log.append("\n")

        # Update round count
        metrics['rounds_completed'] = round_num + 1
        
        if valves.debug_mode:
            current_cumulative_time = asyncio.get_event_loop().time() - overall_start_time
            debug_log.append(f"**🏁 Completed Round {round_num + 1}. Cumulative time: {current_cumulative_time:.2f}s. (Debug Mode)**\n")
    
    if not claude_declared_final and conversation_history:
        # This shouldn't happen with the fix above, but keep as fallback
        last_remote_msg = conversation_history[-1][1] if conversation_history[-1][0] == "assistant" else (conversation_history[-2][1] if len(conversation_history) > 1 and conversation_history[-2][0] == "assistant" else "No suitable final message from the remote model found.")
        actual_final_answer = f"Protocol ended without explicit final answer. The remote model's last response was: \"{last_remote_msg}\""
        if valves.show_conversation:
            conversation_log.append(f"⚠️ Protocol ended without the remote model providing a final answer.\n")

    # Calculate final metrics
    total_execution_time = asyncio.get_event_loop().time() - overall_start_time
    avg_confidence = sum(metrics['confidence_scores']) / len(metrics['confidence_scores']) if metrics['confidence_scores'] else 0.0
    
    # Estimate tokens (rough approximation)
    for role, msg in conversation_history:
        metrics['estimated_tokens'] += len(msg) // 4  # Rough token estimate
    
    if valves.debug_mode:
        debug_log.append(f"**⏱️ Total Minion protocol execution time: {total_execution_time:.2f}s. (Debug Mode)**")

    output_parts = []
    if valves.show_conversation:
        output_parts.append("## 🗣️ Collaboration Conversation")
        output_parts.extend(conversation_log)
        output_parts.append("---")
    if valves.debug_mode:
        output_parts.append("### 🔍 Debug Log")
        output_parts.extend(debug_log)
        output_parts.append("---")

    output_parts.append(f"## 🎯 Final Answer")
    output_parts.append(actual_final_answer)

    stats = _calculate_token_savings(conversation_history, context, query)
    output_parts.append(f"\n## 📊 Efficiency Stats")
    output_parts.append(f"- **Protocol:** Minion (conversational)")
    output_parts.append(f"- **Remote model:** {valves.remote_model}")
    output_parts.append(f"- **Local model:** {valves.local_model}")
    output_parts.append(f"- **Conversation rounds:** {len(conversation_history) // 2}")
    output_parts.append(f"- **Context size:** {len(context):,} characters")
    output_parts.append(f"")
    output_parts.append(f"## 💰 Token Savings Analysis ({valves.remote_model})")
    output_parts.append(f"- **Traditional approach:** ~{stats['traditional_tokens']:,} tokens")
    output_parts.append(f"- **Minion approach:** ~{stats['minion_tokens']:,} tokens")
    output_parts.append(f"- **💰 Token Savings:** ~{stats['percentage_savings']:.1f}%")
    
    # Add conversation metrics
    output_parts.append(f"\n## 📈 Conversation Metrics")
    output_parts.append(f"- **Rounds used:** {metrics['rounds_completed']} of {valves.max_rounds}")
    output_parts.append(f"- **Questions asked:** {metrics['rounds_completed']}")
    output_parts.append(f"- **Average confidence:** {avg_confidence:.2f} ({['LOW', 'MEDIUM', 'HIGH'][int(avg_confidence * 2.99)]})")
    output_parts.append(f"- **Confidence distribution:**")
    for level, count in metrics['confidence_distribution'].items():
        if count > 0:
            output_parts.append(f"  - {level}: {count} response(s)")
    output_parts.append(f"- **Completion method:** {'Early completion detected' if metrics['completion_via_detection'] else 'Reached max rounds or explicit completion'}")
    output_parts.append(f"- **Total duration:** {total_execution_time*1000:.0f}ms")
    output_parts.append(f"- **Estimated tokens:** ~{metrics['estimated_tokens']:,}")
    output_parts.append(f"- **Chunk processing:** {metrics['context_size']:,} chars (max chunk size: {metrics['chunk_size_used']:,})")
    
    return "\n".join(output_parts)

# Partials File: partials/minion_pipe_method.py
import asyncio
from typing import Any, List, Callable, Dict
from fastapi import Request


async def _call_claude_directly(valves: Any, query: str, call_claude_func: Callable) -> str: # Renamed for clarity
    """Fallback to direct Claude call when no context is available"""
    return await call_claude_func(valves, f"Please answer this question: {query}")

async def minion_pipe(
    pipe_self: Any,
    body: Dict[str, Any], # Typed body
    __user__: Dict[str, Any], # Typed __user__
    __request__: Request,
    __files__: List[Dict[str, Any]] = [], # Typed __files__
    __pipe_id__: str = "minion-claude",
) -> str:
    """Execute the Minion protocol with Claude"""
    try:
        # Validate configuration
        if not pipe_self.valves.anthropic_api_key: # Add ollama key check if necessary
            return "❌ **Error:** Please configure your Anthropic API key in the function settings."

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

        if not context:
            # Pass the imported call_claude to _call_claude_directly
            direct_response = await _call_claude_directly(pipe_self.valves, user_query, call_claude_func=call_claude)
            return (
                "ℹ️ **Note:** No significant context detected. Using standard Claude response.\n\n"
                + direct_response
            )

        # Handle chunking for large documents
        chunks = create_chunks(context, pipe_self.valves.chunk_size, pipe_self.valves.max_chunks)
        if not chunks and context:
            return "❌ **Error:** Context provided, but failed to create any processable chunks. Check chunk_size setting."
        
        if len(chunks) > 1:
            # Multiple chunks - need to process each chunk and combine results
            chunk_results = []
            for i, chunk in enumerate(chunks):
                chunk_header = f"## 📄 Chunk {i+1} of {len(chunks)}\n"
                
                try:
                    chunk_result = await _execute_minion_protocol(
                        valves=pipe_self.valves, 
                        query=user_query, 
                        context=chunk, 
                        call_claude_func=call_claude,
                        call_ollama_func=call_ollama,
                        LocalAssistantResponseModel=LocalAssistantResponse,
                        ConversationStateModel=ConversationState,
                        QuestionDeduplicatorModel=QuestionDeduplicator
                    )
                    chunk_results.append(chunk_header + chunk_result)
                except Exception as e:
                    chunk_results.append(f"{chunk_header}❌ **Error processing chunk {i+1}:** {str(e)}")
            
            # Combine all chunk results
            combined_result = "\n\n---\n\n".join(chunk_results)
            
            # Add summary header
            summary_header = f"""# 🔗 Multi-Chunk Analysis Results
            
**Document processed in {len(chunks)} chunks** (max {pipe_self.valves.chunk_size:,} characters each)

{combined_result}

---

## 📋 Summary
The document was automatically divided into {len(chunks)} chunks for processing. Each chunk was analyzed independently using the Minion protocol. Review the individual chunk results above for comprehensive coverage of the document."""
            
            return summary_header
        else:
            # Single chunk or no chunking needed
            result: str = await _execute_minion_protocol(
                valves=pipe_self.valves, 
                query=user_query, 
                context=chunks[0] if chunks else context, 
                call_claude_func=call_claude,  # Pass imported function
                call_ollama_func=call_ollama,  # Pass imported function
                LocalAssistantResponseModel=LocalAssistantResponse, # Pass imported class
                ConversationStateModel=ConversationState,
                QuestionDeduplicatorModel=QuestionDeduplicator
            )
            return result

    except Exception as e:
        import traceback # Keep import here as it's conditional
        error_details: str = traceback.format_exc() if pipe_self.valves.debug_mode else str(e)
        return f"❌ **Error in Minion protocol:** {error_details}"


class Pipe:
    class Valves(MinionValves):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Minion v0.3.6b (Conversational)"

    def pipes(self):
        """Define the available models"""
        return [
            {
                "id": "minion-claude",
                "name": f" ({self.valves.local_model} + {self.valves.remote_model})",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "minion-claude",
    ) -> str:
        """Execute the Minion protocol with Claude"""
        return await minion_pipe(self, body, __user__, __request__, __files__, __pipe_id__)