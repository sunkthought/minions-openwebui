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

async def call_openai_api(
    prompt: str, 
    api_key: str, 
    model: str = "gpt-4o", 
    temperature: float = 0.1, 
    max_tokens: int = 4096,
    timeout: int = 60
) -> str:
    """Call OpenAI's API with error handling and retry logic"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"OpenAI API error: {response.status} - {error_text}"
                )
            result = await response.json()
            
            if result.get("choices") and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("Unexpected response format from OpenAI API or empty content.")

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

async def call_supervisor_model(valves: BaseModel, prompt: str) -> str:
    """Call the configured supervisor model (Claude or OpenAI)"""
    provider = getattr(valves, 'supervisor_provider', 'anthropic')
    
    if provider == 'openai':
        api_key = valves.openai_api_key
        model = getattr(valves, 'openai_model', 'gpt-4o')
        max_tokens = getattr(valves, 'max_tokens_claude', 4096)
        timeout = getattr(valves, 'timeout_claude', 60)
        
        if not api_key:
            raise Exception("OpenAI API key is required when using OpenAI as supervisor provider")
        
        return await call_openai_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    elif provider == 'anthropic':
        if not valves.anthropic_api_key:
            raise Exception("Anthropic API key is required when using Anthropic as supervisor provider")
        
        return await call_claude(valves, prompt)
    
    else:
        raise Exception(f"Unsupported supervisor provider: {provider}. Use 'anthropic' or 'openai'")
