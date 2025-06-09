# Partials File: partials/model_capabilities.py
from typing import Dict, Any, Optional

MODEL_CAPABILITIES = {
    # OpenAI Models
    "gpt-4o": {
        "max_tokens": 128000,
        "supports_json": True,
        "supports_functions": True,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "provider": "openai"
    },
    "gpt-4-turbo": {
        "max_tokens": 128000,
        "supports_json": True,
        "supports_functions": True,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "provider": "openai"
    },
    "gpt-4": {
        "max_tokens": 8192,
        "supports_json": True,
        "supports_functions": True,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
        "provider": "openai"
    },
    
    # Anthropic Models
    "claude-3-5-sonnet-20241022": {
        "max_tokens": 200000,
        "supports_json": True,
        "supports_functions": False,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "provider": "anthropic"
    },
    "claude-3-5-haiku-20241022": {
        "max_tokens": 200000,
        "supports_json": True,
        "supports_functions": False,
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.005,
        "provider": "anthropic"
    },
    "claude-3-opus-20240229": {
        "max_tokens": 200000,
        "supports_json": True,
        "supports_functions": False,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "provider": "anthropic"
    },
}

def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """Get capabilities for a specific model"""
    if model_name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model_name].copy()
    
    # Fallback defaults for unknown models
    return {
        "max_tokens": 4096,
        "supports_json": False,
        "supports_functions": False,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "provider": "unknown"
    }

def detect_ollama_capabilities(model_name: str) -> Dict[str, Any]:
    """Detect Ollama model capabilities based on model family"""
    model_lower = model_name.lower()
    
    # Base capabilities for Ollama models
    capabilities = {
        "max_tokens": 4096,
        "supports_json": False,
        "supports_functions": False,
        "cost_per_1k_input": 0.0,  # Local models are free
        "cost_per_1k_output": 0.0,
        "provider": "ollama"
    }
    
    # Detect context length based on model family
    if "llama3.2" in model_lower:
        capabilities["max_tokens"] = 128000 if ":1b" in model_lower or ":3b" in model_lower else 4096
        capabilities["supports_json"] = True
    elif "llama3.1" in model_lower:
        capabilities["max_tokens"] = 128000
        capabilities["supports_json"] = True
    elif "llama3" in model_lower:
        capabilities["max_tokens"] = 8192
        capabilities["supports_json"] = True
    elif "qwen2.5" in model_lower:
        capabilities["max_tokens"] = 32768
        capabilities["supports_json"] = True
    elif "qwen2" in model_lower:
        capabilities["max_tokens"] = 32768
        capabilities["supports_json"] = True
    elif "gemma2" in model_lower:
        capabilities["max_tokens"] = 8192
        capabilities["supports_json"] = True
    elif "mistral" in model_lower or "mixtral" in model_lower:
        capabilities["max_tokens"] = 32768
        capabilities["supports_json"] = True
    elif "phi3" in model_lower:
        capabilities["max_tokens"] = 4096
        capabilities["supports_json"] = True
    
    return capabilities

def get_effective_model_capabilities(valves: Any) -> Dict[str, Any]:
    """Get effective capabilities for the configured models"""
    supervisor_provider = getattr(valves, 'supervisor_provider', 'anthropic')
    
    if supervisor_provider == 'openai':
        supervisor_model = getattr(valves, 'openai_model', 'gpt-4o')
    else:
        supervisor_model = getattr(valves, 'remote_model', 'claude-3-5-haiku-20241022')
    
    local_model = getattr(valves, 'local_model', 'llama3.2')
    
    supervisor_caps = get_model_capabilities(supervisor_model)
    local_caps = detect_ollama_capabilities(local_model)
    
    return {
        "supervisor": supervisor_caps,
        "local": local_caps
    }

def adjust_parameters_for_capabilities(valves: Any, capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust parameters based on model capabilities"""
    adjustments = {}
    
    supervisor_caps = capabilities["supervisor"]
    local_caps = capabilities["local"]
    
    # Adjust max tokens for supervisor based on capabilities
    current_max_tokens = getattr(valves, 'max_tokens_claude', 4096)
    if current_max_tokens > supervisor_caps["max_tokens"]:
        adjustments["max_tokens_claude"] = supervisor_caps["max_tokens"]
    
    # Adjust local model context if available
    current_local_context = getattr(valves, 'local_model_context_length', 4096)
    if current_local_context > local_caps["max_tokens"]:
        adjustments["local_model_context_length"] = local_caps["max_tokens"]
    
    # Adjust structured output usage based on support
    if hasattr(valves, 'use_structured_output') and valves.use_structured_output:
        if not local_caps["supports_json"]:
            adjustments["use_structured_output"] = False
    
    return adjustments