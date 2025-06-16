"""
Centralized import management for MinionS/Minions OpenWebUI functions.
This module defines all imports used across partials and provides
organized import groups for the generator script.
"""

# Standard library imports grouped by functionality
STANDARD_LIBRARY = {
    'async': [
        'import asyncio',
    ],
    'data_handling': [
        'import json',
        'import re',
        'import hashlib',
    ],
    'url_handling': [
        'from urllib.parse import urlparse',
    ],
    'dataclasses': [
        'from dataclasses import dataclass',
    ],
    'error_handling': [
        'import traceback',
    ],
    'introspection': [
        'import inspect',
    ],
}

# Typing imports - commonly used across all files
TYPING_IMPORTS = [
    'from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable, AsyncGenerator',
]

# Third-party imports
THIRD_PARTY = {
    'http': [
        'import aiohttp',
    ],
    'data_validation': [
        'from pydantic import BaseModel, Field',
    ],
    'web_framework': [
        'from fastapi import Request',
    ],
    'enum': [
        'from enum import Enum',
    ],
}

# Import groups for different partial types
IMPORT_GROUPS = {
    'common_header': [],
    
    'common_imports': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
        *THIRD_PARTY['http'],
        *THIRD_PARTY['data_validation'],
    ],
    
    'models': [
        *TYPING_IMPORTS,
        *THIRD_PARTY['data_validation'],
        *THIRD_PARTY['enum'],
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['dataclasses'],
    ],
    
    'valves': [
        *THIRD_PARTY['data_validation'],
    ],
    
    'api_calls': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
        *STANDARD_LIBRARY['introspection'],
        *THIRD_PARTY['http'],
        *THIRD_PARTY['data_validation'],
    ],
    
    'protocol_logic': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
        *STANDARD_LIBRARY['error_handling'],
    ],
    
    'pipe_method': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
        *THIRD_PARTY['web_framework'],
    ],
    
    'utils': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['data_handling'],
        'import hashlib',
    ],
    
    # v0.3.9 Open WebUI Integration partials
    'web_search_integration': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
    ],
    
    'rag_retriever': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['data_handling'],
    ],
    
    'citation_manager': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['data_handling'],
        *STANDARD_LIBRARY['url_handling'],
    ],
    
    'task_visualizer': [
        *TYPING_IMPORTS,
        *THIRD_PARTY['enum'],
    ],
    
    'streaming_support': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
        'import time',
    ],
    
    'tool_execution_bridge': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
        'import uuid',
        'from datetime import datetime',
    ],
    
    'minions_streaming_protocol': [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'],
    ],
}

# Conditional imports that should be handled specially
CONDITIONAL_IMPORTS = {
    'debug_mode': [
        'import traceback',
    ],
}

# Internal imports between partials (for reference, not included in generated code)
INTERNAL_DEPENDENCIES = {
    'minion_pipe_method': [
        'common_api_calls',
        'minion_protocol_logic',
        'minion_models',
        'common_context_utils',
        'common_file_processing',
        'minion_prompts',
    ],
    'minions_pipe_method': [
        'common_api_calls',
        'minions_protocol_logic',
        'common_file_processing',
        'minions_models',
        'common_context_utils',
        'minions_decomposition_logic',
        'minions_prompts',
        'minion_sufficiency_analyzer',
        'minion_convergence_detector',
    ],
    'minion_convergence_detector': [
        'minions_models',  # Has circular dependency handling
    ],
    'minions_protocol_logic': [
        'minions_models',
    ],
}

def get_imports_for_partial(partial_name: str) -> List[str]:
    """
    Get the appropriate imports for a specific partial file.
    
    Args:
        partial_name: Name of the partial file (without .py extension)
        
    Returns:
        List of import statements
    """
    # Determine the category based on partial name
    if partial_name.endswith('_models'):
        category = 'models'
    elif partial_name.endswith('_valves'):
        category = 'valves'
    elif partial_name == 'common_api_calls':
        category = 'api_calls'
    elif partial_name.endswith('_protocol_logic'):
        category = 'protocol_logic'
    elif partial_name.endswith('_pipe_method'):
        category = 'pipe_method'
    elif partial_name in ['common_context_utils', 'common_file_processing']:
        category = 'utils'
    elif partial_name == 'common_imports':
        category = 'common_imports'
    # v0.3.9 Open WebUI Integration partials
    elif partial_name == 'web_search_integration':
        category = 'web_search_integration'
    elif partial_name == 'rag_retriever':
        category = 'rag_retriever'
    elif partial_name == 'citation_manager':
        category = 'citation_manager'
    elif partial_name == 'task_visualizer':
        category = 'task_visualizer'
    elif partial_name == 'streaming_support':
        category = 'streaming_support'
    elif partial_name == 'tool_execution_bridge':
        category = 'tool_execution_bridge'
    elif partial_name == 'minions_streaming_protocol':
        category = 'minions_streaming_protocol'
    else:
        category = 'common_imports'  # Default fallback
    
    return IMPORT_GROUPS.get(category, [])

def get_all_unique_imports() -> List[str]:
    """
    Get all unique imports across all categories.
    
    Returns:
        Sorted list of unique import statements
    """
    all_imports = set()
    
    # Add all standard library imports
    for group in STANDARD_LIBRARY.values():
        all_imports.update(group)
    
    # Add typing imports
    all_imports.update(TYPING_IMPORTS)
    
    # Add third-party imports
    for group in THIRD_PARTY.values():
        all_imports.update(group)
    
    return sorted(list(all_imports))

def get_minimal_imports() -> List[str]:
    """
    Get minimal set of imports for basic functionality.
    
    Returns:
        List of essential import statements
    """
    return [
        *TYPING_IMPORTS,
        *STANDARD_LIBRARY['async'],
        *STANDARD_LIBRARY['data_handling'][:1],  # Just json
        *THIRD_PARTY['data_validation'],
    ]

def format_imports_block(imports: List[str]) -> str:
    """
    Format a list of imports into a properly organized import block.
    
    Args:
        imports: List of import statements
        
    Returns:
        Formatted import block as string
    """
    # Separate imports by type
    import_lines = []
    from_imports = []
    
    for imp in imports:
        if imp.startswith('from '):
            from_imports.append(imp)
        else:
            import_lines.append(imp)
    
    # Sort each group
    import_lines.sort()
    from_imports.sort()
    
    # Combine with proper spacing
    result = []
    if import_lines:
        result.extend(import_lines)
    if import_lines and from_imports:
        result.append('')  # Empty line between groups
    if from_imports:
        result.extend(from_imports)
    
    return '\n'.join(result)