"""
Generator script for Minion/MinionS Open WebUI functions.
Assembles partial files into complete function files based on profiles in generation_config.json.
Supports the new modular architecture with improved imports, constants, error handling, and debugging.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set

def load_config(config_file: str) -> Dict[str, Any]:
    """Load the generation configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def load_partial(partials_dir: str, filename: str) -> str:
    """Load content from a partial file."""
    filepath = os.path.join(partials_dir, filename)
    with open(filepath, 'r') as f:
        return f.read()

def get_new_utility_modules() -> Set[str]:
    """Get the set of new utility modules that should be handled specially."""
    return {
        "imports_registry",
        "constants", 
        "error_handling",
        "debug_utils",
        "protocol_base",
        "protocol_state"
    }

def should_use_centralized_imports(partials: List[str]) -> bool:
    """Check if profile should use centralized import management."""
    # If imports_registry is in partials, use centralized imports
    return "imports_registry.py" in partials

def process_partial_content(content: str, filename: str, all_partials: List[str], use_centralized_imports: bool = False) -> str:
    """Process partial content to handle imports correctly."""
    lines = content.split('\n')
    processed_lines = []
    
    # Get list of partial modules (without .py extension)
    partial_modules = [p.replace('.py', '') for p in all_partials if p.endswith('.py')]
    new_utility_modules = get_new_utility_modules()
    
    skip_until_close = False
    
    # Special handling for imports_registry - skip its content since imports will be handled centrally
    if filename == "imports_registry.py" and use_centralized_imports:
        return ""  # Skip this file's content, imports will be added separately
    
    for line in lines:
        # Check if we're in a multi-line import that should be skipped
        if skip_until_close:
            if ')' in line:
                skip_until_close = False
            continue
            
        # Skip imports from other partials that will be concatenated
        if line.strip().startswith('from .'):
            # Extract module name
            if ' import ' in line:
                module_name = line.split('from .')[1].split(' import')[0].strip()
                if module_name in partial_modules:
                    # Check if it's a multi-line import
                    if '(' in line and ')' not in line:
                        skip_until_close = True
                    continue
        
        # If using centralized imports, skip standard import statements from non-utility files
        if use_centralized_imports and filename not in [f"{mod}.py" for mod in new_utility_modules]:
            if line.strip().startswith(('import ', 'from typing import', 'from pydantic import', 'from fastapi import')):
                # Skip standard imports that are handled centrally
                if not any(util_mod in line for util_mod in new_utility_modules):
                    continue
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def replace_placeholders(content: str, placeholders: Dict[str, str]) -> str:
    """Replace placeholders in content."""
    for key, value in placeholders.items():
        content = content.replace(f"{{{key}}}", value)
    return content

def generate_centralized_imports(partials_dir: str, partials: List[str]) -> str:
    """Generate centralized imports for the new modular structure."""
    # For now, use a comprehensive set of imports that covers all needs
    # This can be enhanced later to be more selective based on partials used
    return """# Centralized imports for v0.3.8 modular architecture

# Standard library imports
import asyncio
import json
import re
import hashlib
import traceback
import inspect
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

# Typing imports
from typing import (
    List, Dict, Any, Optional, Tuple, Callable, Awaitable, AsyncGenerator,
    Union, Set, TypedDict, Protocol
)

# Third-party imports
import aiohttp
from pydantic import BaseModel, Field, ValidationError
from fastapi import Request"""

def generate_pipe_class(profile: Dict[str, Any], function_type: str) -> str:
    """Generate the final Pipe class definition."""
    pipe_name = profile.get("target_pipe_name_in_init", "Generated Pipe")
    pipe_id = profile.get("target_pipe_id_in_init", "generated-pipe")
    
    # Determine which valves class to use
    valves_class = "MinionValves" if function_type == "minion" else "MinionsValves"
    
    # Determine which pipe method to call
    pipe_method = "minion_pipe" if function_type == "minion" else "minions_pipe_method"
    
    class_def = f'''
class Pipe:
    class Valves({valves_class}):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "{pipe_name}"

    def pipes(self):
        """Define the available models"""
        return [
            {{
                "id": "{pipe_id}",
                "name": f" ({{self.valves.local_model}} + {{self.valves.remote_model}})",
            }}
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __files__: List[dict] = [],
        __pipe_id__: str = "{pipe_id}",
    ) -> str:
        """Execute the {'Minion' if function_type == 'minion' else 'MinionS'} protocol with Claude"""
        return await {pipe_method}(self, body, __user__, __request__, __files__, __pipe_id__)'''
    
    return class_def

def generate_function(profile: Dict[str, Any], partials_dir: str, function_type: str) -> str:
    """Generate a complete function file from partials."""
    output_parts = []
    
    # Get all partials for processing
    all_partials = profile["partials_concat_order"]
    use_centralized_imports = should_use_centralized_imports(all_partials)
    
    print(f"Using centralized imports: {use_centralized_imports}")
    
    # 1. Process header
    header_content = load_partial(partials_dir, "common_header.py")
    header_content = replace_placeholders(header_content, profile["header_placeholders"])
    output_parts.append(header_content)
    
    # 2. Add centralized imports if using new structure
    if use_centralized_imports:
        centralized_imports = generate_centralized_imports(partials_dir, all_partials)
        output_parts.append(centralized_imports)
    
    # 3. Load all partials in order
    for partial_file in all_partials:
        if partial_file == "common_header.py":
            continue  # Already processed
        
        # Skip common_imports.py if using centralized imports
        if use_centralized_imports and partial_file == "common_imports.py":
            continue
        
        try:
            content = load_partial(partials_dir, partial_file)
        except FileNotFoundError:
            print(f"Warning: Partial file '{partial_file}' not found, skipping...")
            continue
        
        # Process content to remove inter-partial imports
        processed_content = process_partial_content(content, partial_file, all_partials, use_centralized_imports)
        
        # Only add non-empty content
        if processed_content.strip():
            output_parts.append(processed_content)
    
    # 4. Generate the final Pipe class
    pipe_class = generate_pipe_class(profile, function_type)
    output_parts.append(pipe_class)
    
    # Join all parts with double newlines
    return "\n\n".join(output_parts)

def main():
    parser = argparse.ArgumentParser(
        description="Generate Minion/MinionS function files from partials."
    )
    parser.add_argument(
        "function_type",
        choices=["minion", "minions"],
        help="Type of function to generate"
    )
    parser.add_argument(
        "--profile",
        help="Profile name from config (defaults to minion_default or minions_default)"
    )
    parser.add_argument(
        "--config_file",
        default="generation_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--partials_dir",
        default="partials",
        help="Directory containing partial files"
    )
    parser.add_argument(
        "--output_dir",
        default="generated_functions",
        help="Output directory for generated files"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config_file)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1
    
    # Determine profile to use
    if args.profile:
        profile_name = args.profile
    else:
        profile_name = f"{args.function_type}_default"
    
    if profile_name not in config:
        print(f"Error: Profile '{profile_name}' not found in config")
        print(f"Available profiles: {', '.join(config.keys())}")
        return 1
    
    profile = config[profile_name]
    
    # Create output directory if needed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate the function
    try:
        print(f"Generating {args.function_type} function using profile '{profile_name}'...")
        function_content = generate_function(profile, args.partials_dir, args.function_type)
        
        # Determine output filename
        output_filename = profile["output_filename_template"].format(
            profile_name=profile_name,
            function_type=args.function_type
        )
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Write the file
        with open(output_path, 'w') as f:
            f.write(function_content)
        
        print(f"Successfully generated: {output_path}")
        print(f"Profile used: {profile_name}")
        print(f"Description: {profile.get('description', 'No description')}")
        
    except Exception as e:
        print(f"Error generating function: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())