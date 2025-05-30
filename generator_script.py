#!/usr/bin/env python3
"""
Generator script for Minion/MinionS Open WebUI functions.
Assembles partial files into complete function files based on profiles in generation_config.json.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

def load_config(config_file: str) -> Dict[str, Any]:
    """Load the generation configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def load_partial(partials_dir: str, filename: str) -> str:
    """Load content from a partial file."""
    filepath = os.path.join(partials_dir, filename)
    with open(filepath, 'r') as f:
        return f.read()

def replace_placeholders(content: str, placeholders: Dict[str, str]) -> str:
    """Replace placeholders in content."""
    for key, value in placeholders.items():
        content = content.replace(f"{{{key}}}", value)
    return content

def generate_imports(profile: Dict[str, Any]) -> str:
    """Generate import statements based on the profile's specific_imports_map."""
    imports = []
    
    # Group imports by module
    imports_by_module = {}
    for import_name, module_name in profile.get("specific_imports_map", {}).items():
        if module_name not in imports_by_module:
            imports_by_module[module_name] = []
        imports_by_module[module_name].append(import_name)
    
    # Generate import statements
    for module, items in imports_by_module.items():
        if len(items) == 1:
            imports.append(f"from {module} import {items[0]}")
        else:
            imports.append(f"from {module} import {', '.join(sorted(items))}")
    
    return '\n'.join(imports)

def generate_pipe_class(profile: Dict[str, Any], function_type: str) -> str:
    """Generate the final Pipe class definition."""
    class_name = profile.get("target_pipe_class_name", "Pipe")
    pipe_name = profile.get("target_pipe_name_in_init", "Generated Pipe")
    pipe_id = profile.get("target_pipe_id_in_init", "generated-pipe")
    
    # Determine which valves class to use
    valves_class = "MinionValves" if function_type == "minion" else "MinionsValves"
    
    # Determine which pipe method to call
    pipe_method = "minion_pipe" if function_type == "minion" else "minions_pipe"
    
    # Build the execute protocol dependencies setup
    deps_setup = []
    for arg_name, global_name in profile.get("execute_protocol_dependencies_map", {}).items():
        deps_setup.append(f"        self.{arg_name} = {global_name}")
    
    # Additional setup for protocol-specific functions
    if function_type == "minion":
        main_protocol_func = "execute_minion_protocol"
    else:
        main_protocol_func = "execute_minions_protocol"
        # MinionS needs additional helper functions
        deps_setup.extend([
            "        self.parse_tasks_func = parse_minions_tasks",
            "        self.create_chunks_func = create_minions_chunks", 
            "        self.execute_tasks_on_chunks_func = execute_minions_tasks_on_chunks",
            "        self.parse_local_response_func = parse_minions_local_response"
        ])
    
    deps_setup_str = '\n'.join(deps_setup)
    
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
        
        # Set up references to functions needed by the pipe method
        self.extract_context_from_messages = extract_context_from_messages
        self.extract_context_from_files = extract_context_from_files
        self.call_claude_api = call_claude
        self.call_ollama_api = call_ollama
        self.{main_protocol_func} = {main_protocol_func}
        
        # Protocol-specific dependencies
{deps_setup_str}
        
        # Call the main pipe method
        return await {pipe_method}(
            self,
            body,
            __user__,
            __request__,
            __files__,
            __pipe_id__
        )'''
    
    return class_def

def generate_function(profile: Dict[str, Any], partials_dir: str, function_type: str) -> str:
    """Generate a complete function file from partials."""
    output_parts = []
    
    # 1. Process header
    header_content = load_partial(partials_dir, "common_header.py")
    header_content = replace_placeholders(header_content, profile["header_placeholders"])
    output_parts.append(header_content)
    
    # 2. Load common imports
    imports_content = load_partial(partials_dir, "common_imports.py")
    output_parts.append(imports_content)
    
    # 3. Generate specific imports
    specific_imports = generate_imports(profile)
    if specific_imports:
        output_parts.append(specific_imports)
    
    # 4. Load all partials in order (skip header and imports as they're handled above)
    for partial_file in profile["partials_concat_order"]:
        if partial_file in ["common_header.py", "common_imports.py"]:
            continue
        
        content = load_partial(partials_dir, partial_file)
        output_parts.append(content)
    
    # 5. Generate the final Pipe class
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