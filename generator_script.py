#!/usr/bin/env python3
"""
Generator script for Minion/MinionS Open WebUI functions.
Assembles partials into complete, single-file functions for Open WebUI.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

def load_config(config_file: str) -> Dict[str, Any]:
    """Load the generation configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

def read_partial_file(partials_dir: str, filename: str) -> str:
    """Read content from a partial file."""
    filepath = os.path.join(partials_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: Partial file '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
        sys.exit(1)

def apply_header_placeholders(header_content: str, placeholders: Dict[str, str]) -> str:
    """Replace placeholders in header content with actual values."""
    result = header_content
    for placeholder, value in placeholders.items():
        result = result.replace(f"{{{placeholder}}}", value)
    return result

def extract_content_after_imports(content: str) -> str:
    """Extract content after import statements, preserving the actual code."""
    lines = content.split('\n')
    content_lines = []
    in_imports = True
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines and comments at the start
        if in_imports and (not stripped or stripped.startswith('#')):
            continue
            
        # Skip import statements
        if in_imports and (stripped.startswith('import ') or stripped.startswith('from ')):
            continue
            
        # Once we hit non-import content, include everything
        in_imports = False
        content_lines.append(line)
    
    return '\n'.join(content_lines)

def generate_pipe_class(profile: Dict[str, Any]) -> str:
    """Generate the final Function class definition for Open WebUI."""
    pipe_name = profile["target_pipe_name_in_init"]
    pipe_id = profile["target_pipe_id_in_init"]
    
    # Determine the function type from profile name or config
    function_type = "minion" if "minion" in profile.get("description", "").lower() and "minions" not in profile.get("description", "").lower() else "minions"
    
    # Get the valves class name and pipe method name
    if function_type == "minion":
        valves_class = "MinionValves"
        pipe_method = "minion_pipe"
        main_protocol_func = "execute_minion_protocol"
    else:
        valves_class = "MinionsValves"
        pipe_method = "minions_pipe"
        main_protocol_func = "execute_minions_protocol"
    
    # Build the dependency assignments
    dependency_assignments = []
    for arg_name, global_func_name in profile.get("execute_protocol_dependencies_map", {}).items():
        dependency_assignments.append(f"        self.{arg_name} = {global_func_name}")
    
    # Additional protocol logic imports for MinionS
    other_assignments = []
    if function_type == "minions":
        other_logic_funcs = [
            "parse_minions_tasks", "create_minions_chunks", 
            "execute_minions_tasks_on_chunks", "parse_minions_local_response"
        ]
        for func_name in other_logic_funcs:
            if func_name.startswith("parse_minions_"):
                attr_name = func_name.replace("parse_minions_", "parse_").replace("_func", "") + "_func"
            elif func_name.startswith("create_minions_"):
                attr_name = func_name.replace("create_minions_", "create_").replace("_func", "") + "_func"
            elif func_name.startswith("execute_minions_"):
                attr_name = func_name.replace("execute_minions_", "execute_").replace("_func", "") + "_func"
            else:
                attr_name = func_name.replace("_func", "") + "_func"
            other_assignments.append(f"        self.{attr_name} = {func_name}")
    
    function_class = f'''
# --- Final Function Class Definition for Open WebUI ---
class Function:
    class Valves({valves_class}):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, Any]]:
        local_model_name = getattr(self.valves, 'local_model', 'local_model')
        remote_model_name = getattr(self.valves, 'remote_model', 'remote_model')
        return [{{
            "id": "{pipe_id}",
            "name": "{pipe_name} (" + local_model_name + " + " + remote_model_name + ")",
        }}]

    async def pipe(self, body: Dict[str, Any], __user__: Dict[str, Any], __request__: Any, __files__: List[Dict[str, Any]] = [], __pipe_id__: str = ""):
        # Assign common utilities (available globally due to concatenation)
        self.extract_context_from_messages = extract_context_from_messages 
        self.extract_context_from_files = extract_context_from_files
        self.call_claude_api = call_claude 
        self.call_ollama_api = call_ollama
        
        # Assign the main protocol execution function
        self.{main_protocol_func} = {main_protocol_func}
        
        # Assign dependencies for the main protocol execution function
{chr(10).join(dependency_assignments)}
{chr(10).join(other_assignments)}
        
        # Call the main standalone pipe method
        return await {pipe_method}(
            self, 
            body, 
            __user__, 
            __request__, 
            __files__, 
            __pipe_id__ if __pipe_id__ else "{pipe_id}"
        )

# --- End of Final Function Class Definition ---'''
    
    return function_class

def generate_function_file(profile_name: str, config: Dict[str, Any], partials_dir: str, output_dir: str) -> str:
    """Generate a complete function file from partials based on the profile."""
    
    if profile_name not in config:
        print(f"Error: Profile '{profile_name}' not found in configuration.")
        sys.exit(1)
    
    profile = config[profile_name]
    print(f"Generating function using profile: {profile_name}")
    print(f"Description: {profile['description']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    output_filename = profile["output_filename_template"].format(profile_name=profile_name)
    output_path = os.path.join(output_dir, output_filename)
    
    # Start building the complete file content
    file_parts = []
    
    # 1. Process header with placeholders
    header_content = read_partial_file(partials_dir, "common_header.py")
    formatted_header = apply_header_placeholders(header_content, profile["header_placeholders"])
    file_parts.append(formatted_header)
    file_parts.append("\n# Dynamically Generated Header End\n")
    
    # 2. Add common imports (exactly as they are)
    imports_content = read_partial_file(partials_dir, "common_imports.py")
    file_parts.append("# --- Start of content from: common_imports.py ---")
    file_parts.append(imports_content)
    file_parts.append("# --- End of content from: common_imports.py ---\n")
    
    # 3. Process each partial file in the specified order
    for partial_filename in profile["partials_concat_order"]:
        # Skip header and imports as we've already processed them
        if partial_filename in ["common_header.py", "common_imports.py"]:
            continue
            
        print(f"  Processing partial: {partial_filename}")
        
        # Read the partial content
        partial_content = read_partial_file(partials_dir, partial_filename)
        
        # Extract content after import statements
        clean_content = extract_content_after_imports(partial_content)
        
        # Add section markers for clarity
        file_parts.append(f"# --- Start of content from: {partial_filename} ---")
        file_parts.append(clean_content)
        file_parts.append(f"# --- End of content from: {partial_filename} ---\n")
    
    # 4. Generate and add the final Function class
    function_class_code = generate_pipe_class(profile)
    file_parts.append(function_class_code)
    
    # 5. Combine all parts
    complete_content = "\n".join(file_parts)
    
    # 6. Write to output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(complete_content)
        print(f"Successfully generated: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error writing output file '{output_path}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate Minion/MinionS function files from partials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generator_script.py minion_default
  python generator_script.py minions_default --output_dir ./my_functions/
  python generator_script.py minion_default --config_file custom_config.json
        """
    )
    
    parser.add_argument(
        "profile", 
        help="Profile name from the configuration file (e.g., 'minion_default', 'minions_default')"
    )
    parser.add_argument(
        "--config_file",
        default="generation_config.json",
        help="Path to the generation configuration JSON file (default: generation_config.json)"
    )
    parser.add_argument(
        "--partials_dir",
        default="partials",
        help="Directory containing partial files (default: partials)"
    )
    parser.add_argument(
        "--output_dir",
        default="generated_functions",
        help="Directory to save generated function files (default: generated_functions)"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Handle list profiles option
    if args.list_profiles:
        print("Available profiles:")
        for profile_name, profile_data in config.items():
            print(f"  {profile_name}: {profile_data.get('description', 'No description')}")
        return
    
    # Validate inputs
    if not os.path.isdir(args.partials_dir):
        print(f"Error: Partials directory '{args.partials_dir}' does not exist.")
        sys.exit(1)
    
    # Generate the function file
    output_path = generate_function_file(
        args.profile, 
        config, 
        args.partials_dir, 
        args.output_dir
    )
    
    print(f"\n✅ Function generation complete!")
    print(f"📁 Output file: {output_path}")
    print(f"📋 You can now copy this file to your Open WebUI functions directory.")

if __name__ == "__main__":
    main()