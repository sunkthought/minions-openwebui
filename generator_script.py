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

def process_partial_content(content: str, filename: str, all_partials: List[str]) -> str:
    """Process partial content to handle imports correctly."""
    lines = content.split('\n')
    processed_lines = []
    
    # Get list of partial modules (without .py extension)
    partial_modules = [p.replace('.py', '') for p in all_partials if p.endswith('.py')]
    
    skip_until_close = False
    
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
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def replace_placeholders(content: str, placeholders: Dict[str, str]) -> str:
    """Replace placeholders in content."""
    for key, value in placeholders.items():
        content = content.replace(f"{{{key}}}", value)
    return content

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
    
    # 1. Process header
    header_content = load_partial(partials_dir, "common_header.py")
    header_content = replace_placeholders(header_content, profile["header_placeholders"])
    output_parts.append(header_content)
    
    # 2. Load all partials in order
    for partial_file in all_partials:
        if partial_file == "common_header.py":
            continue  # Already processed
        
        content = load_partial(partials_dir, partial_file)
        
        # Process content to remove inter-partial imports
        processed_content = process_partial_content(content, partial_file, all_partials)
        
        output_parts.append(processed_content)
    
    # 3. Generate the final Pipe class
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