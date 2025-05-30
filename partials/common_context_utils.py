from typing import List, Dict, Any
from pydantic import BaseModel # For valves type hint

# Forward declaration for type hinting if Valves is defined in a shared place later
ValvesType = Any # Replace with actual Valves type if available

def extract_context_from_messages(
    messages: List[Dict[str, Any]]
) -> str:
    """Extract context from conversation history"""
    context_parts = []

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Assume messages longer than 200 chars contain context/documents
            if len(content) > 200:
                context_parts.append(content)

    return "\n\n".join(context_parts)

async def extract_file_content(
    valves: ValvesType,  # For debug_mode access
    file_info: Dict[str, Any]
) -> str:
    """Extract text content from a single file using Open WebUI's file API"""
    try:
        file_id = file_info.get("id")
        file_name = file_info.get("name", "unknown")

        if not file_id:
            return f"[Could not get file ID for {file_name}]"

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            # Note: In a shared utility, direct printing might not be ideal.
            # Consider logging or returning debug info.
            # For now, keeping it similar to original for simplicity.
            # However, the original returns the debug string, effectively stopping further processing for this file.
            return f"[DEBUG] File ID: {file_id}, Name: {file_name}, Info: {str(file_info)}]"


        # If the file info contains content directly, use it
        if "content" in file_info:
            return file_info["content"]
        
        # Fallback for Open WebUI typical file structure if content not pre-loaded
        # This part is speculative based on typical OpenWebUI file handling if content isn't directly in `file_info`
        # The original code implies that 'content' should be there or it's a non-extractable scenario.
        # The original code's debug return for "File ID..." happens BEFORE checking "content" in file_info,
        # which means if debug_mode is on, it *never* extracts content.
        # Replicating that logic: if debug_mode is on and it's not a special case, it returns the debug string.
        # The original logic for non-debug mode is to check 'content', then provide a generic "File detected" message.

        # Simulating original behavior more closely:
        # The original code had the debug return *before* the "content" in file_info check.
        # If debug_mode is true, the file_info string is returned.
        # If not, it proceeds. The `hasattr` check for valves.debug_mode is good.

        # Placeholder for actual file content fetching if needed for OpenWebUI
        # For example, if __request__ object or similar is needed to fetch by ID:
        # This would require passing __request__ or a file_fetcher callable.
        # For now, the original logic seems to rely on "content" being present
        # or provides a "File detected" message.

        file_type = file_info.get("type", "unknown")
        file_size = file_info.get("size", "unknown")
        
        # Default message if content not found and not in debug (or debug didn't return early)
        return f"[File detected: {file_name} (Type: {file_type}, Size: {file_size})\nNote: File content extraction needs to be configured or content is not directly available in provided file_info]"

    except Exception as e:
        # Again, consider logging here
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[Error extracting file content: {str(e)}]"
        return f"[Error extracting file content]" # Simplified error for non-debug

async def extract_context_from_files(
    valves: ValvesType, # For debug_mode access and passing to extract_file_content
    files: List[Dict[str, Any]]
) -> str:
    """Extract text content from uploaded files using Open WebUI's file system"""
    try:
        if not files:
            return ""

        files_content = []

        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            files_content.append(f"[DEBUG] Found {len(files)} uploaded files")

        for file_info in files:
            if isinstance(file_info, dict):
                # Call the refactored standalone function, passing valves
                content = await extract_file_content(valves, file_info)
                if content:
                    # Check if the content is one of the bracketed messages (errors/debugs)
                    # and avoid prepending "FILE: " if it is.
                    if content.startswith("[") and content.endswith("]"):
                        # Optionally, only add debug/error messages if in debug mode
                        if hasattr(valves, 'debug_mode') and valves.debug_mode:
                            files_content.append(content)
                        # Or, always add them if they are considered important enough
                        # files_content.append(content) 
                    else:
                        file_name = file_info.get("name", "unknown_file")
                        files_content.append(
                            f"=== FILE: {file_name} ===\n{content}"
                        )
                        
        return "\n\n".join(files_content) if files_content else ""

    except Exception as e:
        if hasattr(valves, 'debug_mode') and valves.debug_mode:
            return f"[File extraction error: {str(e)}]"
        return "" # Return empty string on error if not in debug mode
