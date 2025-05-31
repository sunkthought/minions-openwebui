import logging
from typing import List, Any, Dict, Type, Callable, Awaitable, Tuple # Added Type, Tuple
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
    call_claude_func: Callable[..., Awaitable[str]], # More specific callable type
    query: str,
    # context_len: int, # Replaced with scratchpad_content for more context
    scratchpad_content: str,
    num_chunks: int, # For the prompt
    max_tasks_per_round: int,
    current_round: int, # For logging
    conversation_log: List[str], # For logging if show_conversation
    debug_log: List[str] # For detailed debug logging
) -> Tuple[List[str], str]: # Added str to Tuple for claude_response
    """
    Constructs the decomposition prompt, calls Claude, and parses tasks.
    Returns a tuple: (list_of_task_strings, claude_response_string)
    """
    # This prompt construction will be based on the one in _execute_minions_protocol
    # from minions_pipe_method.py
    
    decomposition_prompt = f'''You are a supervisor LLM in a multi-round process. Your goal is to answer: "{query}"
Context has been split into {num_chunks} chunks. A local LLM will process these chunks for each task you define.
Scratchpad (previous findings): {scratchpad_content if scratchpad_content else "Nothing yet."}

Based on the scratchpad and the original query, identify up to {max_tasks_per_round} specific, simple tasks for the local assistant.
If the information in the scratchpad is sufficient to answer the query, respond ONLY with the exact phrase "FINAL ANSWER READY." followed by the comprehensive answer.
Otherwise, list the new tasks clearly. Ensure tasks are actionable. Avoid redundant tasks.
Format tasks as a simple list (e.g., 1. Task A, 2. Task B).'''

    if valves.show_conversation:
        conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round}):** Sending prompt:\n```\n{decomposition_prompt}\n```")

    start_time_claude_decomp = 0.0 # Ensure it's float
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

        # The "FINAL ANSWER READY." check will be done by the caller (_execute_minions_protocol)
        # This function will always try to parse tasks.

        tasks = _parse_tasks_helper(claude_response, max_tasks_per_round, debug_log, valves)
        
        if valves.debug_mode:
            debug_log.append(f"   Identified {len(tasks)} tasks for Round {current_round} from decomposition response.")
            for task_idx, task_item in enumerate(tasks):
                debug_log.append(f"    Task {task_idx+1} (Round {current_round}): {task_item[:100]}...")
        
        return tasks, claude_response # Return claude_response as well for "FINAL ANSWER READY" check

    except Exception as e:
        error_msg = f"❌ Error calling Claude for decomposition in round {current_round}: {e}"
        if valves.show_conversation:
            conversation_log.append(error_msg)
        if valves.debug_mode:
            debug_log.append(f"   {error_msg}")
        # In case of error, return empty list of tasks and an error message string
        return [], f"CLAUDE_ERROR: {error_msg}"

def _execute_generated_code(generated_code: str, JobManifest_cls: Type, logger: logging.Logger) -> List[Any]:
    """
    Safely executes LLM-generated Python code expected to define a list of JobManifest objects.
    JobManifest_cls is the class type for JobManifest model.
    Returns a list of JobManifest_cls instances.
    """
    # Prepare a safe execution environment
    # Import JobManifest_cls into the local scope for exec
    local_scope: Dict[str, Any] = {"JobManifest": JobManifest_cls, "job_manifests": []}

    # Potentially strip markdown code block fences if present
    code_to_execute = generated_code.strip()
    if code_to_execute.startswith("```python"):
        code_to_execute = code_to_execute[9:] # Remove ```python
        if code_to_execute.strip().endswith("```"):
            code_to_execute = code_to_execute.strip()[:-3] # Remove ```
    elif code_to_execute.startswith("```"): # Handle if just ``` not ```python
        code_to_execute = code_to_execute[3:]
        if code_to_execute.strip().endswith("```"):
            code_to_execute = code_to_execute.strip()[:-3]

    code_to_execute = code_to_execute.strip()

    try:
        # Execute the generated code
        # Pass a restricted global scope, JobManifest_cls is the only specific model allowed.
        # No builtins are passed to exec directly, relies on default available ones.
        exec(code_to_execute, {"JobManifest": JobManifest_cls, "__builtins__": {}}, local_scope)

        result = local_scope.get("job_manifests")

        if isinstance(result, list) and all(isinstance(item, JobManifest_cls) for item in result):
            if not result: # Empty list is valid if LLM decides no tasks needed
                logger.info("Generated code produced an empty list of JobManifests.")
            return result # type: ignore # We've checked it's List[JobManifest_cls]
        elif result is None and code_to_execute.startswith("[") and code_to_execute.endswith("]"):
            # Fallback: try to eval if the code itself is a list literal
            logger.info("No 'job_manifests' variable found, attempting to eval code as list literal.")
            try:
                # We need to ensure JobManifest_cls is available in the eval context as well
                # Restricted builtins for safety. Only allowing 'list' and 'dict' for constructing JobManifests if needed.
                eval_result = eval(code_to_execute, {"JobManifest": JobManifest_cls, "__builtins__": {"list": list, "dict": dict}}, {})
                if isinstance(eval_result, list) and all(isinstance(item, JobManifest_cls) for item in eval_result):
                    if not eval_result:
                         logger.info("Evaluated code produced an empty list of JobManifests.")
                    return eval_result # type: ignore
                else:
                    logger.warning(f"Evaluated code did not produce a List[JobManifest_cls]. Type: {type(eval_result)}")
                    raise ValueError("Generated code, when evaluated, did not produce a list of JobManifest_cls objects.")
            except Exception as e_eval:
                logger.error(f"Error evaluating generated code as a list literal: {e_eval}")
                logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
                logger.debug(f"Problematic code for eval: {logged_code}")
                raise ValueError(f"Generated code could not be executed to find 'job_manifests' nor evaluated as a list literal. Eval error: {e_eval}") from e_eval
        elif result is not None:
            logger.warning(f"Generated code assigned 'job_manifests', but it was not List[JobManifest_cls]. Type: {type(result)}")
            raise ValueError(f"Generated code assigned 'job_manifests', but it was not a list of JobManifest_cls objects. Found type: {type(result)}")
        else:
            logger.warning("Generated code did not assign to 'job_manifests' and does not appear to be a direct list literal.")
            raise ValueError("Generated code did not produce a list of JobManifest_cls objects, and 'job_manifests' variable was not found or correctly assigned.")

    except SyntaxError as e:
        logger.error(f"Syntax error in generated code: {e}")
        logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
        logger.debug(f"Problematic code for syntax error: {logged_code}")
        raise ValueError(f"Syntax error in generated code: {e}") from e
    except NameError as e:
        logger.error(f"Name error in generated code: {e}. Ensure JobManifest_cls is correctly used and all necessary variables are defined if code isn't self-contained.")
        logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
        logger.debug(f"Problematic code for name error: {logged_code}")
        raise ValueError(f"Name error in generated code: {e}. This might happen if the code tries to use undefined variables or modules other than JobManifest_cls.") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution of generated code: {e}")
        logged_code = code_to_execute[:500] + "..." if len(code_to_execute) > 500 else code_to_execute
        logger.debug(f"Problematic code for general exception: {logged_code}")
        raise ValueError(f"An unexpected error occurred during execution of generated code: {e}") from e
