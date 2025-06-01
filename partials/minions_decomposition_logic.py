from typing import List, Callable, Any, Dict # Added Dict
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
    call_claude_func: Callable[..., Awaitable[str]],
    query: str,
    scratchpad_content: str,
    num_chunks: int,
    max_tasks_per_round: int,
    current_round: int,
    conversation_log: List[str],
    debug_log: List[str]
) -> Tuple[List[str], str, str]:  # Added third return value for the prompt
    """
    Constructs the decomposition prompt, calls Claude, and parses tasks.
    Returns: (tasks, claude_response, decomposition_prompt)
    """
    # Base decomposition prompt
    base_decomposition_prompt = f'''You are a supervisor LLM in a multi-round process. Your goal is to answer: "{query}"
Context has been split into {num_chunks} chunks. A local LLM will process these chunks for each task you define.
Scratchpad (previous findings): {scratchpad_content if scratchpad_content else "Nothing yet."}

Based on the scratchpad and the original query, identify up to {max_tasks_per_round} specific, simple tasks for the local assistant.
If the information in the scratchpad is sufficient to answer the query, respond ONLY with the exact phrase "FINAL ANSWER READY." followed by the comprehensive answer.
Otherwise, list the new tasks clearly. Ensure tasks are actionable. Avoid redundant tasks.
Format tasks as a simple list (e.g., 1. Task A, 2. Task B).'''

    # Enhance the prompt with task formulation guidance
    decomposition_prompt = _enhance_decomposition_prompt(base_decomposition_prompt, valves)

    if valves.show_conversation:
        conversation_log.append(f"**🤖 Claude (Decomposition - Round {current_round}):** Sending prompt:\n```\n{decomposition_prompt}\n```")

    start_time_claude_decomp = 0
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

        tasks = _parse_tasks_helper(claude_response, max_tasks_per_round, debug_log, valves)
        
        if valves.debug_mode:
            debug_log.append(f"   Identified {len(tasks)} tasks for Round {current_round} from decomposition response.")
            for task_idx, task_item in enumerate(tasks):
                debug_log.append(f"    Task {task_idx+1} (Round {current_round}): {task_item[:100]}...")
        
        return tasks, claude_response, decomposition_prompt  # Return the prompt too

    except Exception as e:
        error_msg = f"❌ Error calling Claude for decomposition in round {current_round}: {e}"
        if valves.show_conversation:
            conversation_log.append(error_msg)
        if valves.debug_mode:
            debug_log.append(f"   {error_msg}")
        return [], f"CLAUDE_ERROR: {error_msg}", ""  # Return empty prompt on error
    
def _enhance_decomposition_prompt(base_prompt: str, valves: Any) -> str:
    """
    Enhances the decomposition prompt with additional guidance to ensure
    tasks are formulated to receive string responses.
    """
    task_formulation_guidance = '''

IMPORTANT TASK FORMULATION RULES:
1. Each task should request information that can be expressed as plain text
2. Avoid tasks that implicitly request structured data (like "Create a table of..." or "List with categories...")
3. Instead of "Extract and categorize X by Y", use "Describe X including information about Y"
4. Tasks should be answerable with narrative text, not data structures

GOOD TASK EXAMPLES:
- "Summarize the key advancements described in each Part, presenting them as a narrative"
- "Describe the AI winters mentioned, including their timeframes and characteristics"
- "Explain the progression of AI development across different periods"

BAD TASK EXAMPLES (avoid these):
- "Create a structured list of Parts with their key points"
- "Extract and categorize advancements by Part number"
- "Build a timeline table of AI winters"

Remember: The local assistant will return text strings, not structured data.'''
    
    return base_prompt + task_formulation_guidance
