# Partials File: partials/minion_prompts.py
from typing import List, Tuple, Any, Optional
from .prompt_templates import PromptTemplates

# This file will store prompt generation functions for the Minion (single-turn) protocol.

def get_minion_initial_claude_prompt(query: str, context_len: int, valves: Any) -> str:
    """
    Returns the initial prompt for Claude in the Minion protocol.
    Enhanced with better question generation guidance.
    """
    return PromptTemplates.get_minion_initial_claude_prompt(
        query=query,
        context_len=context_len,
        max_rounds=valves.max_rounds
    )

def get_minion_conversation_claude_prompt(history: List[Tuple[str, str]], original_query: str, valves: Any) -> str:
    """
    Returns the prompt for Claude during subsequent conversation rounds in the Minion protocol.
    Enhanced with better guidance for follow-up questions.
    """
    current_round = len(history) // 2 + 1
    conversation_history = PromptTemplates.build_conversation_history(history)
    
    return PromptTemplates.get_minion_conversation_claude_prompt(
        original_query=original_query,
        current_round=current_round,
        max_rounds=valves.max_rounds,
        conversation_history=conversation_history
    )

def get_minion_local_prompt(context: str, query: str, claude_request: str, valves: Any) -> str:
    """
    Returns the prompt for the local Ollama model in the Minion protocol.
    Enhanced with better guidance for structured, useful responses.
    """
    return PromptTemplates.get_minion_local_prompt(
        context=context,
        query=query,
        claude_request=claude_request,
        use_structured_output=valves.use_structured_output
    )

def get_minion_initial_claude_prompt_with_state(query: str, context_len: int, valves: Any, conversation_state: Optional[Any] = None, phase_guidance: Optional[str] = None) -> str:
    """
    Enhanced version of initial prompt that includes conversation state if available.
    """
    base_prompt = get_minion_initial_claude_prompt(query, context_len, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        if state_summary:
            base_prompt = PromptTemplates.enhance_prompt_with_state(
                base_prompt, state_summary
            )
    
    # Add phase guidance if provided
    if phase_guidance and valves.enable_flow_control:
        base_prompt = PromptTemplates.enhance_prompt_with_phase_guidance(
            base_prompt, phase_guidance
        )
    
    return base_prompt

def get_minion_conversation_claude_prompt_with_state(history: List[Tuple[str, str]], original_query: str, valves: Any, conversation_state: Optional[Any] = None, previous_questions: Optional[List[str]] = None, phase_guidance: Optional[str] = None) -> str:
    """
    Enhanced version of conversation prompt that includes conversation state if available.
    """
    base_prompt = get_minion_conversation_claude_prompt(history, original_query, valves)
    
    if conversation_state and valves.track_conversation_state:
        state_summary = conversation_state.get_state_summary()
        topics_covered = getattr(conversation_state, 'topics_covered', [])
        key_findings_count = len(getattr(conversation_state, 'key_findings', []))
        information_gaps_count = len(getattr(conversation_state, 'information_gaps', []))
        
        base_prompt = PromptTemplates.enhance_prompt_with_state(
            base_prompt, state_summary, topics_covered, 
            key_findings_count, information_gaps_count
        )
    
    # Add deduplication guidance if previous questions provided
    if previous_questions and valves.enable_deduplication:
        base_prompt = PromptTemplates.enhance_prompt_with_deduplication(
            base_prompt, previous_questions
        )
    
    # Add phase guidance if provided
    if phase_guidance and valves.enable_flow_control:
        base_prompt = PromptTemplates.enhance_prompt_with_phase_guidance(
            base_prompt, phase_guidance
        )
    
    return base_prompt