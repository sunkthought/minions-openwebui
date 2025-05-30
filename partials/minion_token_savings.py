from typing import List, Dict, Any, Tuple

def calculate_minion_token_savings(
    conversation_history: List[Tuple[str, str]], 
    context: str, 
    query: str,
    chars_per_token: float = 3.5  # Average characters per token, can be adjusted
) -> Dict[str, Any]:
    """
    Calculates token savings for the Minion (conversational) protocol.

    Args:
        conversation_history: A list of tuples, where each tuple is (role, message_content).
                              'assistant' role typically refers to the remote model.
        context: The full context string that would have been sent in a traditional approach.
        query: The user's query string.
        chars_per_token: An estimated average number of characters per token.

    Returns:
        A dictionary containing:
            'traditional_tokens': Estimated tokens if context+query were sent directly.
            'minion_tokens': Estimated tokens used by the remote model in the Minion protocol.
            'token_savings': Difference between traditional and Minion tokens.
            'percentage_savings': Percentage of tokens saved.
    """
    # Calculate tokens for the traditional approach (sending full context + query)
    traditional_tokens = int((len(context) + len(query)) / chars_per_token)
    
    # Calculate tokens for the Minion approach
    # This typically counts tokens from messages involving the remote model (e.g., Claude)
    # In this specific Minion protocol, 'assistant' messages in history are Claude's.
    minion_protocol_remote_model_tokens = 0
    for role, message_content in conversation_history:
        if role == "assistant":  # Messages from/to the remote model
            minion_protocol_remote_model_tokens += int(len(message_content) / chars_per_token)

    # The 'minion_tokens' are those specifically attributed to the remote model's involvement
    minion_tokens = minion_protocol_remote_model_tokens
    
    token_savings = traditional_tokens - minion_tokens
    percentage_savings = (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
    
    return {
        'traditional_tokens': traditional_tokens,
        'minion_tokens': minion_tokens,
        'token_savings': token_savings,
        'percentage_savings': percentage_savings
    }
