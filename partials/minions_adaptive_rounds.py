# Partials File: partials/minions_adaptive_rounds.py
from typing import List, Dict, Any
from .minions_models import RoundAnalysis, TaskResult

def analyze_round_results(
    current_results: List[TaskResult], 
    previous_results: List[TaskResult], 
    query: str,
    valves: Any
) -> RoundAnalysis:
    """Analyze if another round would be beneficial"""
    
    # Calculate information gain
    information_gain = _calculate_information_gain(current_results, previous_results)
    
    # Calculate average confidence
    average_confidence = _calculate_average_confidence(current_results)
    
    # Calculate coverage ratio (simplified heuristic)
    coverage_ratio = _estimate_coverage_ratio(current_results, query)
    
    # Decision logic
    should_continue = _should_continue_rounds(
        information_gain, 
        average_confidence, 
        coverage_ratio, 
        valves
    )
    
    # Generate reason
    reason = _generate_stopping_reason(
        information_gain, 
        average_confidence, 
        coverage_ratio, 
        should_continue,
        valves
    )
    
    return RoundAnalysis(
        information_gain=information_gain,
        average_confidence=average_confidence,
        coverage_ratio=coverage_ratio,
        should_continue=should_continue,
        reason=reason
    )

def _calculate_information_gain(current_results: List[TaskResult], previous_results: List[TaskResult]) -> float:
    """Calculate information gain between rounds"""
    if not previous_results:
        return 1.0  # First round always has maximum gain
    
    if not current_results:
        return 0.0
    
    # Simple heuristic: compare unique content
    current_content = set()
    previous_content = set()
    
    for result in current_results:
        if result.answer:
            current_content.add(result.answer.lower().strip())
        if result.explanation:
            current_content.add(result.explanation.lower().strip())
    
    for result in previous_results:
        if result.answer:
            previous_content.add(result.answer.lower().strip())
        if result.explanation:
            previous_content.add(result.explanation.lower().strip())
    
    # Calculate ratio of new content
    if not current_content:
        return 0.0
    
    new_content = current_content - previous_content
    return len(new_content) / len(current_content) if current_content else 0.0

def _calculate_average_confidence(results: List[TaskResult]) -> float:
    """Calculate average confidence from results"""
    if not results:
        return 0.0
    
    confidence_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    total_confidence = 0.0
    
    for result in results:
        conf = result.confidence.upper() if result.confidence else "LOW"
        total_confidence += confidence_map.get(conf, 0.3)
    
    return total_confidence / len(results)

def _estimate_coverage_ratio(results: List[TaskResult], query: str) -> float:
    """Estimate how well the results cover the query"""
    if not results:
        return 0.0
    
    # Simple heuristic: count results with substantive answers
    substantive_results = 0
    for result in results:
        if result.answer and len(result.answer.strip()) > 10:
            substantive_results += 1
        elif result.explanation and len(result.explanation.strip()) > 20:
            substantive_results += 1
    
    # Estimate coverage based on ratio of substantive results
    # This is a simplified heuristic
    coverage = min(1.0, substantive_results / max(1, len(results)))
    
    # Boost coverage if we have high confidence results
    high_conf_count = sum(1 for r in results if r.confidence and r.confidence.upper() == "HIGH")
    if high_conf_count > 0:
        coverage = min(1.0, coverage + (high_conf_count * 0.1))
    
    return coverage

def _should_continue_rounds(
    information_gain: float, 
    average_confidence: float, 
    coverage_ratio: float, 
    valves: Any
) -> bool:
    """Determine if we should continue with more rounds"""
    
    min_info_gain = getattr(valves, 'min_info_gain', 0.1)
    confidence_threshold = getattr(valves, 'confidence_threshold_adaptive', 0.8)
    
    # Stop if information gain is too low
    if information_gain < min_info_gain:
        return False
    
    # Stop if we have high confidence and good coverage
    if average_confidence >= confidence_threshold and coverage_ratio >= 0.7:
        return False
    
    # Continue if we're still learning and not at maximum confidence
    return True

def _generate_stopping_reason(
    information_gain: float,
    average_confidence: float, 
    coverage_ratio: float,
    should_continue: bool,
    valves: Any
) -> str:
    """Generate a human-readable reason for the decision"""
    
    if not should_continue:
        if information_gain < getattr(valves, 'min_info_gain', 0.1):
            return f"Low information gain ({information_gain:.2f}) - diminishing returns detected"
        elif average_confidence >= getattr(valves, 'confidence_threshold_adaptive', 0.8):
            return f"High confidence threshold reached ({average_confidence:.2f}) with good coverage ({coverage_ratio:.2f})"
        else:
            return "Multiple stopping criteria met"
    else:
        reasons = []
        if information_gain >= getattr(valves, 'min_info_gain', 0.1):
            reasons.append(f"good information gain ({information_gain:.2f})")
        if average_confidence < getattr(valves, 'confidence_threshold_adaptive', 0.8):
            reasons.append(f"confidence can be improved ({average_confidence:.2f})")
        if coverage_ratio < 0.7:
            reasons.append(f"coverage can be improved ({coverage_ratio:.2f})")
        
        return f"Continue: {', '.join(reasons) if reasons else 'standard criteria met'}"

def should_stop_early(
    round_num: int,
    all_round_results: List[List[TaskResult]],
    query: str,
    valves: Any
) -> tuple[bool, str]:
    """
    Determine if we should stop early based on adaptive round control
    Returns (should_stop, reason)
    """
    
    if not getattr(valves, 'adaptive_rounds', True):
        return False, "Adaptive rounds disabled"
    
    min_rounds = getattr(valves, 'min_rounds', 1)
    if round_num < min_rounds:
        return False, f"Minimum rounds not reached ({round_num + 1}/{min_rounds})"
    
    if len(all_round_results) < 2:
        return False, "Need at least 2 rounds for analysis"
    
    current_results = all_round_results[-1]
    previous_results = all_round_results[-2] if len(all_round_results) > 1 else []
    
    analysis = analyze_round_results(current_results, previous_results, query, valves)
    
    return not analysis.should_continue, analysis.reason