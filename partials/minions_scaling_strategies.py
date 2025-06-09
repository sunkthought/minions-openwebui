# Partials File: partials/minions_scaling_strategies.py
import asyncio
import random
from typing import List, Dict, Any, Callable
from .minions_models import TaskResult, ScalingStrategy, RepeatedSamplingResult, DecomposedTask, ChunkingStrategy

async def apply_repeated_sampling_strategy(
    valves: Any,
    task: str,
    chunks: List[str],
    call_ollama_func: Callable,
    TaskResultModel: Any,
    num_samples: int = 3
) -> RepeatedSamplingResult:
    """
    Apply repeated sampling strategy: execute the same task multiple times
    with slight variations and aggregate results
    """
    original_result = None
    sample_results = []
    
    # Execute the task multiple times with different temperatures
    base_temp = getattr(valves, 'local_model_temperature', 0.7)
    temperatures = [base_temp, base_temp + 0.1, base_temp - 0.1] if num_samples == 3 else [base_temp + i * 0.1 for i in range(num_samples)]
    
    for i, temp in enumerate(temperatures[:num_samples]):
        # Modify valves temperature temporarily
        original_temp = valves.local_model_temperature if hasattr(valves, 'local_model_temperature') else 0.7
        valves.local_model_temperature = max(0.1, min(2.0, temp))  # Clamp between 0.1 and 2.0
        
        try:
            # Execute task on first chunk (or combined chunks if small)
            chunk_content = chunks[0] if chunks else ""
            
            task_prompt = f"""Task: {task}

Content to analyze:
{chunk_content}

Provide your analysis in the required format."""
            
            response = await call_ollama_func(
                valves,
                task_prompt,
                use_json=True,
                schema=TaskResultModel
            )
            
            # Parse the response
            from .minions_protocol_logic import parse_local_response
            parsed_result = parse_local_response(
                response, 
                is_structured=True, 
                use_structured_output=valves.use_structured_output,
                debug_mode=valves.debug_mode,
                TaskResultModel=TaskResultModel
            )
            
            task_result = TaskResultModel(**parsed_result)
            
            if i == 0:
                original_result = task_result
            else:
                sample_results.append(task_result)
                
        except Exception as e:
            if valves.debug_mode:
                print(f"DEBUG: Repeated sampling attempt {i+1} failed: {e}")
            # Create a fallback result
            fallback_result = TaskResultModel(
                explanation=f"Sampling attempt {i+1} failed: {str(e)}",
                answer=None,
                confidence="LOW"
            )
            if i == 0:
                original_result = fallback_result
            else:
                sample_results.append(fallback_result)
        finally:
            # Restore original temperature
            valves.local_model_temperature = original_temp
    
    # Aggregate results
    aggregated_result = _aggregate_sampling_results(original_result, sample_results, valves)
    
    # Calculate consistency score
    consistency_score = _calculate_consistency_score(original_result, sample_results)
    
    return RepeatedSamplingResult(
        original_result=original_result,
        sample_results=sample_results,
        aggregated_result=aggregated_result,
        confidence_boost=0.1 if consistency_score > 0.7 else 0.0,
        consistency_score=consistency_score
    )

async def apply_finer_decomposition_strategy(
    valves: Any,
    task: str,
    factor: int = 2
) -> DecomposedTask:
    """
    Apply finer decomposition strategy: break task into smaller subtasks
    """
    decomposition_prompt = f"""Break down the following task into {factor} smaller, more specific subtasks:

Original task: {task}

Create {factor} subtasks that together would fully address the original task. Each subtask should be:
1. More specific and focused than the original
2. Independent enough to be executed separately
3. Together they should cover all aspects of the original task

Respond with just the subtasks, one per line, numbered:"""

    try:
        # Use supervisor model to decompose the task
        response = await call_supervisor_model(valves, decomposition_prompt)
        
        # Parse subtasks from response
        subtasks = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                cleaned_task = line.split('.', 1)[-1].strip() if '.' in line else line.strip()
                if cleaned_task:
                    subtasks.append(cleaned_task)
        
        # Fallback if parsing failed
        if not subtasks:
            subtasks = [f"{task} (Part {i+1})" for i in range(factor)]
        
        return DecomposedTask(
            original_task=task,
            subtasks=subtasks[:factor]  # Ensure we don't exceed the requested factor
        )
        
    except Exception as e:
        if valves.debug_mode:
            print(f"DEBUG: Task decomposition failed: {e}")
        
        # Fallback decomposition
        return DecomposedTask(
            original_task=task,
            subtasks=[f"{task} (Part {i+1})" for i in range(factor)]
        )

def apply_context_chunking_strategy(
    context: str,
    chunk_size_reduction_factor: float = 0.5,
    overlap_ratio: float = 0.1
) -> ChunkingStrategy:
    """
    Apply context chunking strategy: create smaller chunks with overlap
    """
    from .common_file_processing import create_chunks
    
    # Calculate smaller chunk size
    original_chunk_size = len(context) // 3  # Rough estimate
    new_chunk_size = max(1000, int(original_chunk_size * chunk_size_reduction_factor))
    
    # Create overlapping chunks
    chunks = []
    overlap_chars = int(new_chunk_size * overlap_ratio)
    start = 0
    
    while start < len(context):
        end = min(start + new_chunk_size, len(context))
        chunk = context[start:end]
        chunks.append(chunk)
        
        if end >= len(context):
            break
            
        # Move start position accounting for overlap
        start = end - overlap_chars
    
    return ChunkingStrategy(
        chunk_size=new_chunk_size,
        overlap_ratio=overlap_ratio,
        chunks_created=len(chunks),
        overlap_chars=overlap_chars
    )

def _aggregate_sampling_results(original: TaskResult, samples: List[TaskResult], valves: Any) -> TaskResult:
    """Aggregate results from repeated sampling"""
    if not samples:
        return original
    
    # Count confidence levels
    all_results = [original] + samples
    confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    
    answers = []
    explanations = []
    
    for result in all_results:
        conf = result.confidence.upper() if result.confidence else "LOW"
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        if result.answer:
            answers.append(result.answer)
        if result.explanation:
            explanations.append(result.explanation)
    
    # Determine aggregate confidence
    total_results = len(all_results)
    if confidence_counts["HIGH"] >= total_results * 0.6:
        aggregate_confidence = "HIGH"
    elif confidence_counts["HIGH"] + confidence_counts["MEDIUM"] >= total_results * 0.6:
        aggregate_confidence = "MEDIUM"
    else:
        aggregate_confidence = "LOW"
    
    # Aggregate answer (take most common or combine)
    if answers:
        # Simple approach: take the original if it exists, otherwise first answer
        aggregate_answer = original.answer if original.answer else (answers[0] if answers else None)
    else:
        aggregate_answer = None
    
    # Aggregate explanation
    if explanations:
        aggregate_explanation = f"Aggregated from {len(all_results)} samples: {explanations[0]}"
        if len(explanations) > 1 and valves.debug_mode:
            aggregate_explanation += f" (Additional perspectives: {len(explanations)-1})"
    else:
        aggregate_explanation = "No explanations provided in sampling"
    
    return TaskResult(
        explanation=aggregate_explanation,
        answer=aggregate_answer,
        confidence=aggregate_confidence,
        citation=original.citation if hasattr(original, 'citation') else None
    )

def _calculate_consistency_score(original: TaskResult, samples: List[TaskResult]) -> float:
    """Calculate how consistent the sampling results are"""
    if not samples:
        return 1.0
    
    all_results = [original] + samples
    
    # Compare confidence levels
    confidences = [r.confidence.upper() if r.confidence else "LOW" for r in all_results]
    confidence_consistency = len(set(confidences)) == 1
    
    # Compare answer presence
    answers = [bool(r.answer and r.answer.strip()) for r in all_results]
    answer_consistency = len(set(answers)) <= 1
    
    # Simple consistency score
    consistency_score = 0.0
    if confidence_consistency:
        consistency_score += 0.5
    if answer_consistency:
        consistency_score += 0.5
    
    return consistency_score