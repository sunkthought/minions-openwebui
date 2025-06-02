# Partials File: partials/minion_convergence_detector.py
from typing import Optional, List, Dict, Any, Tuple

# Attempt to import RoundMetrics and Valves type hints for clarity,
# but handle potential circular dependency or generation-time issues
# by using 'Any' if direct import is problematic during generation.
try:
    from .minions_models import RoundMetrics
    # Assuming valves structure will be available, or use Any
    # from .minions_valves import MinionSValves # This might not exist as a direct importable type
    ValvesType = Any # Placeholder for valve types from pipe_self.valves
except ImportError:
    RoundMetrics = Any
    ValvesType = Any

class ConvergenceDetector:
    def __init__(self, debug_mode: bool = False):
        """Initializes the ConvergenceDetector."""
        self.debug_mode = debug_mode
        if self.debug_mode:
            print("DEBUG [ConvergenceDetector]: Initialized.")

    def calculate_round_convergence_metrics(
        self,
        current_round_metric: RoundMetrics,
        previous_round_metric: Optional[RoundMetrics]
    ) -> Dict[str, Any]:
        """
        Calculates specific convergence-related metrics for the current round.
        These will be used to update the current_round_metric object.
        """
        calculated_metrics = {}

        # 1. Information Gain Rate (using new_findings_count_this_round)
        # Ensure current_round_metric has the attribute, otherwise default to 0. Useful if RoundMetrics is Any.
        calculated_metrics["information_gain_rate"] = float(getattr(current_round_metric, "new_findings_count_this_round", 0))

        # 2. Novel Findings Percentage for this round
        new_findings = getattr(current_round_metric, "new_findings_count_this_round", 0)
        duplicate_findings = getattr(current_round_metric, "duplicate_findings_count_this_round", 0)
        total_findings_this_round = new_findings + duplicate_findings

        if total_findings_this_round > 0:
            calculated_metrics["novel_findings_percentage_this_round"] = new_findings / total_findings_this_round
        else:
            calculated_metrics["novel_findings_percentage_this_round"] = 0.0

        # 3. Task Failure Rate Trend
        trend = "N/A"
        if previous_round_metric:
            current_success_rate = getattr(current_round_metric, "success_rate", 0.0)
            previous_success_rate = getattr(previous_round_metric, "success_rate", 0.0)
            tolerance = 0.05
            if current_success_rate < previous_success_rate - tolerance:
                trend = "increasing_failures"
            elif current_success_rate > previous_success_rate + tolerance:
                trend = "decreasing_failures"
            else:
                trend = "stable_failures"
        calculated_metrics["task_failure_rate_trend"] = trend

        # 4. Predicted Value of Next Round (simple heuristic)
        predicted_value = "medium"
        novelty_current_round = calculated_metrics["novel_findings_percentage_this_round"]

        if novelty_current_round > 0.5:
            predicted_value = "high"
        elif novelty_current_round < 0.1:
            predicted_value = "low"

        if trend == "increasing_failures" and predicted_value == "medium":
            predicted_value = "low"
        elif trend == "decreasing_failures" and predicted_value == "medium":
            predicted_value = "high"

        calculated_metrics["predicted_value_of_next_round"] = predicted_value

        if self.debug_mode:
            round_num_debug = getattr(current_round_metric, "round_number", "Unknown")
            print(f"DEBUG [ConvergenceDetector]: Calculated metrics for round {round_num_debug}: {calculated_metrics}")

        return calculated_metrics

    def check_for_convergence(
        self,
        current_round_metric: RoundMetrics,
        sufficiency_score: float,
        total_rounds_executed: int,
        effective_novelty_to_use: float, # New parameter for dynamic threshold
        effective_sufficiency_to_use: float, # New parameter for dynamic threshold
        valves: ValvesType, # Still needed for min_rounds_before_convergence_check, convergence_rounds_min_novelty
        all_round_metrics: List[RoundMetrics]
    ) -> Tuple[bool, str]:
        """
        Checks if convergence criteria are met using potentially dynamic thresholds.
        """
        min_rounds_for_conv_check = getattr(valves, "min_rounds_before_convergence_check", 2)
        if total_rounds_executed < min_rounds_for_conv_check:
             if self.debug_mode:
                print(f"DEBUG [ConvergenceDetector]: Skipping convergence check for round {getattr(current_round_metric, 'round_number', 'N/A')}, min rounds for convergence check not met ({total_rounds_executed}/{min_rounds_for_conv_check}).")
             return False, ""

        # Convergence criteria:
        # 1. Low novelty for a certain number of consecutive rounds (using effective_novelty_to_use).
        # 2. Sufficiency score is above a threshold (using effective_sufficiency_to_use).

        low_novelty_streak = 0
        required_streak_length = getattr(valves, "convergence_rounds_min_novelty", 2)
        # Use the passed effective_novelty_to_use instead of getattr(valves, "convergence_novelty_threshold", 0.10)

        if len(all_round_metrics) >= required_streak_length:
            is_streak = True
            for i in range(required_streak_length):
                metric_to_check = all_round_metrics[-(i+1)]

                if not hasattr(metric_to_check, 'novel_findings_percentage_this_round') or \
                   getattr(metric_to_check, 'novel_findings_percentage_this_round') >= effective_novelty_to_use:
                    is_streak = False
                    break
            if is_streak:
                low_novelty_streak = required_streak_length

        current_round_num_debug = getattr(current_round_metric, "round_number", "N/A")
        # Use the passed effective_sufficiency_to_use instead of getattr(valves, "convergence_sufficiency_threshold", 0.7)

        if self.debug_mode:
            print(f"DEBUG [ConvergenceDetector]: Checking convergence for round {current_round_num_debug}:")
            print(f"  Using Effective Novelty Threshold: {effective_novelty_to_use:.2f}, Required streak: {required_streak_length}")
            print(f"  Actual low novelty streak achieved for check: {low_novelty_streak} (needs to be >= {required_streak_length})")
            print(f"  Sufficiency score: {sufficiency_score:.2f}, Using Effective Sufficiency Threshold: {effective_sufficiency_to_use:.2f}")

        if low_novelty_streak >= required_streak_length and sufficiency_score >= effective_sufficiency_to_use:
            reason = (
                f"Convergence detected: Novelty < {effective_novelty_to_use*100:.0f}% "
                f"for {low_novelty_streak} round(s) AND "
                f"Sufficiency ({sufficiency_score:.2f}) >= {effective_sufficiency_to_use:.2f}."
            )
            if self.debug_mode:
                print(f"DEBUG [ConvergenceDetector]: {reason}")
            return True, reason

        return False, ""
