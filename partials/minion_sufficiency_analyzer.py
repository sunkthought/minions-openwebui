# Partials File: partials/minion_sufficiency_analyzer.py
import re
from typing import Dict, List, Tuple, Any

class InformationSufficiencyAnalyzer:
    """
    Analyzes text to determine if it sufficiently addresses components of an initial query.
    """
    def __init__(self, query: str, debug_mode: bool = False):
        """
        Initializes the analyzer with the user's query and debug mode.
        """
        self.query = query
        self.debug_mode = debug_mode
        self.components: Dict[str, Dict[str, Any]] = {} # Stores {component_name: {"keywords": [...], "is_addressed": False, "confidence": 0.0}}
        self._identify_components()

    def _identify_components(self):
        """
        Identifies key components or topics from the user's query.
        Uses heuristics like quoted phrases, capitalized words, and generic fallbacks.
        """
        # Basic keyword extraction. This is a simple heuristic and can be expanded.
        # It looks for Nouns, Proper Nouns, and Adjectives, trying to form simple topics.
        # Example: "Compare the budget and timeline for Project Alpha and Project Beta"
        # Might identify: "budget", "timeline", "Project Alpha", "Project Beta"
        # Then forms components like "budget Project Alpha", "timeline Project Alpha", etc.

        # For simplicity in this iteration, we'll use a more direct approach:
        # Look for quoted phrases or capitalized words as potential components.
        # Or, define a few generic components if query is too simple.

        # Let's try to find quoted phrases first
        quoted_phrases = re.findall(r'"([^"]+)"', self.query)
        for phrase in quoted_phrases:
            self.components[phrase] = {"keywords": [kw.lower() for kw in phrase.split()], "is_addressed": False, "confidence": 0.0}

        # If no quoted phrases, look for capitalized words/phrases (potential proper nouns or topics)
        if not self.components:
            common_fillers = [
                "Here", "The", "This", "That", "There", "It", "Who", "What", "When", "Where", "Why", "How",
                "Is", "Are", "Was", "Were", "My", "Your", "His", "Her", "Its", "Our", "Their", "An",
                "As", "At", "But", "By", "For", "From", "In", "Into", "Of", "On", "Or", "Over",
                "So", "Then", "To", "Under", "Up", "With", "I"
            ]
            common_fillers_lower = [f.lower() for f in common_fillers]

            # Regex to find sequences of capitalized words, possibly including 'and', 'or', 'for', 'the'
            potential_topics = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+(?:and|or|for|the|[A-Z][a-zA-Z]*))*\b', self.query)

            topics = []
            for pt in potential_topics:
                is_multi_word = ' ' in pt
                # A single word is a common filler if its lowercased version is in common_fillers_lower
                is_common_filler_single_word = not is_multi_word and pt.lower() in common_fillers_lower
                # A single word is significant if it's an acronym (all upper) or longer than 3 chars
                is_significant_single_word = not is_multi_word and (pt.isupper() or len(pt) > 3)

                if is_multi_word: # Always include multi-word capitalized phrases
                    topics.append(pt)
                elif is_significant_single_word and not is_common_filler_single_word:
                    # Include significant single words only if they are NOT common fillers
                    topics.append(pt)

            if self.debug_mode and potential_topics:
                 print(f"DEBUG [SufficiencyAnalyzer]: Potential capitalized topics found: {potential_topics}")
                 print(f"DEBUG [SufficiencyAnalyzer]: Filtered topics after common word/length check: {topics}")

            for topic in topics:
                # Avoid adding overlapping sub-phrases if a larger phrase is already a component
                is_sub_phrase = False
                for existing_comp in self.components.keys():
                    if topic in existing_comp and topic != existing_comp:
                        is_sub_phrase = True
                        break
                if not is_sub_phrase:
                    self.components[topic] = {"keywords": [kw.lower() for kw in topic.split()], "is_addressed": False, "confidence": 0.0}

        # If still no components (e.g., simple query like "summarize this"), create generic ones.
        if not self.components:
            if "compare" in self.query.lower() or "contrast" in self.query.lower():
                self.components["comparison points"] = {"keywords": ["compare", "contrast", "similarit", "difference"], "is_addressed": False, "confidence": 0.0} # Added common keywords
                self.components["subject 1 details"] = {"keywords": [], "is_addressed": False, "confidence": 0.0} # Placeholder, keyword matching might be hard
                self.components["subject 2 details"] = {"keywords": [], "is_addressed": False, "confidence": 0.0} # Placeholder
            elif "summarize" in self.query.lower() or "overview" in self.query.lower():
                self.components["main points"] = {"keywords": ["summary", "summarize", "overview", "main point", "key aspect"], "is_addressed": False, "confidence": 0.0}
                self.components["details"] = {"keywords": ["detail", "specific", "elaborate"], "is_addressed": False, "confidence": 0.0}
            else: # Default fallback component
                self.components["overall query"] = {"keywords": [kw.lower() for kw in self.query.split()[:5]], "is_addressed": False, "confidence": 0.0} # Use first few words of query

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Identified components: {list(self.components.keys())}")

    def update_components(self, text_to_analyze: str, round_avg_confidence: float):
        """
        Updates the status of query components based on the provided text and confidence.
        Marks components as addressed if their keywords are found in the text.
        """
        # In this version, we'll use round_avg_confidence as a proxy for the confidence
        # of the information that might address a component.
        # A more advanced version could try to link specific task confidences.
        text_lower = text_to_analyze.lower()
        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Updating components based on text (first 100 chars): {text_lower[:100]}...")

        for comp_name, comp_data in self.components.items():
            if not comp_data["is_addressed"]:
                # If keywords are defined, require all keywords for the component to be present.
                # This is a strict rule and might need adjustment (e.g., any keyword, or a percentage).
                if comp_data["keywords"]:
                    all_keywords_present = all(kw in text_lower for kw in comp_data["keywords"])
                    if all_keywords_present:
                        comp_data["is_addressed"] = True
                        comp_data["confidence"] = round_avg_confidence # Use round's average confidence
                        if self.debug_mode:
                            print(f"DEBUG [SufficiencyAnalyzer]: Component '{comp_name}' ADDRESSED by keyword match. Confidence set to {round_avg_confidence:.2f}")
                # If no keywords (e.g. generic components like "subject 1 details"), this logic won't address them.
                # This is a limitation of the current basic keyword approach for generic components.
                # For this iteration, such components might remain unaddressed unless their names/generic keywords appear.

    def calculate_sufficiency_score(self) -> Tuple[float, float, Dict[str, bool]]:
        """
        Calculates the overall sufficiency score based on component coverage and confidence.
        Returns score, coverage percentage, and status of each component.
        """
        if not self.components:
            return 0.0, 0.0, {}

        addressed_components_count = 0
        total_confidence_of_addressed = 0.0
        component_status_for_metrics: Dict[str, bool] = {}

        for comp_name, comp_data in self.components.items():
            component_status_for_metrics[comp_name] = comp_data["is_addressed"]
            if comp_data["is_addressed"]:
                addressed_components_count += 1
                total_confidence_of_addressed += comp_data["confidence"]

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Addressed components: {addressed_components_count}/{len(self.components)}")

        component_coverage_percentage = (addressed_components_count / len(self.components)) if len(self.components) > 0 else 0.0

        avg_confidence_of_addressed = (total_confidence_of_addressed / addressed_components_count) if addressed_components_count > 0 else 0.0

        # Score is a product of coverage and average confidence of what's covered.
        sufficiency_score = component_coverage_percentage * avg_confidence_of_addressed

        if self.debug_mode:
            print(f"DEBUG [SufficiencyAnalyzer]: Coverage: {component_coverage_percentage:.2f}, Avg Confidence of Addressed: {avg_confidence_of_addressed:.2f}, Sufficiency Score: {sufficiency_score:.2f}")

        return sufficiency_score, component_coverage_percentage, component_status_for_metrics

    def get_analysis_details(self) -> Dict[str, Any]:
        """
        Returns a dictionary with the sufficiency score, coverage, and component status.
        """
        sufficiency_score, component_coverage_percentage, component_status = self.calculate_sufficiency_score()
        return {
            "sufficiency_score": sufficiency_score,
            "component_coverage_percentage": component_coverage_percentage,
            "information_components_status": component_status # Changed key name slightly to avoid conflict if used directly in RoundMetrics
        }
