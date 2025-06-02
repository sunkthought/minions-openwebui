import re
from typing import List, Dict, Optional, TypedDict, Pattern, Tuple, Match

# Updated QueryTemplate data structure
class QueryTemplate(TypedDict):
    """
    Represents a template for query reformulation.

    Attributes:
        name: The name of the template.
        pattern: A compiled regex pattern to match against the query.
        sub_query_templates: A list of templates for generating sub-queries.
        variable_slots: A list of variable names corresponding to regex capture groups.
                        Order matters if groups are not named. If named, names should match.
    """
    name: str
    pattern: Pattern[str] 
    sub_query_templates: List[str]
    variable_slots: List[str]


class QueryReformulator:
    """
    A class for reformulating queries based on predefined templates.
    """

    def __init__(self, debug_mode: bool = False):
        """
        Initializes the QueryReformulator.

        Args:
            debug_mode: If True, enables debug logging.
        """
        self.templates: List[QueryTemplate] = []
        self.debug_mode = debug_mode
        self._load_templates()

    def _log_debug(self, message: str):
        """
        Logs a debug message if debug_mode is enabled.

        Args:
            message: The message to log.
        """
        if self.debug_mode:
            print(f"[DEBUG] QueryReformulator: {message}")

    def _load_templates(self):
        """
        Loads or defines the query reformulation templates.
        This method should be implemented to populate the self.templates list.
        """
        self._log_debug("Loading templates...")
        self.templates = [] # Clear any existing templates

        # 1. Comparative Template
        self.templates.append({
            "name": "Comparative",
            "pattern": re.compile(r"compare (?P<X>.*?) and (?P<Y>.*?)(?: regarding (?P<CONTEXT>.*?))?$", re.IGNORECASE),
            "sub_query_templates": [
                "What is {X}?",
                "What is {Y}?",
                "What are the differences between {X} and {Y}?",
                "What are the differences between {X} and {Y} regarding {CONTEXT}?" # This will only format if CONTEXT is present
            ],
            "variable_slots": ["X", "Y", "CONTEXT"]
        })
        self.templates.append({ # Variant for "difference between"
            "name": "Comparative_Difference",
            "pattern": re.compile(r"what is the difference between (?P<X>.*?) and (?P<Y>.*?)(?: regarding (?P<CONTEXT>.*?))?$", re.IGNORECASE),
            "sub_query_templates": [
                "What is {X}?",
                "What is {Y}?",
                "Key differences: {X} vs {Y}",
                "Key differences: {X} vs {Y} focusing on {CONTEXT}"
            ],
            "variable_slots": ["X", "Y", "CONTEXT"]
        })

        # 2. Temporal Analysis Template
        self.templates.append({
            "name": "Temporal Analysis",
            "pattern": re.compile(r"how has (?P<X>.*?) changed(?: from (?P<period1>.*?) to (?P<period2>.*?))?(?: over time)?\??$", re.IGNORECASE),
            "sub_query_templates": [
                "What was {X} in {period1}?",
                "What is {X} in {period2}?",
                "What is the trend for {X} between {period1} and {period2}?",
                "Overall trend for {X}." # Fallback if periods are not specified
            ],
            "variable_slots": ["X", "period1", "period2"]
        })
        self.templates.append({ # Variant for "trend of X from Y to Z"
            "name": "Temporal Analysis_Trend",
            "pattern": re.compile(r"what is the trend of (?P<X>.*?)(?: from (?P<period1>.*?) to (?P<period2>.*?))?\??$", re.IGNORECASE),
            "sub_query_templates": [
                "What was {X} in {period1}?",
                "What is {X} in {period2}?",
                "Show trend for {X} from {period1} to {period2}.",
                "Show overall trend for {X}."
            ],
            "variable_slots": ["X", "period1", "period2"]
        })

        # 3. Comprehensive Analysis Template
        self.templates.append({
            "name": "Comprehensive Analysis",
            "pattern": re.compile(r"(?:analyze|tell me everything about|full report on|give me a comprehensive overview of) (?P<X>.*?)$", re.IGNORECASE),
            "sub_query_templates": [
                "What is {X}?",
                "What are the key metrics for {X}?",
                "What are the recent trends for {X}?",
                "What are the implications or future outlook for {X}?"
            ],
            "variable_slots": ["X"]
        })

        # 4. Causal Analysis Template
        self.templates.append({
            "name": "Causal Analysis_Why",
            "pattern": re.compile(r"why did (?P<X>.*?) (?:happen|occur|transpire)\??$", re.IGNORECASE),
            "sub_query_templates": [
                "Describe {X}.",
                "What factors influenced {X}?",
                "What was the primary cause of {X}?"
            ],
            "variable_slots": ["X"]
        })
        self.templates.append({
            "name": "Causal Analysis_WhatCaused",
            "pattern": re.compile(r"what (?:is|was|were) the cause[s]? of (?P<X>.*?)\??$", re.IGNORECASE),
            "sub_query_templates": [
                "Describe the event: {X}.",
                "Identify factors contributing to {X}.",
                "What is the main cause of {X}?"
            ],
            "variable_slots": ["X"]
        })

        self._log_debug(f"Loaded {len(self.templates)} templates.")

    def _match_template(self, query: str) -> Tuple[Optional[QueryTemplate], Optional[Match[str]], float]:
        """
        Matches the given query against the loaded templates.

        Args:
            query: The original query string.

        Returns:
            A tuple containing:
                - The matched QueryTemplate (Optional[QueryTemplate])
                - The regex match object (Optional[Match[str]])
                - A confidence score (float), 0.0 if no match.
        """
        self._log_debug(f"Attempting to match query: '{query}'")
        for t in self.templates:
            match = t["pattern"].search(query)
            if match:
                # For now, using a fixed high confidence for any direct regex match.
                # This could be refined later, e.g., based on template specificity.
                confidence = 0.9 
                self._log_debug(f"Query matched template: '{t['name']}' with pattern: '{t['pattern'].pattern}'. Confidence: {confidence}")
                return t, match, confidence
        self._log_debug(f"No template matched the query: '{query}'")
        return None, None, 0.0

    def _extract_variables(self, match_obj: Match[str], template: QueryTemplate) -> Dict[str, str]:
        """
        Extracts variables from the regex match object based on the template's variable_slots.

        Args:
            match_obj: The regex match object.
            template: The matched QueryTemplate containing the variable_slots.

        Returns:
            A dictionary of extracted variables.
        """
        variable_slots = template["variable_slots"]
        self._log_debug(f"Extracting variables using pattern: '{template['pattern'].pattern}' for slots: {variable_slots}")
        
        variables: Dict[str, str] = {}

        # Try extracting by named groups first
        named_groups = match_obj.groupdict()
        if named_groups:
            for slot_name in variable_slots:
                variables[slot_name] = named_groups.get(slot_name) 
        else:
            # Fallback to ordered groups if no named groups are defined in regex
            groups = match_obj.groups()
            for i, slot_name in enumerate(variable_slots):
                if i < len(groups):
                    variables[slot_name] = groups[i]
                else:
                    variables[slot_name] = None 

        # Ensure all variable slots defined in the template are present in the dict, even if None
        for slot in variable_slots:
            if slot not in variables: # This case should ideally not be hit if regex and slots are well-defined
                variables[slot] = None
        
        # This log is now done in the reformulate method after extraction.
        # self._log_debug(f"Extracted variables: {variables}") 
        return variables

    def _generate_sub_queries(self, template: QueryTemplate, variables: Dict[str, str]) -> List[str]:
        """
        Generates sub-queries based on the template and extracted variables.

        Args:
            template: The matched QueryTemplate.
            variables: A dictionary of extracted variables.

        Returns:
            A list of generated sub-query strings.
        """
        self._log_debug(f"Generating sub-queries for template: '{template['name']}' with variables: {variables}")
        sub_queries: List[str] = []
        
        # Filter out None values from variables to avoid issues with .format() if a slot is optional and not present
        # However, .format() with {None} results in "None", which might be acceptable or desired.
        # For more control, we can conditionally add sub-queries.
        
        valid_variables = {k: v for k, v in variables.items() if v is not None}

        for sub_template_str in template["sub_query_templates"]:
            try:
                # Check if all necessary placeholders in this specific sub_template_str can be filled
                # This is a simple check; more sophisticated checks might involve parsing the sub_template_str
                # to see which variables it actually uses.
                # For now, we attempt to format and skip if a key is truly missing and essential for *that* sub_template.
                
                # Example: "Compare {X} and {Y} regarding {CONTEXT}"
                # If CONTEXT is None, we might not want this sub-query.
                # A simple way: if a variable slot in the template is None, and that slot is in the sub_template_str, skip.
                # More robust: check if all {variables} in sub_template_str are in valid_variables
                
                # Quick check for optional variables like CONTEXT
                if "{CONTEXT}" in sub_template_str and "CONTEXT" not in valid_variables and template['name'] in ["Comparative", "Comparative_Difference"]:
                    # Skip this specific sub-query if CONTEXT is needed but not available
                    self._log_debug(f"Skipping sub-query due to missing optional CONTEXT: '{sub_template_str}'")
                    continue
                if ("{period1}" in sub_template_str or "{period2}" in sub_template_str) and \
                   ("period1" not in valid_variables or "period2" not in valid_variables) and \
                   template['name'] in ["Temporal Analysis", "Temporal Analysis_Trend"]:
                     # Skip if period1 or period2 is needed but not available for specific sub-queries
                     is_overall_trend_subquery = "Overall trend for" in sub_template_str or "Show overall trend for" in sub_template_str
                     if not is_overall_trend_subquery: # Don't skip the "Overall trend" subquery
                        self._log_debug(f"Skipping sub-query due to missing period data: '{sub_template_str}'")
                        continue


                # Use all variables (including those that are None) for formatting,
                # as "None" might be an acceptable string representation.
                # If a key is missing entirely (which shouldn't happen if _extract_variables is correct),
                # .format(**variables) would raise a KeyError.
                sub_queries.append(sub_template_str.format(**variables))
            except KeyError as e:
                # This might happen if a variable slot defined in `variable_slots` was not actually captured by the regex
                # or if a sub_template string uses a variable not in `variable_slots`.
                self._log_debug(f"KeyError during sub-query formatting for '{sub_template_str}': {e}. Variables available: {variables}. Template slots: {template['variable_slots']}")
        
        self._log_debug(f"Generated sub-queries: {sub_queries}")
        return sub_queries

    def reformulate(self, original_query: str, query_metadata: Optional[Dict] = None) -> List[str]:
        """
        Reformulates the original query into a list of sub-queries or alternative queries.

        Args:
            original_query: The user's original query.
            query_metadata: Optional dictionary containing metadata about the query.

        Returns:
            A list of reformulated query strings.
        """
        self._log_debug(f"Starting reformulation for query: '{original_query}'")
        if query_metadata and self.debug_mode: # Check debug_mode before logging potentially large metadata
            self._log_debug(f"Received query_metadata. Keys: {list(query_metadata.keys())}")

        matched_template, match_obj, confidence_score = self._match_template(original_query)

        if not matched_template or not match_obj:
            self._log_debug(f"No suitable template found for query: '{original_query}'. Confidence: {confidence_score}")
            return [original_query] 

        self._log_debug(f"Matched template: '{matched_template['name']}' with confidence: {confidence_score}")
        
        variables = self._extract_variables(match_obj, matched_template)
        self._log_debug(f"Extracted variables: {variables}")
        
        # Robust check for essential variables based on template's needs.
        # 'X' is almost always essential. 'Y' is essential for comparative.
        # Optional vars like 'CONTEXT', 'period1', 'period2' can be None.
        essential_vars_missing = False
        if matched_template['variable_slots']: # Only check if slots are defined
            if 'X' in matched_template['variable_slots'] and variables.get('X') is None:
                essential_vars_missing = True
            if 'Y' in matched_template['variable_slots'] and \
               matched_template['name'] in ["Comparative", "Comparative_Difference"] and \
               variables.get('Y') is None:
                essential_vars_missing = True
        
        if essential_vars_missing:
            self._log_debug(f"Essential variable(s) missing for template '{matched_template['name']}'. Variables: {variables}. Returning original query.")
            return [original_query]

        reformulated_queries = self._generate_sub_queries(matched_template, variables)

        if not reformulated_queries:
            # This might happen if all sub_query_templates were skipped due to missing optional variables,
            # or if a template genuinely has no sub_query_templates defined.
            self._log_debug(f"No sub-queries generated for template '{matched_template['name']}' with variables {variables}. Returning original query.")
            return [original_query] 

        self._log_debug(f"Reformulation complete. Generated {len(reformulated_queries)} queries: {reformulated_queries}")
        return reformulated_queries

if __name__ == '__main__':
    # Example usage (for testing purposes)
    reformulator = QueryReformulator(debug_mode=True) # _load_templates is called in __init__

    print(f"\n--- Testing {len(reformulator.templates)} Loaded Templates ---")

    # Define some dummy metadata for one test
    sample_metadata_dict = {
        "original_query": "compare product Alpha and product Beta",
        "detected_entities": ["Alpha", "Beta"],
        "query_type_hint": "comparison"
    }

    queries_to_test_data = [
        # Query, Metadata (Optional)
        ("compare apples and oranges", None),
        ("What is the difference between Product A and Product B?", None),
        ("compare dogs and cats regarding trainability", None),
        ("What is the difference between Python and Java regarding performance?", None),
        # Temporal Analysis
        ("how has user engagement changed over time?", None),
        ("How has Microsoft's stock changed from 2020 to 2023?", None),
        ("what is the trend of global warming?", None),
        ("what is the trend of CO2 emissions from 1990 to 2020?", None),
        # Comprehensive Analysis
        ("Analyze the current market trends for renewable energy.", None),
        ("Tell me everything about the platypus", None),
        ("Full report on Q3 financial performance", None),
        ("give me a comprehensive overview of machine learning", None),
        # Causal Analysis
        ("Why did the stock market crash in 2008?", None),
        ("What caused the recent power outage?", None),
        ("why did the dinosaurs disappear?", None),
        ("what were the causes of the french revolution?", None),
        # No match
        ("This query will not match any template.", None),
        ("What's the weather like today?", None), 
        # Test with metadata
        ("compare product Alpha and product Beta", sample_metadata_dict),
    ]

    # Add queries that test fallback for missing essential variables
    queries_to_test_data.append(("compare and ", None)) 
    queries_to_test_data.append(("analyze", None))      

    for query, metadata in queries_to_test_data:
        display_query = f"{query}"
        if metadata:
            display_query += f" (with metadata keys: {list(metadata.keys())})"
        print(f"\nOriginal Query: {display_query}")
        
        reformulated = reformulator.reformulate(query, query_metadata=metadata)
        print(f"Reformulated Queries: {reformulated}")

    reformulator._log_debug("--- End of Reformulator Tests ---")
