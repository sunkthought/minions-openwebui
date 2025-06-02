# Partials File: partials/query_analyzer.py
import re
from typing import List, Dict, Any, Literal, TypedDict, Optional, Tuple
from enum import Enum

# Attempt to import spaCy, fall back to basic entity extraction if not available
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except ImportError:
    NLP = None
    print("Warning: spaCy not found. Falling back to basic entity extraction for QueryAnalyzer.")

class QueryType(Enum):
    QUESTION = "question"
    COMMAND = "command"
    ANALYSIS_REQUEST = "analysis_request"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"

class ScopeIndicator(Enum):
    SPECIFIC = "specific"
    BROAD = "broad"
    COMPREHENSIVE = "comprehensive"
    UNKNOWN = "unknown"

class Entity(TypedDict):
    text: str
    label: str # e.g., PERSON, ORG, DATE, METRIC, CONCEPT
    start_char: int
    end_char: int

class TemporalReference(TypedDict):
    text: str
    type: str # e.g., YEAR, QUARTER, DATE_RANGE
    start_char: int
    end_char: int

class QueryPattern(TypedDict):
    type: str # e.g., MULTI_PART, NESTED_QUERY, IMPLICIT_QUERY, FOLLOW_UP
    text: str # The part of the query that matches the pattern

class QueryMetadata(TypedDict):
    original_query: str
    query_type: QueryType
    entities: List[Entity]
    temporal_refs: List[TemporalReference]
    action_verbs: List[str]
    scope: ScopeIndicator
    ambiguity_markers: List[str] # Specific words/phrases identified
    detected_patterns: List[QueryPattern]
    # Iteration 2 fields (placeholders for now)
    ambiguity_score: float
    decomposability_score: float

class QueryAnalyzer:
    def __init__(self, query: str, debug_mode: bool = False):
        self.original_query = query
        self.query_lower = query.lower()
        self.debug_mode = debug_mode
        if NLP:
            self.doc = NLP(query)
        else:
            self.doc = None

        # Predefined lists
        self.action_verbs_keywords = {
            "analyze": ["analyze", "analysis", "examine", "study"],
            "compare": ["compare", "contrast", "difference", "vs", "versus"],
            "summarize": ["summarize", "summary", "recap", "overview", "gist"],
            "extract": ["extract", "find", "get", "list", "show", "identify", "retrieve"],
            "find": ["find", "locate", "search for", "where is"],
            "command": ["generate", "create", "write", "build", "develop", "draft"],
            "question": ["what", "who", "when", "where", "why", "how", "is", "are", "do", "does", "can", "could", "would", "should"]
        }
        self.scope_keywords = {
            ScopeIndicator.SPECIFIC: ["specific", "particular", "only", "just"],
            ScopeIndicator.BROAD: ["broad", "general", "overall"],
            ScopeIndicator.COMPREHENSIVE: ["all", "every", "comprehensive", "full", "complete", "entire"]
        }
        self.ambiguity_marker_keywords = [
            "it", "they", "them", "this", "that", "those", "these",
            "recent", "recently", "current", "previous", "next", "last",
            "some", "any", "few", "several", "many", "much",
            "better", "worse", "more", "less", "larger", "smaller", "higher", "lower"
        ]
        self.multi_part_conjunctions = [" and ", " or ", " but also ", " as well as "]
        self.implicit_query_starters = ["revenue figures", "sales data", "performance metrics"] # examples

    def _log_debug(self, message: str):
        if self.debug_mode:
            print(f"QueryAnalyzer DEBUG: {message}")

    def extract_query_type(self) -> QueryType:
        self._log_debug(f"Extracting query type for: '{self.original_query}'")
        if any(verb in self.query_lower for verb in self.action_verbs_keywords["compare"]):
            return QueryType.COMPARISON
        if any(verb in self.query_lower for verb in self.action_verbs_keywords["analysis_request"] + self.action_verbs_keywords["summarize"] + self.action_verbs_keywords["extract"]):
             # Prioritize analysis/extraction if those keywords are present
            if any(verb in self.query_lower for verb in self.action_verbs_keywords["analysis_request"]):
                return QueryType.ANALYSIS_REQUEST
            if any(q_word in self.query_lower.split() for q_word in self.action_verbs_keywords["question"]): # "summarize what happened"
                 return QueryType.QUESTION
            return QueryType.COMMAND # "summarize this document" could be a command

        if any(verb in self.query_lower.split() for verb in self.action_verbs_keywords["command"]): # Check if first word is a command verb
            # Check if it's actually a question like "Generate a list of..." vs "Can you generate..."
            if not any(q_word == self.query_lower.split()[0] for q_word in self.action_verbs_keywords["question"] if self.query_lower.split()):
                 return QueryType.COMMAND

        # Default to question if common question words are present
        if any(self.query_lower.startswith(q_word) for q_word in self.action_verbs_keywords["question"]):
            return QueryType.QUESTION
        if "?" in self.original_query: # Final check for question mark
            return QueryType.QUESTION

        self._log_debug("No specific query type matched, defaulting to UNKNOWN.")
        return QueryType.UNKNOWN

    def extract_entities(self) -> List[Entity]:
        self._log_debug("Extracting entities...")
        entities: List[Entity] = []
        if self.doc: # spaCy available
            for ent in self.doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                })
            self._log_debug(f"Found {len(entities)} entities using spaCy: {entities}")
        else: # Basic regex fallback
            # Example: Capitalized words (simple NNP) or known metric terms
            for match in re.finditer(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", self.original_query):
                 entities.append({
                    "text": match.group(0),
                    "label": "UNKNOWN_CAPITALIZED", # Generic label for fallback
                    "start_char": match.start(),
                    "end_char": match.end()
                })
            # Add more regex for dates, currency, etc. as needed for basic fallback
            self._log_debug(f"Found {len(entities)} entities using basic regex: {entities}")
        return entities

    def extract_temporal_references(self) -> List[TemporalReference]:
        self._log_debug("Extracting temporal references...")
        temporal_refs: List[TemporalReference] = []
        # Regex for years (e.g., 2023, FY2023, Q3 2023)
        year_patterns = [
            (r"(19|20)\d{2}", "YEAR"),
            (r"FY\s*(19|20)\d{2}", "FISCAL_YEAR"),
            (r"Q[1-4]\s*(?:(?:19|20)\d{2})?", "QUARTER_YEAR"), # Q3, Q3 2023
            (r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*(?:19|20)\d{2})?", "MONTH_DAY_YEAR"), # Jan 1st, 2023
            (r"(?:this|last|next)\s+(?:year|quarter|month|week)", "RELATIVE_TIME")
        ]
        for pattern, ref_type in year_patterns:
            for match in re.finditer(pattern, self.original_query, re.IGNORECASE):
                temporal_refs.append({
                    "text": match.group(0),
                    "type": ref_type,
                    "start_char": match.start(),
                    "end_char": match.end()
                })
        self._log_debug(f"Found {len(temporal_refs)} temporal references: {temporal_refs}")
        return temporal_refs

    def extract_action_verbs(self) -> List[str]:
        self._log_debug("Extracting action verbs...")
        verbs = set()
        # spaCy based lemmatization if available
        if self.doc:
            for token in self.doc:
                if token.pos_ == "VERB":
                    lemma = token.lemma_.lower()
                    for action_cat, keywords in self.action_verbs_keywords.items():
                        if lemma in keywords or token.lower_ in keywords:
                            verbs.add(lemma) # Store the lemma
                            break
        else: # Fallback to keyword spotting
            for action_cat, keywords in self.action_verbs_keywords.items():
                for keyword in keywords:
                    if keyword in self.query_lower:
                        # Attempt to find a root form if no NLP (very basic)
                        verbs.add(keyword.split()[0]) # e.g., "analyze" from "analyze the data"

        self._log_debug(f"Found action verbs: {list(verbs)}")
        return list(verbs)

    def extract_scope_indicator(self) -> ScopeIndicator:
        self._log_debug("Extracting scope indicator...")
        for scope_enum, keywords in self.scope_keywords.items():
            if any(keyword in self.query_lower for keyword in keywords):
                self._log_debug(f"Matched scope: {scope_enum}")
                return scope_enum
        self._log_debug("No specific scope matched, defaulting to UNKNOWN.")
        return ScopeIndicator.UNKNOWN

    def extract_ambiguity_markers(self) -> List[str]:
        self._log_debug("Extracting ambiguity markers...")
        markers = []
        for marker in self.ambiguity_marker_keywords:
            if marker in self.query_lower.split(): # Check for whole word
                 # More sophisticated checks can be added in Iteration 2
                markers.append(marker)
        self._log_debug(f"Found ambiguity markers: {markers}")
        return markers

    def detect_patterns(self) -> List[QueryPattern]:
        self._log_debug("Detecting query patterns...")
        patterns: List[QueryPattern] = []

        # Multi-part questions
        for conj in self.multi_part_conjunctions:
            if conj in self.query_lower:
                parts = self.original_query.lower().split(conj)
                if len(parts) > 1 and "?" in parts[0] and ("?" in parts[1] or any(parts[1].startswith(qw) for qw in self.action_verbs_keywords["question"])):
                    patterns.append({"type": "MULTI_PART", "text": self.original_query})
                    self._log_debug(f"Detected MULTI_PART pattern using '{conj}'")
                    break
                elif len(parts) > 1 and "?" in self.original_query : # Simpler check if one part has '?' and conjunction exists
                     patterns.append({"type": "MULTI_PART", "text": self.original_query})
                     self._log_debug(f"Detected MULTI_PART pattern (general) using '{conj}'")
                     break


        # Nested queries (simple check for now, e.g., "in sectors where...")
        if "where..." in self.query_lower or "that have" in self.query_lower or "which are" in self.query_lower:
            if "find all companies that have revenue > $1M in sectors where" in self.query_lower: # more specific example
                 patterns.append({"type": "NESTED_QUERY", "text": self.original_query})
                 self._log_debug("Detected NESTED_QUERY pattern (specific example)")
            elif "where" in self.query_lower.split() and self.query_lower.index("where") > 0 : # General "where" clause
                 patterns.append({"type": "NESTED_QUERY", "text": self.original_query})
                 self._log_debug("Detected NESTED_QUERY pattern (general 'where')")


        # Implicit queries (e.g., "Revenue figures")
        for starter in self.implicit_query_starters:
            if self.query_lower.startswith(starter) and "?" not in self.original_query:
                patterns.append({"type": "IMPLICIT_QUERY", "text": self.original_query})
                self._log_debug(f"Detected IMPLICIT_QUERY pattern with starter '{starter}'")
                break

        # Follow-up patterns (very basic for now)
        follow_up_phrases = ["how about", "what about", "and for", "and then", "tell me more about that"]
        if any(self.query_lower.startswith(phrase) for phrase in follow_up_phrases):
            patterns.append({"type": "FOLLOW_UP", "text": self.original_query})
            self._log_debug("Detected FOLLOW_UP pattern")

        self._log_debug(f"Detected patterns: {patterns}")
        return patterns

    def analyze(self) -> QueryMetadata:
        self._log_debug(f"Starting analysis for query: '{self.original_query}'")

        query_type = self.extract_query_type()
        entities = self.extract_entities()
        temporal_refs = self.extract_temporal_references()
        action_verbs = self.extract_action_verbs()
        scope = self.extract_scope_indicator()
        ambiguity_markers = self.extract_ambiguity_markers()
        detected_patterns = self.detect_patterns()

        # Placeholder for Iteration 2 scores
        ambiguity_score = 0.0
        decomposability_score = 0.0

        # Basic decomposability hint for Iteration 1 (can be refined)
        if query_type == QueryType.COMPARISON or \
           any(p['type'] == 'MULTI_PART' for p in detected_patterns) or \
           any(p['type'] == 'NESTED_QUERY' for p in detected_patterns) or \
           len(action_verbs) > 1 :
            decomposability_score = 0.6 # Arbitrary higher value if complex patterns detected

        metadata: QueryMetadata = {
            "original_query": self.original_query,
            "query_type": query_type,
            "entities": entities,
            "temporal_refs": temporal_refs,
            "action_verbs": action_verbs,
            "scope": scope,
            "ambiguity_markers": ambiguity_markers,
            "detected_patterns": detected_patterns,
            "ambiguity_score": ambiguity_score,       # Placeholder
            "decomposability_score": decomposability_score # Placeholder
        }
        self._log_debug(f"Analysis complete. Metadata: {metadata}")
        return metadata

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    queries_to_test = [
        "What is the total revenue for Q3 2023 and how does it compare to Q2 2023?",
        "Analyze the sales performance in the EMEA region for last year.",
        "Generate a report on employee satisfaction.",
        "List all products launched since January 2022.",
        "Find companies with revenue > $1M in sectors where R&D spending is high.",
        "Revenue figures for this quarter.",
        "How about the profit margin?",
        "Tell me everything about project Phoenix.",
        "Is the new feature performing better?",
        "The company's performance was good. But what about its competitors?"
    ]

    for q in queries_to_test:
        print(f"--- Analyzing Query: \"{q}\" ---")
        analyzer = QueryAnalyzer(q, debug_mode=True)
        meta = analyzer.analyze()
        print(f"Query Type: {meta['query_type'].value}")
        print(f"Entities: {meta['entities']}")
        print(f"Temporal Refs: {meta['temporal_refs']}")
        print(f"Action Verbs: {meta['action_verbs']}")
        print(f"Scope: {meta['scope'].value}")
        print(f"Ambiguity Markers: {meta['ambiguity_markers']}")
        print(f"Detected Patterns: {meta['detected_patterns']}")
        print(f"Decomposability Score: {meta['decomposability_score']}")
        print("---------------------------------------\n")

    # Test spaCy fallback
    print("\n--- Testing spaCy fallback (if spaCy is not installed) ---")
    _temp_nlp = NLP # store current NLP
    NLP = None # Simulate spaCy not being available
    analyzer_no_spacy = QueryAnalyzer("What is the profit for ACME Corp in 2024?", debug_mode=True)
    meta_no_spacy = analyzer_no_spacy.analyze()
    print(f"Entities (no spaCy): {meta_no_spacy['entities']}")
    NLP = _temp_nlp # restore NLP
    print("---------------------------------------\n")
