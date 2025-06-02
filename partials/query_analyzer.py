# Partials File: partials/query_analyzer.py
import re
from typing import List, Dict, Any, Literal, TypedDict, Optional, Tuple
from enum import Enum

# Attempt to import spaCy, fall back to basic entity extraction if not available
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    # If using older spaCy, sentencizer might need to be added explicitly if not part of default 'en_core_web_sm'
    if NLP and not NLP.has_pipe("sentencizer") and not NLP.has_pipe("parser"): # Parser often includes sentence boundaries
        try:
            NLP.add_pipe("sentencizer")
            print("QueryAnalyzer: Added sentencizer to spaCy model.")
        except Exception as e:
            print(f"QueryAnalyzer: Warning - could not add sentencizer to spaCy model: {e}. Sentence-based pronoun ambiguity might be affected.")

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

class AmbiguityDetail(TypedDict):
    type: str # e.g., PRONOUN, TEMPORAL, SCOPE, COMPARATIVE, ENTITY
    text: str # The ambiguous phrase or term
    suggestion: Optional[str] # Optional suggestion for clarification

class QueryMetadata(TypedDict):
    original_query: str
    query_type: QueryType
    entities: List[Entity]
    temporal_refs: List[TemporalReference]
    action_verbs: List[str]
    scope: ScopeIndicator
    ambiguity_markers: List[str] # Specific words/phrases initially identified
    detected_patterns: List[QueryPattern]
    # Iteration 2 fields
    ambiguity_score: float
    decomposability_score: float # Will be refined later or use previous logic
    detailed_ambiguity_report: List[AmbiguityDetail]


class QueryAnalyzer:
    def __init__(self, query: str, debug_mode: bool = False):
        self.original_query = query
        self.query_lower = query.lower()
        self.debug_mode = debug_mode
        self.entities_cache: Optional[List[Entity]] = None # Cache for entities

        if NLP:
            self.doc = NLP(query)
        else:
            self.doc = None

        # Predefined lists (some might be expanded for ambiguity detection)
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
        self.ambiguity_marker_keywords = [ # General markers, specific checks will be in ambiguity detectors
            "it", "they", "them", "this", "that", "those", "these",
            "recent", "recently", "current", "previous", "next", "last", # Can be temporal ambiguity
            "some", "any", "few", "several", "many", "much",
            "better", "worse", "more", "less", "larger", "smaller", "higher", "lower" # Can be comparative ambiguity
        ]
        self.pronouns_list = ["it", "they", "them", "this", "that", "those", "these", "he", "she", "him", "her", "itself", "themselves"]
        self.vague_temporal_terms = ["recently", "last time", "current", "soon", "later", "previous", "next time", "earlier"]
        self.vague_scope_terms = ["performance", "results", "data", "issues", "status", "updates", "information", "details", "summary", "overview"]
        self.comparative_terms = ["better", "worse", "more", "less", "larger", "smaller", "higher", "lower", "increased", "decreased", "improved", "reduced"]

        self.multi_part_conjunctions = [" and ", " or ", " but also ", " as well as "]
        self.implicit_query_starters = ["revenue figures", "sales data", "performance metrics"]

    def _log_debug(self, message: str):
        if self.debug_mode:
            print(f"QueryAnalyzer DEBUG: {message}")

    def extract_query_type(self) -> QueryType:
        # (Code from previous version, ensure it's the corrected one)
        self._log_debug(f"Extracting query type for: '{self.original_query}'")
        if any(verb in self.query_lower for verb in self.action_verbs_keywords["compare"]):
            self._log_debug("Query type: COMPARISON (matched 'compare' keywords)")
            return QueryType.COMPARISON

        is_analysis = any(verb in self.query_lower for verb in self.action_verbs_keywords["analyze"])
        is_summarize = any(verb in self.query_lower for verb in self.action_verbs_keywords["summarize"])
        is_extract = any(verb in self.query_lower for verb in self.action_verbs_keywords["extract"])

        if is_analysis or is_summarize or is_extract:
            if is_analysis:
                self._log_debug("Query type: ANALYSIS_REQUEST (matched 'analyze' keywords)")
                return QueryType.ANALYSIS_REQUEST

            is_question_format = any(q_word in self.query_lower for q_word in self.action_verbs_keywords["question"])

            if is_question_format and (is_summarize or is_extract):
                try:
                    first_action_verb_indices = []
                    if is_summarize:
                        for v in self.action_verbs_keywords["summarize"]:
                            if v in self.query_lower: first_action_verb_indices.append(self.query_lower.find(v))
                    if is_extract:
                        for v in self.action_verbs_keywords["extract"]:
                            if v in self.query_lower: first_action_verb_indices.append(self.query_lower.find(v))

                    first_question_word_indices = []
                    for qv in self.action_verbs_keywords["question"]:
                        if qv in self.query_lower: first_question_word_indices.append(self.query_lower.find(qv))

                    if first_question_word_indices and first_action_verb_indices:
                        if min(first_question_word_indices) < min(first_action_verb_indices):
                            self._log_debug("Query type: QUESTION (matched summarize/extract keywords but also question format)")
                            return QueryType.QUESTION
                except (ValueError, IndexError):
                    pass

            self._log_debug("Query type: COMMAND (matched summarize/extract keywords, not clearly a question)")
            return QueryType.COMMAND

        if any(self.query_lower.startswith(verb) for verb in self.action_verbs_keywords["command"]):
            if not any(q_word == self.query_lower.split()[0] for q_word in self.action_verbs_keywords["question"] if self.query_lower.split()):
                 self._log_debug("Query type: COMMAND (starts with command verb)")
                 return QueryType.COMMAND

        if any(self.query_lower.startswith(q_word) for q_word in self.action_verbs_keywords["question"]):
            self._log_debug("Query type: QUESTION (starts with question word)")
            return QueryType.QUESTION
        if "?" in self.original_query:
            self._log_debug("Query type: QUESTION (contains '?')")
            return QueryType.QUESTION

        self._log_debug("No specific query type matched, defaulting to UNKNOWN.")
        return QueryType.UNKNOWN

    def extract_entities(self) -> List[Entity]:
        if self.entities_cache is not None:
            return self.entities_cache

        self._log_debug("Extracting entities...")
        entities: List[Entity] = []
        if self.doc:
            for ent in self.doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                })
            self._log_debug(f"Found {len(entities)} entities using spaCy: {entities}")
        else:
            for match in re.finditer(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", self.original_query):
                 entities.append({
                    "text": match.group(0),
                    "label": "UNKNOWN_CAPITALIZED",
                    "start_char": match.start(),
                    "end_char": match.end()
                })
            self._log_debug(f"Found {len(entities)} entities using basic regex: {entities}")
        self.entities_cache = entities
        return entities

    def extract_temporal_references(self) -> List[TemporalReference]:
        # (Code from previous version)
        self._log_debug("Extracting temporal references...")
        temporal_refs: List[TemporalReference] = []
        year_patterns = [
            (r"(19|20)\d{2}", "YEAR"),
            (r"FY\s*(19|20)\d{2}", "FISCAL_YEAR"),
            (r"Q[1-4]\s*(?:(?:19|20)\d{2})?", "QUARTER_YEAR"),
            (r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*(?:19|20)\d{2})?", "MONTH_DAY_YEAR"),
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
        # (Code from previous version)
        self._log_debug("Extracting action verbs...")
        verbs = set()
        if self.doc:
            for token in self.doc:
                if token.pos_ == "VERB":
                    lemma = token.lemma_.lower()
                    for _, keywords in self.action_verbs_keywords.items(): # Check all categories
                        if lemma in keywords or token.lower_ in keywords:
                            verbs.add(lemma)
                            break
        else:
            for _, keywords in self.action_verbs_keywords.items():
                for keyword in keywords:
                    # More robust check for keyword presence
                    if f" {keyword} " in f" {self.query_lower} " or \
                       self.query_lower.startswith(f"{keyword} ") or \
                       self.query_lower.endswith(f" {keyword}"):
                        verbs.add(keyword.split()[0])

        self._log_debug(f"Found action verbs: {list(verbs)}")
        return list(verbs)

    def extract_scope_indicator(self) -> ScopeIndicator:
        # (Code from previous version)
        self._log_debug("Extracting scope indicator...")
        for scope_enum, keywords in self.scope_keywords.items():
            if any(keyword in self.query_lower for keyword in keywords):
                self._log_debug(f"Matched scope: {scope_enum}")
                return scope_enum
        self._log_debug("No specific scope matched, defaulting to UNKNOWN.")
        return ScopeIndicator.UNKNOWN

    def extract_ambiguity_markers(self) -> List[str]:
        # (Code from previous version - this becomes less important as specific detectors are added)
        self._log_debug("Extracting general ambiguity markers (words)...")
        markers = []
        # This method will become less critical as specific detectors are more robust.
        # For now, it can still provide a quick list of potentially ambiguous words.
        for marker_token in self.query_lower.split():
            if marker_token in self.ambiguity_marker_keywords:
                 markers.append(marker_token)
        self._log_debug(f"Found general ambiguity marker words: {markers}")
        return markers

    def detect_patterns(self) -> List[QueryPattern]:
        # (Code from previous version)
        self._log_debug("Detecting query patterns...")
        patterns: List[QueryPattern] = []
        for conj in self.multi_part_conjunctions:
            if conj in self.query_lower:
                parts = self.original_query.lower().split(conj)
                if len(parts) > 1 and "?" in self.original_query :
                     patterns.append({"type": "MULTI_PART", "text": self.original_query})
                     self._log_debug(f"Detected MULTI_PART pattern (general) using '{conj}'")
                     break
        if "where..." in self.query_lower or "that have" in self.query_lower or "which are" in self.query_lower:
            if "find all companies that have revenue > $1M in sectors where" in self.query_lower:
                 patterns.append({"type": "NESTED_QUERY", "text": self.original_query})
                 self._log_debug("Detected NESTED_QUERY pattern (specific example)")
            elif "where" in self.query_lower.split() and self.query_lower.index("where") > 0 :
                 patterns.append({"type": "NESTED_QUERY", "text": self.original_query})
                 self._log_debug("Detected NESTED_QUERY pattern (general 'where')")
        for starter in self.implicit_query_starters:
            if self.query_lower.startswith(starter) and "?" not in self.original_query:
                patterns.append({"type": "IMPLICIT_QUERY", "text": self.original_query})
                self._log_debug(f"Detected IMPLICIT_QUERY pattern with starter '{starter}'")
                break
        follow_up_phrases = ["how about", "what about", "and for", "and then", "tell me more about that"]
        if any(self.query_lower.startswith(phrase) for phrase in follow_up_phrases):
            patterns.append({"type": "FOLLOW_UP", "text": self.original_query})
            self._log_debug("Detected FOLLOW_UP pattern")
        self._log_debug(f"Detected patterns: {patterns}")
        return patterns

    # --- Iteration 2: Ambiguity Detection Methods ---
    def _detect_pronoun_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        if not self.doc: # spaCy not available
            for pronoun in self.pronouns_list:
                if pronoun in self.query_lower.split():
                    report.append({
                        "type": "PRONOUN", "text": pronoun,
                        "suggestion": "Clarify what this pronoun refers to (spaCy not available for context)."
                    })
            return report

        # spaCy is available
        # A simple approach: check if pronoun is too far from a preceding noun/proper noun
        # More advanced: use coreference resolution if available and performant enough
        for sent in self.doc.sents:
            for token in sent:
                if token.lower_ in self.pronouns_list:
                    # Basic check: is there a recent noun phrase (potential antecedent)?
                    # This is a heuristic. Real coreference is complex.
                    has_recent_antecedent = False
                    # Check current sentence before the pronoun
                    for i in range(token.i - 1, sent.start, -1): # Iterate backwards from pronoun in the same sentence
                        prev_token = self.doc[i]
                        if prev_token.pos_ in ["NOUN", "PROPN"]:
                            has_recent_antecedent = True
                            break
                        if i < token.i - 5 : # Stop looking back too far within the sentence
                            break

                    if not has_recent_antecedent:
                         # Could also check previous sentence if this is not the first sentence
                        if sent.start > 0: # Not the first sentence in the query
                            prev_sent = self.doc[sent.start-1].sent # Get previous sentence
                            for prev_token in reversed(list(prev_sent)): # Iterate backwards through previous sentence
                                if prev_token.pos_ in ["NOUN", "PROPN"]:
                                    has_recent_antecedent = True
                                    break
                                if prev_token.i < prev_sent.end - 5: # stop looking too far
                                    break


                        if not has_recent_antecedent:
                            report.append({
                                "type": "PRONOUN", "text": token.text,
                                "suggestion": f"Clarify what '{token.text}' refers to. Consider rephrasing or providing more context."
                            })
        self._log_debug(f"Pronoun ambiguity report: {report}")
        return report

    def _detect_temporal_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        # Get already identified specific temporal refs to avoid double flagging
        specific_temporal_texts = [ref['text'].lower() for ref in self.extract_temporal_references()]

        for term in self.vague_temporal_terms:
            if term in self.query_lower:
                # Check if this vague term is part of an already identified specific reference
                is_part_of_specific = False
                for specific_ref_text in specific_temporal_texts:
                    if term in specific_ref_text: # e.g. "last year" is specific, "last" alone might be vague
                        is_part_of_specific = True
                        break
                if not is_part_of_specific:
                    report.append({
                        "type": "TEMPORAL", "text": term,
                        "suggestion": f"'{term}' is vague. Specify a date, range, or event (e.g., 'current quarter', 'last month')."
                    })
        self._log_debug(f"Temporal ambiguity report: {report}")
        return report

    def _detect_scope_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        # Check if vague scope terms are used without specific entities nearby
        # This is a heuristic. Context is key.
        entities = self.extract_entities()
        entity_texts_lower = [e['text'].lower() for e in entities]

        for term in self.vague_scope_terms:
            if term in self.query_lower:
                # Simple check: is the vague term accompanied by a specific entity?
                # e.g., "sales data" vs "data for product X"
                is_clarified_by_entity = False
                term_index = self.query_lower.find(term)
                if term_index != -1:
                    # Look for entities within a window around the term
                    window_start = max(0, term_index - 20)
                    window_end = min(len(self.query_lower), term_index + len(term) + 20)
                    query_window = self.query_lower[window_start:window_end]
                    if any(e_text in query_window for e_text in entity_texts_lower):
                        is_clarified_by_entity = True

                if not is_clarified_by_entity:
                    report.append({
                        "type": "SCOPE", "text": term,
                        "suggestion": f"'{term}' is vague. Specify for what entity, product, or area (e.g., '{term} for sales team', '{term} of Q3')."
                    })
        self._log_debug(f"Scope ambiguity report: {report}")
        return report

    def _detect_comparative_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        for term in self.comparative_terms:
            if term in self.query_lower.split(): # Check for whole word
                # Simple check: is there a "than" or "to" nearby, or multiple entities being compared?
                # This is a basic heuristic.
                has_baseline_indicator = " than " in self.query_lower or " to " in self.query_lower or " compared to " in self.query_lower
                entities = self.extract_entities() # Ensure entities are extracted

                # Count entities that are not dates or quantities, as those usually aren't the items being compared directly
                relevant_entities_count = len([e for e in entities if e['label'] not in ['DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY', 'CARDINAL', 'ORDINAL']])

                if not has_baseline_indicator and relevant_entities_count < 2:
                    report.append({
                        "type": "COMPARATIVE", "text": term,
                        "suggestion": f"'{term}' implies a comparison. Specify what it's being compared to or provide a baseline."
                    })
        self._log_debug(f"Comparative ambiguity report: {report}")
        return report

    def _detect_entity_ambiguity(self) -> List[AmbiguityDetail]:
        report: List[AmbiguityDetail] = []
        entities = self.extract_entities()

        # Count occurrences of ORG, PRODUCT, etc.
        org_entities = [e['text'] for e in entities if e['label'] == 'ORG']
        product_entities = [e['text'] for e in entities if e['label'] == 'PRODUCT'] # Assuming PRODUCT label if spaCy provides it

        generic_references = {
            "the company": (org_entities, "ORG"),
            "the project": (product_entities, "PRODUCT"), # Example, adjust label as needed
            "the product": (product_entities, "PRODUCT")
        }

        for generic_term, (entity_list, entity_type_label) in generic_references.items():
            if generic_term in self.query_lower:
                if len(entity_list) > 1:
                    report.append({
                        "type": "ENTITY", "text": generic_term,
                        "suggestion": f"'{generic_term}' is ambiguous. Multiple {entity_type_label}s detected: {', '.join(entity_list)}. Specify which one."
                    })
                elif not entity_list and not any(e['label'] == entity_type_label for e in entities): # No specific entity of this type mentioned
                     report.append({
                        "type": "ENTITY", "text": generic_term,
                        "suggestion": f"'{generic_term}' is used, but no specific {entity_type_label} was identified in the query. Please specify."
                    })

        self._log_debug(f"Entity ambiguity report: {report}")
        return report

    def _calculate_ambiguity_score(self, detailed_report: List[AmbiguityDetail]) -> float:
        if not detailed_report:
            return 0.0

        # Simple scoring: each ambiguity type adds a fixed amount, with a cap.
        # Could be weighted by type later.
        # Max score 1.0
        # Each detected ambiguity adds 0.25 to the score.
        score = len(detailed_report) * 0.25

        self._log_debug(f"Calculated ambiguity score: {min(score, 1.0)} from {len(detailed_report)} ambiguities.")
        return min(score, 1.0) # Cap at 1.0

    def analyze(self) -> QueryMetadata:
        self._log_debug(f"Starting analysis for query: '{self.original_query}' (Iteration 2)")

        # Clear entity cache for fresh analysis if called multiple times (though typically once per instance)
        self.entities_cache = None

        query_type = self.extract_query_type()
        entities = self.extract_entities() # Call it once and cache for other ambiguity detectors
        temporal_refs = self.extract_temporal_references()
        action_verbs = self.extract_action_verbs()
        scope = self.extract_scope_indicator()
        general_ambiguity_markers = self.extract_ambiguity_markers() # Still useful for a quick glance
        detected_patterns = self.detect_patterns()

        # Iteration 2: Detailed Ambiguity Detection
        detailed_ambiguity_report: List[AmbiguityDetail] = []
        detailed_ambiguity_report.extend(self._detect_pronoun_ambiguity())
        detailed_ambiguity_report.extend(self._detect_temporal_ambiguity())
        detailed_ambiguity_report.extend(self._detect_scope_ambiguity())
        detailed_ambiguity_report.extend(self._detect_comparative_ambiguity())
        detailed_ambiguity_report.extend(self._detect_entity_ambiguity())

        ambiguity_score = self._calculate_ambiguity_score(detailed_ambiguity_report)

        # Decomposability score (can be refined, using Iteration 1 logic for now)
        decomposability_score = 0.0
        if query_type == QueryType.COMPARISON or \
           any(p['type'] == 'MULTI_PART' for p in detected_patterns) or \
           any(p['type'] == 'NESTED_QUERY' for p in detected_patterns) or \
           len(action_verbs) > 1 :
            decomposability_score = 0.6
        if ambiguity_score > 0.5: # High ambiguity might make it harder to decompose reliably
            decomposability_score = max(0, decomposability_score - 0.2)


        metadata: QueryMetadata = {
            "original_query": self.original_query,
            "query_type": query_type,
            "entities": entities,
            "temporal_refs": temporal_refs,
            "action_verbs": action_verbs,
            "scope": scope,
            "ambiguity_markers": general_ambiguity_markers, # General list
            "detected_patterns": detected_patterns,
            "ambiguity_score": ambiguity_score,
            "decomposability_score": decomposability_score,
            "detailed_ambiguity_report": detailed_ambiguity_report # New field
        }
        self._log_debug(f"Analysis complete (Iteration 2). Metadata: {metadata}")
        return metadata

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    queries_to_test_iter2 = [
        "What is its status and how does it compare to the previous one?", # Pronoun, Comparative, Temporal
        "Analyze performance data.", # Scope, Temporal (current implied)
        "Is the new system better?", # Comparative
        "Tell me about the company's recent results.", # Entity, Temporal, Scope
        "List all projects for the client.", # Entity (if multiple clients or no client identified)
        "Generate a summary of their findings.", # Pronoun
        "Compare sales in 2023 to 2022 for product X and product Y.", # Low ambiguity
        "What are the current risks?", # Temporal, Scope
        "Find issues related to the new deployment.", # Scope
        "How much did revenue increase for the division?" # Comparative, Entity (if multiple divisions)
    ]

    for q in queries_to_test_iter2:
        print(f"--- Analyzing Query (Iter 2): \"{q}\" ---")
        analyzer = QueryAnalyzer(q, debug_mode=True)
        meta = analyzer.analyze()
        print(f"Query Type: {meta['query_type'].value}")
        # print(f"Entities: {meta['entities']}")
        # print(f"Temporal Refs: {meta['temporal_refs']}")
        # print(f"Action Verbs: {meta['action_verbs']}")
        # print(f"Scope: {meta['scope'].value}")
        # print(f"Ambiguity Markers (general): {meta['ambiguity_markers']}")
        # print(f"Detected Patterns: {meta['detected_patterns']}")
        print(f"Ambiguity Score: {meta['ambiguity_score']:.2f}")
        print(f"Detailed Ambiguity Report: {meta['detailed_ambiguity_report']}")
        print(f"Decomposability Score: {meta['decomposability_score']:.2f}")
        print("---------------------------------------\n")

    # Test with no spaCy again if needed
    # _temp_nlp = NLP; NLP = None
    # analyzer_no_spacy = QueryAnalyzer("What is its profit?", debug_mode=True)
    # meta_no_spacy = analyzer_no_spacy.analyze()
    # print(f"Ambiguity Report (no spaCy): {meta_no_spacy['detailed_ambiguity_report']}")
    # NLP = _temp_nlp
