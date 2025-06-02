# Partials File: partials/query_expander.py
import re
from typing import List, Dict, Any, Optional

# Attempt to import spaCy for lemmatization, fall back if not available
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except ImportError:
    NLP = None
    print("Warning: spaCy not found. QueryExpander's synonym matching might be less effective.")

class QueryExpander:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

        # Domain-specific contextual phrases/keywords
        # These are just examples; more sophisticated domain detection might be needed.
        self.domain_keywords = {
            "finance": ["revenue", "profit", "loss", "stock", "market", "investment", "ebitda", "financial"],
            "research": ["study", "research", "paper", "findings", "experiment", "analyze", "methodology"],
            "coding": ["code", "function", "class", "module", "debug", "algorithm", "error", "python", "java"],
            "writing": ["draft", "edit", "chapter", "section", "narrative", "style", "tone", "grammar"]
        }

        # Default examples for context injection (can be expanded)
        self.default_doc_type_context_examples = {
            "default": { # General context, less specific
                "prepend": [],
                "append": []
            },
            "research": {
                "prepend": ["According to the research paper, ", "Based on the study, "],
                "append": [" considering the methodologies described.", " in the context of this research."]
            },
            "finance": {
                "prepend": ["Regarding the financial data, ", "From a financial perspective, "],
                "append": [" as per the financial statements.", " considering market conditions."]
            },
            "writing": {
                "prepend": ["In terms of the draft, ", "Considering the writing style, "],
                "append": [" to improve clarity and flow.", " keeping the target audience in mind."]
            },
            "coding": {
                "prepend": ["For the given code, ", "Regarding the algorithm, "],
                "append": [" ensure efficiency and proper error handling.", " following best coding practices."]
            }
        }

        # Templates for query completion
        self.default_completion_templates = {
            # Default domain
            r"^(?:what about|how about|and)\s+(.+)$": r"What is the status of ?", # "what about sales" -> "What is the status of sales?"
            r"^([A-Za-z\s]+)\s+(figures|data|details|information|summary|overview)$": r"What are the  ?", # "sales figures" -> "What are the sales figures?"
            r"^([A-Za-z\s]+)\s+(report|analysis)$": r"Provide a  .", # "market report" -> "Provide a market report."
            r"^(compare|comparison)\s+([\w\s]+)\s+(?:and|vs|to)\s+([\w\s]+)$": r"Compare  with .", # "compare A to B"
            r"^(compare|comparison)\s+([\d]{4})\s+([\d]{4})$": r"Compare the results between  and .", # "Comparison 2023 2024"
            r"^(risk factors|key risks)$": r"What are the main risk factors mentioned?",
            # Finance specific (examples)
            r"^(Q[1-4])\s+(revenue|sales|profit)$": r"What was the  for ?", # "Q3 revenue"
            r"^(((?:[A-Z]{2,5}))\s+stock price)$": r"What is the current stock price for ?", # "MSFT stock price"
            # Research specific (examples)
            r"^(findings for|results of)\s+(.+)$": r"What are the findings for ?",
            # Coding specific (examples)
            r"^(fix|debug)\s+(.+)$": r"How can I fix the error in ?",
             r"^(generate code for|write a function to)\s+(.+)$": r"Generate Python code to ." # Default to Python or make configurable
        }

        # Synonym sets (can be greatly expanded)
        self.default_synonym_sets = {
            "default": {
                "show": ["display", "list", "present"],
                "important": ["critical", "key", "significant", "main"],
                "issue": ["problem", "challenge", "difficulty", "concern"]
            },
            "finance": {
                "profit": ["earnings", "income", "net income", "bottom line"],
                "revenue": ["sales", "turnover", "top line"],
                "company": ["organization", "firm", "business", "corporation"],
                "ebitda": ["earnings before interest, taxes, depreciation and amortization"]
            },
            "research": {
                "study": ["paper", "investigation", "analysis", "examination"],
                "findings": ["results", "conclusions", "discoveries"],
                "correlation": ["relationship", "association", "connection"]
            },
            "coding": {
                "error": ["bug", "defect", "issue", "glitch"],
                "function": ["method", "subroutine", "procedure"],
                "fix": ["debug", "resolve", "patch", "correct"]
            },
            "writing": {
                "summary": ["abstract", "synopsis", "overview", "recapitulation"],
                "improve": ["enhance", "refine", "polish", "strengthen"],
                "clarity": ["lucidity", "clearness", "precision"]
            }
        }

    def _log_debug(self, message: str):
        if self.debug_mode:
            print(f"QueryExpander DEBUG: {message}")

    def _get_lemma(self, word: str) -> str:
        if NLP:
            doc = NLP(word)
            return doc[0].lemma_.lower()
        return word.lower() # Fallback if no spaCy

    def _infer_domain(self, query: str, provided_domain: Optional[str] = None) -> str:
        if provided_domain and provided_domain in self.domain_keywords:
            return provided_domain

        query_lower = query.lower()
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                self._log_debug(f"Inferred domain: {domain}")
                return domain
        self._log_debug("Domain not inferred, using 'default'.")
        return "default"

    def add_document_type_context(self, query: str,
                                  doc_type_examples: Optional[Dict[str, Any]] = None,
                                  domain_hint: Optional[str] = None) -> str:
        self._log_debug(f"Original query for context injection: '{query}'")
        if doc_type_examples is None:
            doc_type_examples = self.default_doc_type_context_examples

        domain = self._infer_domain(query, domain_hint)
        domain_contexts = doc_type_examples.get(domain, doc_type_examples.get("default", {}))

        expanded_query = query
        if domain_contexts:
            if domain_contexts.get("prepend"):
                # Simple: pick the first prepend phrase if multiple exist
                phrase_to_prepend = domain_contexts["prepend"][0]
                if not query.lower().startswith(phrase_to_prepend.lower()): # Avoid double prepending
                    expanded_query = phrase_to_prepend + expanded_query
                    self._log_debug(f"Prepended context for domain '{domain}': '{phrase_to_prepend}'")

            if domain_contexts.get("append"):
                # Simple: pick the first append phrase
                phrase_to_append = domain_contexts["append"][0]
                if not query.lower().endswith(phrase_to_append.lower()): # Avoid double appending
                    expanded_query += phrase_to_append
                    self._log_debug(f"Appended context for domain '{domain}': '{phrase_to_append}'")

        if expanded_query != query:
            self._log_debug(f"Query after context injection: '{expanded_query}'")
        else:
            self._log_debug("No context injected.")
        return expanded_query

    def complete_query(self, query: str,
                       completion_templates: Optional[Dict[str, str]] = None,
                       domain_hint: Optional[str] = None) -> str:
        self._log_debug(f"Original query for completion: '{query}'")
        if completion_templates is None:
            completion_templates = self.default_completion_templates

        # Domain could be used to select a subset of templates in the future
        # current_domain = self._infer_domain(query, domain_hint)

        for pattern, template in completion_templates.items():
            try:
                if re.match(pattern, query, re.IGNORECASE):
                    # Check if the query ALREADY looks like the completed version to avoid re-completing
                    # This is a simple check; might need refinement.
                    # Example: if template is "What is ?" and query is "What is sales?", don't change.
                    # A rough check: if the template's structure (non-group parts) is already in query.
                    temp_template_test = template.replace(r"", ".*").replace(r"", ".*").replace(r"", ".*")
                    if re.match(f"^{temp_template_test}$", query, re.IGNORECASE):
                        self._log_debug(f"Query '{query}' already seems complete for pattern '{pattern}'. Skipping.")
                        continue

                    expanded_query = re.sub(pattern, template, query, flags=re.IGNORECASE)
                    if expanded_query != query:
                        self._log_debug(f"Query completed: '{pattern}' -> '{expanded_query}'")
                        return expanded_query # Return on first match for simplicity
            except re.error as e:
                self._log_debug(f"Regex error for pattern '{pattern}': {e}")
                continue

        self._log_debug("No query completion applied.")
        return query

    def expand_synonyms(self, query: str,
                        synonym_sets: Optional[Dict[str, Any]] = None,
                        domain_hint: Optional[str] = None,
                        max_expansions_per_term: int = 1) -> str:
        self._log_debug(f"Original query for synonym expansion: '{query}'")
        if synonym_sets is None:
            synonym_sets = self.default_synonym_sets

        domain = self._infer_domain(query, domain_hint)

        # Combine default synonyms with domain-specific ones, domain takes precedence
        active_synonyms: Dict[str, List[str]] = {}
        active_synonyms.update(synonym_sets.get("default", {}))
        active_synonyms.update(synonym_sets.get(domain, {})) # Domain specific overrides/adds

        if not active_synonyms:
            self._log_debug("No active synonym sets for expansion.")
            return query

        # Use spaCy for tokenization and lemmatization if available
        # Otherwise, simple word splitting and lowercasing
        words_to_process = []
        if NLP and hasattr(self, 'doc') and self.doc: # Assuming self.doc is populated if NLP is available
            # doc = NLP(query) # Re-process to ensure fresh tokens if query was modified
            words_to_process = [(token.text, token.lemma_.lower()) for token in self.doc]
        else:
            words_to_process = [(word, word.lower()) for word in query.split()]

        expanded_query_parts = []
        original_query_parts = query.split() # For reconstruction if no spaCy
        word_idx_no_spacy = 0

        for original_term, term_lemma in words_to_process:
            found_synonym = False
            if term_lemma in active_synonyms:
                syns = active_synonyms[term_lemma]
                if syns:
                    # Build "term OR synonym1 OR synonym2" string
                    # Limit number of synonyms to avoid overly long queries
                    syns_to_add = syns[:max_expansions_per_term]
                    expansion_str = f"{original_term} OR {' OR '.join(syns_to_add)}"
                    # Basic check to avoid redundant expansion like "profit OR profit"
                    if original_term.lower() in [s.lower() for s in syns_to_add]:
                        expansion_str = original_term # Just use original term

                    expanded_query_parts.append(f"({expansion_str})")
                    self._log_debug(f"Expanded '{original_term}' (lemma: {term_lemma}) to '({expansion_str})' using domain '{domain}'")
                    found_synonym = True

            if not found_synonym:
                expanded_query_parts.append(original_term)

            word_idx_no_spacy +=1

        if not expanded_query_parts: # Should not happen if query has content
            return query

        expanded_query = " ".join(expanded_query_parts)
        if expanded_query != query:
            self._log_debug(f"Query after synonym expansion: '{expanded_query}'")
        else:
            self._log_debug("No synonyms applied.")
        return expanded_query

    def expand(self, query: str,
               apply_context_injection: bool = True,
               apply_completion: bool = True,
               apply_synonyms: bool = True,
               domain_hint: Optional[str] = None) -> str:
        self._log_debug(f"Starting expansion for query: '{query}'")
        current_query = query

        if apply_context_injection:
            current_query = self.add_document_type_context(current_query, domain_hint=domain_hint)

        if apply_completion:
            current_query = self.complete_query(current_query, domain_hint=domain_hint)

        if apply_synonyms:
            # Re-initialize self.doc for synonym expansion if spaCy is used,
            # as query might have changed after completion/context steps.
            if NLP:
                self.doc = NLP(current_query)
            current_query = self.expand_synonyms(current_query, domain_hint=domain_hint)

        self._log_debug(f"Final expanded query: '{current_query}'")
        return current_query

if __name__ == '__main__':
    expander = QueryExpander(debug_mode=True)

    queries_to_test_iter3 = [
        ("Q3 revenue", "finance"),
        ("Comparison 2023 2024", "default"),
        ("risk factors", "finance"),
        ("fix the login function", "coding"),
        ("summarize chapter 3", "writing"),
        ("market analysis", "finance"), # Test context injection + completion
        ("company profit", "finance"), # Test synonym expansion
        ("important research issue", "research") # Test synonym + context
    ]

    for q, domain in queries_to_test_iter3:
        print(f"--- Expanding Query (Iter 3): \"{q}\" (Domain Hint: {domain}) ---")

        # Test individual methods
        print("  Testing Completion:")
        completed = expander.complete_query(q, domain_hint=domain)
        print(f"  Completed: '{completed}'")

        print("  Testing Context Injection:")
        contexted = expander.add_document_type_context(completed, domain_hint=domain)
        print(f"  Context Injected: '{contexted}'")

        print("  Testing Synonym Expansion:")
        # Re-init doc for synonym expansion if spaCy is used
        if NLP: expander.doc = NLP(contexted)
        synonymized = expander.expand_synonyms(contexted, domain_hint=domain)
        print(f"  Synonymized: '{synonymized}'")

        print("  Testing Full Expansion Pipeline:")
        expanded_q = expander.expand(q, domain_hint=domain)
        print(f"  Fully Expanded: '{expanded_q}'")
        print("---------------------------------------\n")

    print("--- Testing query not needing much expansion ---")
    q_no_expand = "What is the detailed revenue breakdown for Q1 2024 for all US regions?"
    expanded_q_no_expand = expander.expand(q_no_expand, domain_hint="finance")
    print(f"  Fully Expanded: '{expanded_q_no_expand}'")
    print("---------------------------------------\n")
