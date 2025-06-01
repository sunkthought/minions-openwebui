import re
from enum import Enum

class QueryComplexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"

class QueryComplexityClassifier:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        # Keywords indicating complexity
        self.complex_keywords = [
            "analyze", "compare", "contrast", "summarize", "explain in detail",
            "discuss", "critique", "evaluate", "recommend", "predict", "what if",
            "how does", "why does", "implications"
        ]
        self.medium_keywords = [
            "list", "describe", "details of", "tell me about", "what are the"
        ]
        # Question words (simple ones often start fact-based questions)
        self.simple_question_starters = ["what is", "who is", "when was", "where is", "define"]

    def classify_query(self, query: str) -> QueryComplexity:
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Query='{query_lower}', WordCount={word_count}")

        # Rule 1: Complex Keywords
        for keyword in self.complex_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched complex keyword '{keyword}'")
                return QueryComplexity.COMPLEX

        # Rule 2: Word Count for Complex
        if word_count > 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched complex by word count (>25)")
            return QueryComplexity.COMPLEX

        # Rule 3: Word Count for Simple (and simple question starters)
        if word_count < 10:
            is_simple_starter = any(query_lower.startswith(starter) for starter in self.simple_question_starters)
            if is_simple_starter:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched simple by word count (<10) and starter.")
                return QueryComplexity.SIMPLE
            # If short but not a clear simple starter, could be medium or simple.
            # For now, let's lean towards medium if no other rules hit.
            # A short query like "impact of AI" is not simple.

        # Rule 4: Medium Keywords
        for keyword in self.medium_keywords:
            if keyword in query_lower:
                if self.debug_mode:
                    print(f"DEBUG QueryComplexityClassifier: Matched medium keyword '{keyword}'")
                return QueryComplexity.MEDIUM

        # Rule 5: Word Count for Medium (defaulting after complex/simple checks)
        if word_count >= 10 and word_count <= 25:
            if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Matched medium by word count (10-25)")
            return QueryComplexity.MEDIUM

        # Default or fallback classification
        # If very short and not caught by simple starters, it might be an implicit medium query.
        if word_count < 10:
             if self.debug_mode:
                print(f"DEBUG QueryComplexityClassifier: Defaulting short query to MEDIUM (no simple starter)")
             return QueryComplexity.MEDIUM # e.g. "AI ethics"

        # If it's longer but didn't hit complex keywords, it's likely medium.
        if self.debug_mode:
            print(f"DEBUG QueryComplexityClassifier: Defaulting to MEDIUM (no other rules matched clearly)")
        return QueryComplexity.MEDIUM

if __name__ == '__main__': # For basic testing
    classifier = QueryComplexityClassifier(debug_mode=True)
    queries_to_test = [
        ("What is the capital of France?", QueryComplexity.SIMPLE),
        ("Define artificial intelligence.", QueryComplexity.SIMPLE),
        ("List the main causes of climate change and their effects.", QueryComplexity.MEDIUM), # Medium keyword
        ("Tell me about the process of photosynthesis.", QueryComplexity.MEDIUM),
        ("Compare and contrast renewable and non-renewable energy sources, including their environmental impact.", QueryComplexity.COMPLEX), # Complex keyword
        ("Analyze the economic impact of the COVID-19 pandemic on the tourism industry worldwide.", QueryComplexity.COMPLEX), # Complex keyword
        ("Summarize the plot of Hamlet.", QueryComplexity.COMPLEX), # Complex keyword
        ("How does a blockchain work?", QueryComplexity.COMPLEX), # Complex keyword (how does)
        ("The impact of social media on society.", QueryComplexity.MEDIUM), # 7 words, no strong simple/complex keyword
        ("Renewable energy.", QueryComplexity.MEDIUM), # 2 words, no simple starter
        ("Explain the theory of relativity in detail and discuss its major implications for modern physics.", QueryComplexity.COMPLEX), # >25 words
        ("What are the benefits of exercise?", QueryComplexity.MEDIUM) # Medium keyword
    ]

    for q, expected in queries_to_test:
        result = classifier.classify_query(q)
        print(f"Query: '{q}' -> Classified as: {result.value}, Expected: {expected.value} (Correct: {result == expected})")
