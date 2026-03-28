"""
Sign Grammar Converter

Converts natural language grammar to sign language grammar.
Sign languages have different grammatical structures:
  - Topic-Comment structure
  - No articles (a, an, the)
  - No linking verbs (is, am, are)
  - Time markers come first
  - Questions use facial grammar

Example:
    English: "I am going to the school tomorrow"
    Sign:    "TOMORROW SCHOOL I GO"
"""


class SignGrammarConverter:

    def __init__(self):
        self.stop_words = {"is", "am", "are", "was", "were", "the", "a", "an", "to"}

        self.time_words = {
            "yesterday", "today", "tomorrow", "now",
            "morning", "afternoon", "evening", "night",
            "later", "before", "after"
        }

    def convert(self, text):
        """
        Convert English sentence to sign language gloss order.

        Steps:
            1. Remove stop words
            2. Move time markers to the front
            3. Uppercase all tokens
        """

        words = text.lower().split()

        time_tokens = []
        content_tokens = []

        for word in words:
            if word in self.stop_words:
                continue
            elif word in self.time_words:
                time_tokens.append(word.upper())
            else:
                content_tokens.append(word.upper())

        # Sign grammar: TIME + TOPIC + COMMENT
        sign_gloss = time_tokens + content_tokens

        return sign_gloss

    def gloss_to_string(self, gloss):
        """Convert gloss list to display string."""
        return " ".join(gloss)
