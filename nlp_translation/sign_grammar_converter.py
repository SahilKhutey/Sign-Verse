"""
Sign Grammar Converter

Converts natural language grammar to sign language grammar for multiple languages.
Sign languages have different grammatical structures:
  - ASL: Time-Topic-Comment
  - DGS: Verb-Final (Dative-Accusative-Verb)
  - ISL: SOV structure
  - TSL: Similar to Turkish (SOV/OSV)
"""

class SignGrammarConverter:

    def __init__(self, language="ASL"):
        self.language = language.upper()
        self.stop_words = {"is", "am", "are", "was", "were", "the", "a", "an", "to"}

        self.time_words = {
            "yesterday", "today", "tomorrow", "now",
            "morning", "afternoon", "evening", "night",
            "later", "before", "after"
        }

    def convert(self, text, lang_hint=None):
        """
        Convert English sentence to sign language gloss order based on language rules.
        """
        lang = (lang_hint or self.language).upper()
        words = text.lower().split()

        # Step 1: Remove stop words
        filtered = [w for w in words if w not in self.stop_words]
        if not filtered:
            return []

        # Step 2: Extract time words
        time_tokens = [w.upper() for w in filtered if w in self.time_words]
        content_tokens = [w.upper() for w in filtered if w not in self.time_words]

        # Step 3: Apply Language-Specific Reordering
        if lang == "ASL":
            # ASL: TIME + TOPIC + COMMENT
            sign_gloss = time_tokens + content_tokens
        elif lang in ("DGS", "ISL", "TSL", "LSA"):
            # SOV: Subject + Object + Verb
            # Heuristic: In English (SVO), move the second word (usually verb) to the end.
            if len(content_tokens) >= 3:
                subject = content_tokens[0]
                verb = content_tokens[1]
                objects = content_tokens[2:]
                sign_gloss = time_tokens + [subject] + objects + [verb]
            elif len(content_tokens) == 2:
                # "I eat" -> "I eat" (already correct or subject-verb)
                sign_gloss = time_tokens + content_tokens
            else:
                sign_gloss = time_tokens + content_tokens
        else:
            # Default to Time-first
            sign_gloss = time_tokens + content_tokens

        return [f"<2{lang}>"] + sign_gloss

    def gloss_to_string(self, gloss):
        """Convert gloss list to display string."""
        return " ".join(gloss)

    def spoken_to_gloss(self, text, lang="ASL"):
        """Alias for convert to maintain backward compatibility."""
        gloss_list = self.convert(text, lang_hint=lang)
        return " ".join(gloss_list)
