class TextToSignConverter:

    def __init__(self):
        from nlp_translation.sign_grammar_converter import SignGrammarConverter
        self._converter = SignGrammarConverter()

    def convert(self, text):
        """
        Convert natural language text to sign gloss tokens (baseline rule-based).

        Example:
            "I am going to school tomorrow"
            -> ["TOMORROW", "I", "GOING", "SCHOOL"]
        """
        return self._converter.convert(text)
