"""
Text -> Gloss Conversion

This module provides the simple baseline conversion used throughout the repo:
- remove stop-words
- move time markers to the front
- uppercase output tokens (gloss-like)

For learned translation, see:
- `nlp_translation/models/seq2seq_model.py`
- `nlp_translation/models/transformer_model.py`
"""

from __future__ import annotations

from typing import List

from nlp_translation.sign_grammar_converter import SignGrammarConverter


_converter = SignGrammarConverter()


def text_to_gloss(text: str) -> List[str]:
    """Convert English text to a list of gloss tokens (baseline rule-based)."""
    return _converter.convert(text)


class TextToGloss:
    """Small OO wrapper for compatibility with service layers."""

    def __init__(self):
        self._converter = SignGrammarConverter()

    def convert(self, text: str) -> List[str]:
        return self._converter.convert(text)
