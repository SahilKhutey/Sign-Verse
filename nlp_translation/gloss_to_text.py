"""
Gloss -> Text Conversion

Baseline reconstruction is intentionally simple:
- join tokens
- capitalize for readability

For learned translation, use a seq2seq/transformer model.
"""

from __future__ import annotations

from typing import Iterable, List


def gloss_to_text(gloss_tokens: Iterable[str]) -> str:
    toks: List[str] = [str(t) for t in gloss_tokens if str(t).strip()]
    return " ".join(w.capitalize() for w in toks)


class GlossToText:
    """OO wrapper for compatibility with service layers."""

    def convert(self, gloss_tokens: Iterable[str]) -> str:
        return gloss_to_text(gloss_tokens)
