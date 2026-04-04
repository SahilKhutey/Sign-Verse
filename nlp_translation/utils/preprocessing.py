"""
Text Preprocessing Utilities for NLP Translation

Handles text cleaning, normalization, and augmentation
for sign language translation pipeline.
"""

import re
import string


def clean_text(text):
    """Remove extra whitespace, punctuation, and normalize."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_gloss(gloss):
    """Normalize sign gloss format to uppercase tokens."""
    return " ".join(gloss.upper().split())


def augment_text(text):
    """
    Simple text augmentation:
    - Synonym substitution placeholder
    - Random word dropout
    """
    words = text.split()
    if len(words) > 3:
        # Drop a random word (simple augmentation)
        import random
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    return " ".join(words)


def split_sentences(text):
    """Split text into individual sentences."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]
