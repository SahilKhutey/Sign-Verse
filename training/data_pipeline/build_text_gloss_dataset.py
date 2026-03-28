"""
Build a larger text<->gloss dataset from available sources.

Sources:
  - datasets/text_sign_pairs/*.csv (text, gloss)
  - training-data/labels.csv (label_name + optional text column)
  - datasets/isl_dataset/annotations.json (if present)

Output:
  datasets/text_sign_pairs/expanded_pairs.csv
"""

from __future__ import annotations

import csv
import json
import os
from typing import List, Tuple

from nlp_translation.sign_grammar_converter import SignGrammarConverter


def _read_pairs_from_csv(path: str) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            gloss = (row.get("gloss") or "").strip()
            if text and gloss:
                pairs.append((text, gloss))
    return pairs


def _pairs_from_labels_csv(path: str) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label_name") or "").strip()
            text = (row.get("text") or "").strip()
            if text and label:
                pairs.append((text, label))
            elif label:
                # Use label as gloss, synthesize text (simple baseline)
                pairs.append((label.replace("_", " ").title(), label))
    return pairs


def _pairs_from_isl_annotations(path: str) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect entries like {"text": "...", "gloss": "..."} or similar
    for entry in data if isinstance(data, list) else data.values():
        text = (entry.get("text") or entry.get("sentence") or "").strip()
        gloss = (entry.get("gloss") or entry.get("tokens") or "").strip()
        if text and gloss:
            pairs.append((text, gloss))
    return pairs


def _dedupe(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for text, gloss in pairs:
        key = (text.strip().upper(), gloss.strip().upper())
        if key in seen:
            continue
        seen.add(key)
        out.append((text, gloss))
    return out


def main():
    base_dir = os.path.join("datasets", "text_sign_pairs")
    out_path = os.path.join(base_dir, "expanded_pairs.csv")

    pairs = []
    # Existing small CSVs
    pairs += _read_pairs_from_csv(os.path.join(base_dir, "training_data.csv"))
    pairs += _read_pairs_from_csv(os.path.join(base_dir, "validation_data.csv"))

    # Unified labels (if available)
    pairs += _pairs_from_labels_csv(os.path.join("training-data", "labels.csv"))

    # ISL annotations (if available)
    pairs += _pairs_from_isl_annotations(os.path.join("datasets", "isl_dataset", "annotations.json"))

    # Augment by generating gloss from text (rule-based) when gloss missing
    converter = SignGrammarConverter()
    augmented = []
    for text, gloss in pairs:
        if gloss:
            augmented.append((text, gloss))
        else:
            gen_gloss = " ".join(converter.convert(text))
            if gen_gloss:
                augmented.append((text, gen_gloss))

    pairs = _dedupe(augmented)

    os.makedirs(base_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "gloss"])
        writer.writeheader()
        for text, gloss in pairs:
            writer.writerow({"text": text, "gloss": gloss})

    print(f"Wrote {len(pairs):,} pairs to {out_path}")


if __name__ == "__main__":
    main()
