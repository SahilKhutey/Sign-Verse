"""
Build a multi-lingual text<->gloss dataset from available and synthetic sources.
Support for ASL, DGS, TSL, ISL, and LSA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from typing import Iterable, List, Tuple

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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


def _pairs_from_labels_csv(path: str, lang: str = "ASL") -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label_name") or "").strip()
            text = (row.get("text") or "").strip()
            if text and label:
                pairs.append((text, f"[{lang}] {label}"))
            elif label:
                # Placeholder for label name if no text description exists.
                pairs.append((label.replace("_", " ").title(), f"[{lang}] {label}"))
    return pairs


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_gloss(gloss: str) -> str:
    gloss = " ".join(gloss.strip().split())
    return gloss.upper()


def _dedupe(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for text, gloss in pairs:
        key = (_normalize_text(text).upper(), _normalize_gloss(gloss))
        if key in seen:
            continue
        seen.add(key)
        out.append((_normalize_text(text), _normalize_gloss(gloss)))
    return out


def _filter_pairs(
    pairs: Iterable[Tuple[str, str]],
    min_len: int,
    max_len: int,
) -> List[Tuple[str, str]]:
    filtered = []
    for text, gloss in pairs:
        t_len = len(text.split())
        g_len = len(gloss.split())
        if t_len < min_len or g_len < min_len:
            continue
        if t_len > max_len or g_len > max_len:
            continue
        filtered.append((text, gloss))
    return filtered


def _split_by_hash(
    pairs: List[Tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    train, val, test = [], [], []
    total = max(1, train_ratio + val_ratio + test_ratio)
    train_cut = train_ratio / total
    val_cut = (train_ratio + val_ratio) / total

    for text, gloss in pairs:
        key = f"{text}||{gloss}".encode("utf-8")
        h = hashlib.sha1(key).hexdigest()
        bucket = int(h[:6], 16) / float(0xFFFFFF)
        if bucket < train_cut:
            train.append((text, gloss))
        elif bucket < val_cut:
            val.append((text, gloss))
        else:
            test.append((text, gloss))
    return train, val, test


def _write_csv(path: str, pairs: List[Tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "gloss"])
        writer.writeheader()
        for text, gloss in pairs:
            writer.writerow({"text": text, "gloss": gloss})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join("datasets", "text_sign_pairs"))
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    args = parser.parse_args()

    base_dir = args.out_dir
    out_train = os.path.join(base_dir, "train.csv")
    out_val = os.path.join(base_dir, "val.csv")
    out_test = os.path.join(base_dir, "test.csv")

    # Languages supported.
    LANGS = ["ASL", "DGS", "TSL", "ISL", "LSA"]
    converter = SignGrammarConverter()

    # Load synthetic labels from labels.csv as a primary source of signs.
    labels_path = os.path.join("training-data", "labels.csv")
    raw_labels = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_labels = [row.get("label_name") for row in reader if row.get("label_name")]

    if not raw_labels:
        # Fallback if labels.csv is missing.
        raw_labels = [f"SIGN_{i:04d}" for i in range(1000)]

    augmented_pairs = []

    # Map chunks of labels to different languages for multi-lingual diversity.
    chunk_size = max(1, len(raw_labels) // len(LANGS))
    for i, lang in enumerate(LANGS):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < len(LANGS) - 1 else len(raw_labels)
        lang_labels = raw_labels[start_idx:end_idx]

        # Generate translation templates for each label in this language's context.
        templates = [
            "I like {label}",
            "Show me {label}",
            "Translate {label}",
            "Learn {label}",
            "Repeat {label}",
            "Can you sign {label}?",
            "What is {label}?",
            "This is {label}",
            "See {label}",
            "Wait for {label}"
        ]

        for label in lang_labels:
            clean_label = label.replace("_", " ").title()
            for tpl in random.sample(templates, 3):  # 3 variations per sign
                raw_text = tpl.format(label=clean_label)
                # Prepend language hint to source text
                text = f"[{lang}] {raw_text}"
                # Convert to gloss using the multi-lingual grammar rules.
                # The converter automatically adds the [LANG] marker to the gloss output.
                gloss_list = converter.convert(raw_text, lang_hint=lang)
                augmented_pairs.append((text, " ".join(gloss_list)))

    deduped = _dedupe(augmented_pairs)
    filtered = _filter_pairs(deduped, args.min_len, args.max_len)
    
    if args.max_pairs and len(filtered) > args.max_pairs:
        filtered = random.sample(filtered, args.max_pairs)

    train, val, test = _split_by_hash(
        filtered,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    _write_csv(out_train, train)
    _write_csv(out_val, val)
    _write_csv(out_test, test)

    print(f"✅ Multi-Lingual Dataset Build Summary")
    print(f"  Total Pairs: {len(filtered):,}")
    print(f"  Languages:   {', '.join(LANGS)}")
    print(f"  Train:       {len(train):,}")
    print(f"  Val:         {len(val):,}")
    print(f"  Test:        {len(test):,}")


if __name__ == "__main__":
    main()
