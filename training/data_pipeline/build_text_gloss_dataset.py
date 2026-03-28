"""
Build a larger text<->gloss dataset from available sources.

Sources:
  - datasets/text_sign_pairs/*.csv (text, gloss)
  - training-data/labels.csv (label_name + optional text column)
  - datasets/isl_dataset/annotations.json (if present)

Output:
  datasets/text_sign_pairs/expanded_pairs.csv
  datasets/text_sign_pairs/train.csv
  datasets/text_sign_pairs/val.csv
  datasets/text_sign_pairs/test.csv
  reports/text_gloss_dataset_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from typing import Iterable, List, Tuple

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


def _read_pairs_from_folder(folder: str) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.exists(folder):
        return pairs
    for name in os.listdir(folder):
        if not name.endswith(".csv"):
            continue
        pairs.extend(_read_pairs_from_csv(os.path.join(folder, name)))
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
                pairs.append((label.replace("_", " ").title(), label))
    return pairs


def _pairs_from_isl_annotations(path: str) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data if isinstance(data, list) else data.values():
        text = (entry.get("text") or entry.get("sentence") or "").strip()
        gloss = (entry.get("gloss") or entry.get("tokens") or "").strip()
        if text and gloss:
            pairs.append((text, gloss))
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


def _length_stats(pairs: List[Tuple[str, str]]) -> dict:
    if not pairs:
        return {"text": {}, "gloss": {}}
    text_lens = [len(t.split()) for t, _ in pairs]
    gloss_lens = [len(g.split()) for _, g in pairs]
    return {
        "text": {
            "min": min(text_lens),
            "max": max(text_lens),
            "avg": round(sum(text_lens) / max(1, len(text_lens)), 2),
        },
        "gloss": {
            "min": min(gloss_lens),
            "max": max(gloss_lens),
            "avg": round(sum(gloss_lens) / max(1, len(gloss_lens)), 2),
        },
    }


def _unique_counts(pairs: List[Tuple[str, str]]) -> dict:
    texts = {_normalize_text(t).upper() for t, _ in pairs}
    glosses = {_normalize_gloss(g) for _, g in pairs}
    return {"unique_texts": len(texts), "unique_glosses": len(glosses)}


def _top_tokens(pairs: List[Tuple[str, str]], limit: int = 20) -> dict:
    text_counts = {}
    gloss_counts = {}
    for text, gloss in pairs:
        for tok in _normalize_text(text).split():
            tok = tok.lower()
            text_counts[tok] = text_counts.get(tok, 0) + 1
        for tok in _normalize_gloss(gloss).split():
            tok = tok.lower()
            gloss_counts[tok] = gloss_counts.get(tok, 0) + 1
    text_top = sorted(text_counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    gloss_top = sorted(gloss_counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return {
        "text_top": text_top,
        "gloss_top": gloss_top,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join("datasets", "text_sign_pairs"))
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--warn-dedupe-rate", type=float, default=0.35)
    parser.add_argument("--warn-min-pairs", type=int, default=200)
    args = parser.parse_args()

    base_dir = args.out_dir
    out_expanded = os.path.join(base_dir, "expanded_pairs.csv")
    out_train = os.path.join(base_dir, "train.csv")
    out_val = os.path.join(base_dir, "val.csv")
    out_test = os.path.join(base_dir, "test.csv")

    pairs = []
    pairs += _read_pairs_from_folder(base_dir)
    pairs += _pairs_from_labels_csv(os.path.join("training-data", "labels.csv"))
    pairs += _pairs_from_isl_annotations(os.path.join("datasets", "isl_dataset", "annotations.json"))

    converter = SignGrammarConverter()
    augmented = []
    for text, gloss in pairs:
        if gloss:
            augmented.append((text, gloss))
        else:
            gen_gloss = " ".join(converter.convert(text))
            if gen_gloss:
                augmented.append((text, gen_gloss))

    deduped = _dedupe(augmented)
    filtered = _filter_pairs(deduped, args.min_len, args.max_len)
    if args.max_pairs and len(filtered) > args.max_pairs:
        filtered = filtered[: args.max_pairs]

    train, val, test = _split_by_hash(
        filtered,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    _write_csv(out_expanded, filtered)
    _write_csv(out_train, train)
    _write_csv(out_val, val)
    _write_csv(out_test, test)

    dedupe_removed = max(0, len(augmented) - len(deduped))
    dedupe_rate = round(dedupe_removed / max(1, len(augmented)), 4)
    warnings = []
    if dedupe_rate >= args.warn_dedupe_rate:
        warnings.append(f"High dedupe rate: {dedupe_rate:.2f}")
    if len(filtered) < args.warn_min_pairs:
        warnings.append(f"Low dataset size: {len(filtered)} pairs")

    report = {
        "total_pairs": len(filtered),
        "train_pairs": len(train),
        "val_pairs": len(val),
        "test_pairs": len(test),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "dedupe_removed": dedupe_removed,
        "dedupe_rate": dedupe_rate,
        "length_stats": _length_stats(filtered),
        "unique_counts": _unique_counts(filtered),
        "top_tokens": _top_tokens(filtered),
        "warnings": warnings,
        "sources": {
            "csv_folder": base_dir,
            "labels_csv": os.path.join("training-data", "labels.csv"),
            "isl_annotations": os.path.join("datasets", "isl_dataset", "annotations.json"),
        },
        "outputs": {
            "expanded": out_expanded,
            "train": out_train,
            "val": out_val,
            "test": out_test,
        },
    }
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "text_gloss_dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(filtered):,} pairs to {out_expanded}")
    print(f"Train/Val/Test: {len(train):,} / {len(val):,} / {len(test):,}")


if __name__ == "__main__":
    main()
