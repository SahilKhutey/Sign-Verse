"""
Dataset Registry — Metadata and download info for all
publicly available sign language datasets.

Datasets:
    Dataset         Samples    Vocab    Language    Type
    WLASL           21,083     2000     ASL         word-level
    AUTSL           38,336      226     TSL         word-level
    RWTH-PHOENIX    45,760     1200     DGS         sentence-level
    How2Sign        80,000    16000     ASL         sentence-level
    MS-ASL          25,000     1000     ASL         word-level
    LSA64            3,200       64     LSA         word-level
    ─────────────────────────────────────────────────────
    TOTAL          213,379    ~20k     multilingual
"""

DATASETS = {
    "WLASL": {
        "description": "Word-Level American Sign Language",
        "url": "https://github.com/dxli94/WLASL",
        "annotation_url": "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json",
        "samples": 21083,
        "vocab_size": 2000,
        "type": "word_level",
        "language": "ASL",
        "license": "Attribution Non-Commercial 4.0",
        "requires_request": False,
        "annotation_format": "json",
        "splits": ["train", "val", "test"],
    },
    "AUTSL": {
        "description": "American Turkish Sign Language",
        "url": "https://chalearnlap.cvc.uab.cat/dataset/40/description/",
        "samples": 38336,
        "vocab_size": 226,
        "type": "word_level",
        "language": "TSL",
        "license": "Research use",
        "requires_request": True,
        "annotation_format": "csv",
        "splits": ["train", "val", "test"],
    },
    "RWTH_PHOENIX": {
        "description": "RWTH-PHOENIX-Weather 2014T — sentence-level DGS",
        "url": "https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/",
        "download_url": "https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/phoenix-2014-T.v3.tar.gz",
        "samples": 45760,
        "vocab_size": 1200,
        "type": "sentence_level",
        "language": "DGS",
        "license": "Research use",
        "requires_request": True,
        "annotation_format": "csv",
        "splits": ["train", "dev", "test"],
        "size_gb": 53.0,
    },
    "How2Sign": {
        "description": "American Sign Language sentence-level",
        "url": "https://how2sign.github.io/",
        "samples": 80000,
        "vocab_size": 16000,
        "type": "sentence_level",
        "language": "ASL",
        "license": "CC BY 4.0",
        "requires_request": False,
        "annotation_format": "csv",
        "splits": ["train", "val", "test"],
        "size_gb": 300.0,
    },
    "MS_ASL": {
        "description": "Microsoft ASL dataset",
        "url": "https://www.microsoft.com/en-us/research/project/ms-asl/",
        "samples": 25000,
        "vocab_size": 1000,
        "type": "word_level",
        "language": "ASL",
        "license": "Research use",
        "requires_request": True,
        "annotation_format": "json",
        "splits": ["train", "val", "test"],
    },
    "LSA64": {
        "description": "Argentine Sign Language — 64 words",
        "url": "http://facundoq.github.io/datasets/lsa64/",
        "download_url": "https://mega.nz/folder/hTBTDSIL#9rDtKspGThlHY2iqGUgcvQ",
        "samples": 3200,
        "vocab_size": 64,
        "type": "word_level",
        "language": "LSA",
        "license": "Research use",
        "requires_request": False,
        "annotation_format": "directory",
        "splits": ["full"],
    },
}


# Helper to get all languages
ALL_LANGUAGES = sorted(list(set([info["language"] for info in DATASETS.values()])))

# Standardized Language Tags Mapping
LANGUAGE_TAGS = {
    "ASL": "<2ASL>",
    "DGS": "<2DGS>",
    "TSL": "<2TSL>",
    "ISL": "<2ISL>",
    "LSA": "<2LSA>"
}


def get_dataset(name):
    """Get dataset metadata by name."""
    return DATASETS.get(name)


def list_datasets(enabled_only=False):
    """List all registered datasets."""
    return list(DATASETS.keys())


def total_samples():
    """Calculate total samples across all datasets."""
    return sum(d["samples"] for d in DATASETS.values())


def print_summary():
    """Print a summary table of all datasets."""
    print(f"\n{'Dataset':<20} {'Samples':>10} {'Vocab':>8} {'Lang':<6} {'Tag':<10} {'Type':<16}")
    print("-" * 75)
    for name, info in DATASETS.items():
        lang = info['language']
        tag = LANGUAGE_TAGS.get(lang, "N/A")
        print(f"{name:<20} {info['samples']:>10,} {info['vocab_size']:>8,} "
              f"{lang:<6} {tag:<10} {info['type']:<16}")
    print("-" * 75)
    print(f"{'TOTAL':<20} {total_samples():>10,}\n")


if __name__ == "__main__":
    print_summary()
