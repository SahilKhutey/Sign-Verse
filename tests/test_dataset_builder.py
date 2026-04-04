import csv
import json
import os
import shutil
import sys
import uuid

from training.data_pipeline import build_text_gloss_dataset as builder


def _write_pairs_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "gloss"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_dedupe_and_normalize():
    pairs = [
        (" Hello   world ", "hello world"),
        ("hello world", "HELLO WORLD"),
        ("Bye", "BYE"),
    ]
    out = builder._dedupe(pairs)
    assert len(out) == 2
    assert out[0] == ("Hello world", "HELLO WORLD")
    assert out[1] == ("Bye", "BYE")


def test_split_by_hash_is_deterministic():
    pairs = [(f"text {i}", f"GLOSS {i}") for i in range(30)]
    a_train, a_val, a_test = builder._split_by_hash(pairs, 0.8, 0.1, 0.1)
    b_train, b_val, b_test = builder._split_by_hash(pairs, 0.8, 0.1, 0.1)
    assert a_train == b_train
    assert a_val == b_val
    assert a_test == b_test
    assert len(a_train) + len(a_val) + len(a_test) == len(pairs)


def test_main_writes_report_with_expected_keys(monkeypatch):
    td = os.path.join(os.getcwd(), f".tmp_dataset_builder_{uuid.uuid4().hex}")
    os.makedirs(td, exist_ok=True)
    try:
        out_dir = os.path.join(td, "datasets", "text_sign_pairs")
        train_data = os.path.join(out_dir, "training_data.csv")
        _write_pairs_csv(
            train_data,
            [
                {"text": "hello there", "gloss": "HELLO THERE"},
                {"text": "hello there", "gloss": "hello there"},  # duplicate after normalize
                {"text": "good morning", "gloss": "GOOD MORNING"},
            ],
        )

        os.makedirs(os.path.join(td, "training-data"), exist_ok=True)
        os.makedirs(os.path.join(td, "datasets", "isl_dataset"), exist_ok=True)

        monkeypatch.chdir(td)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_text_gloss_dataset.py",
                "--out-dir",
                str(out_dir),
                "--warn-min-pairs",
                "20",
                "--warn-dedupe-rate",
                "0.2",
            ],
        )
        builder.main()

        report_path = os.path.join(td, "reports", "text_gloss_dataset_report.json")
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        for key in [
            "total_pairs",
            "train_pairs",
            "val_pairs",
            "test_pairs",
            "dedupe_rate",
            "length_stats",
            "unique_counts",
            "top_tokens",
            "warnings",
        ]:
            assert key in report

        assert isinstance(report["top_tokens"]["text_top"], list)
        assert isinstance(report["top_tokens"]["gloss_top"], list)
        assert any("Low dataset size" in msg for msg in report["warnings"])
    finally:
        shutil.rmtree(td, ignore_errors=True)
