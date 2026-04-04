"""
Generate synthetic sentence-level sign videos by concatenating dictionary clips.

Outputs:
  - training-data/synthetic_sign_video_pairs.csv
  - reports/synthetic_sign_video_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nlp_translation.concatenative_synthesis import ConcatenativeSynthesis


def _read_text_gloss_pairs(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-csv",
        default=os.path.join("datasets", "text_sign_pairs", "expanded_pairs.csv"),
    )
    parser.add_argument(
        "--dictionary-manifest",
        default=os.path.join("datasets", "sign_dictionary", "manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("datasets", "synthetic_sentence_videos"),
    )
    parser.add_argument(
        "--output-manifest",
        default=os.path.join("training-data", "synthetic_sign_video_pairs.csv"),
    )
    parser.add_argument(
        "--report-path",
        default=os.path.join("reports", "synthetic_sign_video_report.json"),
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    pairs = _read_text_gloss_pairs(args.pairs_csv)
    if not pairs:
        raise RuntimeError(f"No text/gloss pairs found in {args.pairs_csv}")

    synth = ConcatenativeSynthesis(
        dictionary_manifest=args.dictionary_manifest,
        strict_exists=True,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(args.max_samples) if args.max_samples > 0 else len(pairs)

    rows = []
    attempted = 0
    rendered = 0
    skipped_low_coverage = 0
    for i, (text, gloss) in enumerate(pairs):
        if attempted >= max_samples:
            break
        attempted += 1
        gloss_tokens = [tok for tok in gloss.split() if tok.strip()]
        planned = synth.build_plan(gloss_tokens)
        coverage = float(planned.get("coverage") or 0.0)
        if coverage < float(args.min_coverage):
            skipped_low_coverage += 1
            continue

        out_path = out_dir / f"sample_{i:06d}.mp4"
        render_stats = {
            "output_path": str(out_path),
            "frames_written": 0,
            "rendered_tokens": 0,
            "errors": [],
        }
        if not args.plan_only:
            if not (args.skip_existing and out_path.exists()):
                render_stats = synth.synthesize_plan(
                    plan=planned["plan"],
                    output_path=str(out_path),
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                )
            if out_path.exists():
                rendered += 1

        rows.append(
            {
                "text": text,
                "gloss": gloss,
                "coverage": round(coverage, 4),
                "found_tokens": int(planned.get("found_tokens") or 0),
                "total_tokens": int(planned.get("total_tokens") or 0),
                "missing_tokens": " ".join(planned.get("missing_tokens") or []),
                "video_path": render_stats.get("output_path"),
                "frames_written": int(render_stats.get("frames_written") or 0),
                "rendered_tokens": int(render_stats.get("rendered_tokens") or 0),
                "errors": "|".join(render_stats.get("errors") or []),
            }
        )

    os.makedirs(os.path.dirname(args.output_manifest) or ".", exist_ok=True)
    with open(args.output_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "gloss",
                "coverage",
                "found_tokens",
                "total_tokens",
                "missing_tokens",
                "video_path",
                "frames_written",
                "rendered_tokens",
                "errors",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": args.pairs_csv,
        "dictionary_manifest": args.dictionary_manifest,
        "attempted_samples": attempted,
        "written_samples": len(rows),
        "rendered_videos": rendered,
        "skipped_low_coverage": skipped_low_coverage,
        "plan_only": bool(args.plan_only),
        "min_coverage": float(args.min_coverage),
        "outputs": {
            "videos_dir": str(out_dir),
            "manifest": args.output_manifest,
        },
    }
    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote synthetic pairs: {len(rows)}")
    print(f"Rendered videos: {rendered}")
    print(f"Manifest: {args.output_manifest}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
