"""
Run concatenative text->sign video synthesis locally.

Examples:
  python scripts/run_concatenative_synthesis.py --text "hello how are you" --plan-only
  python scripts/run_concatenative_synthesis.py --text "hello how are you" --output datasets/synthetic_sentence_videos/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is importable when script is run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp_translation.concatenative_synthesis import ConcatenativeSynthesis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--dictionary-manifest",
        default=os.path.join("datasets", "sign_dictionary", "manifest.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("datasets", "synthetic_sentence_videos", "demo.mp4"),
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--strict-manifest", action="store_true")
    args = parser.parse_args()

    synth = ConcatenativeSynthesis(
        dictionary_manifest=args.dictionary_manifest,
        strict_exists=args.strict_manifest,
    )
    if args.plan_only:
        out = synth.plan_from_text(args.text)
    else:
        out = synth.synthesize_text(
            text=args.text,
            output_path=args.output,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
