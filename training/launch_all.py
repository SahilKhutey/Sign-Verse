"""
Consolidated Training Launcher — wraps run_all_training.py with better UX.

Features:
    - --dry-run: 2 epochs, 2 batches — validates full pipeline in ~1 min
    - Better error output with stage-by-stage timing
    - JSON summary written to logs/launch_summary.json

Usage:
    python training/launch_all.py                 # full pipeline
    python training/launch_all.py --dry-run       # quick validation
    python training/launch_all.py --from-stage foundation --dry-run
"""

import os
import sys
import argparse
import time
import json
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(
        description="SignVerse — Master Training Launcher"
    )
    parser.add_argument("--stage", choices=[
        "download", "preprocess", "validate", "tokenize",
        "gesture", "foundation", "transformer", "diffusion", "llm", "registry"
    ], help="Run a single stage")
    parser.add_argument("--from-stage", choices=[
        "download", "preprocess", "validate", "tokenize",
        "gesture", "foundation", "transformer", "diffusion", "llm", "registry"
    ], help="Start from a specific stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 epochs per model for validation")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download guidance stage")
    parser.add_argument("--config",
                        default="training/configs/training_config.yaml")
    
    # Production arguments
    parser.add_argument("--datasets", nargs="+", help="Datasets to use (e.g. wlasl autsl)")
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--resume", help="Resume from 'latest' or specific checkpoint")
    parser.add_argument("--monitor", choices=["wandb", "tensorboard", "none"], 
                        help="Monitoring tool to use")
    args = parser.parse_args()

    print(f"\n{'=' * 62}")
    print(f"  SignVerse AI — Master Training Pipeline")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'DRY-RUN (2 epochs)' if args.dry_run else 'FULL TRAINING'}")
    print(f"{'=' * 62}")

    # ── Load config and override for dry-run ──
    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    if args.dry_run:
        # Override all epoch counts
        for section in ["gesture_model", "foundation_model", "sign_transformer",
                        "diffusion_model", "multimodal_llm"]:
            if section not in config:
                config[section] = {}
            config[section]["epochs"] = 2
            config[section]["batch_size"] = min(config[section].get("batch_size", 8), 8)
            config[section]["num_workers"] = 0

    # ── Multi-GPU and production overrides ──
    global_overrides = {}
    if args.gpus: global_overrides["gpus"] = args.gpus
    if args.batch_size: global_overrides["batch_size"] = args.batch_size
    if args.monitor: global_overrides["monitor"] = args.monitor
    if args.resume: global_overrides["resume"] = (args.resume == "latest" or args.resume.lower() == "true")

    for section in ["gesture_model", "foundation_model", "sign_transformer",
                    "diffusion_model", "multimodal_llm"]:
        if section not in config:
            config[section] = {}
        config[section].update(global_overrides)
        if args.datasets:
            config[section]["datasets"] = args.datasets

    # ── Import stage functions ──
    try:
        from training.run_all_training import (
            STAGES, STAGE_ORDER, stage_download, stage_preprocess,
            stage_validate, stage_tokenize, stage_train_gesture,
            stage_train_foundation, stage_train_transformer,
            stage_train_diffusion, stage_train_llm, stage_save_registry
        )
    except ImportError as e:
        print(f"  Error importing training stages: {e}")
        sys.exit(1)

    # ── Determine stages to run ──
    if args.stage:
        stages_to_run = [args.stage]
    else:
        start_idx = 0
        if args.from_stage:
            start_idx = STAGE_ORDER.index(args.from_stage)
        if args.skip_download:
            start_idx = max(start_idx, STAGE_ORDER.index("preprocess"))
        stages_to_run = STAGE_ORDER[start_idx:]

    # ── Run stages with timing ──
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    results = {}
    t_total = time.time()

    for stage_name in stages_to_run:
        print(f"\n{'-' * 62}")
        print(f"  Stage: {stage_name}")
        print(f"{'-' * 62}")

        t0 = time.time()
        try:
            result = STAGES[stage_name](config)
            elapsed = time.time() - t0
            results[stage_name] = {
                "status": "OK",
                "time_s": round(elapsed, 1),
                "result": str(result) if result is not None else None
            }
            print(f"\n  [OK] {stage_name} completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            results[stage_name] = {
                "status": "FAILED",
                "time_s": round(elapsed, 1),
                "error": str(e),
                "traceback": tb
            }
            print(f"\n  [FAILED] {stage_name} FAILED in {elapsed:.1f}s")
            print(f"  Error: {e}")
            print(f"\n  Traceback:\n{tb}")

    total_time = time.time() - t_total

    # ── Summary ──
    print(f"\n{'=' * 62}")
    print(f"  Training Summary")
    print(f"{'=' * 62}")

    passed = 0
    failed = 0
    for stage, res in results.items():
        status = res.get("status", "OK")
        mark = "[OK]" if status == "OK" else "[FAILED]"
        t = f"({res.get('time_s', '')}s)"
        print(f"  {mark}  {stage:<20} {t}")
        if status == "OK":
            passed += 1
        else:
            failed += 1

    print(f"\n  Passed: {passed} | Failed: {failed}")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Models saved to: models/")

    # ── Save JSON log ──
    summary_path = f"logs/launch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "mode": "dry-run" if args.dry_run else "full",
            "results": results,
            "total_time_s": round(total_time, 1),
            "passed": passed,
            "failed": failed
        }, f, indent=2, default=str)

    print(f"  Summary: {summary_path}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
