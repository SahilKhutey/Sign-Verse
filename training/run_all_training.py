"""
Master Training Orchestrator — PRODUCTION VERSION
Runs the full SignVerse training pipeline end-to-end.

Pipeline stages:
    1. Dataset download guidance
    2. Preprocessing (frames → pose → keypoints)
    3. Data validation
    4. Gesture tokenization (for foundation model)
    5. Train Gesture Recognition Model
    6. Train Foundation Transformer
    7. Train Sign Transformer (Sign → Text)
    8. Train Gesture Diffusion Model
    9. Train Multimodal LLM
    10. Save model registry

Usage:
    python training/run_all_training.py --gpus 4 --batch_size 32 --monitor wandb
"""

import os
import sys
import argparse
import time
import json
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "configs", "training_config.yaml")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def banner(title, width=62):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")

def stage_header(n, title):
    print(f"\n{'-'*62}")
    print(f"  Stage {n}: {title}")
    print(f"{'-'*62}")

# ──────────────────────────────────────────────────────────────────
# Stage 1: Dataset info / download guidance
# ──────────────────────────────────────────────────────────────────
def stage_download(config):
    stage_header(1, "Dataset Download Check")
    from training.data_pipeline.dataset_registry import DATASETS, print_summary
    print_summary()

    raw_dir = "datasets/raw_videos"
    found = []
    missing = []

    for name in DATASETS:
        ds_dir = os.path.join(raw_dir, name)
        if os.path.exists(ds_dir) and os.listdir(ds_dir):
            found.append(name)
        else:
            missing.append(name)

    print(f"  Found datasets:   {found}")
    print(f"  Missing datasets: {missing}")

    if missing:
        print(f"\n  -- How to download missing datasets --")
        print(f"  Run: python training/data_pipeline/download_datasets.py --all\n")

    return found

# ──────────────────────────────────────────────────────────────────
# Stage 2: Preprocessing
# ──────────────────────────────────────────────────────────────────
def stage_preprocess(config):
    stage_header(2, "Unified Preprocessing")
    from training.data_pipeline.unified_preprocessor import UnifiedPreprocessor

    pre_cfg = config.get("preprocessing", {})
    preprocessor = UnifiedPreprocessor(pre_cfg)
    
    # Filter datasets if specified in config
    target_datasets = pre_cfg.get("datasets", None)
    total = preprocessor.run_all(datasets=target_datasets)

    if total == 0:
        print("  No videos found. Generating synthetic training data...")
        _generate_synthetic_data(config)

    return total

def _generate_synthetic_data(config):
    """Generate synthetic keypoint data if no real data is found."""
    import numpy as np
    n_classes = int(config.get("gesture_model", {}).get("num_classes", 50) or 50)
    samples_per_class = config.get("gesture_model", {}).get("samples_per_class", 10)
    
    # Dramatically reduce for dry-run
    if n_classes > 50:
        print(f"  [DRY RUN] Reducing classes from {n_classes} to 50 for speed")
        n_classes = 50
        samples_per_class = 2
    seq_len = 60
    feature_dim = 225

    os.makedirs("training-data/keypoints", exist_ok=True)
    rows = []
    label_map = {}

    for label_id in range(n_classes):
        sign = f"SIGN_{label_id:04d}"
        label_map[sign] = label_id
        for s in range(samples_per_class):
            seq = (np.random.randn(seq_len, feature_dim) * 0.1).astype(np.float32)
            fname = f"synthetic_{sign}_{s:03d}.npy"
            np.save(f"training-data/keypoints/{fname}", seq)
            rows.append({
                "filename": fname, "label_id": label_id,
                "label_name": sign, "dataset": "SYNTHETIC",
                "split": "train" if s < 8 else "val",
                "text": f"This is {sign}"
            })

    import csv
    with open("training-data/labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label_id", "label_name", "dataset", "split", "text"])
        writer.writeheader()
        writer.writerows(rows)

    with open("training-data/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

# ──────────────────────────────────────────────────────────────────
# Stage 3: Validation
# ──────────────────────────────────────────────────────────────────
def stage_validate(config):
    stage_header(3, "Data Validation")
    from training.data_pipeline.data_validation import validate_dataset
    result = validate_dataset()
    return result.get("valid", False)

def stage_text_gloss_dataset(config):
    stage_header(4, "Text-Gloss Dataset Build")
    from training.data_pipeline.build_text_gloss_dataset import main as build_text_gloss
    build_text_gloss()
    return True

# ──────────────────────────────────────────────────────────────────
# Stage 4: Tokenize
# ──────────────────────────────────────────────────────────────────
def stage_tokenize(config):
    stage_header(5, "Gesture Tokenization")
    from training.data_pipeline.gesture_tokenizer import GestureTokenizer
    import numpy as np

    kp_dirs = ["training-data/keypoints", "training-data/synthetic"]
    kp_dir = next((d for d in kp_dirs if os.path.exists(d) and os.listdir(d)), None)
    
    token_dir = "datasets/gesture_tokens"
    os.makedirs(token_dir, exist_ok=True)

    if not kp_dir:
        print("  No keypoints found in standard directories.")
        return

    print(f"  Using keypoints from: {kp_dir}")
    n_clusters = config.get("foundation_model", {}).get("vocab_size", 512)
    is_dry_run = config.get("gesture_model", {}).get("epochs", 100) <= 2
    n_init = 1 if is_dry_run else 10
    max_iter = 50 if is_dry_run else 300
    
    tokenizer = GestureTokenizer(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    
    # Fit on a substantial sample (up to 2k for full-scale, or 500 for dry-run)
    all_files = [f for f in os.listdir(kp_dir) if f.endswith('.npy')]
    fit_sample_size = 500 if is_dry_run else 2000
    fit_files = all_files[:fit_sample_size]
    
    print(f"  Fitting tokenizer on {len(fit_files)} samples...")
    all_f = []
    for f in fit_files:
        try:
            all_f.append(np.load(os.path.join(kp_dir, f)))
        except: continue
        
    tokenizer.fit(np.concatenate(all_f, axis=0))
    tokenizer.save("models/gesture_tokenizer.pkl")

    # Tokenize ALL files
    print(f"  Tokenizing {len(all_files)} samples...")
    for i, f in enumerate(all_files):
        try:
            tokens = tokenizer.tokenize_sequence(np.load(os.path.join(kp_dir, f)))
            np.save(os.path.join(token_dir, f.replace(".npy", "_tokens.npy")), tokens)
            if (i + 1) % 5000 == 0:
                print(f"    Processed {i + 1}/{len(all_files)}...")
        except: continue

# ──────────────────────────────────────────────────────────────────
# Stages 5-9: Training
# ──────────────────────────────────────────────────────────────────
def stage_train_gesture(config):
    stage_header(6, "Training Gesture Model")
    from ai_models.gesture_recognition.train import train
    cfg = config.get("gesture_model", {})
    cfg.setdefault("data_dir", "training-data")
    return train(cfg)

def stage_train_foundation(config):
    stage_header(7, "Training Foundation Model")
    from training.train_foundation_model import train
    return train(config.get("foundation_model", {}))

def stage_train_transformer(config):
    stage_header(8, "Training Sign Transformer")
    from ai_models.sign_transformer.train_transformer import train
    cfg = config.get("sign_transformer", {})
    return train(cfg)

def stage_train_diffusion(config):
    stage_header(9, "Training Diffusion Model")
    from training.train_diffusion import train
    return train(config.get("diffusion_model", {}))

def stage_train_llm(config):
    stage_header(10, "Training Multimodal LLM")
    from ai_models.multimodal_llm.training_pipeline import train
    cfg = config.get("multimodal_llm", {})
    return train(cfg)

def stage_train_co_speech(config):
    stage_header(11, "Training Co-speech Generation")
    from training.train_co_speech import train
    return train(config.get("co_speech", {}))

# ──────────────────────────────────────────────────────────────────
# Stage 10: Registry
# ──────────────────────────────────────────────────────────────────
def stage_save_registry(config):
    stage_header(12, "Saving Registry")
    models_dir = "models"
    registry = {}
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pt"):
                registry[f] = {"path": os.path.join(models_dir, f), "size_mb": round(os.path.getsize(os.path.join(models_dir, f))/1e6, 2)}
    with open(os.path.join(models_dir, "model_registry.json"), "w") as f:
        json.dump(registry, f, indent=2)

# ──────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────
STAGES = {
    "download":    stage_download,
    "preprocess":  stage_preprocess,
    "validate":    stage_validate,
    "text_gloss":  stage_text_gloss_dataset,
    "tokenize":    stage_tokenize,
    "gesture":     stage_train_gesture,
    "foundation":  stage_train_foundation,
    "transformer": stage_train_transformer,
    "diffusion":   stage_train_diffusion,
    "llm":         stage_train_llm,
    "co_speech":   stage_train_co_speech,
    "registry":    stage_save_registry,
}

STAGE_ORDER = ["download", "preprocess", "validate", "text_gloss", "tokenize", "gesture", "foundation", "transformer", "diffusion", "llm", "co_speech", "registry"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGES.keys()))
    parser.add_argument("--from-stage", choices=list(STAGES.keys()))
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--skip-download", action="store_true")
    # Production overrides
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--monitor")
    args = parser.parse_args()

    config = load_config()
    
    # Apply overrides
    overrides = {}
    if args.gpus: overrides["gpus"] = args.gpus
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.monitor: overrides["monitor"] = args.monitor

    for k in STAGES.keys():
        section = f"{k}_model" if k in ["gesture", "foundation"] else k
        if section in config:
            config[section].update(overrides)
        elif k in ["transformer", "diffusion", "llm"]:
            section = f"{k}_model" if "model" not in k else k
            # Check for alternative naming
            actual_section = next((s for s in config if k in s), None)
            if actual_section: config[actual_section].update(overrides)

    banner("SignVerse AI — Master Training Pipeline")
    t_start = time.time()
    results = {}

    start_idx = STAGE_ORDER.index(args.from_stage) if args.from_stage else 0
    if args.skip_download: start_idx = max(start_idx, STAGE_ORDER.index("preprocess"))
    
    stages_to_run = [args.stage] if args.stage else STAGE_ORDER[start_idx:]

    for name in stages_to_run:
        try:
            t0 = time.time()
            res = STAGES[name](config)
            results[name] = {"status": "OK", "time_s": round(time.time() - t0, 1)}
            print(f"  [OK] {name} completed")
        except Exception as e:
            results[name] = {"status": "ERROR", "error": str(e)}
            print(f"  [FAIL] {name} failed: {e}")
            import traceback; traceback.print_exc()

    banner("Training Summary")
    for s, r in results.items():
        print(f"  {'[OK]' if r['status'] == 'OK' else '[FAIL]'}  {s:<20} ({r.get('time_s', '??')}s)")

    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} min\n")

if __name__ == "__main__":
    main()
