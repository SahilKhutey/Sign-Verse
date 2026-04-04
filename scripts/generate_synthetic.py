import os
import numpy as np
import argparse

def generate_synthetic_data(samples=100000, output_dir="training-data"):
    # Subdirectories
    synthetic_dir = os.path.join(output_dir, "synthetic")
    keypoints_dir = os.path.join(output_dir, "keypoints")
    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)
    
    print(f"Generating {samples} synthetic gesture samples...")
    
    import csv
    import json
    
    num_classes = 100
    rows = []
    label_map = {f"SIGN_{i:04d}": i for i in range(num_classes)}
    
    for i in range(samples):
        # Generate random base motion
        seq_len = np.random.randint(20, 60)
        base_motion = (np.random.randn(seq_len, 225) * 0.05).astype(np.float32)
        
        label_id = i % num_classes
        label_name = f"SIGN_{label_id:04d}"
        fname = f"sample_{i:06d}.npy"
        
        # Save to both locations to be safe for all models
        np.save(os.path.join(synthetic_dir, fname), base_motion)
        np.save(os.path.join(keypoints_dir, fname), base_motion)
        
        rows.append({
            "filename": fname,
            "label_id": label_id,
            "label_name": label_name,
            "dataset": "SYNTHETIC",
            "split": "train" if (i % 10) < 9 else "val",
            "text": f"This is the sign for {label_name}"
        })
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{samples}...")

    # Save manifest
    with open(os.path.join(output_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label_id", "label_name", "dataset", "split", "text"])
        writer.writeheader()
        writer.writerows(rows)
        
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Success! {samples} samples and manifest saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="100k")
    parser.add_argument("--variations", type=int, default=10)
    
    args = parser.parse_args()
    
    size_num = 100000 if args.size == "100k" else int(args.size)
    
    generate_synthetic_data(samples=size_num)
