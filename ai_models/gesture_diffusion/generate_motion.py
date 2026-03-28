"""
Generate Motion — CLI and API for gesture motion generation.

Usage:
    python generate_motion.py --tokens 12 34 55 --frames 60
"""

import torch
import numpy as np
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ai_models.gesture_diffusion.diffusion_model import GestureDiffusionPipeline


def generate_from_tokens(token_ids, seq_len=30, model_path=None):
    """Generate 3D motion from a list of gesture token IDs."""
    pipeline = GestureDiffusionPipeline(model_path=model_path)
    tokens = torch.tensor([token_ids], dtype=torch.long)
    motion = pipeline.generate(gesture_tokens=tokens, seq_len=seq_len)
    return motion  # (1, seq_len, 150)


def save_motion(motion, output_path):
    """Save motion as numpy array."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, motion)
    print(f"Saved motion: {output_path} — shape: {motion.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gesture motion")
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[12, 34, 55], help="Gesture token IDs")
    parser.add_argument("--frames", type=int, default=30,
                        help="Output sequence length (frames)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained diffusion model")
    parser.add_argument("--output", type=str,
                        default="exports/generated_motion.npy")
    args = parser.parse_args()

    print(f"Generating motion for tokens: {args.tokens}, frames: {args.frames}")
    motion = generate_from_tokens(args.tokens, args.frames, args.model)
    save_motion(motion[0], args.output)
