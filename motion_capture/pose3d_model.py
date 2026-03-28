"""
3D Pose Reconstruction — 2D Keypoints → 3D Skeleton

Lifts 2D/normalized pose keypoints to full 3D skeleton coordinates
that can drive avatar rigs in Unity or Unreal Engine.

Pipeline:
    2D keypoints → Temporal MLP → 3D skeleton coordinates

Input:  ~225 values (body 99 + hands 126)
Output: 150 values (50 joints × xyz)

Export formats: BVH, FBX (for Unity/Blender)
"""

import torch
import torch.nn as nn
import numpy as np


class Pose3DReconstructor(nn.Module):
    """
    Lifts 2D pose keypoints to 3D skeleton coordinates.
    Uses a residual MLP for stable training.
    """

    def __init__(self, input_dim=225, hidden_dim=512, output_dim=150):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) 2D keypoints

        Returns:
            (batch, output_dim) 3D joint positions
        """
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class ResidualBlock(nn.Module):
    """Residual MLP block with dropout and layer norm."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)


class TemporalPose3D(nn.Module):
    """
    Temporal 3D pose reconstruction using sequence of frames.
    Uses a Transformer to capture temporal motion patterns.

    Input:  (batch, seq_len, 225) — sequence of 2D poses
    Output: (batch, seq_len, 150) — sequence of 3D skeletons
    """

    def __init__(self, input_dim=225, output_dim=150, d_model=256,
                 nhead=4, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, output_dim) 3D skeleton sequence
        """
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output_proj(x)


def skeleton_to_bvh(skeleton_sequence, output_path, fps=30):
    """
    Convert 3D skeleton sequence to BVH format placeholder.
    For full BVH export, integrate with a BVH library.

    Args:
        skeleton_sequence: numpy array (frames, 150)
        output_path: path to save BVH file
        fps: frames per second
    """
    frames = len(skeleton_sequence)

    with open(output_path, "w") as f:
        f.write("HIERARCHY\n")
        f.write("ROOT Hips\n")
        f.write("{\n")
        f.write("  OFFSET 0.0 0.0 0.0\n")
        f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        f.write("  End Site\n")
        f.write("  {\n")
        f.write("    OFFSET 0.0 1.0 0.0\n")
        f.write("  }\n")
        f.write("}\n")
        f.write("MOTION\n")
        f.write(f"Frames: {frames}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")

        for frame in skeleton_sequence:
            values = " ".join(f"{v:.4f}" for v in frame[:6])
            f.write(f"{values}\n")

    print(f"BVH saved to {output_path} ({frames} frames @ {fps}fps)")
