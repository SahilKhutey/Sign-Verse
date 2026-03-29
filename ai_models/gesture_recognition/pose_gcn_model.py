"""
Clean-room Pose-GCN style baseline for sign recognition from keypoint sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_chain_adjacency(num_nodes: int) -> torch.Tensor:
    a = torch.eye(num_nodes, dtype=torch.float32)
    for i in range(num_nodes - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    # Row-normalize for stable message passing.
    row_sum = a.sum(dim=1, keepdim=True).clamp(min=1.0)
    return a / row_sum


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(float(dropout)),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N), adj: (N, N)
        x_g = torch.einsum("bctn,nm->bctm", x, adj)
        x_g = self.gcn(x_g)
        out = self.tcn(x_g) + self.residual(x)
        return F.relu(out, inplace=True)


class PoseGCNSignModel(nn.Module):
    """
    Input:  (B, T, D) where D = num_nodes * 3
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_nodes: int = 75,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim = self.num_nodes * 3

        self.register_buffer("adjacency", _build_chain_adjacency(self.num_nodes))
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(3, 64, dropout=dropout),
                STGCNBlock(64, 128, dropout=dropout),
                STGCNBlock(128, 256, dropout=dropout),
            ]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        b, t, d = x.shape
        if d < self.in_dim:
            pad = x.new_zeros((b, t, self.in_dim - d))
            x = torch.cat([x, pad], dim=-1)
        elif d > self.in_dim:
            x = x[..., : self.in_dim]

        # (B,T,N,3) -> (B,3,T,N)
        x = x.view(b, t, self.num_nodes, 3).permute(0, 3, 1, 2).contiguous()
        for blk in self.blocks:
            x = blk(x, self.adjacency)
        # Global average over time and nodes.
        x = x.mean(dim=-1).mean(dim=-1)  # (B,C)
        return self.classifier(x)
