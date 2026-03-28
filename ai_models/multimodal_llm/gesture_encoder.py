"""
Multimodal Sign LLM — Gesture Encoder

Encodes sign video/keypoint sequences into embeddings
for the fusion transformer.

Input modalities:
    - Gesture video (keypoint sequences)
    - Sign tokens from tokenizer

Output: (batch, seq_len, d_model) contextual gesture embeddings
"""

import torch
import torch.nn as nn
import math


class GestureEncoder(nn.Module):
    """
    Video Transformer encoder for gesture understanding.

    Processes sequences of pose keypoints and produces
    rich contextual embeddings for the fusion transformer.
    """

    def __init__(self, input_dim=225, d_model=512, nhead=8,
                 num_layers=8, dim_feedforward=2048, dropout=0.1,
                 max_seq_len=1024):
        super().__init__()

        self.d_model = d_model

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

        # Deep transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
                activation="gelu"
            ),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim) gesture keypoint sequence
            padding_mask: (batch, seq_len) boolean mask

        Returns:
            (batch, seq_len, d_model)
        """
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)
