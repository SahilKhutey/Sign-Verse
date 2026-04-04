"""
Sign Transformer Encoder

Encodes a sequence of keypoints (or gesture tokens) into
a rich contextual representation.

Input:  (batch, seq_len, feature_dim) — raw keypoints
Output: (batch, seq_len, d_model) — contextualized embeddings
"""

import torch
import torch.nn as nn
import math


class SignEncoder(nn.Module):

    def __init__(self, feature_dim=126, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_seq_len=512):
        super().__init__()

        self.d_model = d_model

        # Project raw features to model dim
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            ),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)
