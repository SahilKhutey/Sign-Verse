"""
Sign Transformer Decoder

Autoregressive text decoder with cross-attention to sign encoder.

Input:  encoder memory + text token ids
Output: text token logits
"""

import torch
import torch.nn as nn
import math


class SignDecoder(nn.Module):

    def __init__(self, text_vocab_size, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_seq_len=256):
        super().__init__()

        self.d_model = d_model

        # Keep PAD=0 stable for loss masking and key-padding masks.
        self.embedding = nn.Embedding(text_vocab_size, d_model, padding_idx=0)

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            ),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, text_vocab_size)

    def forward(self, tgt_tokens, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        t = tgt_tokens.size(1)
        x = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = x + self.pe[:, :t]
        x = self.dropout(x)

        if tgt_mask is None:
            # Use a bool mask so it matches the bool key_padding_mask type.
            tgt_mask = torch.triu(
                torch.ones((t, t), device=tgt_tokens.device, dtype=torch.bool),
                diagonal=1,
            )

        x = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        x = self.norm(x)
        return self.output_proj(x)
