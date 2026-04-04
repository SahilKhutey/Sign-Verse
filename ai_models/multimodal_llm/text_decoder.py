"""
Multimodal Sign LLM — Text Decoder

Cross-modal text decoder that attends to gesture encoder output.

Supports:
    - Sign → Text (translation)
    - Text → Sign token guidance
    - Sign → Speech (via TTS integration)
"""

import torch
import torch.nn as nn
import math


class TextDecoder(nn.Module):
    """
    Autoregressive text decoder with cross-attention over gesture memory.
    """

    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_layers=8, dim_feedforward=2048, dropout=0.1,
                 max_seq_len=512):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
                activation="gelu"
            ),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None):
        t = tgt.size(1)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = x + self.pe[:, :t]
        x = self.dropout(x)

        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                t, device=tgt.device
            )

        x = self.decoder(x, memory, tgt_mask=tgt_mask)
        x = self.norm(x)
        return self.output_proj(x)

    def greedy_decode(self, memory, max_len=100, sos=1, eos=2):
        """Greedy autoregressive decoding."""
        device = memory.device
        generated = [sos]

        for _ in range(max_len):
            tgt = torch.tensor([generated], device=device)
            logits = self.forward(tgt, memory)
            next_tok = logits[:, -1, :].argmax().item()
            if next_tok == eos:
                break
            generated.append(next_tok)

        return generated[1:]
