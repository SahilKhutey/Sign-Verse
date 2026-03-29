import torch
import torch.nn as nn
import math
from models.sign_foundation_transformer import PositionalEncoding

class ManualTransformerLayer(nn.Module):
    """Decomposed Transformer Layer for ONNX stability in Python/C#."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEdgeWrapper(nn.Module):
    """
    Adaptive Transformer Wrapper for Edge Optimization.
    Decomposes large TransformerEncoder blocks for ONNX-Lite stability.
    """
    def __init__(self, vocab_size=512, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=64)
        self.layers = nn.ModuleList([
            ManualTransformerLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens, mask=None):
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.output_layer(x)

class MLPEdgeWrapper(nn.Module):
    """
    Optimized MLP Wrapper for non-transformer models (e.g., Diffusion, GestureMLP).
    Integrates LayerNorm and ReLU for INT8 acceleration.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_edge_architecture(model_type, **kwargs):
    """Factory to retrieve the appropriate optimized architecture."""
    if model_type == "transformer":
        return TransformerEdgeWrapper(**kwargs)
    elif model_type == "mlp":
        return MLPEdgeWrapper(**kwargs)
    else:
        raise ValueError(f"Unsupported edge architecture type: {model_type}")
