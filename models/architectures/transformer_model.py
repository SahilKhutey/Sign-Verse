"""
Transformer architecture for pose sequence modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class PoseTransformer(nn.Module):
    """
    Transformer model for pose sequence processing.
    Suitable for both pose refinement and action recognition.
    """
    
    def __init__(self, 
                 input_dim: int = 51,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_classes: Optional[int] = None,
                 max_seq_len: int = 300):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        
        if num_classes:
            # For classification tasks
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            # For sequence-to-sequence tasks (pose refinement)
            self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)
        x = self.dropout(x)
        
        if self.num_classes:
            # Global average pooling + classification
            x = x.mean(dim=1)  # (batch_size, hidden_dim)
            return self.classifier(x)
        else:
            # Sequence output for refinement
            return self.output_proj(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention maps for visualization."""
        # Store attention weights
        attention_maps = []
        
        def hook_fn(module, input, output):
            # The output of self_attn is (attn_output, attn_weights)
            # In PyTorch's TransformerEncoderLayer, we need to handle this carefully
            pass # Placeholder for hooks if needed specifically for the attention layer
            
        # Register hooks... 
        # (Omitted for brevity in this architectural file, typically used for interpretability)
        return attention_maps
