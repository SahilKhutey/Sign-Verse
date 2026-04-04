"""
LSTM architecture for action recognition from pose sequences.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal sequence classification.
    """
    
    def __init__(self,
                 input_dim: int = 51,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_classes: int = 18,  # Default: 18 sign language actions
                 bidirectional: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        classifier_input_dim = hidden_dim * self.num_directions
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 to help with long-term dependencies
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            logits: Classification logits
            lstm_out: All hidden states from the LSTM
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward (automatic zero-initialization of hidden/cell states if not provided)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use last hidden state for classification
        if self.bidirectional:
            # hn contains (num_layers * num_directions, batch_size, hidden_dim)
            # We take the final layer's (num_layers-1) forward and backward states
            # For PyTorch, indices are 0 to num_layers*num_directions - 1
            # Forward: -2, Backward: -1
            last_output = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last_output = hn[-1]
        
        # Classification
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        
        return logits, lstm_out
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple attention mechanism for interpretability.
        """
        logits, lstm_out = self.forward(x)
        
        # Learnable attention approximation using classifier weights mapping
        # x is (batch, seq, directions*hidden)
        # weight is (classes, directions*hidden)
        # mapped is (batch, seq, classes)
        attention_weights = torch.softmax(
            torch.matmul(lstm_out, self.classifier.weight.t()), 
            dim=1
        )
        
        return attention_weights
