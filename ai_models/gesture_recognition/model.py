"""
Gesture Recognition Model — CNN + BiLSTM + Transformer

Architecture:
    Hand + body keypoints (MediaPipe)
    → BiLSTM  (temporal sequence)
    → Transformer Encoder  (attention over time)
    → Linear  (sign token prediction)

Input:  (batch, seq_len, 126) keypoint sequences
Output: (batch, num_classes) sign token logits
"""

import torch
import torch.nn as nn


class GestureModel(nn.Module):

    def __init__(self, input_size=126, hidden_size=256,
                 num_classes=500, num_lstm_layers=2, nhead=8,
                 num_transformer_layers=4):
        super().__init__()

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Project BiLSTM output (hidden*2 → hidden)
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Transformer encoder for global attention
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )

        self.norm = nn.LayerNorm(hidden_size)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size) keypoint sequence

        Returns:
            logits: (batch, num_classes)
        """
        # BiLSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)

        # Transformer encoder
        x = self.transformer(lstm_out)
        x = self.norm(x)

        # Use last timestep for classification
        x = self.fc(x[:, -1])

        return x
