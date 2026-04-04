import torch
import torch.nn as nn

class AudioToGestureTransformer(nn.Module):
    """
    Transformer-based model for generating gesture trajectories from audio features.
    Architecture:
        Audio Encoder -> Emotion Context Injection -> Transformer Decoder -> Motion Head
    """
    def __init__(self, audio_dim=128, emotion_dim=32, motion_dim=225, nhead=8, num_layers=4):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, 256)
        self.emotion_proj = nn.Linear(emotion_dim, 256)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.motion_head = nn.Linear(256, motion_dim)

    def forward(self, audio_features, emotion_embedding):
        # audio_features: (batch, seq_len, 128)
        # emotion_embedding: (batch, 32)
        
        # Project to d_model=256
        audio_emb = self.audio_proj(audio_features) # (batch, seq_len, 256)
        context = self.emotion_proj(emotion_embedding).unsqueeze(0) # (1, batch, 256)
        
        # Transpose audio_emb to (seq_len, batch, 256) for Transformer
        audio_emb = audio_emb.transpose(0, 1)
        
        # Use emotion as the 'memory' for the transformer decoder
        # output: (seq_len, batch, 256)
        output = self.transformer_decoder(audio_emb, context)
        
        # Motion head: (seq_len, batch, 225)
        motion = self.motion_head(output)
        
        # Transpose back to (batch, seq_len, 225)
        return motion.transpose(0, 1)
