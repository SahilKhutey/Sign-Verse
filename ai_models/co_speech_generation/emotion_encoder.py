import torch
import torch.nn as nn

class EmotionEncoder(nn.Module):
    """
    Encodes categorical emotion labels into dense embeddings for gesture conditioning.
    Supports standard emotions: Neutral, Happy, Sad, Angry, Surprised.
    """
    def __init__(self, num_emotions=8, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, embedding_dim)
        
    def forward(self, emotion_ids):
        # emotion_ids: (batch,)
        return self.embedding(emotion_ids)
