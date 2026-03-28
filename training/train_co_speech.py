import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_models.co_speech_generation.audio_to_gesture import AudioToGestureTransformer
from ai_models.co_speech_generation.emotion_encoder import EmotionEncoder
from training.utils.trainer_base import BaseTrainer

class CoSpeechDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=60):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Synthetic audio features (seq_len, 128)
        audio = torch.randn(self.seq_len, 128)
        # Synthetic motion (seq_len, 225)
        motion = torch.randn(self.seq_len, 225)
        # Random emotion ID (0-7)
        emotion_id = torch.randint(0, 8, (1,)).item()
        
        return {
            "audio": audio,
            "motion": motion,
            "emotion_id": emotion_id
        }

class CoSpeechTrainer(BaseTrainer):
    def __init__(self, model, emotion_encoder, config):
        super().__init__(config)
        self.model = model.to(self.device)
        self.emotion_encoder = emotion_encoder.to(self.device)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.emotion_encoder.parameters()), 
            lr=config.get("lr", 1e-4)
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            audio = batch["audio"].transpose(0, 1).to(self.device) # (seq, batch, dim)
            motion_gt = batch["motion"].transpose(0, 1).to(self.device)
            emotion_ids = batch["emotion_id"].to(self.device)
            
            emotion_emb = self.emotion_encoder(emotion_ids)
            
            self.optimizer.zero_grad()
            motion_pred = self.model(audio, emotion_emb)
            
            loss = self.criterion(motion_pred, motion_gt)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(dataloader)

def train(config):
    model = AudioToGestureTransformer()
    emotion_encoder = EmotionEncoder()
    
    dataset = CoSpeechDataset(num_samples=config.get("num_samples", 100))
    dataloader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=True)
    
    trainer = CoSpeechTrainer(model, emotion_encoder, config)
    
    epochs = config.get("epochs", 2)
    print(f"Starting Co-speech training for {epochs} epochs...")
    
    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/co_speech_model_best.pt")
    print("Co-speech model saved.")
    return True

if __name__ == "__main__":
    train({"epochs": 2, "batch_size": 16, "num_samples": 48})
