"""
Utility classes for AI model training and lifecycle management.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from loguru import logger
import time
from typing import Dict, Any, List

class ModelTrainer:
    """Manages the training loop, validation, and checkpointing for PyTorch models."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model.to(self.device)
        self.criterion = nn.MSELoss() # Default for pose refinement
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on {self.device}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for i, batch in enumerate(self.train_loader):
            features = batch['features'].to(self.device)
            # Simple self-supervised task for refinement: predict original from original
            self.optimizer.zero_grad()
            output = self.model(features.unsqueeze(1)) # Add sequence dim
            loss = self.criterion(output.squeeze(1), features)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                output = self.model(features.unsqueeze(1))
                loss = self.criterion(output.squeeze(1), features)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self):
        """Standard training loop."""
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch}/{self.config.max_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_pose_model.pt")
                
        return history

    def save_checkpoint(self, filename):
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': getattr(self.model, 'num_layers', 4)
            }
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
