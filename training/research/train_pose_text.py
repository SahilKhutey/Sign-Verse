import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from models.v3_foundation import MultimodalFoundationModel, GestureGPT
import os
import argparse

def create_wds_loader(urls, batch_size=32):
    """
    Creates a WebDataset loader for Pose + Text training.
    Assumes .tar shards contain .npy (pose) and .json (metadata/text).
    """
    dataset = (
        wds.WebDataset(urls)
        .shuffle(1000)
        .decode("numpy")
        .to_tuple("npy", "json")
        .map(lambda x: (torch.tensor(x[0], dtype=torch.float32), x[1]))
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4,
        pin_memory=True
    )
    return loader

class PoseTextTrainer:
    """
    Trainer for the Pose + Text Multimodal Core.
    Fuses pose sequences with natural language translations.
    """
    def __init__(self, model_dim=512, vocab_size=30000):
        self.encoder = MultimodalFoundationModel(dim=model_dim, num_layers=12)
        self.decoder = GestureGPT(dim=model_dim, vocab=vocab_size, num_layers=12)
        
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=1e-4
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def train_step(self, pose, text_tokens):
        """
        Cross-modal translation step: Pose -> Encoder -> Decoder -> Text Logits
        """
        self.encoder.train()
        self.decoder.train()
        
        # 1. Encode Pose
        memory = self.encoder(pose=pose) # (batch, seq, dim)
        
        # 2. Decode Text (Teacher Forcing)
        # Shift target for autoregressive training
        tgt_in = text_tokens[:, :-1]
        tgt_out = text_tokens[:, 1:]
        
        # Simple embedding for blueprint (in production, use TextEncoder from Foundation)
        # Assuming decoder handles embedding internally for indices
        logits = self.decoder(memory, tgt_in)
        
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def main():
    parser = argparse.ArgumentParser(description="SignVerse Pose-to-Text Core Training")
    parser.add_argument("--shards", type=str, required=True, help="WebDataset shards path pattern (e.g. shards/*.tar)")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    trainer = PoseTextTrainer()
    loader = create_wds_loader(args.shards)
    
    print(f"Starting Pose-to-Text Foundation Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        for batch_idx, (pose, metadata) in enumerate(loader):
            # In production, metadata['text'] would be tokenized here
            # Mocking tokenized text (batch, seq_len)
            text_tokens = torch.randint(1, 30000, (pose.size(0), 20)).cuda()
            pose = pose.cuda()
            
            loss = trainer.train_step(pose, text_tokens)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
