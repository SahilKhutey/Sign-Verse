"""
Training Pipeline for Continuous Sign Language Model

Uses CTC loss for sequence-to-sequence alignment without
requiring frame-level labels.

Training requires:
    - GPU: RTX 3090 / A100
    - RAM: 32-64 GB
    - Dataset: 100-500 GB
    - Time: 3-10 days
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
Training Pipeline for Continuous Sign Language Model

Uses CTC loss for sequence-to-sequence alignment without
requiring frame-level labels.

Training requires:
    - GPU: RTX 3090 / A100
    - RAM: 32-64 GB
    - Dataset: 100-500 GB
    - Time: 3-10 days
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.continuous_sign_model import ContinuousSignModel
from training.utils.ewc_utils import EWC, ewc_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from training.data_pipeline.feedback_dataset import FeedbackDataset, collate_feedback_batch
from training.data_pipeline.multimodal_dataset import MultimodalDataset
from torch.utils.data import DataLoader

def get_model(vocab_size=1500, checkpoint=None):
    model = ContinuousSignModel(vocab_size=vocab_size).to(device)
    if checkpoint and os.path.exists(checkpoint):
        try:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            print(f"Loaded checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
    return model

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

def train_session(model, dataloader, epochs=5, mode="train", ewc=None, importance=1000):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if mode == "train" else 1e-5)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for video_batch, labels in dataloader:
            video_batch = video_batch.to(device)
            outputs = model(video_batch) # (N, T, C)

            # CTC loss requires (T, N, C) format
            log_probs = outputs.permute(1, 0, 2).log_softmax(dim=2)

            input_lengths = torch.full(
                (video_batch.size(0),),
                outputs.size(1),
                dtype=torch.long
            ).to(device)

            # Convert text labels to token indices (simplified for this step)
            # In a full impl, use a proper tokenizer. Here we assume label is a list of ids or we map it.
            # Assuming labels are already tokenized lists for feedback
            target_lengths = torch.tensor(
                [len(l) for l in labels],
                dtype=torch.long
            ).to(device)
            
            # Flatten targets for CTC
            targets = torch.cat([torch.tensor(l) for l in labels]).to(device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            if mode == "fine-tune" and ewc:
                ewc_penalty = ewc.penalty(model)
                loss = loss + importance * ewc_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        print(f"Epoch {epoch + 1}/{epochs} — Loss: {epoch_loss / max(batch_count, 1):.4f}")

    save_suffix = "fine_tuned" if mode == "fine-tune" else "standard"
    save_path = f"models/continuous_sign_model_{save_suffix}.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse Continuous Training Engine")
    parser.add_argument("--mode", type=str, choices=["train", "fine-tune"], default="train", help="Training mode")
    parser.add_argument("--importance", type=float, default=1000.0, help="EWC importance factor")
    parser.add_argument("--checkpoint", type=str, default="models/continuous_sign_model.pt", help="Path to base model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--feedback_dir", type=str, default="datasets/user_feedback/validated", help="Feedback data")
    args = parser.parse_args()

    print(f"Starting SignVerse Continuous Training - Mode: {args.mode}")
    print(f"Device: {device}")

    model = get_model(checkpoint=args.checkpoint)

    if args.mode == "fine-tune":
        print(f"Loading feedback from: {args.feedback_dir}")
        dataset = FeedbackDataset(data_dir=args.feedback_dir)
        if len(dataset) == 0:
            print("No feedback samples found. Exiting.")
            sys.exit(0)
            
        dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True, collate_fn=collate_feedback_batch)
        
        print("Calculating Fisher Information Matrix for EWC...")
        # Load a small subset of the base dataset to calculate importance
        base_dataset = MultimodalDataset(split="train") # Ensure this exists or use a dummy
        base_loader = DataLoader(base_dataset, batch_size=8, shuffle=True)
        
        ewc = EWC(model, base_loader, device=device)
        
        train_session(model, dataloader, epochs=args.epochs, mode="fine-tune", ewc=ewc, importance=args.importance)
    else:
        # Standard training logic
        base_dataset = MultimodalDataset(split="train")
        dataloader = DataLoader(base_dataset, batch_size=16, shuffle=True)
        train_session(model, dataloader, epochs=args.epochs, mode="train")
