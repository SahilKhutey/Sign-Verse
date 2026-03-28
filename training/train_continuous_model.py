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
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.continuous_sign_model import ContinuousSignModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ContinuousSignModel(vocab_size=1500).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

ctc_loss = nn.CTCLoss()


def train(dataloader, epochs=50):
    """
    Train the continuous sign language model.

    Args:
        dataloader: PyTorch DataLoader yielding (video_batch, labels)
        epochs: Number of training epochs
    """

    model.train()

    for epoch in range(epochs):

        epoch_loss = 0.0
        batch_count = 0

        for video_batch, labels in dataloader:

            video_batch = video_batch.to(device)

            outputs = model(video_batch)

            # CTC loss requires (T, N, C) format
            log_probs = outputs.permute(1, 0, 2).log_softmax(dim=2)

            input_lengths = torch.full(
                (video_batch.size(0),),
                outputs.size(1),
                dtype=torch.long
            )

            target_lengths = torch.tensor(
                [len(l) for l in labels],
                dtype=torch.long
            )

            loss = ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)

        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "models/continuous_sign_model.pt")
    print("Model saved.")


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Dataloader not configured — provide a dataset to begin training.")
