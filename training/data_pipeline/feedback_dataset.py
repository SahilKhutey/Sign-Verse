import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple

class FeedbackDataset(Dataset):
    """
    Dataset for loading user feedback corrections saved as JSON files.
    Expected JSON structure:
    {
        "original_text": "...",
        "corrected_text": "...",
        "language": "...",
        "sequence_data": [[x1, y1, z1, ...], [x2, y2, z2, ...]] # Landmarks
    }
    """
    def __init__(
        self, 
        data_dir: str = "datasets/user_feedback/validated",
        seq_len: int = 60,
        feature_dim: int = 225
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.samples = []
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith(".json"):
                    self.samples.append(os.path.join(data_dir, filename))
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        file_path = self.samples[idx]
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Extract sequence data
        sequence = np.array(data["sequence_data"], dtype=np.float32)
        
        # Padding/Truncating
        T, D = sequence.shape
        if T < self.seq_len:
            sequence = np.pad(sequence, ((0, self.seq_len - T), (0, 0)), mode="constant")
        else:
            sequence = sequence[:self.seq_len]
            
        if D < self.feature_dim:
            sequence = np.pad(sequence, ((0, 0), (0, self.feature_dim - D)), mode="constant")
        else:
            sequence = sequence[:, :self.feature_dim]
            
        # Normalization (Z-score similar to MultimodalDataset)
        mean = sequence.mean(axis=0, keepdims=True)
        std = sequence.std(axis=0, keepdims=True) + 1e-6
        sequence = (sequence - mean) / std
        
        return torch.tensor(sequence), data["corrected_text"]

def collate_feedback_batch(batch):
    """
    Custom collate function for FeedbackDataset.
    Returns:
        sequences: (B, T, D) tensor
        labels: List of corrected_text strings
    """
    sequences, labels = zip(*batch)
    return torch.stack(sequences), list(labels)
