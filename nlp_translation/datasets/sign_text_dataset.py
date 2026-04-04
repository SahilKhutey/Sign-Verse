"""
Sign-Text Dataset Loader

PyTorch Dataset for sign language text pairs.
Supports loading from CSV files with columns: text, gloss
"""

import torch
from torch.utils.data import Dataset
import csv


class SignTextDataset(Dataset):
    """
    Dataset for sign language translation.

    Expected CSV format:
        text,gloss
        "I am going to school","SCHOOL I GO"
    """

    def __init__(self, csv_path, tokenizer, max_len=50, augment: bool = False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.pairs = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append((row["text"], row["gloss"]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text, gloss = self.pairs[idx]

        if self.augment:
            text, gloss = self._augment_pair(text, gloss)

        src_tokens = self.tokenizer.encode(text)
        tgt_tokens = self.tokenizer.encode(gloss)

        # Pad or truncate
        src_tokens = self._pad(src_tokens)
        tgt_tokens = self._pad(tgt_tokens)

        return (
            torch.tensor(src_tokens, dtype=torch.long),
            torch.tensor(tgt_tokens, dtype=torch.long)
        )

    def _pad(self, tokens):
        """Pad or truncate token list to max_len."""
        if len(tokens) >= self.max_len:
            return tokens[:self.max_len]
        return tokens + [0] * (self.max_len - len(tokens))

    def _augment_pair(self, text: str, gloss: str):
        """
        Simple augmentation:
          - token dropout on source or target
          - minor word order shuffle on gloss
        """
        import random

        def dropout(s: str, p: float):
            toks = s.split()
            if len(toks) <= 3:
                return s
            keep = [t for t in toks if random.random() > p]
            return " ".join(keep) if keep else s

        def shuffle(s: str):
            toks = s.split()
            if len(toks) <= 3:
                return s
            mid = toks[1:-1]
            random.shuffle(mid)
            return " ".join([toks[0]] + mid + [toks[-1]])

        if random.random() < 0.5:
            text = dropout(text, 0.1)
        if random.random() < 0.5:
            gloss = dropout(gloss, 0.1)
        if random.random() < 0.3:
            gloss = shuffle(gloss)

        return text, gloss
