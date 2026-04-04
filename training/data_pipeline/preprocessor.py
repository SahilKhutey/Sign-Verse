"""
SignVerse Preprocessor — Distributed Data Loader
Uses WebDataset for streaming-grade training at 10B parameter scale.
Handles 848-dim motion intelligence vectors.
"""

import webdataset as wds
import torch
import numpy as np
import io

class SignPreprocessor:
    def __init__(self, tar_pattern, batch_size=32, shuffle=1000):
        self.tar_pattern = tar_pattern
        self.batch_size = batch_size
        self.shuffle = shuffle

    def decode_and_preprocess(self, sample):
        """Decodes .npy vectors and .txt labels from tar shard."""
        # sample is a dict: {".npy": bytes, ".txt": bytes, "__key__": str}
        vector_bytes = sample.get("npy")
        label_bytes = sample.get("txt")
        
        if vector_bytes is None:
            return None
            
        vectors = np.load(io.BytesIO(vector_bytes))
        label = label_bytes.decode("utf-8") if label_bytes else ""
        
        return {
            "vectors": torch.tensor(vectors, dtype=torch.float32),
            "label": label,
            "key": sample["__key__"]
        }

    def get_dataloader(self):
        """Returns a high-performance streaming DataLoader."""
        dataset = (
            wds.WebDataset(self.tar_pattern)
            .shuffle(self.shuffle)
            .map(self.decode_and_preprocess)
            .where(lambda x: x is not None)
            .batched(self.batch_size)
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None, # Already batched by WebDataset
            num_workers=4,
            pin_memory=True
        )

if __name__ == "__main__":
    # Test with mock tar shard
    print("SignPreprocessor initialized for distributed training (WebDataset).")
    # loader = SignPreprocessor("datasets/shards/train-*.tar").get_dataloader()
