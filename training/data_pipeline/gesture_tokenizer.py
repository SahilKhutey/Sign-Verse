"""
Gesture Tokenizer — Vector Quantization for Sign Language

Converts continuous pose/motion sequences into discrete gesture tokens,
analogous to word tokens in language models.

Pipeline:
    Pose vectors → KMeans clustering → Token IDs → Vocabulary

Example vocabulary:
    G001 = HELLO
    G002 = THANK_YOU
    G003 = YOU
    G004 = GOOD

This enables training a transformer on gesture "text" using
self-supervised objectives (next-token prediction).
"""

import numpy as np
import pickle
import os


class GestureTokenizer:

    def __init__(self, n_clusters=512, n_init=10, max_iter=300, vocab_path=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.codebook = None
        self.token_labels = {}

        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def fit(self, features):
        """
        Build gesture vocabulary from pose feature vectors.

        Args:
            features: numpy array of shape (N, feature_dim)
                      where N = total frames across all videos
        """
        from sklearn.cluster import KMeans

        print(f"Fitting gesture tokenizer with {self.n_clusters} clusters "
              f"on {len(features)} samples...")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=42,
            verbose=1
        )

        self.kmeans.fit(features)
        self.codebook = self.kmeans.cluster_centers_

        print(f"Gesture vocabulary created: {self.n_clusters} tokens")

    def tokenize(self, features):
        """
        Convert pose features to gesture token IDs.

        Args:
            features: numpy array (N, feature_dim) — one row per frame

        Returns:
            numpy array of token IDs
        """
        if self.codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        return self.kmeans.predict(features)

    def tokenize_sequence(self, frame_features):
        """
        Tokenize a video sequence and collapse consecutive repeats.

        Args:
            frame_features: list of feature vectors, one per frame

        Returns:
            list of unique consecutive token IDs
        """
        raw_tokens = self.tokenize(np.array(frame_features))

        # Collapse consecutive duplicates
        collapsed = [raw_tokens[0]]
        for t in raw_tokens[1:]:
            if t != collapsed[-1]:
                collapsed.append(t)

        return collapsed

    def assign_labels(self, token_id, label):
        """Assign a human-readable label to a token ID."""
        self.token_labels[token_id] = label

    def decode(self, token_ids):
        """Convert token IDs back to labels."""
        return [
            self.token_labels.get(tid, f"G{tid:03d}")
            for tid in token_ids
        ]

    def save(self, path):
        """Save tokenizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "codebook": self.codebook,
            "kmeans": self.kmeans,
            "labels": self.token_labels,
            "n_clusters": self.n_clusters
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        """Load tokenizer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.codebook = data["codebook"]
        self.kmeans = data["kmeans"]
        self.token_labels = data["labels"]
        self.n_clusters = data["n_clusters"]
        print(f"Tokenizer loaded: {self.n_clusters} tokens")
