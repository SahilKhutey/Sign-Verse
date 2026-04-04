"""
Augmentation Pipeline — Data augmentation for sign language sequences.

Techniques:
    1. Gaussian noise          — small perturbations to keypoints
    2. Time warping            — random speed variation
    3. Random cropping         — subsample sequence
    4. Horizontal flip         — mirror sign (switch L/R hands)
    5. Scale jitter            — slight scale variation
    6. Frame dropout           — randomly zero some frames

Applied during training to improve generalization across:
    - Signers with different body proportions
    - Variable signing speeds
    - Camera angles and distances
"""

import numpy as np


class AugmentationPipeline:

    def __init__(self, config=None):
        cfg = config or {}
        self.noise_std = cfg.get("noise_std", 0.005)
        self.time_warp_factor = cfg.get("time_warp_factor", 0.2)
        self.scale_range = cfg.get("scale_range", (0.85, 1.15))
        self.flip_prob = cfg.get("flip_prob", 0.3)
        self.dropout_prob = cfg.get("dropout_prob", 0.05)
        self.crop_ratio = cfg.get("crop_ratio", 0.8)

    def augment(self, seq):
        """
        Apply random augmentation chain to a sequence.

        Args:
            seq: numpy (frames, feature_dim)

        Returns:
            Augmented sequence (same shape)
        """
        seq = seq.copy()

        # Apply each augmentation with its own probability
        if np.random.rand() < 0.8:
            seq = self.add_noise(seq)
        if np.random.rand() < 0.5:
            seq = self.time_warp(seq)
        if np.random.rand() < 0.5:
            seq = self.scale_jitter(seq)
        if np.random.rand() < self.flip_prob:
            seq = self.horizontal_flip(seq)
        if np.random.rand() < 0.3:
            seq = self.frame_dropout(seq)

        return seq

    def add_noise(self, seq):
        """Add small Gaussian noise to keypoints."""
        noise = np.random.randn(*seq.shape).astype(np.float32) * self.noise_std
        return seq + noise

    def time_warp(self, seq):
        """Randomly speed up or slow down the sequence."""
        T = len(seq)
        factor = 1.0 + np.random.uniform(-self.time_warp_factor, self.time_warp_factor)
        new_len = max(2, int(T * factor))
        indices = np.linspace(0, T - 1, new_len)
        warped = np.array([seq[min(int(i), T-1)] for i in indices])
        # Restore original length via linear interp
        return self._resize_seq(warped, T)

    def scale_jitter(self, seq):
        """Apply random scale to all values."""
        scale = np.random.uniform(*self.scale_range)
        return seq * scale

    def horizontal_flip(self, seq, hand_dim=63, body_dim=99):
        """
        Mirror sign by swapping left/right hand features.
        Assumes feature layout: [right_hand | left_hand | body]
        """
        flipped = seq.copy()
        total = seq.shape[1]
        if total >= hand_dim * 2:
            # Swap left and right hands
            right = flipped[:, :hand_dim].copy()
            left = flipped[:, hand_dim:hand_dim*2].copy()
            flipped[:, :hand_dim] = left
            flipped[:, hand_dim:hand_dim*2] = right
        return flipped

    def frame_dropout(self, seq):
        """Randomly zero out some frames."""
        T = len(seq)
        mask = np.random.rand(T) > self.dropout_prob
        result = seq.copy()
        result[~mask] = 0.0
        return result

    def random_crop(self, seq, target_len=None):
        """Randomly crop a subsequence."""
        T = len(seq)
        target = target_len or max(2, int(T * self.crop_ratio))
        if T <= target:
            return seq
        start = np.random.randint(0, T - target)
        return seq[start:start + target]

    def _resize_seq(self, seq, target_len):
        """Resize sequence to target length via linear interpolation."""
        T = len(seq)
        if T == target_len:
            return seq
        old_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, target_len)
        resized = np.zeros((target_len, seq.shape[1]), dtype=seq.dtype)
        for feat in range(seq.shape[1]):
            resized[:, feat] = np.interp(new_idx, old_idx, seq[:, feat])
        return resized


class MixUp:
    """
    MixUp augmentation for sequences — blend two samples.
    Helps model generalize across signing styles.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def apply(self, seq_a, label_a, seq_b, label_b, num_classes):
        """Mix two sequences and their labels."""
        lam = np.random.beta(self.alpha, self.alpha)

        # Ensure same length
        min_len = min(len(seq_a), len(seq_b))
        seq_a = seq_a[:min_len]
        seq_b = seq_b[:min_len]

        mixed_seq = lam * seq_a + (1 - lam) * seq_b

        # Soft labels
        label_a_onehot = np.zeros(num_classes)
        label_b_onehot = np.zeros(num_classes)
        label_a_onehot[label_a] = 1.0
        label_b_onehot[label_b] = 1.0
        mixed_label = lam * label_a_onehot + (1 - lam) * label_b_onehot

        return mixed_seq, mixed_label
