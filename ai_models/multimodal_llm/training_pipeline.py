"""
Multimodal Sign LLM — Production Training Pipeline

Full architecture:
    GestureEncoder (Video Transformer) ← sign keypoints
    + Text Embedding                   ← text context
    → FusionTransformer                ← multi-modal fusion
    → TextDecoder                      → natural language output

FIXED: Replaced random tensor batches with real MultimodalDataset.
Uses BaseTrainer for AMP, checkpointing, and TensorBoard.

Usage:
    python -m ai_models.multimodal_llm.training_pipeline
    python -m ai_models.multimodal_llm.training_pipeline --resume --dry-run
"""

import os
import sys
import argparse
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ge = _load('gesture_encoder_mod', os.path.join(os.path.dirname(__file__), 'gesture_encoder.py'))
_td = _load('text_decoder_mod', os.path.join(os.path.dirname(__file__), 'text_decoder.py'))
GestureEncoder = _ge.GestureEncoder
TextDecoder = _td.TextDecoder

from training.utils.trainer_base import BaseTrainer
from training.data_pipeline.multimodal_dataset import MultimodalDataset, SyntheticMultimodalDataset


# ──────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────

class FusionTransformer(nn.Module):
    """Fuses gesture and text representations across sequence dimension."""

    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True, activation="gelu",
                dropout=0.1
            ),
            num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gesture_emb, text_emb=None):
        inputs = [gesture_emb]
        if text_emb is not None:
            inputs.append(text_emb)
        x = torch.cat(inputs, dim=1)
        return self.norm(self.fusion(x))


class MultimodalSignLLM(nn.Module):
    """
    Complete Multimodal Sign Language LLM.

    Input:  gesture keypoint sequence + optional text tokens
    Output: text token logits
    """

    def __init__(self, gesture_dim=225, text_vocab=10000, d_model=512):
        super().__init__()
        self.gesture_encoder = GestureEncoder(input_dim=gesture_dim, d_model=d_model)
        self.text_embedding = nn.Embedding(text_vocab, d_model, padding_idx=0)
        self.fusion = FusionTransformer(d_model=d_model)
        self.text_decoder = TextDecoder(vocab_size=text_vocab, d_model=d_model)
        self.vocab_size = text_vocab

    def forward(self, gesture_seq, text_input=None, text_target=None):
        """
        Args:
            gesture_seq:  (batch, time, gesture_dim)
            text_input:   (batch, src_len) optional text input tokens
            text_target:  (batch, tgt_len) decoding target tokens

        Returns:
            text_logits: (batch, tgt_len, vocab_size)
        """
        gesture_emb = self.gesture_encoder(gesture_seq)

        text_emb = None
        if text_input is not None:
            text_emb = self.text_embedding(text_input)

        fused = self.fusion(gesture_emb, text_emb)

        if text_target is not None:
            return self.text_decoder(text_target, fused)

        return self.text_decoder.greedy_decode(fused)


# ──────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────

class MultimodalLLMTrainer(BaseTrainer):

    def build_model(self) -> nn.Module:
        return MultimodalSignLLM(
            gesture_dim=self.config.get("gesture_dim", 225),
            text_vocab=self.config.get("text_vocab", 10000),
            d_model=self.config.get("d_model", 512),
        )

    def build_datasets(self):
        seq_len = self.config.get("seq_len", 60)
        text_seq_len = self.config.get("text_seq_len", 25)
        vocab_size = self.config.get("text_vocab", 10000)

        try:
            train_ds = MultimodalDataset(
                keypoint_dir="training-data/keypoints",
                labels_csv="training-data/labels.csv",
                seq_len=seq_len,
                text_seq_len=text_seq_len,
                split="train"
            )
            val_ds = MultimodalDataset(
                keypoint_dir="training-data/keypoints",
                labels_csv="training-data/labels.csv",
                seq_len=seq_len,
                text_seq_len=text_seq_len,
                split="val"
            )
        except RuntimeError:
            print("  [MultimodalLLM] Falling back to synthetic dataset")
            train_ds = SyntheticMultimodalDataset(
                n_samples=200, seq_len=seq_len,
                feature_dim=225, text_seq_len=text_seq_len,
                vocab_size=vocab_size
            )
            val_ds = SyntheticMultimodalDataset(
                n_samples=50, seq_len=seq_len,
                feature_dim=225, text_seq_len=text_seq_len,
                vocab_size=vocab_size
            )

        return train_ds, val_ds

    def compute_loss(self, batch):
        gesture_seq, text_input, text_target = batch

        logits = self.model(gesture_seq, text_input=text_input, text_target=text_input)
        # logits: (batch, tgt_len, vocab)  |  text_target: (batch, tgt_len)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            text_target.reshape(-1),
            ignore_index=0
        )


# ──────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────

def train(config: dict):
    """Entry point called by run_all_training.py Stage 9."""
    config.setdefault("model_name", "multimodal_llm")
    trainer = MultimodalLLMTrainer(config)
    return trainer.fit()


def main():
    parser = argparse.ArgumentParser(description="Multimodal Sign LLM Training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    try:
        import yaml
        with open("training/configs/training_config.yaml") as f:
            config = yaml.safe_load(f).get("multimodal_llm", {})
    except Exception:
        config = {}

    config["model_name"] = "multimodal_llm"
    config["resume"] = args.resume

    if args.dry_run:
        config["epochs"] = 2
        config["batch_size"] = 4
        print("  [DRY RUN] 2 epochs")
    if args.epochs:
        config["epochs"] = args.epochs

    train(config)


if __name__ == "__main__":
    main()
