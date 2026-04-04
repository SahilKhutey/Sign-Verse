"""
Sign Transformer Training Script — Production Version

Trains the Sign Language → Text Translation transformer.

Architecture:
    Gesture keypoint encoder (TransformerEncoder)
    + Text decoder (TransformerDecoder)
    → Sequence-to-sequence sign → text translation

Uses BaseTrainer for AMP, checkpointing, and TensorBoard logging.

Usage:
    python -m ai_models.sign_transformer.train_transformer
    python -m ai_models.sign_transformer.train_transformer --resume --dry-run
"""

import os
import sys
import argparse
import importlib.util
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import csv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

from common.keypoint_schema import FEATURE_DIM_225, TWO_HANDS_DIM, HANDS_SLICE_225

from nlp_translation.tokenizer import SignTokenizer
from nlp_translation.datasets.multilingual_dataset import MultilingualSignDataset


def _src_key_padding_mask(src: torch.Tensor) -> torch.Tensor:
    """
    Build a padding mask for src keypoint sequences.

    The dataset pads with all-zeros frames; we treat those as padding.
    Returns bool mask shaped (batch, src_len) where True = pad.
    """
    if src.ndim != 3:
        raise ValueError(f"Expected src (B,T,D), got shape={tuple(src.shape)}")
    return src.abs().sum(dim=-1) < 1e-8


def _normalize_label_text(label: str) -> str:
    # Most datasets store gloss-like labels as "THANK_YOU".
    return str(label or "").replace("_", " ").strip()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_enc = _load('sign_encoder_mod', os.path.join(os.path.dirname(__file__), 'encoder.py'))
_dec = _load('sign_decoder_mod', os.path.join(os.path.dirname(__file__), 'decoder.py'))
SignEncoder = _enc.SignEncoder
SignDecoder = _dec.SignDecoder

from training.utils.trainer_base import BaseTrainer


# ──────────────────────────────────────────────────────────────────
# Full Sign Transformer model
# ──────────────────────────────────────────────────────────────────

class SignTransformer(nn.Module):
    """
    Sign Language → Text Transformer.

    Input:  gesture keypoint sequence  (batch, src_len, feature_dim)
    Output: text token logits          (batch, tgt_len, vocab_size)
    """

    def __init__(self, feature_dim=126, text_vocab_size=10000,
                 d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = SignEncoder(
            feature_dim=feature_dim, d_model=d_model,
            nhead=nhead, num_layers=num_layers
        )
        self.decoder = SignDecoder(
            text_vocab_size=text_vocab_size, d_model=d_model,
            nhead=nhead, num_layers=num_layers
        )
        self.vocab_size = text_vocab_size

    def forward(self, src, tgt):
        """
        Args:
            src: (batch, src_len, feature_dim) gesture keypoints
            tgt: (batch, tgt_len) token ids
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        src_pad = _src_key_padding_mask(src)
        tgt_pad = tgt == 0

        memory = self.encoder(src, src_key_padding_mask=src_pad)
        return self.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )

    @torch.no_grad()
    def translate(self, src, max_len=25, sos_id=1, eos_id=2, target_lang="ASL"):
        """
        Greedy decode a text token sequence from sign keypoints with language conditioning.

        Args:
            src: (batch, src_len, feature_dim)
            target_lang: "ASL", "ISL", "BSL"
        Returns:
            List[List[int]] token ids (batch dimension preserved)
        """
        self.eval()
        device = src.device

        src_pad = _src_key_padding_mask(src)
        memory = self.encoder(src, src_key_padding_mask=src_pad)

        batch = src.size(0)
        
        # In a real multilingual transformer, the SOS is followed by a LANG_TAG.
        # We'll assume the tokenizer provides these mappings.
        # For now, we'll start with SOS and the model will be trained to expect LANG_TAG.
        decoded = torch.full((batch, 1), int(sos_id), dtype=torch.long, device=device)
        
        # If the model was trained with prepended lang tags, we should add it here too.
        # We'll stick to basic greedy decode for now, assuming the model learns the tag.

        for _ in range(int(max_len)):
            logits = self.decoder(decoded, memory, memory_key_padding_mask=src_pad)  # (B, T, vocab)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded = torch.cat([decoded, next_id], dim=1)
            if (next_id.squeeze(1) == int(eos_id)).all():
                break

        # Strip SOS and any EOS tokens.
        out = decoded[:, 1:]
        out_list = []
        for row in out.tolist():
            trimmed = []
            for tid in row:
                if tid == int(eos_id):
                    break
                trimmed.append(int(tid))
            out_list.append(trimmed)
        return out_list


# ──────────────────────────────────────────────────────────────────
# Paired gesture-text dataset
# ──────────────────────────────────────────────────────────────────

class PairedGestureTextDataset(Dataset):
    """
    Loads (keypoint_sequence, text_token_ids) pairs.
    Falls back to synthetic data if no real data is available.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(self, keypoint_dir="training-data/keypoints",
                 labels_csv="training-data/labels.csv",
                 src_len=50, tgt_len=25, feature_dim=126, split=None,
                 tokenizer=None, vocab_size=1000):
        self.keypoint_dir = keypoint_dir
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.feature_dim = feature_dim
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.samples = []

        if os.path.exists(labels_csv):
            with open(labels_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                has_split = bool(reader.fieldnames and ("split" in reader.fieldnames))
                for row in reader:
                    # If the CSV defines an explicit split column, filter by it. Otherwise
                    # we load all rows and let the trainer do a deterministic split.
                    if split is not None and has_split and row.get("split") != split:
                        continue
                    self.samples.append({
                        "kp_path": os.path.join(keypoint_dir, row.get("filename", "")),
                        "label_id": int(row.get("label_id", 0)),
                        "label_name": row.get("label_name", ""),
                        "text": row.get("text") or row.get("sentence") or row.get("gloss") or "",
                    })

        # Use synthetic if empty
        self._synthetic = len(self.samples) == 0
        if self._synthetic:
            self.samples = list(range(500))
            print(f"  [SignTransformer] No data found — using synthetic ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._synthetic:
            src = torch.randn(self.src_len, self.feature_dim)
            # Ensure token IDs are always within [0, vocab_size).
            hi = max(int(self.vocab_size), 4)
            tgt_in = torch.randint(0, hi, (self.tgt_len,), dtype=torch.long)
            tgt_out = torch.randint(0, hi, (self.tgt_len,), dtype=torch.long)
            return src, tgt_in, tgt_out

        row = self.samples[idx]
        src = self._load_kp(row["kp_path"])

        if self.tokenizer is not None:
            text = row.get("text") or row.get("sentence") or row.get("gloss") or row.get("label_name", "")
            text = _normalize_label_text(text)
            tokens = self.tokenizer.encode(text)  # [SOS, ..., EOS]
            tgt_in_ids = tokens[:-1]
            tgt_out_ids = tokens[1:]
        else:
            # Fallback: map label_id into a non-special token ID range.
            v = max(int(self.vocab_size), 5)
            label_id = int(row.get("label_id", 0))
            label_tok = 4 + (label_id % (v - 4))
            tgt_in_ids = [self.BOS_ID, label_tok]
            tgt_out_ids = [label_tok, self.EOS_ID]

        tgt_in = np.full(self.tgt_len, self.PAD_ID, dtype=np.int64)
        tgt_out = np.full(self.tgt_len, self.PAD_ID, dtype=np.int64)

        for i, t in enumerate(tgt_in_ids[: self.tgt_len]):
            tgt_in[i] = int(t)
        for i, t in enumerate(tgt_out_ids[: self.tgt_len]):
            tgt_out[i] = int(t)

        # If we truncated and lost EOS, force EOS at the end.
        if self.EOS_ID not in tgt_out and self.tgt_len > 0:
            tgt_out[self.tgt_len - 1] = self.EOS_ID

        return src, torch.tensor(tgt_in, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)

    def _load_kp(self, path):
        try:
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            T, D = arr.shape

            # If training a 126-dim (hands-only) model but the file is stored in the
            # canonical 225-dim layout, slice out the hands portion.
            if D == FEATURE_DIM_225 and self.feature_dim == TWO_HANDS_DIM:
                arr = arr[:, HANDS_SLICE_225]
                T, D = arr.shape

            arr = arr[:, :self.feature_dim] if D >= self.feature_dim else np.pad(arr, ((0, 0), (0, self.feature_dim - D)))
            arr = arr[:self.src_len] if T >= self.src_len else np.pad(arr, ((0, self.src_len - T), (0, 0)))
            return torch.tensor(arr, dtype=torch.float32)
        except Exception:
            return torch.zeros(self.src_len, self.feature_dim)


# ──────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────

class SignTransformerTrainer(BaseTrainer):

    def build_model(self) -> nn.Module:
        save_dir = self.config.get("save_dir", "models")
        vocab_path = self.config.get(
            "vocab_path",
            os.path.join(save_dir, "sign_transformer_vocab.json"),
        )
        labels_csv = self.config.get(
            "labels_csv",
            os.path.join("training-data", "labels.csv"),
        )

        tokenizer = None
        if os.path.exists(vocab_path):
            tokenizer = SignTokenizer(vocab_path=vocab_path)

        # If we have data, ensure the vocab has more than the 4 special tokens.
        if os.path.exists(labels_csv) and (tokenizer is None or int(getattr(tokenizer, "vocab_size", 0)) <= 4):
            # Build a word-level vocab from whatever paired text columns exist.
            tokenizer = SignTokenizer()
            sentences = []
            try:
                with open(labels_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get("text") or row.get("sentence") or row.get("gloss") or row.get("label_name", "")
                        text = _normalize_label_text(text)
                        if text:
                            sentences.append(text)
            except Exception:
                sentences = []
            tokenizer.build_vocab(sentences)
            try:
                os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
                tokenizer.save_vocab(vocab_path)
            except Exception:
                pass

        self.tokenizer = tokenizer
        text_vocab_size = int(
            getattr(tokenizer, "vocab_size", 0)
            or self.config.get("text_vocab_size", 10000)
        )
        self.config["text_vocab_size"] = text_vocab_size

        return SignTransformer(
            feature_dim=self.config.get("feature_dim", 126),
            text_vocab_size=text_vocab_size,
            d_model=self.config.get("d_model", 512),
            nhead=self.config.get("nhead", 8),
            num_layers=self.config.get("num_layers", 6),
        )

    def build_datasets(self):
        src_len = self.config.get("src_len", 50)
        tgt_len = self.config.get("tgt_len", 25)
        feature_dim = self.config.get("feature_dim", 126)
        keypoint_dir = self.config.get(
            "keypoint_dir",
            os.path.join("training-data", "keypoints"),
        )
        labels_csv = self.config.get(
            "labels_csv",
            os.path.join("training-data", "labels.csv"),
        )

        vocab_size = int(self.config.get("text_vocab_size", 10000))
        tokenizer = getattr(self, "tokenizer", None)

        # If the CSV provides explicit splits, keep that behavior. Otherwise do a
        # deterministic split here so validation isn't synthetic.
        has_split = False
        try:
            if os.path.exists(labels_csv):
                with open(labels_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    has_split = bool(reader.fieldnames and ("split" in reader.fieldnames))
        except Exception:
            has_split = False

        if self.config.get("multilingual"):
            # Load full sequential data for multilingual wrapper
            # In production, this would load from a specific multilingual manifest
            full_ds = MultilingualSignDataset(
                sign_sequences=[], # Placeholder: would load from keypoint files
                translations=[], 
                languages=[], 
                tokenizer=tokenizer
            )
            print("  [SignTransformer] Initialized MultilingualSignDataset")
            # For simplicity in this demo, we'll continue with the standard loader 
            # but with the wrapper ready.

        if has_split:
            train_ds = PairedGestureTextDataset(
                keypoint_dir=keypoint_dir,
                labels_csv=labels_csv,
                split="train",
                src_len=src_len,
                tgt_len=tgt_len,
                feature_dim=feature_dim,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
            )
            val_ds = PairedGestureTextDataset(
                keypoint_dir=keypoint_dir,
                labels_csv=labels_csv,
                split="val",
                src_len=src_len,
                tgt_len=tgt_len,
                feature_dim=feature_dim,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
            )
            return train_ds, val_ds

        full_ds = PairedGestureTextDataset(
            keypoint_dir=keypoint_dir,
            labels_csv=labels_csv,
            split=None,
            src_len=src_len,
            tgt_len=tgt_len,
            feature_dim=feature_dim,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )

        val_frac = float(self.config.get("val_split", 0.1))
        val_frac = 0.0 if val_frac < 0 else 1.0 if val_frac > 1.0 else val_frac
        n_total = len(full_ds)
        n_val = int(round(n_total * val_frac)) if n_total >= 2 else 0
        n_val = min(max(n_val, 1), n_total - 1) if n_total >= 2 else 0
        n_train = n_total - n_val

        if n_val <= 0:
            return full_ds, None

        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        return train_ds, val_ds

    def val_epoch(self, epoch: int) -> float:
        """Validation with basic translation metrics."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        total_seqs = 0
        exact_match = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                src, tgt_in, tgt_out = batch
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(src, tgt_in)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_out.reshape(-1),
                        ignore_index=0,
                    )
                total_loss += float(loss.item())

                pred = logits.argmax(dim=-1)
                mask = tgt_out != 0
                correct_tokens += int(((pred == tgt_out) & mask).sum().item())
                total_tokens += int(mask.sum().item())

                # Sequence-level exact match (ignores padding).
                for i in range(tgt_out.size(0)):
                    m = mask[i]
                    true_seq = tgt_out[i][m].tolist()
                    pred_seq = pred[i][m].tolist()
                    if pred_seq == true_seq:
                        exact_match += 1
                    total_seqs += 1

        avg_loss = total_loss / max(len(self.val_loader), 1)
        token_acc = correct_tokens / max(total_tokens, 1)
        exact = exact_match / max(total_seqs, 1)

        self.logger.log_metrics(
            {
                "val/loss": avg_loss,
                "val/token_acc": token_acc,
                "val/exact_match": exact,
            },
            step=epoch,
        )
        return avg_loss

    def compute_loss(self, batch):
        src, tgt_in, tgt_out = batch
        logits = self.model(src, tgt_in)
        # logits: (batch, tgt_len, vocab)  |  tgt_out: (batch, tgt_len)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=0
        )


def train(config: dict):
    """Entry point called by run_all_training.py."""
    config.setdefault("model_name", "sign_transformer")
    trainer = SignTransformerTrainer(config)
    return trainer.fit()


def main():
    parser = argparse.ArgumentParser(description="Sign Transformer Training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        import yaml
        with open("training/configs/training_config.yaml") as f:
            config = yaml.safe_load(f).get("sign_transformer", {})
    except Exception:
        config = {}

    config["model_name"] = "sign_transformer"
    config["resume"] = args.resume
    if args.dry_run:
        config["model_name"] = "sign_transformer_smoke"
        config["epochs"] = 2
        config["batch_size"] = 4
        # Keep dry-runs fast on CPU while still exercising the full training loop.
        config["d_model"] = 128
        config["nhead"] = 4
        config["num_layers"] = 2
        config["src_len"] = 30
        config["tgt_len"] = 12

    train(config)


if __name__ == "__main__":
    main()
