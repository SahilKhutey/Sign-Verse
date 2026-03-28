"""
Train Text<->Gloss Transformer models.

Outputs:
  - models/nlp_vocab.json
  - models/nlp_text2gloss.pt
  - models/nlp_gloss2text.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from nlp_translation.datasets.sign_text_dataset import SignTextDataset
from nlp_translation.tokenizer import SignTokenizer
from nlp_translation.models.transformer_model import TransformerSeq2Seq


def build_tokenizer(train_csv: str, val_csv: str | None, vocab_path: str) -> SignTokenizer:
    tokenizer = SignTokenizer()
    def ingest(path: str):
        if not path or not os.path.exists(path):
            return
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tokenizer.build_vocab([row["text"], row["gloss"]])
    ingest(train_csv)
    ingest(val_csv)
    tokenizer.save_vocab(vocab_path)
    return tokenizer


def _wer(ref: List[str], hyp: List[str]) -> float:
    # Simple word error rate (Levenshtein distance)
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / max(1, n)


def _ngram_counts(tokens: List[str], n: int):
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _bleu_score(ref_tokens: List[str], hyp_tokens: List[str], max_n: int = 4) -> float:
    # Simple BLEU with smoothing (add-one)
    if not hyp_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = _ngram_counts(ref_tokens, n)
        hyp_counts = _ngram_counts(hyp_tokens, n)
        match = 0
        total = max(1, sum(hyp_counts.values()))
        for gram, cnt in hyp_counts.items():
            match += min(cnt, ref_counts.get(gram, 0))
        precisions.append((match + 1) / (total + 1))

    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = torch.exp(torch.tensor(1.0 - ref_len / max(1, hyp_len))).item()

    score = bp
    for p in precisions:
        score *= p ** (1 / max_n)
    return float(score)


def _decode_ids(tokenizer: SignTokenizer, ids: List[int]) -> List[str]:
    text = tokenizer.decode(ids)
    return [t for t in text.split() if t]


def _evaluate(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    tokenizer: SignTokenizer,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    pad_id = tokenizer.word2idx[tokenizer.PAD_TOKEN]
    total_loss = 0.0
    total_wer = 0.0
    total_bleu = 0.0
    n_batches = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += float(loss.item())

            # Batched greedy decode for evaluation
            pred_ids_batch = model.greedy_decode_batch(
                src_tokens=src,
                max_len=tgt.size(1),
                sos_id=tokenizer.word2idx[tokenizer.SOS_TOKEN],
                eos_id=tokenizer.word2idx[tokenizer.EOS_TOKEN],
            )
            for i in range(min(pred_ids_batch.size(0), tgt.size(0))):
                pred_ids = pred_ids_batch[i].tolist()
                ref_ids = tgt[i].tolist()
                ref_tokens = _decode_ids(tokenizer, ref_ids)
                hyp_tokens = _decode_ids(tokenizer, pred_ids)
                total_wer += _wer(ref_tokens, hyp_tokens)
                total_bleu += _bleu_score(ref_tokens, hyp_tokens)

            n_batches += 1
    return (
        total_loss / max(1, n_batches),
        total_wer / max(1, n_batches),
        total_bleu / max(1, n_batches),
    )


def train_one(
    dataset: SignTextDataset,
    tokenizer: SignTokenizer,
    save_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    device: str,
    val_loader: DataLoader = None,
) -> None:
    model = TransformerSeq2Seq(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=1024,
        pad_id=tokenizer.word2idx[tokenizer.PAD_TOKEN],
    ).to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    pad_id = tokenizer.word2idx[tokenizer.PAD_TOKEN]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            opt.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            total += float(loss.item())
        avg = total / max(1, len(loader))
        print(f"epoch {epoch+1}/{epochs} loss={avg:.4f}")

        if val_loader is not None:
            val_loss, val_wer, val_bleu = _evaluate(model, val_loader, tokenizer, device)
            print(f"  val_loss={val_loss:.4f} val_wer={val_wer:.4f} val_bleu={val_bleu:.4f}")
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    if val_loader is None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="datasets/text_sign_pairs/training_data.csv")
    parser.add_argument("--val-csv", default="datasets/text_sign_pairs/validation_data.csv")
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--short-max-len", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    vocab_path = os.path.join(args.save_dir, "nlp_vocab.json")
    tokenizer = build_tokenizer(args.train_csv, args.val_csv, vocab_path)

    # text -> gloss
    max_len = args.max_len
    if args.curriculum:
        max_len = args.short_max_len
    dataset_text2gloss = SignTextDataset(args.train_csv, tokenizer, max_len=max_len, augment=args.augment)
    val_text2gloss = SignTextDataset(args.val_csv, tokenizer, max_len=args.max_len) if args.val_csv else None
    # gloss -> text (swap fields by reusing dataset with inverted CSV rows)
    dataset_gloss2text = SignTextDataset(args.train_csv, tokenizer, max_len=max_len, augment=args.augment)
    val_gloss2text = SignTextDataset(args.val_csv, tokenizer, max_len=args.max_len) if args.val_csv else None
    # Override pairs to swap direction
    dataset_gloss2text.pairs = [(g, t) for (t, g) in dataset_gloss2text.pairs]
    if val_gloss2text is not None:
        val_gloss2text.pairs = [(g, t) for (t, g) in val_gloss2text.pairs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text2gloss_path = os.path.join(args.save_dir, "nlp_text2gloss.pt")
    gloss2text_path = os.path.join(args.save_dir, "nlp_gloss2text.pt")

    print("Training text->gloss")
    train_one(
        dataset_text2gloss,
        tokenizer,
        text2gloss_path,
        args.epochs if not args.curriculum else args.warmup_epochs,
        args.batch_size,
        args.lr,
        args.d_model,
        args.nhead,
        args.num_layers,
        device,
        val_loader=DataLoader(val_text2gloss, batch_size=args.batch_size) if val_text2gloss else None,
    )

    if args.curriculum:
        print("Curriculum phase: extending to full max_len")
        dataset_text2gloss = SignTextDataset(args.train_csv, tokenizer, max_len=args.max_len, augment=args.augment)
        train_one(
            dataset_text2gloss,
            tokenizer,
            text2gloss_path,
            max(1, args.epochs - args.warmup_epochs),
            args.batch_size,
            args.lr,
            args.d_model,
            args.nhead,
            args.num_layers,
            device,
            val_loader=DataLoader(val_text2gloss, batch_size=args.batch_size) if val_text2gloss else None,
        )

    print("Training gloss->text")
    train_one(
        dataset_gloss2text,
        tokenizer,
        gloss2text_path,
        args.epochs if not args.curriculum else args.warmup_epochs,
        args.batch_size,
        args.lr,
        args.d_model,
        args.nhead,
        args.num_layers,
        device,
        val_loader=DataLoader(val_gloss2text, batch_size=args.batch_size) if val_gloss2text else None,
    )

    if args.curriculum:
        print("Curriculum phase: extending to full max_len")
        dataset_gloss2text = SignTextDataset(args.train_csv, tokenizer, max_len=args.max_len, augment=args.augment)
        dataset_gloss2text.pairs = [(g, t) for (t, g) in dataset_gloss2text.pairs]
        train_one(
            dataset_gloss2text,
            tokenizer,
            gloss2text_path,
            max(1, args.epochs - args.warmup_epochs),
            args.batch_size,
            args.lr,
            args.d_model,
            args.nhead,
            args.num_layers,
            device,
            val_loader=DataLoader(val_gloss2text, batch_size=args.batch_size) if val_gloss2text else None,
        )


if __name__ == "__main__":
    main()
