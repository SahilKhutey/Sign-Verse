"""
CTC Decoder for Continuous Sign Language Recognition.

Decodes model logits into readable sentences using
greedy CTC decoding (argmax per timestep, collapse repeats).
"""

import torch


def decode_predictions(logits, labels):
    """
    Greedy CTC decode: argmax per timestep → collapse repeated tokens → sentence.

    Args:
        logits: (batch, seq_len, vocab_size) tensor
        labels: list of label strings indexed by class ID

    Returns:
        List of decoded sentence strings
    """

    predictions = torch.argmax(logits, dim=-1)

    sentences = []

    for sequence in predictions:

        words = []
        prev = -1

        for token_id in sequence:

            tid = token_id.item()

            # Skip blank token (index 0) and collapse repeats
            if tid != 0 and tid != prev:
                if tid < len(labels):
                    words.append(labels[tid])

            prev = tid

        sentences.append(" ".join(words))

    return sentences
