"""
Evaluation Metrics for Sign Language Translation

BLEU score, Word Error Rate (WER), and accuracy
for evaluating translation quality.
"""


def bleu_score(reference, hypothesis, n=4):
    """
    Compute simple BLEU score between reference and hypothesis.

    Args:
        reference: list of reference tokens
        hypothesis: list of hypothesis tokens
        n: max n-gram order
    """
    from collections import Counter

    scores = []

    for i in range(1, n + 1):
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i)]))
        hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i)]))

        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = max(sum(hyp_ngrams.values()), 1)

        scores.append(overlap / total)

    if 0 in scores:
        return 0.0

    import math
    avg = sum(math.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = min(1.0, len(hypothesis) / max(len(reference), 1))

    return bp * math.exp(avg)


def word_error_rate(reference, hypothesis):
    """
    Compute Word Error Rate (WER) using edit distance.

    Args:
        reference: list of reference tokens
        hypothesis: list of hypothesis tokens

    Returns:
        WER as a float (0.0 = perfect, 1.0 = all wrong)
    """
    r_len = len(reference)
    h_len = len(hypothesis)

    # Dynamic programming edit distance
    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[r_len][h_len] / max(r_len, 1)


def accuracy(predictions, targets):
    """Compute token-level accuracy."""
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / max(len(targets), 1)
