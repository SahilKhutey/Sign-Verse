"""
Sign Language Tokenizer

Handles tokenization for sign language translation:
  - Text to token IDs
  - Token IDs back to text
  - Vocabulary management
  - Special tokens (PAD, SOS, EOS, UNK)
"""

import json
import os


class SignTokenizer:

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    # Language Tags
    ASL_TAG = "<ASL>"
    ISL_TAG = "<ISL>"
    BSL_TAG = "<BSL>"

    def __init__(self, vocab_path=None):
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
            self.ASL_TAG: 4,
            self.ISL_TAG: 5,
            self.BSL_TAG: 6
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 7

        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)

    def build_vocab(self, sentences):
        """Build vocabulary from a list of sentences."""
        for sentence in sentences:
            for word in sentence.upper().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text):
        """Convert text to list of token IDs."""
        tokens = [self.word2idx.get(self.SOS_TOKEN)]
        for word in text.upper().split():
            tokens.append(self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]))
        tokens.append(self.word2idx.get(self.EOS_TOKEN))
        return tokens

    def decode(self, token_ids):
        """Convert token IDs back to text."""
        words = []
        for tid in token_ids:
            word = self.idx2word.get(tid, self.UNK_TOKEN)
            if word in (self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN):
                continue
            words.append(word)
        return " ".join(words)

    def save_vocab(self, path):
        """Save vocabulary to JSON file."""
        with open(path, "w") as f:
            json.dump(self.word2idx, f, indent=2)

    def load_vocab(self, path):
        """Load vocabulary from JSON file."""
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
