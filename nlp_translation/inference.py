"""
NLP Translation Inference

Provides end-to-end translation between:
  - Text → Sign gloss
  - Sign gloss → Text
"""

from __future__ import annotations

from typing import List, Optional

import os

import torch

from nlp_translation.sign_grammar_converter import SignGrammarConverter
from nlp_translation.tokenizer import SignTokenizer
from nlp_translation.models.transformer_model import TransformerSeq2Seq


class TranslationInference:

    def __init__(self, vocab_path: Optional[str] = None, model_dir: str = "models"):
        self.grammar = SignGrammarConverter()
        self.vocab_path = vocab_path or os.path.join(model_dir, "nlp_vocab.json")
        self.tokenizer = SignTokenizer(self.vocab_path if os.path.exists(self.vocab_path) else None)
        self.model_dir = model_dir

        self._text2gloss = None
        self._gloss2text = None

    def _load_model(self, name: str) -> Optional[TransformerSeq2Seq]:
        path = os.path.join(self.model_dir, name)
        if not os.path.exists(path):
            return None

        model = TransformerSeq2Seq(
            src_vocab_size=self.tokenizer.vocab_size,
            tgt_vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            pad_id=self.tokenizer.word2idx[self.tokenizer.PAD_TOKEN],
        )
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    @property
    def text2gloss(self) -> Optional[TransformerSeq2Seq]:
        if self._text2gloss is None:
            self._text2gloss = self._load_model("nlp_text2gloss.pt")
        return self._text2gloss

    @property
    def gloss2text(self) -> Optional[TransformerSeq2Seq]:
        if self._gloss2text is None:
            self._gloss2text = self._load_model("nlp_gloss2text.pt")
        return self._gloss2text

    def _decode(self, model: TransformerSeq2Seq, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        src = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        out = model.greedy_decode(
            src_tokens=src,
            max_len=50,
            sos_id=self.tokenizer.word2idx[self.tokenizer.SOS_TOKEN],
            eos_id=self.tokenizer.word2idx[self.tokenizer.EOS_TOKEN],
        )
        decoded = out.squeeze(0).tolist()
        return self.tokenizer.decode(decoded).split()

    def text_to_sign(self, text: str):
        """
        Convert natural text to sign language gloss.

        Returns:
            dict with gloss tokens and display string
        """
        if self.text2gloss is not None:
            gloss = self._decode(self.text2gloss, text)
        else:
            gloss = self.grammar.convert(text)
        return {
            "input": text,
            "gloss": gloss,
            "display": self.grammar.gloss_to_string(gloss),
            "source": "model" if self.text2gloss is not None else "rule",
        }

    def sign_to_text(self, gloss_tokens: List[str]) -> str:
        """
        Convert sign gloss tokens back to English text.
        """
        joined = " ".join(gloss_tokens)
        if self.gloss2text is not None:
            words = self._decode(self.gloss2text, joined)
            return " ".join(words).capitalize()
        return " ".join(word.capitalize() for word in gloss_tokens)

    def batch_translate(self, sentences):
        """Translate a batch of sentences to sign gloss."""
        return [self.text_to_sign(s) for s in sentences]
