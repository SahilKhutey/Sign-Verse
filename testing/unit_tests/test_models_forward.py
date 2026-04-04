"""
Forward Pass Shape Tests — All 5 Core Models

Validates that each model produces the expected output tensor shapes
with dummy input. This catches architecture regressions early.

Models tested:
    1. GestureModel:           (batch, seq, 126)  → (batch, num_classes)
    2. SignFoundationModel:    (batch, seq)        → (batch, seq, vocab_size)
    3. SignTransformer:        src + tgt           → (batch, tgt_len, vocab_size)
    4. GestureDiffusionModel:  noisy + timestep    → (batch, seq, motion_dim)
    5. MultimodalSignLLM:      gesture + text_tgt  → (batch, tgt_len, vocab_size)
"""

import sys
import os
import unittest
import importlib
import torch

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from ai_models.gesture_recognition.model import GestureModel
from ai_models.sign_transformer.train_transformer import SignTransformer
from ai_models.multimodal_llm.training_pipeline import MultimodalSignLLM


class TestGestureModelForward(unittest.TestCase):
    """Test GestureModel (BiLSTM + Transformer)."""

    def test_output_shape(self):
        batch, seq_len, input_size = 2, 30, 126
        num_classes = 2000
        model = GestureModel(input_size=input_size, num_classes=num_classes)
        model.eval()

        x = torch.randn(batch, seq_len, input_size)
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape, (batch, num_classes))

    def test_single_sample(self):
        model = GestureModel(input_size=126, num_classes=100)
        model.eval()
        x = torch.randn(1, 30, 126)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 100))


class TestFoundationModelForward(unittest.TestCase):
    """Test SignFoundationModel (GPT-style token prediction)."""

    def test_output_shape(self):
        from models.sign_foundation_transformer import SignFoundationModel

        batch, seq_len = 2, 128
        vocab_size = 512
        model = SignFoundationModel(vocab_size=vocab_size, d_model=128,
                                     nhead=4, num_layers=2)
        model.eval()

        tokens = torch.randint(0, vocab_size, (batch, seq_len))
        with torch.no_grad():
            logits = model(tokens)

        self.assertEqual(logits.shape, (batch, seq_len, vocab_size))

    def test_generation(self):
        from models.sign_foundation_transformer import SignFoundationModel

        vocab_size = 512
        model = SignFoundationModel(vocab_size=vocab_size, d_model=64,
                                     nhead=4, num_layers=1)
        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(prompt, max_len=10, temperature=1.0)

        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], 5)


class TestSignTransformerForward(unittest.TestCase):
    """Test Sign Transformer (Sign → Text)."""

    def test_output_shape(self):
        batch = 2
        src_len, tgt_len, feature_dim = 50, 25, 126
        vocab_size = 10000
        model = SignTransformer(feature_dim=feature_dim,
                                 text_vocab_size=vocab_size,
                                 d_model=128, nhead=4, num_layers=2)
        model.eval()

        src = torch.randn(batch, src_len, feature_dim)
        tgt = torch.randint(0, vocab_size, (batch, tgt_len))

        with torch.no_grad():
            logits = model(src, tgt)

        self.assertEqual(logits.shape, (batch, tgt_len, vocab_size))


class TestDiffusionModelForward(unittest.TestCase):
    """Test GestureDiffusionModel (noise prediction)."""

    def test_output_shape(self):
        from models.gesture_diffusion import GestureDiffusionModel

        batch, seq_len, motion_dim = 2, 30, 150
        model = GestureDiffusionModel(motion_dim=motion_dim, d_model=128,
                                       num_blocks=2)
        model.eval()

        noisy_motion = torch.randn(batch, seq_len, motion_dim)
        timestep = torch.randint(0, 1000, (batch,))

        with torch.no_grad():
            predicted_noise = model(noisy_motion, timestep)

        self.assertEqual(predicted_noise.shape, (batch, seq_len, motion_dim))

    def test_with_conditioning(self):
        from models.gesture_diffusion import GestureDiffusionModel

        batch, seq_len, motion_dim = 2, 30, 150
        model = GestureDiffusionModel(motion_dim=motion_dim, d_model=128,
                                       num_blocks=2)
        model.eval()

        noisy_motion = torch.randn(batch, seq_len, motion_dim)
        timestep = torch.randint(0, 1000, (batch,))
        tokens = torch.randint(0, 512, (batch, 10))

        with torch.no_grad():
            out = model(noisy_motion, timestep, gesture_tokens=tokens)

        self.assertEqual(out.shape, (batch, seq_len, motion_dim))


class TestMultimodalLLMForward(unittest.TestCase):
    """Test MultimodalSignLLM (gesture → text)."""

    def test_output_shape(self):
        batch = 2
        gesture_seq_len = 60
        gesture_dim = 225
        tgt_len = 24
        vocab_size = 1000

        model = MultimodalSignLLM(gesture_dim=gesture_dim,
                                   text_vocab=vocab_size, d_model=128)
        model.eval()

        gesture_seq = torch.randn(batch, gesture_seq_len, gesture_dim)
        text_target = torch.randint(0, vocab_size, (batch, tgt_len))

        with torch.no_grad():
            logits = model(gesture_seq, text_target=text_target)

        self.assertEqual(logits.shape[0], batch)
        self.assertEqual(logits.shape[2], vocab_size)

    def test_with_text_input(self):
        batch = 2
        model = MultimodalSignLLM(gesture_dim=225, text_vocab=1000, d_model=128)
        model.eval()

        gesture = torch.randn(batch, 60, 225)
        text_in = torch.randint(0, 1000, (batch, 10))
        text_tgt = torch.randint(0, 1000, (batch, 20))

        with torch.no_grad():
            logits = model(gesture, text_input=text_in, text_target=text_tgt)

        self.assertEqual(logits.shape[0], batch)
        self.assertEqual(logits.shape[2], 1000)


if __name__ == "__main__":
    unittest.main()
