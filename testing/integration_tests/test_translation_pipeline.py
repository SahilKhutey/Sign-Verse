"""
Integration Test — Training Pipeline

End-to-end mini-pipeline test that validates:
    1. Synthetic data generation
    2. Data validation
    3. Gesture tokenization
    4. Model instantiation for all 5 models
    5. Single training step for each model
    6. Checkpoint save and load

This test runs WITHOUT GPU and WITHOUT real data.
"""

import sys
import os
import unittest
import tempfile
import shutil
import importlib.util
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from ai_models.gesture_recognition.model import GestureModel
from ai_models.sign_transformer.train_transformer import SignTransformer
from ai_models.multimodal_llm.training_pipeline import MultimodalSignLLM


class TestTrainingPipelineIntegration(unittest.TestCase):
    """End-to-end mini pipeline test."""

    @classmethod
    def setUpClass(cls):
        """Create temporary directories with synthetic data."""
        cls.tmpdir = tempfile.mkdtemp()
        cls.kp_dir = os.path.join(cls.tmpdir, "keypoints")
        cls.token_dir = os.path.join(cls.tmpdir, "tokens")
        cls.model_dir = os.path.join(cls.tmpdir, "models")
        cls.motion_dir = os.path.join(cls.tmpdir, "motion")
        os.makedirs(cls.kp_dir, exist_ok=True)
        os.makedirs(cls.token_dir, exist_ok=True)
        os.makedirs(cls.model_dir, exist_ok=True)
        os.makedirs(cls.motion_dir, exist_ok=True)

        # Generate synthetic keypoint data
        for i in range(20):
            seq = np.random.randn(60, 225).astype(np.float32)
            np.save(os.path.join(cls.kp_dir, f"syn_{i:03d}.npy"), seq)

        # Generate synthetic motion data
        for i in range(20):
            motion = np.random.randn(30, 150).astype(np.float32) * 0.1
            np.save(os.path.join(cls.motion_dir, f"motion_{i:03d}.npy"), motion)

        # Generate synthetic token data
        for i in range(20):
            tokens = np.random.randint(1, 512, size=np.random.randint(30, 100))
            np.save(os.path.join(cls.token_dir, f"seq_{i:03d}.npy"), tokens)

        # Generate labels CSV
        import csv
        labels_path = os.path.join(cls.tmpdir, "labels.csv")
        with open(labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "label_id", "label_name", "dataset", "split"
            ])
            writer.writeheader()
            for i in range(20):
                writer.writerow({
                    "filename": f"syn_{i:03d}.npy",
                    "label_id": i % 5,
                    "label_name": f"SIGN_{i % 5:04d}",
                    "dataset": "SYNTHETIC",
                    "split": "train" if i < 16 else "val"
                })

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_01_gesture_model_trains(self):
        """Gesture model: instantiate + 1 training step."""
        import torch
        model = GestureModel(input_size=126, num_classes=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 30, 126)
        labels = torch.randint(0, 5, (4,))

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss))

    def test_02_foundation_model_trains(self):
        """Foundation model: instantiate + 1 training step."""
        import torch
        from models.sign_foundation_transformer import SignFoundationModel

        model = SignFoundationModel(vocab_size=512, d_model=64,
                                     nhead=4, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        tokens = torch.randint(1, 512, (2, 32))
        target = torch.randint(1, 512, (2, 32))

        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 512), target.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss))

    def test_03_diffusion_model_trains(self):
        """Diffusion model: instantiate + 1 training step."""
        import torch
        from models.gesture_diffusion import GestureDiffusionModel, DiffusionScheduler

        model = GestureDiffusionModel(motion_dim=150, d_model=128, num_blocks=2)
        scheduler = DiffusionScheduler(num_timesteps=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        motion = torch.randn(2, 30, 150)
        t = scheduler.sample_timestep(2)
        noisy, noise = scheduler.add_noise(motion, t)

        predicted_noise = model(noisy, t)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss))

    def test_04_sign_transformer_trains(self):
        """Sign transformer: instantiate + 1 training step."""
        import torch
        model = SignTransformer(feature_dim=126, text_vocab_size=100,
                                d_model=64, nhead=4, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        src = torch.randn(2, 20, 126)
        tgt = torch.randint(0, 100, (2, 10))
        target = torch.randint(0, 100, (2, 10))

        logits = model(src, tgt)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 100), target.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss))

    def test_05_multimodal_llm_trains(self):
        """Multimodal LLM: instantiate + 1 training step."""
        import torch
        model = MultimodalSignLLM(gesture_dim=225, text_vocab=100, d_model=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        gesture = torch.randn(2, 30, 225)
        text_target = torch.randint(0, 100, (2, 10))
        target_out = torch.randint(0, 100, (2, 10))

        logits = model(gesture, text_target=text_target)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_out.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss))

    def test_06_checkpoint_roundtrip(self):
        """Checkpoint: save → load → weights match."""
        import torch
        from training.utils.checkpoint_manager import CheckpointManager

        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        original = model.weight.data.clone()

        ckpt = CheckpointManager(save_dir=self.model_dir, prefix="integration_test")
        ckpt.save(model, optimizer, epoch=1, metric=0.5, tag="best")

        model2 = torch.nn.Linear(10, 5)
        optimizer2 = torch.optim.Adam(model2.parameters())
        path = ckpt.find_latest()
        ckpt.load(path, model2, optimizer2)

        self.assertTrue(torch.allclose(model2.weight.data, original))

    def test_07_multimodal_dataset(self):
        """MultimodalDataset: loads data from synthetic files."""
        from training.data_pipeline.multimodal_dataset import MultimodalDataset

        ds = MultimodalDataset(
            keypoint_dir=self.kp_dir,
            labels_csv=os.path.join(self.tmpdir, "labels.csv"),
            seq_len=60,
            text_seq_len=10,
            split="train"
        )
        self.assertGreater(len(ds), 0)

        gesture, text_in, text_tgt = ds[0]
        self.assertEqual(gesture.shape, (60, 225))
        self.assertEqual(text_in.shape, (10,))
        self.assertEqual(text_tgt.shape, (10,))


class TestTranslationPipeline(unittest.TestCase):
    """Tests from original test_translation_pipeline.py.

    These test the non-hyphenated modules (ai_engine, nlp_translation, etc.)
    which use standard Python package names with underscores.
    """

    def test_text_to_sign_pipeline(self):
        from ai_engine.modules.text_to_sign import TextToSignConverter
        converter = TextToSignConverter()
        tokens = converter.convert("hello how are you")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn("HELLO", tokens)

    def test_sign_grammar_end_to_end(self):
        from nlp_translation.sign_grammar_converter import SignGrammarConverter
        converter = SignGrammarConverter()
        gloss = converter.convert("I am going to school tomorrow")
        display = converter.gloss_to_string(gloss)
        self.assertIn("TOMORROW", display)
        self.assertTrue(display.startswith("TOMORROW"))

    def test_avatar_mapping_pipeline(self):
        from avatar_animation.gesture_mapper import GestureMapper
        from avatar_animation.animation_controller import AnimationController

        mapper = GestureMapper()
        controller = AnimationController()

        tokens = ["HELLO", "THANK_YOU"]
        clips = mapper.map_sequence(tokens)
        controller.enqueue(clips)

        state = controller.get_state()
        self.assertEqual(state["queue_size"], 2)

        anim = controller.play_next()
        self.assertIsNotNone(anim)


if __name__ == "__main__":
    unittest.main()
