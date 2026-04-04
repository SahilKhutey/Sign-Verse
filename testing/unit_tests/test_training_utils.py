"""
Unit Tests — Training Utilities

Tests for:
    - CheckpointManager: save, load, find_latest, prune
    - TensorBoardLogger: CSV fallback, metric logging
    - BaseTrainer: AMP disabled on CPU, device resolution
"""

import sys
import os
import unittest
import tempfile
import shutil
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestCheckpointManager(unittest.TestCase):
    """Tests for checkpoint save/load/resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        """Save a checkpoint and load it back — weights should match."""
        import torch
        import torch.nn as nn
        from training.utils.checkpoint_manager import CheckpointManager

        ckpt = CheckpointManager(save_dir=self.tmpdir, prefix="test_model")

        # Create a simple model
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        original_weight = model.weight.data.clone()

        # Save
        path = ckpt.save(model, optimizer, epoch=5, metric=0.42, tag="best")
        self.assertTrue(os.path.exists(path))

        # Check metadata sidecar
        meta_path = path.replace(".pt", "_meta.json")
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["epoch"], 5)
        self.assertAlmostEqual(meta["best_metric"], 0.42, places=2)

        # Load into a fresh model
        model2 = nn.Linear(10, 5)
        optimizer2 = torch.optim.Adam(model2.parameters())
        epoch, metric = ckpt.load(path, model2, optimizer2)

        self.assertEqual(epoch, 5)
        self.assertAlmostEqual(metric, 0.42, places=2)
        self.assertTrue(torch.allclose(model2.weight.data, original_weight))

    def test_find_latest_best(self):
        """find_latest() should prefer 'best' checkpoint."""
        import torch
        import torch.nn as nn
        from training.utils.checkpoint_manager import CheckpointManager

        ckpt = CheckpointManager(save_dir=self.tmpdir, prefix="test_model")
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt.save(model, optimizer, epoch=10, metric=0.5, tag="epoch10")
        ckpt.save(model, optimizer, epoch=20, metric=0.3, tag="best")

        latest = ckpt.find_latest()
        self.assertIsNotNone(latest)
        self.assertIn("best", latest)

    def test_find_latest_epoch_fallback(self):
        """find_latest() falls back to highest epoch if no 'best'."""
        import torch
        import torch.nn as nn
        from training.utils.checkpoint_manager import CheckpointManager

        ckpt = CheckpointManager(save_dir=self.tmpdir, prefix="test_model")
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt.save(model, optimizer, epoch=10, metric=0.5, tag="epoch10")
        ckpt.save(model, optimizer, epoch=20, metric=0.3, tag="epoch20")

        latest = ckpt.find_latest()
        self.assertIsNotNone(latest)
        self.assertIn("epoch20", latest)

    def test_list_checkpoints(self):
        """list_checkpoints() returns all checkpoints."""
        import torch
        import torch.nn as nn
        from training.utils.checkpoint_manager import CheckpointManager

        ckpt = CheckpointManager(save_dir=self.tmpdir, prefix="test_model")
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt.save(model, optimizer, epoch=10, metric=0.5, tag="epoch10")
        ckpt.save(model, optimizer, epoch=20, metric=0.3, tag="best")

        paths = ckpt.list_checkpoints()
        self.assertEqual(len(paths), 2)


class TestTensorBoardLogger(unittest.TestCase):
    """Tests for TensorBoard/CSV logging."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_csv_fallback(self):
        """Logger should write metrics to CSV even without TensorBoard."""
        from training.utils.tensorboard_logger import TensorBoardLogger

        logger = TensorBoardLogger(log_dir=self.tmpdir)
        logger.log_metrics({"train/loss": 0.42, "train/acc": 91.2}, step=10)
        logger.log_metrics({"train/loss": 0.35}, step=20)
        logger.close()

        csv_path = os.path.join(self.tmpdir, "metrics.csv")
        self.assertTrue(os.path.exists(csv_path))

        # Read rows back
        rows = logger.read_metrics(key_filter="train/loss")
        self.assertGreaterEqual(len(rows), 2)

    def test_summary(self):
        """summary() should return latest metric values."""
        from training.utils.tensorboard_logger import TensorBoardLogger

        logger = TensorBoardLogger(log_dir=self.tmpdir)
        logger.log_metrics({"train/loss": 0.42}, step=1)
        logger.log_metrics({"train/loss": 0.30}, step=2)
        logger.close()

        summary = logger.summary()
        self.assertIn("train/loss", summary)


class TestBaseTrainerConfig(unittest.TestCase):
    """Tests for BaseTrainer configuration and device handling."""

    def test_device_resolution_cpu(self):
        """_resolve_device('cpu') should return torch.device('cpu')."""
        import torch
        from training.utils.trainer_base import BaseTrainer

        device = BaseTrainer._resolve_device("cpu")
        self.assertEqual(device, torch.device("cpu"))

    def test_device_resolution_auto(self):
        """_resolve_device('auto') should not crash."""
        import torch
        from training.utils.trainer_base import BaseTrainer

        device = BaseTrainer._resolve_device("auto")
        self.assertIsInstance(device, torch.device)

    def test_amp_disabled_on_cpu(self):
        """AMP should be disabled when device is CPU."""
        from training.utils.trainer_base import BaseTrainer

        config = {"device": "cpu", "mixed_precision": True, "epochs": 1}
        trainer = BaseTrainer(config)
        self.assertFalse(trainer.use_amp)


if __name__ == "__main__":
    unittest.main()
