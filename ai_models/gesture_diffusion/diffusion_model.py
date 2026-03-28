"""
Gesture Diffusion Model — ai-models layer

Wraps the core diffusion model for use inside the ai-models pipeline.
Text → Sign Tokens → Diffusion → 3D Motion → Avatar Animation.
"""

import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from models.gesture_diffusion import GestureDiffusionModel, DiffusionScheduler


class GestureDiffusionPipeline:
    """
    High-level diffusion pipeline:
        Sign tokens → 3D motion sequence (seq_len × motion_dim)
    """

    def __init__(self, model_path=None, motion_dim=150, num_timesteps=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.motion_dim = motion_dim

        self.model = GestureDiffusionModel(motion_dim=motion_dim).to(self.device)

        if model_path and os.path.exists(model_path):
            payload = torch.load(model_path, map_location=self.device)
            state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
            try:
                self.model.load_state_dict(state)
            except Exception:
                # Skip any shape-mismatched tensors to keep inference usable.
                if isinstance(state, dict):
                    cur = self.model.state_dict()
                    filtered = {}
                    for k, v in state.items():
                        if k not in cur:
                            continue
                        try:
                            if hasattr(v, "shape") and hasattr(cur[k], "shape") and v.shape != cur[k].shape:
                                continue
                        except Exception:
                            pass
                        filtered[k] = v
                    self.model.load_state_dict(filtered, strict=False)

        self.model.eval()
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps)

    def generate(self, gesture_tokens=None, seq_len=30, num_samples=1):
        """
        Generate motion sequences from gesture tokens.

        Args:
            gesture_tokens: (1, token_len) LongTensor for conditioning
            seq_len: output motion sequence length (frames)
            num_samples: number of sequences to generate

        Returns:
            numpy array (num_samples, seq_len, motion_dim)
        """
        if gesture_tokens is not None:
            gesture_tokens = gesture_tokens.to(self.device)

        motion = self.scheduler.generate(
            self.model,
            shape=(num_samples, seq_len, self.motion_dim),
            gesture_tokens=gesture_tokens,
            device=self.device
        )

        return motion.cpu().numpy()
