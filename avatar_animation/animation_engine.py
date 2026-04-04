"""
Animation Engine

Connects:
  sign tokens -> animation clips (rule-based mapping)
  generated motion vectors -> skeleton frames for Unity

The goal is to give the system a concrete end-to-end path today, while keeping
interfaces stable for future learned retargeting and facial models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from avatar_animation.gesture_mapper import GestureMapper
from avatar_animation.animation_controller import AnimationController
from avatar_animation.motion_exporter import MotionExporter
from avatar_animation.motion_interpolator import MotionInterpolator
from avatar_animation.controllers.body_controller import BodyController
from avatar_animation.controllers.face_controller import FaceController


class AnimationEngine:
    def __init__(self, fps: int = 30):
        self.fps = int(fps)
        self.mapper = GestureMapper()
        self.controller = AnimationController()
        self.exporter = MotionExporter()
        self.interpolator = MotionInterpolator(fps=self.fps)
        self.body = BodyController(scale=1.0)
        self.face = FaceController()

    def tokens_to_clips(self, tokens: List[str]) -> List[str]:
        """Convert tokens to animation clip IDs (Unity Animator states)."""
        return self.mapper.map_sequence(tokens)

    def enqueue_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """Queue animations for playback in Unity."""
        clips = self.tokens_to_clips(tokens)
        self.controller.enqueue(clips)
        return {"clips": clips, "queued": len(clips)}

    def motion_to_stream(self, motion_sequence: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert a motion sequence (frames x motion_dim) into Unity stream packets.
        """
        return self.exporter.to_unity_stream(motion_sequence, fps=self.fps)

    def build_unity_payload(
        self,
        tokens: List[str],
        motion_sequence: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Build a single payload Unity can consume.
        """
        clips = self.tokens_to_clips(tokens)
        payload: Dict[str, Any] = {
            "type": "sign_sequence",
            "tokens": tokens,
            "clips": clips,
            "fps": self.fps,
            "face": self.face.facial_params(tokens),
        }

        if motion_sequence is not None:
            payload["motion"] = self.motion_to_stream(motion_sequence)

        return payload

    def state(self) -> Dict[str, Any]:
        return {
            "fps": self.fps,
            "controller": self.controller.get_state(),
            "body": self.body.state(),
            "face": self.face.state(),
        }
