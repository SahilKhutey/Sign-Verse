"""
Body Controller

Applies simple post-processing to body joints before they are sent to Unity.

This is intentionally lightweight: it makes it easy to add constraints,
smoothing, and retargeting rules later without rewriting the streaming API.
"""

from __future__ import annotations

from typing import Dict, Any


class BodyController:
    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def apply(self, joints: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Args:
            joints: mapping of joint_name -> {"x":..,"y":..,"z":..}
        Returns:
            Updated joints (scaled).
        """
        if self.scale == 1.0:
            return joints

        out: Dict[str, Dict[str, float]] = {}
        for name, xyz in joints.items():
            out[name] = {
                "x": float(xyz.get("x", 0.0)) * self.scale,
                "y": float(xyz.get("y", 0.0)) * self.scale,
                "z": float(xyz.get("z", 0.0)) * self.scale,
            }
        return out

    def state(self) -> Dict[str, Any]:
        return {"scale": self.scale}
