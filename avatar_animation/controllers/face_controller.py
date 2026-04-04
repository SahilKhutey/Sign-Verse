"""
Face Controller

Generates simple facial-grammar signals (blendshape-like parameters) from tokens.

This is a stub that can be replaced by a learned facial expression model later.
"""

from __future__ import annotations

from typing import Dict, List, Any


class FaceController:
    def __init__(self):
        # Minimal mapping; expand as you add a real face rig.
        self._token_to_face = {
            "QUESTION": {"brow_raise": 1.0, "head_tilt": 0.4},
            "NEGATION": {"brow_furrow": 0.7, "head_shake": 0.6},
            "EMPHASIS": {"eye_widen": 0.5},
        }

    def facial_params(self, tokens: List[str]) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for t in tokens:
            face = self._token_to_face.get(t)
            if not face:
                continue
            for k, v in face.items():
                params[k] = max(float(v), float(params.get(k, 0.0)))
        return params

    def state(self) -> Dict[str, Any]:
        return {"known_face_tokens": sorted(self._token_to_face.keys())}
