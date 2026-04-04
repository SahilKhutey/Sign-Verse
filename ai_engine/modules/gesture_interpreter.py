"""
Gesture Interpreter

Converts low-level gesture IDs/tokens into readable gloss strings.
This is a placeholder for a real label-map / gloss lexicon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GestureInterpreter:
    label_map: Optional[Dict[int, str]] = None

    def interpret_id(self, gesture_id: int) -> str:
        if self.label_map and int(gesture_id) in self.label_map:
            return self.label_map[int(gesture_id)]
        return f"G{int(gesture_id):04d}"
