"""
Letter-level finger-spelling decoder.

Converts noisy per-frame alphabet labels into stable characters and assembled text.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Optional


class FingerSpellingDecoder:
    def __init__(
        self,
        window: int = 8,
        min_majority: int = 5,
        cooldown_frames: int = 4,
        blank_reset_frames: int = 3,
        min_confidence: float = 0.4,
        max_text_chars: int = 256,
    ):
        self.window = max(3, int(window))
        self.min_majority = max(2, int(min_majority))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.blank_reset_frames = max(1, int(blank_reset_frames))
        self.min_confidence = float(min_confidence)
        self.max_text_chars = max(32, int(max_text_chars))

        self.buffer: Deque[Optional[str]] = deque(maxlen=self.window)
        self.text = ""
        self._cooldown = 0
        self._last_committed_label: Optional[str] = None
        self._blank_streak = 0

    def clear(self):
        self.buffer.clear()
        self.text = ""
        self._cooldown = 0
        self._last_committed_label = None
        self._blank_streak = 0

    def backspace(self):
        if self.text:
            self.text = self.text[:-1]

    @staticmethod
    def _normalize_label(label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        lab = str(label).strip().upper()
        if not lab:
            return None
        return lab

    def _majority_label(self) -> Optional[str]:
        valid = [x for x in self.buffer if x]
        if not valid:
            return None
        best, count = Counter(valid).most_common(1)[0]
        if count < self.min_majority:
            return None
        return best

    def _apply_commit(self, label: str) -> Optional[str]:
        committed = None
        if label in {"DEL", "DELETE", "BACKSPACE"}:
            self.backspace()
            committed = "<BACKSPACE>"
        elif label in {"SPACE", "<SPACE>"}:
            if self.text and not self.text.endswith(" "):
                self.text += " "
                committed = " "
        elif label in {"CLEAR", "<CLEAR>"}:
            self.text = ""
            committed = "<CLEAR>"
        elif len(label) == 1 and label.isalpha():
            if len(self.text) < self.max_text_chars:
                self.text += label
                committed = label
        return committed

    def update(self, label: Optional[str], confidence: Optional[float] = None) -> Dict[str, Optional[str] | str]:
        conf = float(confidence) if confidence is not None else 0.0
        normalized = self._normalize_label(label)
        if normalized is None or conf < self.min_confidence:
            normalized = None

        self.buffer.append(normalized)

        if normalized is None:
            self._blank_streak += 1
            if self._cooldown > 0:
                self._cooldown -= 1
            if self._blank_streak >= self.blank_reset_frames:
                self._last_committed_label = None
        else:
            self._blank_streak = 0

        stable = self._majority_label()
        committed = None
        if stable and self._cooldown == 0 and stable != self._last_committed_label:
            committed = self._apply_commit(stable)
            if committed is not None:
                self._last_committed_label = stable
                self._cooldown = self.cooldown_frames

        return {
            "stable_label": stable,
            "committed": committed,
            "text": self.text,
        }
