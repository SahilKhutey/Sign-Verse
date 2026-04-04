"""
Conv3D Research Training Entrypoint (clean-room implementation).

Usage:
  python training/train_research_conv3d.py --epochs 20
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from ai_models.gesture_recognition.train_conv3d_research import main as _main

    _main()


if __name__ == "__main__":
    main()
