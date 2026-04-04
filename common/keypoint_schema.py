"""
Keypoint Schema Constants

Canonical 225-dim per-frame layout used by the unified preprocessor and
real-time feature extractor:
  [body(99), left_hand(63), right_hand(63)]

Notes:
- "Left/Right hand" refer to signer perspective. If a tracker cannot reliably
  assign handedness, use a stable ordering (e.g., first-detected then second).
"""

COORDS = 3

BODY_LANDMARKS = 33
HAND_LANDMARKS = 21

BODY_DIM = BODY_LANDMARKS * COORDS  # 99
ONE_HAND_DIM = HAND_LANDMARKS * COORDS  # 63
TWO_HANDS_DIM = 2 * ONE_HAND_DIM  # 126

FEATURE_DIM_225 = BODY_DIM + TWO_HANDS_DIM  # 225

CANONICAL_ORDER = ("body", "left_hand", "right_hand")

# Canonical slices for FEATURE_DIM_225 = [body(99), left_hand(63), right_hand(63)]
BODY_SLICE_225 = slice(0, BODY_DIM)
LEFT_HAND_SLICE_225 = slice(BODY_DIM, BODY_DIM + ONE_HAND_DIM)
RIGHT_HAND_SLICE_225 = slice(BODY_DIM + ONE_HAND_DIM, FEATURE_DIM_225)
HANDS_SLICE_225 = slice(BODY_DIM, FEATURE_DIM_225)
