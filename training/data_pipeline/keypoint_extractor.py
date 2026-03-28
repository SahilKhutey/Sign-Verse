"""
Keypoint Extractor — Extract skeleton keypoints from video frames.

Uses MediaPipe Hands to extract 21 landmarks per hand.
Output: 63 features per frame (21 landmarks × x, y, z).

Modern sign recognition systems use skeleton keypoints instead of
raw video for better generalization and lower compute cost.
"""

import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands.Hands()


def extract_keypoints(frame):
    """
    Extract hand keypoints from a single frame.

    Args:
        frame: BGR image (numpy array)

    Returns:
        numpy array of keypoints (63 values) or None if no hand detected
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = mp_hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    keypoints = []

    for hand in result.multi_hand_landmarks:
        for lm in hand.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

    return np.array(keypoints)


def extract_video_keypoints(video_path):
    """
    Extract keypoints from all frames of a video.

    Args:
        video_path: Path to video file

    Returns:
        List of numpy arrays (one per frame with detected hands)
    """

    cap = cv2.VideoCapture(video_path)

    all_keypoints = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        kp = extract_keypoints(frame)

        if kp is not None:
            all_keypoints.append(kp)

    cap.release()

    return all_keypoints
