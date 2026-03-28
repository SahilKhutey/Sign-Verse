"""
Dataset Builder — Pose Label Generator

Extracts MediaPipe keypoints from image frames,
saves as .npy files with metadata for training.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json


def generate_pose_labels(frames_dir, output_dir):
    """
    Process all frames in a directory and generate pose keypoints.

    Args:
        frames_dir:  directory of frame images (.jpg/.png)
        output_dir:  directory to save .npy keypoint files
    """
    os.makedirs(output_dir, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )

    metadata = []
    processed = 0

    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        kp = []

        # Body (99)
        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp.extend([0.0] * 99)

        # Left hand (63)
        if result.left_hand_landmarks:
            for lm in result.left_hand_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp.extend([0.0] * 63)

        # Right hand (63)
        if result.right_hand_landmarks:
            for lm in result.right_hand_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp.extend([0.0] * 63)

        arr = np.array(kp, dtype=np.float32)
        npy_name = fname.replace('.jpg', '.npy').replace('.png', '.npy')
        np.save(os.path.join(output_dir, npy_name), arr)
        metadata.append({"frame": fname, "keypoints_file": npy_name, "dim": len(arr)})
        processed += 1

    holistic.close()

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Processed {processed} frames → {output_dir}")
    return processed
