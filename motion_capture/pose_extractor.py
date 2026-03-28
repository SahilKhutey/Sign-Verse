"""
Neural Motion Capture — Full Body Pose Extraction

Extracts body + hand keypoints from video frames for sign language.

Features per frame:
    Part        Points
    Hands       21 × 2 = 42 points (126 values)
    Body        33 points (99 values)
    Face        468 points (1404 values)
    Total       ~500-600 keypoints

Output:
    pose_sequence = [frame1_keypoints, frame2_keypoints, ...]
"""

import mediapipe as mp
import cv2
import numpy as np
import os
import json


class NeuralMotionCapture:

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_pose(self, frame):
        """
        Extract body and hand keypoints from a single frame.
        Returns flat numpy array of all keypoint coordinates.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result = self.pose.process(rgb)
        hand_result = self.hands.process(rgb)

        keypoints = []

        # Body pose: 33 landmarks × 3 = 99 values
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)

        # Hands: up to 2 × 21 landmarks × 3 = 126 values
        hand_count = 0
        if hand_result.multi_hand_landmarks:
            for hand in hand_result.multi_hand_landmarks:
                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                hand_count += 1

        # Pad if fewer than 2 hands detected
        while hand_count < 2:
            keypoints.extend([0.0] * 63)
            hand_count += 1

        return np.array(keypoints)  # 225 values (99 + 63 + 63)

    def extract_video(self, video_path):
        """
        Extract pose sequence from entire video.

        Returns:
            List of numpy arrays, one per frame
        """
        cap = cv2.VideoCapture(video_path)
        pose_sequence = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = self.extract_pose(frame)
            pose_sequence.append(keypoints)
            frame_id += 1

        cap.release()
        print(f"Extracted {frame_id} frames from {video_path}")
        return pose_sequence

    def extract_and_save(self, video_path, output_path):
        """Extract pose sequence and save as numpy file."""
        sequence = self.extract_video(video_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, np.array(sequence))
        print(f"Saved pose sequence to {output_path}")
        return sequence

    def visualize(self, frame, draw=True):
        """Extract pose and optionally draw landmarks on frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(rgb)
        hand_result = self.hands.process(rgb)

        if draw:
            if pose_result.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, pose_result.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
            if hand_result.multi_hand_landmarks:
                for hand in hand_result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )

        return frame

    def run_live(self):
        """Real-time motion capture from webcam with visualization."""
        cap = cv2.VideoCapture(0)
        frame_count = 0

        print("Neural Motion Capture — Press ESC to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = self.extract_pose(frame)
            frame = self.visualize(frame)

            info = f"Keypoints: {len(keypoints)} | Frame: {frame_count}"
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Motion Capture", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture = NeuralMotionCapture()
    capture.run_live()
