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
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

    def extract_pose(self, frame):
        """
        Extract full body, face mesh, and hand keypoints using MediaPipe Holistic.
        Returns a flat numpy array of 543 Landmarks (1629 values).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        keypoints = []

        # 1. Body Pose (33 Landmarks: 0-98 values)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)

        # 2. Face Mesh (468 Landmarks: 99-1502 values)
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 1404)

        # 3. Left Hand (21 Landmarks: 1503-1565 values)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)

        # 4. Right Hand (21 Landmarks: 1566-1628 values)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)

        return np.array(keypoints)  # 1629 values (99 + 1404 + 63 + 63)

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
        results = self.holistic.process(rgb)

        if draw:
            # 1. Face Mesh
            if results.face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                )
            
            # 2. Body Pose
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # 3. Left Hand
            if results.left_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )

            # 4. Right Hand
            if results.right_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
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
