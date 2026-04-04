import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any

class MediaPipePoseExtractor:
    def __init__(self):
        """Initialize the MediaPipe Pose solution."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def extract_from_video(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process a video file and extract pose landmarks for each frame.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            List[Dict]: A list of dictionaries containing frame number and landmarks.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results_list = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb_frame)

            frame_data = {
                "frame_number": frame_count,
                "landmarks": None
            }

            if result.pose_landmarks:
                # Convert landmarks to a serializable list of dictionaries
                frame_data["landmarks"] = []
                for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    frame_data["landmarks"].append({
                        "id": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

            results_list.append(frame_data)
            frame_count += 1

        cap.release()
        return results_list

    def release(self):
        """Release the MediaPipe Pose model resources."""
        self.pose.close()
