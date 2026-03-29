"""
Vision Pipeline — Face Tracking Module

Real-time face detection and keypoint extraction.
Uses MediaPipe Face Mesh for 468-landmark tracking.
"""

import cv2
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import numpy as np


class FaceTracker:
    """Tracks up to 1 face — 468 landmarks × 3 coords = 1404 features."""

    def __init__(self, max_faces=1, detection_confidence=0.5,
                 tracking_confidence=0.5):
        self.mp_face_mesh = mp_face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=False, # standard is 468
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp_drawing
        self.max_faces = max_faces

    def process(self, frame):
        """
        Detect faces in frame.

        Returns:
            keypoints: numpy (1404,) — zeros if no faces detected
            result: full MediaPipe result for visualization
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        keypoints = np.zeros(self.max_faces * 1404)

        if result.multi_face_landmarks:
            for i, face in enumerate(result.multi_face_landmarks[:self.max_faces]):
                offset = i * 1404
                for j, lm in enumerate(face.landmark):
                    if j >= 468:
                        break # Only use the base 468 points if refine is somehow on
                    keypoints[offset + j*3] = lm.x
                    keypoints[offset + j*3 + 1] = lm.y
                    keypoints[offset + j*3 + 2] = lm.z

        return keypoints, result

    def draw(self, frame, result):
        """Draw face mesh on frame."""
        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        return frame
