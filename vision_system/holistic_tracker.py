"""
Holistic Tracker — High Fidelity Pose + Hand + Face Tracking
Powered by MediaPipe Holistic (543 landmarks).
"""

import cv2
import mediapipe as mp
import numpy as np
import time

try:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except ImportError:
    try:
        import mediapipe as mp
        mp_holistic = getattr(mp.solutions, "holistic", None)
        mp_drawing = getattr(mp.solutions, "drawing_utils", None)
        mp_drawing_styles = getattr(mp.solutions, "drawing_styles", None)
    except (ImportError, AttributeError):
        mp_holistic = None
        mp_drawing = None
        mp_drawing_styles = None

class HolisticTracker:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if mp_holistic is None:
            print("Warning: MediaPipe Holistic not found. Using mock tracker.")
            self.use_mock = True
            self.mp_holistic = None
            self.mp_draw = None
            return

        try:
            self.mp_holistic = mp_holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_draw = mp_drawing
            self.mp_drawing_styles = mp_drawing_styles
            self.use_mock = False
        except Exception as e:
            print(f"Warning: MediaPipe Holistic initialization failed ({e}). Using mock tracker.")
            self.use_mock = True
            self.mp_holistic = None
            self.mp_draw = None

    def process(self, frame):
        if self.use_mock:
            self.last_process_time = 0.005 # 5ms mock latency
            return {
                "pose": None, "face": None, "left_hand": None, "right_hand": None,
                "latency_ms": 5
            }

        start = time.time()
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        self.last_process_time = time.time() - start

        return {
            "pose": results.pose_landmarks,
            "face": results.face_landmarks,
            "left_hand": results.left_hand_landmarks,
            "right_hand": results.right_hand_landmarks,
            "latency_ms": int(self.last_process_time * 1000)
        }

    def landmarks_to_numpy(self, landmarks, count):
        """Converts MediaPipe landmarks to (count, 3) numpy array."""
        if landmarks is None:
            return np.zeros((count, 3))
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    def get_full_vectors(self, results):
        """Extracts all landmarks into a flat dictionary of numpy arrays."""
        if self.use_mock:
            return {
                "pose": np.random.randn(33, 3) * 0.01,
                "face": np.random.randn(468, 3) * 0.01,
                "left_hand": np.random.randn(21, 3) * 0.01,
                "right_hand": np.random.randn(21, 3) * 0.01
            }
        return {
            "pose": self.landmarks_to_numpy(results["pose"], 33),
            "face": self.landmarks_to_numpy(results["face"], 468),
            "left_hand": self.landmarks_to_numpy(results["left_hand"], 21),
            "right_hand": self.landmarks_to_numpy(results["right_hand"], 21)
        }

    def draw(self, frame, results):
        """Draws tracked landmarks onto the frame for debugging."""
        if self.use_mock or not self.mp_draw:
            # Draw a simple 'Mock Active' text
            cv2.putText(frame, "PERCEPTION MOCK ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame
            
        annotated_frame = frame.copy()
        # Face
        self.mp_draw.draw_landmarks(
            annotated_frame, results["face"], self.mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        # Pose
        self.mp_draw.draw_landmarks(
            annotated_frame, results["pose"], self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        # Hands
        self.mp_draw.draw_landmarks(annotated_frame, results["left_hand"], self.mp_holistic.HAND_CONNECTIONS)
        self.mp_draw.draw_landmarks(annotated_frame, results["right_hand"], self.mp_holistic.HAND_CONNECTIONS)
        return annotated_frame
