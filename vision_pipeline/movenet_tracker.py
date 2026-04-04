"""
MoveNet Tracker for Edge & AR Tracking

Lightning fast, extremely lightweight pose estimation (17 keypoints).
Optimized for mobile integration, offering bridging for the AR endpoints.
"""

import cv2
import numpy as np
import logging

try:
    import tensorflow as tf
except ImportError:
    tf = None
    logging.warning("TensorFlow not found. Falling back to Mock MoveNet Tracker.")

class MoveNetTracker:
    """
    MoveNet SinglePose Tracker implementation.
    Standardized output: 17x3 array representing COCO-style body keypoints.
    """
    def __init__(self, use_thunder=False):
        self.mock_mode = (tf is None)
        self.model = None
        self.input_size = 256 if use_thunder else 192 # Thunder=256, Lightning=192
        
        if not self.mock_mode:
            try:
                import tensorflow_hub as hub
                model_name = "movenet/singlepose/thunder/4" if use_thunder else "movenet/singlepose/lightning/4"
                url = f"https://tfhub.dev/google/{model_name}"
                self.model = hub.load(url).signatures['serving_default']
                logging.info(f"Loaded MoveNet: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to load MoveNet from TF Hub: {e}. Enabling Mock Mode.")
                self.mock_mode = True

    def process_frame(self, frame):
        """
        Process a BGR frame through MoveNet.
        Returns: 
           keypoints: [17, 3] where values are [x(absolute), y(absolute), confidence].
        """
        if self.mock_mode:
            # Random mock shape [17, 3]
            return np.random.rand(17, 3).astype(np.float32) * 100.0
            
        # MoveNet requires RGB image of shape [1, width, height, 3] of dtype int32
        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = tf.cast(image, dtype=tf.int32)
        image = tf.expand_dims(image, axis=0)
        
        # Inference
        outputs = self.model(image)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0] # Output shape [17, 3]
        
        # MoveNet normally returns [y_norm, x_norm, confidence]
        structured_kp = np.zeros((17, 3), dtype=np.float32)
        
        for i in range(17):
            y_norm, x_norm, conf = keypoints_with_scores[i]
            structured_kp[i] = [x_norm * w, y_norm * h, conf]
            
        return structured_kp

    def convert_to_foundation_schema(self, kp_17):
        """
        Converts 17 COCO keypoints into the 543-keypoint standard Foundation Schema (1629 dims).
        Since MoveNet lacks Hands and Face, they remain zero.
        MoveNet -> MediaPipe Pose roughly matches:
        MN(0) Nose -> MP(0) Nose
        MN(1,2) Eyes -> MP(2,5) Eyes
        MN(5,6) Shoulders -> MP(11,12) Shoulders
        MN(7,8) Elbows -> MP(13,14) Elbows
        MN(9,10) Wrists -> MP(15,16) Wrists
        MN(11,12) Hips -> MP(23,24) Hips
        MN(13,14) Knees -> MP(25,26) Knees
        MN(15,16) Ankles -> MP(27,28) Ankles
        """
        mp_flat = np.zeros(1629, dtype=np.float32)
        map_idx = {
            0: 0, 1: 2, 2: 5, 3: 7, 4: 8, # Face
            5: 11, 6: 12, # Shoulders
            7: 13, 8: 14, # Elbows
            9: 15, 10: 16, # Wrists
            11: 23, 12: 24, # Hips
            13: 25, 14: 26, # Knees
            15: 27, 16: 28  # Ankles
        }
        
        for mn_idx, mp_idx in map_idx.items():
            start = mp_idx * 3
            mp_flat[start:start+3] = kp_17[mn_idx]
            
        return mp_flat
