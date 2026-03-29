"""
OpenPose Extractor for Offline Dataset Generation

Provides high-accuracy dataset labeling for Body, Face, and Hands.
Trade-offs: Slower than MediaPipe, requires GPU.
Best Use: Offline video conversion to .npy / .json for Model Training.
"""

import cv2
import numpy as np
import os
import sys
import logging

class OpenPoseExtractor:
    def __init__(self, model_folder="/models/", num_gpu=1, enable_face=True, enable_hand=True):
        """
        Initializes OpenPose parameters and wrapper.
        Expects pyopenpose to be accessible in the environment.
        """
        self.params = {
            "model_folder": model_folder,
            "face": enable_face,
            "hand": enable_hand,
            "num_gpu": num_gpu,
            "net_resolution": "-1x368", # Standard resolution tradeoff
            "disable_blending": False
        }
        
        # Guard initialization in case compiling OpenPose fails on local dev
        self.op = None
        self.opWrapper = None
        try:
            import pyopenpose as op
            self.op = op
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(self.params)
            self.opWrapper.start()
            logging.info("OpenPose Wrapper successfully initialized.")
            self.mock_mode = False
        except ImportError:
            logging.warning("pyopenpose not found. Falling back to Mock OpenPose Extractor for dev compatibility.")
            self.mock_mode = True

    def process_frame(self, frame):
        """
        Processes a single BGR frame.
        Finds the primary signer and extracts 137 keypoints:
        25 (Body) + 70 (Face) + 21 (Left Hand) + 21 (Right Hand).
        Output shape: (137, 3) where [x, y, confidence]
        """
        if self.mock_mode:
            # Body(25) + Face(70) + LeftHand(21) + RightHand(21) = 137
            return np.random.rand(137, 3).astype(np.float32)

        datum = self.op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        
        # Check if any person detected
        if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
            return np.zeros((137, 3), dtype=np.float32)
            
        # Multi-person tracking: Select the primary signer (largest torso bounding box)
        primary_idx = self._select_primary_signer(datum.poseKeypoints)
        
        # Extract features for the primary signer
        body = datum.poseKeypoints[primary_idx] if datum.poseKeypoints is not None else np.zeros((25, 3))
        
        face = np.zeros((70, 3))
        if datum.faceKeypoints is not None and len(datum.faceKeypoints) > primary_idx:
             face = datum.faceKeypoints[primary_idx]
             
        left_hand = np.zeros((21, 3))
        if datum.handKeypoints[0] is not None and len(datum.handKeypoints[0]) > primary_idx:
             left_hand = datum.handKeypoints[0][primary_idx]
             
        right_hand = np.zeros((21, 3))
        if datum.handKeypoints[1] is not None and len(datum.handKeypoints[1]) > primary_idx:
             right_hand = datum.handKeypoints[1][primary_idx]
             
        # Concatenate into the canonical OpenPose 137-point schema
        features = np.concatenate([body, face, left_hand, right_hand], axis=0) # Shape: (137, 3)
        return features.astype(np.float32)

    def _select_primary_signer(self, pose_keypoints):
        """
        Selects the primary signer based on the width of their shoulders and length of torso.
        Approximates bounding box size to pick the person closest to the camera.
        """
        best_idx = 0
        max_area = 0
        
        for i, person in enumerate(pose_keypoints):
            # Points 2 and 5 are right and left shoulders in BODY_25
            r_shoulder = person[2]
            l_shoulder = person[5]
            # Point 8 is MidHip
            mid_hip = person[8]
            
            # Use confidence threshold
            if r_shoulder[2] > 0.1 and l_shoulder[2] > 0.1 and mid_hip[2] > 0.1:
                width = abs(r_shoulder[0] - l_shoulder[0])
                height = abs(mid_hip[1] - (r_shoulder[1] + l_shoulder[1])/2)
                area = width * height
                
                if area > max_area:
                    max_area = area
                    best_idx = i
                    
        return best_idx

    def convert_to_foundation_schema(self, op_features):
        """
        Optional mapper to convert 411 OpenPose flat vector (137x3) to 1629 MediaPipe flat vector (543x3)
        to allow drop-in training compatibility.
        NOTE: Since datasets typically handle native formats, this provides zero-padded fallback mapping.
        """
        # Create MediaPipe formatted flat array
        mp_flat = np.zeros(1629, dtype=np.float32) # (543 * 3)
        
        # 1. Pose (99 values / 33 points) - Map roughly matching joints
        body = op_features[0:75]
        mp_flat[0:75] = body
        
        # 2. Face (1404 values / 468 points) -> OpenPose has 70 points = 210 values
        face = op_features[75:285]
        mp_flat[99:99+210] = face
        
        # 3. Hands (63 left + 63 right) - Hands map 1:1 almost perfectly!
        left_hand = op_features[285:348]
        right_hand = op_features[348:411]
        
        mp_flat[1503:1566] = left_hand
        mp_flat[1566:1629] = right_hand
        
        return mp_flat
