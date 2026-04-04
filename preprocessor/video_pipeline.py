"""
Unified Video Preprocessor — Finalized Pipeline
Handles parallel batch processing of sign language videos.
"""

import os
import sys
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from training.data_pipeline.pose_extractor import PoseExtractor
from training.data_pipeline.gesture_tokenizer import GestureTokenizer
from training.utils.augmentation import KeypointAugmenter


class UnifiedPreprocessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # MediaPipe Holistic is inside PoseExtractor
        self.pose_extractor = PoseExtractor()
        
        vocab_path = self.config.get("vocab_path", "models/gesture_tokenizer.pkl")
        self.tokenizer = GestureTokenizer(vocab_path=vocab_path)
        
        self.augmentor = KeypointAugmenter(
            sigma=self.config.get("aug_sigma", 0.01),
            min_scale=self.config.get("aug_min_scale", 0.95),
            max_scale=self.config.get("aug_max_scale", 1.05)
        )

    def extract_landmarks(self, video_path: str) -> np.ndarray:
        """Extract landmarks from a single video."""
        print(f"  Extracting landmarks: {os.path.basename(video_path)}")
        poses = self.pose_extractor.extract_video(video_path)
        
        if not poses:
            return np.array([])
            
        # Standardize to (T, 225) using the same logic as training/data_pipeline/unified_preprocessor.py
        from common.keypoint_schema import BODY_DIM, ONE_HAND_DIM, FEATURE_DIM_225
        
        vectors = []
        for p in poses:
            v = np.concatenate([
                np.array(p["pose"].get("body") or np.zeros(BODY_DIM), dtype=np.float32).reshape(-1)[:BODY_DIM],
                np.array(p["pose"].get("left_hand") or np.zeros(ONE_HAND_DIM), dtype=np.float32).reshape(-1)[:ONE_HAND_DIM],
                np.array(p["pose"].get("right_hand") or np.zeros(ONE_HAND_DIM), dtype=np.float32).reshape(-1)[:ONE_HAND_DIM],
            ], axis=0)
            vectors.append(v)
            
        seq = np.stack(vectors, axis=0)
        
        # Consistent dimensioning
        if seq.shape[1] != FEATURE_DIM_225:
            seq = seq[:, :FEATURE_DIM_225]
            if seq.shape[1] < FEATURE_DIM_225:
                seq = np.pad(seq, ((0, 0), (0, FEATURE_DIM_225 - seq.shape[1])))
                
        return seq

    def tokenize(self, landmarks: np.ndarray) -> List[int]:
        """Convert landmarks to discrete tokens."""
        if landmarks.size == 0:
            return []
        
        # Apply normalization if needed (PoseExtractor usually returns unnormalized)
        # We'll use the same normalization as the unified_preprocessor.py
        mean = landmarks.mean(axis=0, keepdims=True)
        std = landmarks.std(axis=0, keepdims=True) + 1e-8
        norm_landmarks = (landmarks - mean) / std
        
        return self.tokenizer.tokenize_sequence(norm_landmarks)

    def process_batch(self, videos: List[str]) -> List[List[int]]:
        """
        Parallel processing pipeline: Videos -> Landmarks -> Tokens.
        """
        print(f"\n[Parallel Processing] Batch size: {len(videos)}")
        
        # We split the extraction into multiple threads
        # Note: MediaPipe might have GIL issues or internal threading, 
        # but for IO-bound video reading and CPU-bound landmarking, threads help.
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Step 1: Extract all landmarks in parallel
            landmark_results = list(executor.map(self.extract_landmarks, videos))
            
            # Step 2: Tokenize (Sequential for now as it's fast enough and fits nicely in a list comp)
            all_tokens = [self.tokenize(lm) for lm in landmark_results]
            
        return all_tokens

    def process_with_augmentation(self, video_path: str, count: int = 5) -> List[List[int]]:
        """Generate multiple token sequences for one video using augmentation."""
        landmarks = self.extract_landmarks(video_path)
        if landmarks.size == 0:
            return []
            
        results = []
        # Original
        results.append(self.tokenize(landmarks))
        
        # Synthetic variations
        for _ in range(count - 1):
            aug_landmarks = self.augmentor(landmarks)
            results.append(self.tokenize(aug_landmarks))
            
        return results
