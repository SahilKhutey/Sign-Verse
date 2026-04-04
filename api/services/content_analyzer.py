import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
from loguru import logger

from .perception_service import perception_service

class ContentAnalyzer:
    def __init__(self):
        self.min_person_height = 100  # Minimum pixel height for good tracking
        self.min_motion_threshold = 5.0  # Minimum optical flow magnitude
        self.default_num_frames = 30  # Increased for better coverage (user feedback)
    
    def sample_frames(self, video_path: str, num_frames: Optional[int] = None) -> List[np.ndarray]:
        """Sample frames evenly from video to assess content quality."""
        num_frames = num_frames or self.default_num_frames
        
        if not os.path.exists(video_path):
            logger.error(f"Video path does not exist: {video_path}")
            return []
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []
            
        frames = []
        # Sample frames at regular intervals
        for i in range(num_frames):
            frame_pos = int(i * total_frames / num_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
                
        cap.release()
        logger.info(f"Sampled {len(frames)} frames from {video_path}")
        return frames
    
    def detect_humans(self, frames: List[np.ndarray]) -> float:
        """Detect human presence in frames."""
        if not frames:
            return 0.0
            
        human_count = 0
        for frame in frames:
            result = perception_service.process_frame(frame)
            if result.get('persons') and len(result['persons']) > 0:
                human_count += 1
                
        return human_count / len(frames)
    
    def compute_motion_score(self, frames: List[np.ndarray]) -> float:
        """Compute optical flow magnitude between consecutive sampled frames."""
        if len(frames) < 2:
            return 0.0
            
        total_flow = 0.0
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Farneback dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_magnitude = np.mean(magnitude)
            
            total_flow += mean_magnitude
            prev_gray = curr_gray
            
        return total_flow / (len(frames) - 1)
    
    def assess_trackability(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Assess how trackable the content is based on human posture and object interactions.
        Returns a score dictionary.
        """
        if not frames:
            return {"total": 0.0, "pose": 0.0, "interaction": 0.0}
            
        pose_scores = []
        interaction_scores = []
        
        for frame in frames:
            result = perception_service.process_frame(frame)
            persons = result.get('persons', [])
            interactions = result.get('interactions', [])
            
            # 1. Pose Quality Score
            if persons:
                # Visibility and size assessment
                person_pose_scores = []
                for p in persons:
                    kps = p.get('keypoints', [])
                    visible_kps = sum(1 for k in kps if k.get('score', 0) > 0.3)
                    vis_ratio = visible_kps / len(kps) if kps else 0
                    
                    bbox = self._estimate_bbox(kps)
                    height_score = 0
                    if bbox:
                        height = bbox[3] - bbox[1]
                        height_score = min(1.0, height / self.min_person_height)
                    
                    person_pose_scores.append((vis_ratio + height_score) / 2)
                pose_scores.append(max(person_pose_scores))
            else:
                pose_scores.append(0.0)
            
            # 2. Interaction Score (Weighted high per user feedback)
            if interactions:
                # Interaction count and confidence
                avg_interaction_conf = sum(i.get('confidence', 0) for i in interactions) / len(interactions)
                interaction_scores.append(min(1.0, avg_interaction_conf * 1.5)) # Boost interaction significance
            else:
                interaction_scores.append(0.0)
                
        avg_pose = sum(pose_scores) / len(pose_scores)
        avg_interaction = sum(interaction_scores) / len(interaction_scores)
        
        # Combined score with emphasis on interactions
        total_score = (avg_pose * 0.4) + (avg_interaction * 0.6)
        
        return {
            "total": total_score,
            "pose": avg_pose,
            "interaction": avg_interaction
        }
    
    def _estimate_bbox(self, keypoints: List[Dict]) -> Optional[List[float]]:
        """Estimate bounding box from keypoints."""
        xs = [k['x'] for k in keypoints if k.get('score', 0) > 0.2]
        ys = [k['y'] for k in keypoints if k.get('score', 0) > 0.2]
        
        if not xs or not ys:
            return None
            
        return [min(xs), min(ys), max(xs), max(ys)]
    
    def classify_heuristics(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Classify content type using object/pose heuristics."""
        if not frames:
            return {}
            
        counts = {"workout": 0, "dance": 0, "craft": 0, "sports": 0, "culinary": 0}
        
        # Sample subset for heuristics to stay fast
        sample = frames[::5] 
        for frame in sample:
            res = perception_service.process_frame(frame)
            objs = [o.get('type') for o in res.get('objects', [])]
            
            if any(t in objs for t in ['dumbbell', 'barbell', 'bench']):
                counts["workout"] += 1
            if any(t in objs for t in ['sports ball', 'tennis racket', 'frisbee']):
                counts["sports"] += 1
            if any(t in objs for t in ['hammer', 'saw', 'wrench', 'tool']):
                counts["craft"] += 1
            if any(t in objs for t in ['knife', 'bowl', 'pan', 'bottle']):
                counts["culinary"] += 1
            
            # Simple pose-based dance heuristic
            if res.get('persons'):
                if self._check_dance_pose_heuristics(res['persons'][0]):
                    counts["dance"] += 1
                    
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    def _check_dance_pose_heuristics(self, person: Dict) -> bool:
        """Check for dance-like poses (arms raised, etc)."""
        kps = {k.get('name'): k for k in person.get('keypoints', [])}
        
        needed = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']
        if not all(n in kps and kps[n]['score'] > 0.4 for n in needed):
            return False
            
        # If elbows are higher than shoulders, assume expressive motion (dance/workout)
        raised = kps['left_elbow']['y'] < kps['left_shoulder']['y'] or \
                 kps['right_elbow']['y'] < kps['right_shoulder']['y']
                 
        return raised

# Global analyzer instance
content_analyzer = ContentAnalyzer()
