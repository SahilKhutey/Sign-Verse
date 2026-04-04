"""
MediaPipe pose estimation pipeline component.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import mediapipe as mp
from prefect import task
from loguru import logger

from configs import get_config
from core.data_models import PoseData, Keypoint
from core.data_manager import data_manager

@task(name="mediapipe_pipeline.process_video")
def process_video(video_path: str, output_dir: Optional[str] = None) -> List[PoseData]:
    """
    Process video through MediaPipe pose estimation.
    
    Args:
        video_path: Path to input video
        output_dir: Custom output directory (optional)
        
    Returns:
        List of PoseData objects for each frame
    """
    config = get_config()
    output_dir = output_dir or config.pipelines.pose_estimation.mediapipe.output_dir
    mp_config = config.pipelines.pose_estimation.mediapipe
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, # Video mode
        model_complexity=mp_config.model_complexity,
        min_detection_confidence=mp_config.min_detection_confidence,
        min_tracking_confidence=mp_config.min_tracking_confidence
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    pose_data_list = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            pose_data = process_landmarks(
                results.pose_landmarks,
                video_path,
                frame_count,
                fps
            )
            pose_data_list.append(pose_data)
            
            # Save individual frame data using DataManager
            if output_dir:
                output_filename = f"{Path(video_path).stem}_frame_{frame_count:06d}.json"
                data_manager.save_pose_data(pose_data, output_filename)
        
        frame_count += 1
    
    cap.release()
    pose.close()
    
    logger.success(f"Processed {frame_count} frames from {video_path}")
    return pose_data_list

def process_landmarks(landmarks, video_path: str, frame_number: int, fps: float) -> PoseData:
    """Convert MediaPipe landmarks to standardized PoseData."""
    config = get_config()
    keypoint_labels = config.keypoint_labels.labels
    
    keypoints = []
    for idx, landmark in enumerate(landmarks.landmark):
        if idx < len(keypoint_labels):
            keypoints.append(Keypoint(
                id=idx,
                name=keypoint_labels[idx],
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                confidence=landmark.visibility,
                visible=landmark.visibility > 0.5
            ))
    
    return PoseData(
        source_video=str(Path(video_path).name),
        frame_number=frame_number,
        timestamp=frame_number / fps if fps > 0 else 0,
        keypoints=keypoints
    )
