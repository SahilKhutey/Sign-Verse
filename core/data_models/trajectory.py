"""
Temporal tracking system for maintaining pose trajectories across frames.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from loguru import logger

from .skeleton import SkeletonFrame, JointType

@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    frame_id: int
    timestamp: float
    joints: Dict[str, Any]  # Joint positions
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PersonTrajectory:
    """Complete trajectory for a single person."""
    person_id: str
    start_frame: int
    end_frame: int
    trajectory: List[TrajectoryPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_point(self, frame: SkeletonFrame):
        """Add a new point to the trajectory."""
        point = TrajectoryPoint(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            joints=self._extract_joints(frame),
            confidence=frame.overall_confidence,
            metadata={
                "source_video": frame.source_video,
                "camera_id": frame.camera_id
            }
        )
        
        self.trajectory.append(point)
        # Ensure temporal order
        self.trajectory.sort(key=lambda x: x.frame_id)
        
        # Update frame range
        self.start_frame = min(self.start_frame, frame.frame_id)
        self.end_frame = max(self.end_frame, frame.frame_id)
    
    def _extract_joints(self, frame: SkeletonFrame) -> Dict[str, Any]:
        """Extract joints from frame."""
        joints = {}
        
        # Extract body joints
        body_joints = frame.body.get_joints_dict()
        for joint_name, vector in body_joints.items():
            joints[f"body_{joint_name}"] = {
                "x": vector.x,
                "y": vector.y,
                "z": vector.z,
                "confidence": vector.confidence
            }
        
        # Extract hand joints
        if frame.left_hand:
            left_hand_joints = frame.left_hand.dict()
            for joint_name, vector in left_hand_joints.items():
                if vector is not None:
                    joints[f"left_hand_{joint_name}"] = {
                        "x": vector["x"],
                        "y": vector["y"],
                        "z": vector["z"],
                        "confidence": vector["confidence"]
                    }
        
        if frame.right_hand:
            right_hand_joints = frame.right_hand.dict()
            for joint_name, vector in right_hand_joints.items():
                if vector is not None:
                    joints[f"right_hand_{joint_name}"] = {
                        "x": vector["x"],
                        "y": vector["y"],
                        "z": vector["z"],
                        "confidence": vector["confidence"]
                    }
        
        return joints
    
    def get_joint_trajectory(self, joint_name: str) -> List[Dict[str, Any]]:
        """Get trajectory for a specific joint."""
        trajectory = []
        for point in self.trajectory:
            if joint_name in point.joints:
                trajectory.append({
                    "frame_id": point.frame_id,
                    "timestamp": point.timestamp,
                    "position": point.joints[joint_name],
                    "confidence": point.confidence
                })
        return trajectory
    
    def get_velocity(self, joint_name: str) -> List[Dict[str, Any]]:
        """Calculate velocity for a specific joint."""
        trajectory = self.get_joint_trajectory(joint_name)
        velocities = []
        
        for i in range(1, len(trajectory)):
            current = trajectory[i]
            previous = trajectory[i-1]
            
            dt = current["timestamp"] - previous["timestamp"]
            if dt > 0:
                dx = current["position"]["x"] - previous["position"]["x"]
                dy = current["position"]["y"] - previous["position"]["y"]
                dz = current["position"]["z"] - previous["position"]["z"]
                
                velocity = np.sqrt(dx**2 + dy**2 + dz**2) / dt
                velocities.append({
                    "frame_id": current["frame_id"],
                    "timestamp": current["timestamp"],
                    "velocity": velocity,
                    "confidence": min(current["confidence"], previous["confidence"])
                })
        
        return velocities
    
    def get_acceleration(self, joint_name: str) -> List[Dict[str, Any]]:
        """Calculate acceleration for a specific joint."""
        velocities = self.get_velocity(joint_name)
        accelerations = []
        
        for i in range(1, len(velocities)):
            current = velocities[i]
            previous = velocities[i-1]
            
            dt = current["timestamp"] - previous["timestamp"]
            if dt > 0:
                acceleration = (current["velocity"] - previous["velocity"]) / dt
                accelerations.append({
                    "frame_id": current["frame_id"],
                    "timestamp": current["timestamp"],
                    "acceleration": acceleration,
                    "confidence": min(current["confidence"], previous["confidence"])
                })
        
        return accelerations

class TemporalTracker:
    """Manages temporal tracking of multiple persons."""
    
    def __init__(self, max_gap_frames: int = 30):
        """
        Initialize temporal tracker.
        
        Args:
            max_gap_frames: Maximum allowed gap between frames for same person
        """
        self.max_gap_frames = max_gap_frames
        self.trajectories: Dict[str, PersonTrajectory] = {}
        self.active_persons: Dict[str, int] = {}  # person_id -> last_frame_id
    
    def update(self, frame: SkeletonFrame) -> bool:
        """
        Update tracker with new frame.
        
        Args:
            frame: SkeletonFrame to add
            
        Returns:
            True if successfully added, False if rejected
        """
        person_id = frame.person_id
        
        # Check if this is a new person or existing one
        if person_id in self.trajectories:
            trajectory = self.trajectories[person_id]
            last_frame = trajectory.trajectory[-1].frame_id if trajectory.trajectory else 0
            
            # Check for reasonable frame gap
            frame_gap = frame.frame_id - last_frame
            if frame_gap > self.max_gap_frames:
                logger.warning(f"Large frame gap for {person_id}: {frame_gap} frames")
                # Consider this a new trajectory or handle accordingly
                return False
            
            trajectory.add_point(frame)
            self.active_persons[person_id] = frame.frame_id
            return True
        else:
            # New person - create trajectory
            trajectory = PersonTrajectory(
                person_id=person_id,
                start_frame=frame.frame_id,
                end_frame=frame.frame_id
            )
            trajectory.add_point(frame)
            self.trajectories[person_id] = trajectory
            self.active_persons[person_id] = frame.frame_id
            return True

    def get_trajectory(self, person_id: str) -> Optional[PersonTrajectory]:
        """Get trajectory for a specific person."""
        return self.trajectories.get(person_id)

    def get_all_trajectories(self) -> Dict[str, PersonTrajectory]:
        """Get all trajectories."""
        return self.trajectories.copy()

    def cleanup_inactive(self, current_frame: int, inactive_threshold: int = 60):
        """Clean up inactive trajectories."""
        to_remove = []
        for person_id, last_frame in self.active_persons.items():
            if current_frame - last_frame > inactive_threshold:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.active_persons[person_id]
            # Keep trajectory data, just mark as inactive
            logger.info(f"Marked person {person_id} as inactive")

    def export_trajectory(self, person_id: str, format: str = "json") -> Dict[str, Any]:
        """Export trajectory in standardized format."""
        trajectory = self.get_trajectory(person_id)
        if not trajectory:
            return {}
        
        return {
            "person_id": person_id,
            "start_frame": trajectory.start_frame,
            "end_frame": trajectory.end_frame,
            "duration": trajectory.trajectory[-1].timestamp - trajectory.trajectory[0].timestamp if trajectory.trajectory else 0,
            "frame_count": len(trajectory.trajectory),
            "joints": list(trajectory.trajectory[0].joints.keys()) if trajectory.trajectory else [],
            "data": [point.__dict__ for point in trajectory.trajectory]
        }

    def find_similar_trajectories(self, query_trajectory: PersonTrajectory, 
                                 similarity_threshold: float = 0.8) -> List[str]:
        """Find similar trajectories using DTW or other similarity measures."""
        similar = []
        
        for person_id, trajectory in self.trajectories.items():
            if trajectory.person_id != query_trajectory.person_id:
                similarity = self._calculate_similarity(query_trajectory, trajectory)
                if similarity >= similarity_threshold:
                    similar.append(person_id)
        
        return similar

    def _calculate_similarity(self, traj1: PersonTrajectory, traj2: PersonTrajectory) -> float:
        """Calculate similarity between two trajectories."""
        # Simple implementation - would use DTW in production
        if not traj1.trajectory or not traj2.trajectory:
            return 0.0
        
        # Compare common joints
        common_joints = set(traj1.trajectory[0].joints.keys()) & set(traj2.trajectory[0].joints.keys())
        if not common_joints:
            return 0.0
        
        # Simple average position similarity
        total_similarity = 0.0
        count = 0
        
        for joint in common_joints:
            # Get positions for this joint
            pos1 = [point.joints[joint] for point in traj1.trajectory if joint in point.joints]
            pos2 = [point.joints[joint] for point in traj2.trajectory if joint in point.joints]
            
            if pos1 and pos2:
                # Simple Euclidean distance-based similarity
                min_len = min(len(pos1), len(pos2))
                distances = []
                
                for i in range(min_len):
                    dist = np.sqrt(
                        (pos1[i]["x"] - pos2[i]["x"])**2 +
                        (pos1[i]["y"] - pos2[i]["y"])**2 +
                        (pos1[i]["z"] - pos2[i]["z"])**2
                    )
                    distances.append(dist)
                
                if distances:
                    avg_distance = np.mean(distances)
                    similarity = 1.0 / (1.0 + avg_distance)  # Convert distance to similarity
                    total_similarity += similarity
                    count += 1
        
        return total_similarity / count if count > 0 else 0.0
