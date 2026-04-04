"""
Example usage of the unified skeleton format and temporal tracking.
"""
import sys
import os
import numpy as np

# Add workspace to path to allow imports from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_models.skeleton import SkeletonFrame, Vector3D, BodyJoints, HeadPose
from core.data_models.trajectory import TemporalTracker

def create_example_skeleton() -> SkeletonFrame:
    """Create an example skeleton frame."""
    return SkeletonFrame(
        frame_id=1,
        timestamp=0.033,  # 30 FPS
        person_id="P1",
        source_video="example_video.mp4",
        
        body=BodyJoints(
            nose=Vector3D(x=0.5, y=0.2, z=0.0, confidence=0.9),
            left_shoulder=Vector3D(x=0.4, y=0.3, z=0.0, confidence=0.8),
            right_shoulder=Vector3D(x=0.6, y=0.3, z=0.0, confidence=0.8),
            left_elbow=Vector3D(x=0.3, y=0.4, z=0.0, confidence=0.7),
            right_elbow=Vector3D(x=0.7, y=0.4, z=0.0, confidence=0.7)
        ),
        
        head_pose=HeadPose(
            yaw=5.0,
            pitch=-2.0,
            roll=1.0,
            confidence=0.8
        ),
        
        overall_confidence=0.85,
        tracking_quality=0.9,
        resolution=(1920, 1080)
    )

def demo_temporal_tracking():
    """Demonstrate temporal tracking."""
    tracker = TemporalTracker()
    
    # Create some example frames
    for frame_id in range(1, 101):  # 100 frames
        frame = SkeletonFrame(
            frame_id=frame_id,
            timestamp=frame_id / 30.0,  # 30 FPS
            person_id="P1",
            source_video="demo_video.mp4",
            
            body=BodyJoints(
                nose=Vector3D(
                    x=0.5 + 0.01 * np.sin(frame_id / 10.0),  # Moving sinusoidally
                    y=0.2 + 0.01 * np.cos(frame_id / 10.0),
                    z=0.0,
                    confidence=0.9
                ),
                left_shoulder=Vector3D(x=0.4, y=0.3, z=0.0, confidence=0.8),
                right_shoulder=Vector3D(x=0.6, y=0.3, z=0.0, confidence=0.8)
            ),
            
            overall_confidence=0.85,
            tracking_quality=0.9
        )
        
        tracker.update(frame)
    
    # Get the trajectory
    trajectory = tracker.get_trajectory("P1")
    if trajectory:
        print(f"Trajectory for P1: {len(trajectory.trajectory)} frames")
        print(f"Duration: {trajectory.trajectory[-1].timestamp - trajectory.trajectory[0].timestamp:.2f}s")
        
        # Get velocity for nose joint
        velocities = trajectory.get_velocity("body_nose")
        if velocities:
            # velocities is a list of Dicts with "velocity" key
            avg_velocity = np.mean([v["velocity"] for v in velocities])
            print(f"Average nose velocity: {avg_velocity:.4f} units/s")
        
        # Export trajectory
        export_data = tracker.export_trajectory("P1")
        print(f"Exported trajectory with {len(export_data['data'])} points")

if __name__ == "__main__":
    # Create example skeleton
    example_frame = create_example_skeleton()
    print("Example Skeleton Frame:")
    print(f"Person: {example_frame.person_id}, Frame: {example_frame.frame_id}")
    print(f"Nose position: {example_frame.body.nose}")
    print(f"Head pose: {example_frame.head_pose}")
    
    # Demo temporal tracking
    print("\n--- Temporal Tracking Demo ---")
    demo_temporal_tracking()
