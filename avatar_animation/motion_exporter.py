"""
Motion Exporter — Connect generated motion to avatar rigs.

Pipeline:
    Generated Motion → Skeleton Mapping → Avatar Rig → Sign Animation

Export formats:
    - BVH  (Blender, MotionBuilder)
    - FBX  (Unity, Unreal Engine)
    - JSON (custom WebSocket streaming)

Full system:
    User Speech → Text → Sign Tokens
    → Gesture Diffusion Model → 3D Motion Sequence
    → Motion Exporter → Avatar Animation → AR/VR Display
"""

import numpy as np
import json
import os


class SkeletonMapper:
    """Maps 3D joint positions to avatar bone structure."""

    # Standard skeleton joint mapping
    JOINT_NAMES = [
        "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftUpLeg", "LeftLeg", "LeftFoot",
        "RightUpLeg", "RightLeg", "RightFoot",
        # Hand joints (simplified)
        "LeftThumb1", "LeftThumb2", "LeftThumb3",
        "LeftIndex1", "LeftIndex2", "LeftIndex3",
        "LeftMiddle1", "LeftMiddle2", "LeftMiddle3",
        "LeftRing1", "LeftRing2", "LeftRing3",
        "LeftPinky1", "LeftPinky2", "LeftPinky3",
        "RightThumb1", "RightThumb2", "RightThumb3",
        "RightIndex1", "RightIndex2", "RightIndex3",
        "RightMiddle1", "RightMiddle2", "RightMiddle3",
        "RightRing1", "RightRing2", "RightRing3",
        "RightPinky1", "RightPinky2", "RightPinky3",
    ]

    def __init__(self):
        self.num_joints = len(self.JOINT_NAMES)

    def map_to_skeleton(self, motion_vector):
        """
        Map a flat motion vector to named joint positions.

        Args:
            motion_vector: flat array of 3D positions

        Returns:
            dict mapping joint name → (x, y, z)
        """
        skeleton = {}
        for i, name in enumerate(self.JOINT_NAMES):
            idx = i * 3
            if idx + 3 <= len(motion_vector):
                skeleton[name] = {
                    "x": float(motion_vector[idx]),
                    "y": float(motion_vector[idx + 1]),
                    "z": float(motion_vector[idx + 2])
                }
        return skeleton


class MotionExporter:
    """Export generated motion in various formats for avatar engines."""

    def __init__(self):
        self.skeleton_mapper = SkeletonMapper()

    def to_json(self, motion_sequence, output_path, fps=30):
        """
        Export motion as JSON for WebSocket streaming to Unity.

        Args:
            motion_sequence: numpy array (frames, motion_dim)
            output_path: save path
            fps: animation framerate
        """
        frames = []

        for i, frame_data in enumerate(motion_sequence):
            skeleton = self.skeleton_mapper.map_to_skeleton(frame_data)
            frames.append({
                "frame": i,
                "time": i / fps,
                "joints": skeleton
            })

        animation = {
            "fps": fps,
            "total_frames": len(frames),
            "duration": len(frames) / fps,
            "frames": frames
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(animation, f, indent=2)

        print(f"Exported {len(frames)} frames to {output_path}")

    def to_bvh(self, motion_sequence, output_path, fps=30):
        """
        Export motion as BVH file for Blender/MotionBuilder.
        """
        from motion_capture.pose3d_model import skeleton_to_bvh
        skeleton_to_bvh(motion_sequence, output_path, fps)

    def to_unity_stream(self, motion_sequence, fps=30):
        """
        Convert motion to Unity-ready streaming format.
        Returns list of frame dicts for WebSocket broadcasting.
        """
        stream_data = []

        for i, frame_data in enumerate(motion_sequence):
            skeleton = self.skeleton_mapper.map_to_skeleton(frame_data)
            stream_data.append({
                "type": "motion_frame",
                "frame_id": i,
                "timestamp": i / fps,
                "joints": skeleton
            })

        return stream_data


def generate_and_export(model_path, output_dir="exports", num_sequences=1):
    """
    Full pipeline: Generate motion → Export to multiple formats.

    Args:
        model_path: path to trained diffusion model
        output_dir: directory for exported files
        num_sequences: number of sequences to generate
    """
    from training.train_diffusion import generate_motion

    os.makedirs(output_dir, exist_ok=True)
    exporter = MotionExporter()

    motions = generate_motion(model_path, num_samples=num_sequences)

    for i in range(num_sequences):
        motion_np = motions[i].cpu().numpy()

        # Export JSON
        exporter.to_json(
            motion_np,
            os.path.join(output_dir, f"gesture_{i:03d}.json")
        )

        # Export BVH
        exporter.to_bvh(
            motion_np,
            os.path.join(output_dir, f"gesture_{i:03d}.bvh")
        )

        print(f"Exported gesture {i + 1}/{num_sequences}")
