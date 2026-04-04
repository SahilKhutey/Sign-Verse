"""
Converts pose data to Blender animations.
"""
import bpy
import mathutils
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PoseAnimator:
    """Converts pose sequences to Blender keyframe animations."""
    
    def __init__(self, armature_obj: bpy.types.Object):
        self.armature = armature_obj
        self.pose_bones = armature_obj.pose.bones
        self.current_frame = 0
    
    def create_animation_from_poses(self, pose_sequence: List[Dict[str, Any]], 
                                  fps: int = 30, start_frame: int = 1):
        """
        Iterate through pose sequence and insert keyframes.
        
        Args:
            pose_sequence: List of pose data dictionaries (canonical PoseData format)
            fps: Playback frames per second
            start_frame: Initial frame offset
        """
        bpy.context.scene.frame_start = start_frame
        bpy.context.scene.frame_end = start_frame + len(pose_sequence)
        bpy.context.scene.render.fps = fps
        
        self.current_frame = start_frame
        
        for frame_idx, pose_data in enumerate(pose_sequence):
            self._set_pose_frame(pose_data, frame_idx + start_frame)
        
        logger.info(f"Created keyframe sequence with {len(pose_sequence)} frames")
    
    def _set_pose_frame(self, pose_data: Dict[str, Any], frame_number: int):
        """Calculate and set bone transformations for a specific frame."""
        bpy.context.scene.frame_set(frame_number)
        
        # Map 2D/3D landmarks to bone rotations (Quaternions)
        bone_rotations = self._calculate_bone_rotations(pose_data)
        
        for bone_name, rotation in bone_rotations.items():
            if bone_name in self.pose_bones:
                bone = self.pose_bones[bone_name]
                
                # Set rotation mode to Quaternion for stable interpolation
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = rotation
                
                # Insert rotation keyframe
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)
        
        self.current_frame = frame_number
    
    def _calculate_bone_rotations(self, pose_data: Dict[str, Any]) -> Dict[str, mathutils.Quaternion]:
        """Convert skeletal landmarks into quaternions for the rig."""
        rotations = {}
        
        # Populate lookup map for O(1) keypoint access
        keypoints = {}
        for kp in pose_data.get('keypoints', []):
            keypoints[kp['name']] = (kp['x'], kp['y'], kp.get('z', 0))
        
        # Arms: Left
        if 'left_shoulder' in keypoints and 'left_elbow' in keypoints:
            rotations['upper_arm.L'] = self._calculate_segment_rotation(
                keypoints['left_shoulder'], 
                keypoints['left_elbow'],
                keypoints.get('left_wrist')
            )
        
        # Arms: Right
        if 'right_shoulder' in keypoints and 'right_elbow' in keypoints:
            rotations['upper_arm.R'] = self._calculate_segment_rotation(
                keypoints['right_shoulder'], 
                keypoints['right_elbow'],
                keypoints.get('right_wrist')
            )
            
        # TODO: Add Legs, Spine, and Neck rotation logic
        
        return rotations
    
    def _calculate_segment_rotation(self, pivot: tuple, target: tuple, distal: Optional[tuple] = None) -> mathutils.Quaternion:
        """Calculate the rotation difference from a segment vector."""
        # Convert raw coordinates to Blender-native mathutils Vectors
        v_pivot = mathutils.Vector(pivot)
        v_target = mathutils.Vector(target)
        
        # Calculate normalized direction of the bone segment
        direction = (v_target - v_pivot).normalized()
        
        # Default reference direction (Down for humanoid arms/legs)
        default_dir = mathutils.Vector((0, 0, -1))
        
        # Calculate rotation difference to align default with direction
        return default_dir.rotation_difference(direction)
    
    def smooth_animation(self, window_size: int = 5):
        """Apply a temporal moving average to mitigate detection jitter."""
        if not self.armature.animation_data or not self.armature.animation_data.action:
            logger.warning("No animation data found on armature to smooth.")
            return

        action = self.armature.animation_data.action
        for fcurve in action.fcurves:
            if 'rotation_quaternion' in fcurve.data_path:
                self._apply_moving_average(fcurve, window_size)
        
        logger.info("Temporal jitter removal (smoothing) complete.")
    
    def _apply_moving_average(self, fcurve: bpy.types.FCurve, window_size: int):
        """Smooth the values of a single F-Curve using a windowed mean."""
        kp_points = fcurve.keyframe_points
        num_keys = len(kp_points)
        
        if num_keys <= window_size:
            return
        
        # Pass 1: Calculate smoothed values
        smoothed_vals = []
        for i in range(num_keys):
            start = max(0, i - window_size // 2)
            end = min(num_keys, i + window_size // 2 + 1)
            
            # co[1] is the value of the keyframe at its timestamp
            window = [kp_points[j].co[1] for j in range(start, end)]
            smoothed_vals.append(sum(window) / len(window))
        
        # Pass 2: Re-apply to F-Curve
        for i, val in enumerate(smoothed_vals):
            kp_points[i].co[1] = val
            # Flatten handles to prevent overshooting at smoothed nodes
            kp_points[i].handle_left[1] = val
            kp_points[i].handle_right[1] = val
