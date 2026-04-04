"""
Physics constraints and motion validation for robotic movement.
"""
import bpy
import mathutils
from typing import Dict, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotionValidator:
    """Validates and constraints motion for robotic safety and physical feasibility."""
    
    def __init__(self, joint_limits: Dict[str, Tuple[float, float]]):
        self.joint_limits = joint_limits
        self.violations = []
    
    def validate_animation(self, armature_obj: bpy.types.Object) -> bool:
        """Validate an entire Blender animation against the registered joint limits."""
        scene = bpy.context.scene
        start_frame = scene.frame_start
        end_frame = scene.frame_end
        
        self.violations = []
        
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            
            for bone_name, limits in self.joint_limits.items():
                if bone_name in armature_obj.pose.bones:
                    bone = armature_obj.pose.bones[bone_name]
                    self._check_bone_limits(bone, limits, frame)
        
        if self.violations:
            logger.warning(f"Motion Validation Failed: {len(self.violations)} constraint violations found.")
            return False
        else:
            logger.info("Motion Validation Passed: Animation sits within all registered joint limits.")
            return True
    
    def _check_bone_limits(self, bone, limits: Tuple[float, float], frame: int):
        """Convert bone orientation to Euler and verify against min/max degree limits."""
        # Convert stable quaternion rotation back to Euler degrees for human-readable validation
        euler_angles = bone.rotation_quaternion.to_euler()
        degrees_angles = [angle * 180 / 3.14159 for angle in euler_angles]
        
        # Check X, Y, Z axes
        for axis, angle in enumerate(degrees_angles):
            min_limit, max_limit = limits
            
            if angle < min_limit or angle > max_limit:
                self.violations.append({
                    'bone': bone.name,
                    'axis': axis,
                    'angle': angle,
                    'min_limit': min_limit,
                    'max_limit': max_limit,
                    'frame': frame
                })
    
    def apply_velocity_limits(self, armature_obj: bpy.types.Object, max_velocity_deg_per_sec: float):
        """Scan animation F-curves and cap the rate of change to match robotic motor speeds."""
        scene = bpy.context.scene
        fps = scene.render.fps
        
        for bone in armature_obj.pose.bones:
            fcurves = self._get_bone_fcurves(bone)
            
            for fcurve in fcurves:
                self._limit_fcurve_velocity(fcurve, max_velocity_deg_per_sec, fps)
    
    def _get_bone_fcurves(self, bone):
        """Retrieve all active F-curves for a given pose bone."""
        if bone.animation_data and bone.animation_data.action:
            return bone.animation_data.action.fcurves
        return []
    
    def _limit_fcurve_velocity(self, fcurve, max_vel: float, fps: int):
        """Clamp the derivative (velocity) of a specific F-curve while preserving direction."""
        kp_points = fcurve.keyframe_points
        num_keys = len(kp_points)
        
        for i in range(1, num_keys):
            prev_p = kp_points[i-1]
            curr_p = kp_points[i]
            
            # co[0] = Frame, co[1] = Value (Degrees)
            frame_diff = curr_p.co[0] - prev_p.co[0]
            if frame_diff <= 0:
                continue
                
            raw_value_diff = curr_p.co[1] - prev_p.co[1]
            abs_value_diff = abs(raw_value_diff)
            
            # Calculate velocity in Degrees/Second
            velocity = (abs_value_diff / frame_diff) * fps
            
            if velocity > max_vel:
                # Calculate maximum allowed value change for this frame skip
                max_allowed_change = (max_vel * frame_diff) / fps
                
                # Apply the cap while maintaining the original direction of motion
                direction = np.sign(raw_value_diff)
                new_value = prev_p.co[1] + (direction * max_allowed_change)
                
                # Update keyframe and flatten handles to prevent Bezier overshoot
                curr_p.co[1] = new_value
                curr_p.handle_left[1] = new_value
                curr_p.handle_right[1] = new_value
