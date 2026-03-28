"""
Motion Interpolator — Smooth transitions between sign animations.

Uses linear interpolation (lerp) and spherical interpolation (slerp)
for natural-looking transitions between skeletal poses.
"""

import numpy as np


class MotionInterpolator:

    def __init__(self, fps=30):
        self.fps = fps

    def lerp(self, start, end, t):
        """Linear interpolation between two values."""
        return start + (end - start) * t

    def lerp_pose(self, pose_a, pose_b, t):
        """
        Interpolate between two skeletal poses.

        Args:
            pose_a: numpy array of joint positions (N, 3)
            pose_b: numpy array of joint positions (N, 3)
            t: interpolation factor (0.0 = pose_a, 1.0 = pose_b)
        """
        return self.lerp(np.array(pose_a), np.array(pose_b), t)

    def generate_transition(self, pose_a, pose_b, duration=0.3):
        """
        Generate intermediate frames for smooth transition.

        Args:
            pose_a: starting pose
            pose_b: ending pose
            duration: transition time in seconds

        Returns:
            List of interpolated poses
        """
        num_frames = max(int(duration * self.fps), 2)
        frames = []

        for i in range(num_frames):
            t = i / (num_frames - 1)
            # Ease in-out for natural motion
            t = self._ease_in_out(t)
            frame = self.lerp_pose(pose_a, pose_b, t)
            frames.append(frame)

        return frames

    def _ease_in_out(self, t):
        """Smooth ease-in-out curve (cubic)."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
