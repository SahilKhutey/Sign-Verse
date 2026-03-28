"""
Animation Controller — Manages avatar animation playback.

Handles:
  - Animation queue management
  - Transition timing between signs
  - Playback speed control
  - Idle state management
"""

import time
from collections import deque


class AnimationController:

    def __init__(self, default_duration=0.8, transition_time=0.2):
        self.queue = deque()
        self.current_animation = None
        self.is_playing = False
        self.default_duration = default_duration
        self.transition_time = transition_time
        self.playback_speed = 1.0

    def enqueue(self, animation_clips):
        """Add animation clips to the queue."""
        for clip in animation_clips:
            self.queue.append({
                "clip": clip,
                "duration": self.default_duration,
                "timestamp": time.time()
            })

    def play_next(self):
        """Play the next animation in the queue."""
        if not self.queue:
            self.current_animation = "idle"
            self.is_playing = False
            return None

        self.current_animation = self.queue.popleft()
        self.is_playing = True
        return self.current_animation

    def get_state(self):
        """Get current animation state."""
        return {
            "current": self.current_animation,
            "queue_size": len(self.queue),
            "is_playing": self.is_playing,
            "speed": self.playback_speed
        }

    def set_speed(self, speed):
        """Set playback speed (0.5 = half, 2.0 = double)."""
        self.playback_speed = max(0.1, min(speed, 3.0))

    def clear(self):
        """Clear all queued animations."""
        self.queue.clear()
        self.current_animation = "idle"
        self.is_playing = False
