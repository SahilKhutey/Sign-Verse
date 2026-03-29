import numpy as np

class TemporalEngine:
    def __init__(self):
        self.prev = None

    def compute(self, current):
        """Compute velocity between current and previous feature vector."""
        if self.prev is None:
            self.prev = current
            return np.zeros_like(current)
        
        velocity = current - self.prev
        self.prev = current
        return velocity

    def reset(self):
        self.prev = None
