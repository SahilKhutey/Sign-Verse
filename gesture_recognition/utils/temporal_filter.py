from collections import deque
from collections import Counter


class TemporalFilter:
    """
    Smooths gesture predictions across multiple frames using majority voting.
    Critical for sign language — single frame predictions are noisy.

    Uses a sliding window of recent predictions and returns the most
    common gesture within that window.
    """

    def __init__(self, window=10):

        self.window = window
        self.buffer = deque(maxlen=window)

    def update(self, gesture):
        """
        Add a gesture prediction and return the smoothed result.
        Returns None until the buffer is full.
        """

        if gesture is not None:
            self.buffer.append(gesture)

        if len(self.buffer) < self.window:
            return None

        most_common = Counter(self.buffer).most_common(1)

        return most_common[0][0]


class TemporalSequenceBuffer:
    """
    Maintain a sliding window of per-frame feature vectors for temporal models.
    """

    def __init__(self, window=30, stride=1):
        self.window = int(window)
        self.stride = int(stride)
        self.buffer = deque(maxlen=self.window)

    def reset(self):
        self.buffer.clear()

    def update(self, features):
        self.buffer.append(features)
        if len(self.buffer) < self.window:
            return None
        if self.stride > 1 and (len(self.buffer) % self.stride != 0):
            return None
        return list(self.buffer)
