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

        if gesture:
            self.buffer.append(gesture)

        if len(self.buffer) < self.window:
            return None

        most_common = Counter(self.buffer).most_common(1)

        return most_common[0][0]
