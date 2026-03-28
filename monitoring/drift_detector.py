import numpy as np
import time
import json
import os

class DriftDetector:
    """
    Model Drift Detector — Monitors inference confidence and input distribution.
    Alerts if the model starts performing poorly on production data.
    """

    def __init__(self, threshold=0.85, window_size=100):
        self.threshold = threshold
        self.window_size = window_size
        self.confidence_history = []
        self.drift_detected = False

    def add_inference(self, confidence: float):
        """Add a new inference result to the monitoring window."""
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.window_size:
            self.confidence_history.pop(0)
        
        self._check_drift()

    def _check_drift(self):
        """Calculate the average confidence and check against threshold."""
        if len(self.confidence_history) < self.window_size:
            return

        avg_confidence = np.mean(self.confidence_history)
        if avg_confidence < self.threshold:
            self.drift_detected = True
            self._trigger_alert(avg_confidence)
        else:
            self.drift_detected = False

    def _trigger_alert(self, current_avg: float):
        """Simulate an alert to a monitoring system (e.g., Slack, PagerDuty)."""
        alert_msg = {
            "level": "CRITICAL",
            "source": "SignVerseModelMonitor",
            "message": "Performance Drift Detected",
            "threshold": self.threshold,
            "current_avg": float(current_avg),
            "timestamp": time.time()
        }
        print(f"!!! ALERT: {json.dumps(alert_msg)}")
        # In production, this would call an external API

    def get_status(self):
        return {
            "drift_detected": self.drift_detected,
            "avg_confidence": np.mean(self.confidence_history) if self.confidence_history else 1.0,
            "samples": len(self.confidence_history)
        }

if __name__ == "__main__":
    detector = DriftDetector(threshold=0.90, window_size=10)
    # Simulate a drift event
    print("Simulating high confidence...")
    for _ in range(10): detector.add_inference(0.95)
    print(f"Status: {detector.get_status()}")

    print("\nSimulating drift...")
    for _ in range(10): detector.add_inference(0.80)
    print(f"Status: {detector.get_status()}")
