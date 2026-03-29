"""
Optical Flow Analyzer

Provides dense physical motion vectors via OpenCV Farneback Optical Flow.
Highly robust against poor lighting and blur where keypoint extractors fail.
"""
import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        
        self.prev_gray = None
        
    def process_frame(self, frame, bbox_points=None):
        """
        Computes dense optical flow magnitude across the full frame to ensure shape stability, 
        then optionally masks to the Region of Interest (bbox_points) to compute energy.
        """
        # Optimization: Downsample if the image is too large for fast optical flow
        h, w = frame.shape[:2]
        scale = 1.0
        if w > 320:
            scale = 320 / w
            frame = cv2.resize(frame, (320, int(h*scale)))
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return 0.0 # First frame has no motion
            
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.prev_gray,
            next=gray,
            flow=None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )
        
        self.prev_gray = gray
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        if bbox_points is not None:
            nx1, ny1, nx2, ny2 = bbox_points
            # Scale coordinates down to match optical flow array
            sx1, sy1 = int(nx1 * scale), int(ny1 * scale)
            sx2, sy2 = int(nx2 * scale), int(ny2 * scale)
            
            # Prevent Out of Bounds
            sx1, sy1 = max(0, sx1), max(0, sy1)
            sx2, sy2 = min(magnitude.shape[1], sx2), min(magnitude.shape[0], sy2)
            
            if sx2 > sx1 and sy2 > sy1:
                roi_mag = magnitude[sy1:sy2, sx1:sx2]
                return float(np.mean(roi_mag))
        
        return float(np.mean(magnitude))
