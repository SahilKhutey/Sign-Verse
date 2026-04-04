import cv2
import numpy as np
import os

def generate_test_video(output_path, duration=5, fps=30, resolution=(640, 480)):
    """Generates a synthetic test video with a moving circle."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    num_frames = duration * fps
    for i in range(num_frames):
        # Create black frame
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Draw a moving circle (simulating a "person" or "object")
        center_x = int(resolution[0] * (0.2 + 0.6 * (i / num_frames)))
        center_y = int(resolution[1] * (0.5 + 0.2 * np.sin(2 * np.pi * i / fps)))
        cv2.circle(frame, (center_x, center_y), 40, (0, 255, 0), -1)
        
        # Draw a static "object"
        cv2.rectangle(frame, (resolution[0]//2 - 20, resolution[1]//2 - 20), 
                      (resolution[0]//2 + 20, resolution[1]//2 + 20), (255, 0, 0), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Synthetic test video generated: {output_path}")

if __name__ == "__main__":
    generate_test_video("data/raw/test_video.mp4")
