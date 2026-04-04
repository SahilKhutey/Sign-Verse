"""
Dataset Builder — Video Frame Extractor

Extracts frames from sign language videos at configurable FPS.
Sources: WLASL, How2Sign, PHOENIX-2014T, ASLLVD
"""

import cv2
import os


def extract_frames(video_path, output_dir, sample_fps=5):
    """
    Extract frames from a video at the given sample rate.

    Args:
        video_path:  path to input video
        output_dir:  directory to save frames
        sample_fps:  how many frames per second to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(video_fps / sample_fps))

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            cv2.imwrite(os.path.join(output_dir, f"{saved:06d}.jpg"), frame)
            saved += 1
        frame_id += 1

    cap.release()
    return saved


def batch_extract(dataset_dir, output_root, sample_fps=5):
    """Process all videos in a directory tree."""
    total = 0
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                video_path = os.path.join(root, f)
                rel = os.path.relpath(root, dataset_dir)
                out_dir = os.path.join(output_root, rel, os.path.splitext(f)[0])
                saved = extract_frames(video_path, out_dir, sample_fps)
                total += saved
                print(f"  {f}: {saved} frames")
    print(f"Total frames extracted: {total}")
    return total
