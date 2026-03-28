"""
Dataset Builder — Extract frames from sign language video datasets.

Supports:
    - WLASL (2000+ ASL words)
    - RWTH-PHOENIX-Weather (45760 samples, 1200 vocab, sentence-level)
    - LSA64
    - AUTSL

Usage:
    extract_frames("path/to/video.mp4", "output/dir/", fps=25)
"""

import os
import cv2


def extract_frames(video_path, output_dir, fps=25):
    """
    Extract frames from a video at the specified sampling rate.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Sample every Nth frame (default: 25)
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % fps == 0:

            filename = os.path.join(output_dir, f"frame_{frame_id}.jpg")

            cv2.imwrite(filename, frame)

        frame_id += 1

    cap.release()


def process_dataset(dataset_dir, output_dir, fps=25):
    """
    Process all videos in a dataset directory.

    Args:
        dataset_dir: Root directory containing video files
        output_dir: Root directory for extracted frames
        fps: Frame sampling rate
    """

    for root, dirs, files in os.walk(dataset_dir):

        for file in files:

            if file.endswith(('.mp4', '.avi', '.mov')):

                video_path = os.path.join(root, file)

                rel_path = os.path.relpath(root, dataset_dir)
                video_name = os.path.splitext(file)[0]

                frame_dir = os.path.join(output_dir, rel_path, video_name)

                extract_frames(video_path, frame_dir, fps)

                print(f"Processed: {video_path}")
