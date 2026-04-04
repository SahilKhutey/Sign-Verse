"""
Run OpenPose Dataset Pipeline

Batch extracts offline video datasets into high-fidelity OpenPose keypoints.
Produces either native `411-dim` (137x3) arrays or optionally mapped
1629-dim sequences to align with the core Foundation schema.
"""

import os
import glob
import cv2
import numpy as np
import argparse
import sys
import logging

from training.data_pipeline.openpose_extractor import OpenPoseExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def process_video(extractor, video_path, output_dir, map_to_foundation=False):
    """
    Process a single video using OpenPose and save keypoints sequence.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_file = os.path.join(output_dir, f"{video_id}.npy")
    
    # Skip if processed
    if os.path.exists(out_file):
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video {video_path}")
        return
        
    sequence = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extractor outputs native 137x3 (411 dim) per frame
        features_native = extractor.process_frame(frame)
        
        if map_to_foundation:
            features = extractor.convert_to_foundation_schema(features_native.flatten())
            sequence.append(features)
        else:
            sequence.append(features_native.flatten())
            
        frame_count += 1
        if frame_count % 100 == 0:
            logging.info(f"Processed {frame_count} frames for {video_id}...")
            
    cap.release()
    
    if len(sequence) > 0:
        seq_array = np.array(sequence, dtype=np.float32)
        np.save(out_file, seq_array)
        logging.info(f"Saved {video_id} -> {seq_array.shape[0]} frames of dim {seq_array.shape[1]}")
    else:
        logging.warning(f"Video {video_id} had 0 valid frames.")

def main():
    parser = argparse.ArgumentParser(description="Offline OpenPose Dataset Processor")
    parser.add_argument("--input", "-i", type=str, required=True, help="Directory containing .mp4 dataset")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for .npy arrays")
    parser.add_argument("--map-foundation", "-m", action="store_true", help="Map outputs to 1629-dim MediaPipe schema")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize OpenPose Extractor (offline configs)
    logging.info("Initializing OpenPose Extractor (Offline Heavy)")
    try:
        extractor = OpenPoseExtractor(num_gpu=1)
    except Exception as e:
        logging.error(f"Failed to initialize extractor: {e}")
        sys.exit(1)
        
    # Gather videos
    videos = glob.glob(os.path.join(args.input, "*.mp4")) + glob.glob(os.path.join(args.input, "*.avi"))
    logging.info(f"Found {len(videos)} videos to process in {args.input}")
    
    for i, vid in enumerate(videos):
        logging.info(f"[{i+1}/{len(videos)}] Processing {os.path.basename(vid)}...")
        process_video(extractor, vid, args.output, map_to_foundation=args.map_foundation)
        
    logging.info("OpenPose Dataset Pipeline successfully finished.")

if __name__ == "__main__":
    main()
