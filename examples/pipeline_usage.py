"""
Example usage of the SignVerse Video Processing Pipeline.
Demonstrates end-to-end processing from video to normalized ML datasets.
"""
import time
import os
from core.pipeline.processing_pipeline import VideoProcessingPipeline, PipelineConfig
from core.processing.normalization import NORMALIZATION_PRESETS

def main():
    """Main execution example for standardized video processing."""
    # Step 1: Configuration for high-fidelity ML training datasets
    config = PipelineConfig(
        detection_confidence=0.6,
        pose_model_complexity=2,
        pose_min_confidence=0.5,
        normalization_preset="ml_training",
        target_fps=30,  # Process at full source frame rate
        storage_base_path="./data",
        database_url="sqlite:///./data/metadata.db"
    )
    
    # Step 2: Initialize the intelligent pipeline
    pipeline = VideoProcessingPipeline(config)
    
    try:
        # Step 3: Process a sample video stream
        video_path = "path/to/your/video.mp4" # Update with actual video path
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
            
        print(f"Starting Intelligent Pipeline for: {video_path}")
        results = pipeline.process_video(video_path, "example_video")
        
        print(f"\nProcessing Completed Successfully!")
        print(f"------------------------------------")
        print(f"Processed 3D Frames: {results['processed_frames']}")
        print(f"Total Persons Tracked: {results['processed_persons']}")
        print(f"Wall Time Tracking: {results['total_time']:.2f}s")
        print(f"Avg Real-time Factor: {results['avg_frame_time']*1000:.1f}ms/frame")
        
        # Step 4: Export to High-Performance ML Tier (Parquet/HDF5)
        print(f"\nExporting Tiered Datasets...")
        export_path = pipeline.export_for_training("example_video", "parquet")
        print(f"Exported Parquet Dataset: {export_path}")
        
        # Step 5: Audit Metadata Tier Statistics
        stats = pipeline.get_processing_stats("example_video")
        print(f"Database Analytics Sync: {stats}")
        
    except Exception as e:
        print(f"\nPipeline Processing Failure: {e}")
        
    finally:
        # Step 6: Graceful Resource Release
        pipeline.close()

def batch_processing_example():
    """Demonstrates high-throughput batch processing of multiple video sources."""
    config = PipelineConfig(
        normalization_preset="ml_training",
        target_fps=15  # Optimization: process every 2nd frame for massive datasets
    )
    
    pipeline = VideoProcessingPipeline(config)
    
    # Define multi-modal video sources
    video_paths = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4"
    ]
    
    all_results = []
    
    print(f"Initiating Batch Intake for {len(video_paths)} videos...")
    
    for video_path in video_paths:
        if not os.path.exists(video_path): continue
            
        try:
            print(f"Handling '{video_path}' tier-flow...")
            results = pipeline.process_video(video_path)
            all_results.append(results)
            
            print(f"Archived {video_path}: {results['processed_frames']} frames ingested.")
            
        except Exception as e:
            print(f"Intake Failure for {video_path}: {e}")
    
    pipeline.close()
    
    # Generate Multi-Video Analytics Summary
    if all_results:
        total_frames = sum(r['processed_frames'] for r in all_results)
        total_time = sum(r['total_time'] for r in all_results)
        
        print(f"\nBatch Processing Intelligence Summary")
        print(f"=====================================")
        print(f"Total Video Ingress: {len(all_results)}")
        print(f"Total ML-Ready Frames: {total_frames}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Aggregated Pipeline FPS: {total_frames/total_time:.1f}")

if __name__ == "__main__":
    main()
    # batch_processing_example()
