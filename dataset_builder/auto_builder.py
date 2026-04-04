"""
SignVerse Dataset Auto-Builder — Scaling to 100M+ Samples
Automatically scrapes, extracts landmarks, and labels sign language clips.

Workflow:
1. Video Source (YouTube/Web) -> yt-dlp
2. Speech-to-Text (Whisper) -> Align text with timestamps.
3. Holistic Tracking -> Extract landmarks at 30+ FPS.
4. Feature Engineering -> 848-dim vectors.
"""

import os
import json
import time
import subprocess
import numpy as np
from vision_system.motion_intelligence import MotionIntelligence

class DatasetAutoBuilder:
    def __init__(self, output_dir="datasets/auto_built"):
        self.output_dir = output_dir
        self.mi = MotionIntelligence()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "vectors"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    def download_video(self, url):
        """Downloads a video from a URL (e.g. YouTube) using yt-dlp."""
        video_id = url.split("v=")[-1]
        output_path = f"temp/{video_id}.mp4"
        os.makedirs("temp", exist_ok=True)
        
        try:
            print(f"Downloading video: {url} -> {output_path}")
            subprocess.run(["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]", "-o", output_path, url], check=True)
            return output_path
        except Exception as e:
            print(f"yt-dlp download failed: {e}")
            return None

    def align_speech(self, video_path):
        """Uses Whisper to get timestamped transcript (aligned to source audio)."""
        from faster_whisper import WhisperModel
        print(f"Aligning speech for: {video_path}")
        try:
            model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
            segments, info = model.transcribe(video_path, beam_size=5)
            
            aligned = []
            for segment in segments:
                aligned.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
            return aligned
        except Exception as e:
            print(f"Whisper alignment failed: {e}")
            return []

    def extract_motion(self, video_path, segments):
        """Performs frame-by-frame 848-dim extraction on aligned segments."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        for seg in segments:
            start_frame = int(seg["start"] * fps)
            end_frame = int(seg["end"] * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            seq_vectors = []
            
            for f in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret: break
                
                out = self.mi.process_frame(frame)
                seq_vectors.append(out["motion_vector"])
                
            if seq_vectors:
                vectors_np = np.stack(seq_vectors)
                sample_id = f"sample_{int(time.time() * 1000)}"
                save_path = os.path.join(self.output_dir, "vectors", f"{sample_id}.npy")
                np.save(save_path, vectors_np)
                
                meta = {
                    "id": sample_id,
                    "text": seg["text"],
                    "fps": fps,
                    "frames": len(seq_vectors),
                    "vector_path": save_path
                }
                
                with open(os.path.join(self.output_dir, "metadata", f"{sample_id}.json"), "w") as f:
                    json.dump(meta, f)
                
                results.append(meta)
        
        cap.release()
        return results

    def run_pipeline(self, url):
        """Complete auto-build pipeline for a single URL source."""
        start = time.time()
        # video_path = self.download_video(url)
        # For now, simulate with a local video if provided or exit
        print(f"Running auto-build for source: {url}")
        # segments = self.align_speech(video_path)
        # manifest = self.extract_motion(video_path, segments)
        print(f"Pipeline finished in {int(time.time() - start)}s")
        # return manifest

if __name__ == "__main__":
    builder = DatasetAutoBuilder()
    builder.run_pipeline("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
