import os
import cv2
import numpy as np
from pytube import YouTube
from tracker import FullBodyTracker
import argparse

class DatasetScraper:
    """
    Automated Sign Language Dataset Scraper.
    Downloads sign language videos, extracts frames, tracks keypoints, 
    and saves them as labeled sequences.
    """
    def __init__(self, output_dir="datasets/scraped"):
        self.output_dir = output_dir
        self.tracker = FullBodyTracker()
        os.makedirs(output_dir, exist_ok=True)

    def download_video(self, url, filename="temp_video"):
        """Downloads a video from YouTube."""
        yt = YouTube(url)
        print(f"Downloading: {yt.title}")
        stream = yt.streams.filter(file_extension="mp4").first()
        file_path = stream.download(output_path="/tmp", filename=filename + ".mp4")
        return file_path

    def process_video(self, video_path, label, seq_len=30):
        """Processes a video into labeled gesture sequences."""
        cap = cv2.VideoCapture(video_path)
        sequence = []
        sample_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract keypoints
            data = self.tracker.process(frame)
            sequence.append(data)
            
            # Save sequence when full
            if len(sequence) == seq_len:
                out_path = os.path.join(self.output_dir, label)
                os.makedirs(out_path, exist_ok=True)
                
                # Random filename to avoid collisions in distributed setup
                save_file = os.path.join(out_path, f"seq_{np.random.randint(1e6):06d}.npy")
                np.save(save_file, np.array(sequence))
                
                sequence = []
                sample_count += 1
                
        cap.release()
        print(f"Processed {sample_count} samples for label: {label}")

    def scrape_list(self, list_of_urls_with_labels):
        """Scrapes a list of YouTube URLs."""
        for url, label in list_of_urls_with_labels:
            try:
                video_path = self.download_video(url, label)
                self.process_video(video_path, label)
                os.remove(video_path) # Clean up
            except Exception as e:
                print(f"Error scraping {url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Dataset Scraper")
    parser.add_argument("--url", type=str, help="YouTube URL to scrape")
    parser.add_argument("--label", type=str, help="Label for the scraped video")
    args = parser.parse_args()
    
    scraper = DatasetScraper()
    if args.url and args.label:
        scraper.scrape_list([(args.url, args.label)])
    else:
        # Example list
        example_urls = [
            ("https://www.youtube.com/watch?v=...", "HELLO"),
            ("https://www.youtube.com/watch?v=...", "THANK_YOU")
        ]
        scraper.scrape_list(example_urls)
