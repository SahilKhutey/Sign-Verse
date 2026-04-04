import os
from typing import Dict, Any, List
from loguru import logger

from .youtube_pipeline import youtube_pipeline
from .content_analyzer import content_analyzer

class VideoScorer:
    def __init__(self):
        # Configuration for selection weights
        self.score_weights = {
            'human_presence': 0.3,
            'motion_quality': 0.3,
            'trackability': 0.2,
            'content_relevance': 0.2
        }
        self.acceptance_threshold = 0.65 # Slightly adjusted per user feedback
    
    def score_video(self, video: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a video based on multiple visual criteria.
        Downloads the video temporarily to perform analysis.
        """
        logger.info(f"Scoring video: {video.get('title', video.get('id'))}")
        
        # Download video temporarily for analysis
        download_result = youtube_pipeline.download_video(video['url'])
        if not download_result['success']:
            return {**video, 'score': 0, 'status': 'failed_download'}
        
        video_path = download_result['path']
        
        try:
            # Sample frames for analysis (using default density in analyzer)
            frames = content_analyzer.sample_frames(video_path)
            if not frames:
                self._cleanup(video_path)
                return {**video, 'score': 0, 'status': 'no_frames'}
            
            # Calculate individual scores
            human_score = content_analyzer.detect_humans(frames)
            motion_score = min(1.0, content_analyzer.compute_motion_score(frames) / 10.0)
            
            # Use the total score from assess_trackability (which includes pose + interaction)
            track_results = content_analyzer.assess_trackability(frames)
            track_score = track_results['total']
            
            # Content classification based on heuristics
            content_scores = content_analyzer.classify_heuristics(frames)
            relevance_score = max(content_scores.values()) if content_scores else 0.0
            
            # Calculate final weighted score
            final_score = (
                self.score_weights['human_presence'] * human_score +
                self.score_weights['motion_quality'] * motion_score +
                self.score_weights['trackability'] * track_score +
                self.score_weights['content_relevance'] * relevance_score
            )
            
            # Determine primary category
            primary_category = "unknown"
            if content_scores:
                primary_category = max(content_scores.items(), key=lambda x: x[1])[0]
            
            # Clean up downloaded video after analysis
            self._cleanup(video_path)
            
            return {
                **video,
                'score': round(final_score, 4),
                'human_score': round(human_score, 4),
                'motion_score': round(motion_score, 4),
                'trackability_score': round(track_score, 4),
                'relevance_score': round(relevance_score, 4),
                'category': primary_category,
                'status': 'accepted' if final_score >= self.acceptance_threshold else 'rejected',
                'score_breakdown': content_scores
            }
            
        except Exception as e:
            logger.error(f"Scoring failed for {video.get('id')}: {e}")
            self._cleanup(video_path)
            return {**video, 'score': 0, 'status': f'error: {str(e)}'}
    
    def score_batch(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch of videos sequentially."""
        scored_videos = []
        for video in videos:
            scored = self.score_video(video)
            scored_videos.append(scored)
        return scored_videos

    def _cleanup(self, video_path: str):
        """Helper to remove temporary video file."""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {video_path}: {e}")

# Global scorer instance
video_scorer = VideoScorer()
