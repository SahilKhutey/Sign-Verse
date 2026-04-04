import re
from typing import Dict, Any, List

class MetadataFilter:
    def __init__(self):
        # Positive keywords indicating human motion content
        self.positive_keywords = [
            'workout', 'exercise', 'training', 'practice', 'routine',
            'dance', 'yoga', 'pilates', 'calisthenics', 'gymnastics',
            'construction', 'carpentry', 'woodworking', 'building',
            'painting', 'drawing', 'art', 'craft', 'sculpting',
            'cooking', 'baking', 'culinary', 'recipe',
            'sports', 'football', 'basketball', 'tennis', 'soccer',
            'martial', 'boxing', 'karate', 'judo', 'taekwondo',
            'writing', 'calligraphy', 'handwriting',
            'blacksmith', 'metalworking', 'forging'
        ]
        
        # Negative keywords to exclude
        self.negative_keywords = [
            'review', 'unboxing', 'reaction', 'compilation', 'highlights',
            'music video', 'lyrics', 'official video', 'trailer',
            'interview', 'podcast', 'talk show', 'news', 'documentary',
            'video game', 'gaming', 'walkthrough',
            'animation', 'animated', 'cartoon',
            'movie', 'film', 'series', 'episode'
        ]
        
        # Channel blacklist
        self.blacklisted_channels = [
            'music', 'news', 'gaming', 'movie', 'trailer'
        ]
    
    def filter_video(self, video: Dict[str, Any]) -> bool:
        """Apply metadata filtering rules to a video"""
        title = video.get('title', '').lower()
        channel = video.get('channel', '').lower()
        duration = video.get('duration', 0)
        view_count = video.get('view_count', 0)
        
        # Duration filters
        if duration < 30:  # Too short (30 seconds)
            return False
        if duration > 3600:  # Too long (1 hour)
            return False
            
        # View count filter (optional)
        if view_count < 100:  # Very low view count
            return False
            
        # Channel blacklist
        if any(blacklisted in channel for blacklisted in self.blacklisted_channels):
            return False
            
        # Negative keyword filter
        if any(negative in title for negative in self.negative_keywords):
            return False
            
        # Positive keyword requirement
        # Check title and channel for positive keywords
        has_positive = any(positive in title for positive in self.positive_keywords) or \
                       any(positive in channel for positive in self.positive_keywords)
                       
        if not has_positive:
            return False
            
        return True
    
    def filter_batch(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a batch of videos"""
        return [video for video in videos if self.filter_video(video)]

# Global filter instance
metadata_filter = MetadataFilter()
