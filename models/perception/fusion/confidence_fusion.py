"""
Confidence fusion for combining confidence scores from multiple sources.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

class ConfidenceFusion:
    """Fuses confidence scores from multiple pose estimation models."""
    
    def __init__(self, fusion_method: str = "product"):
        """
        Initialize confidence fusion.
        
        Args:
            fusion_method: Fusion method ('product', 'average', 'minimum', 'maximum')
        """
        self.fusion_method = fusion_method
    
    def fuse_confidence(self, 
                       confidence_scores: List[Dict[str, Any]],
                       weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Fuse multiple confidence score dictionaries.
        
        Args:
            confidence_scores: List of confidence score dictionaries
            weights: Optional weights for each confidence source
            
        Returns:
            Fused confidence scores
        """
        if not confidence_scores:
            return {}
        
        if weights is None:
            weights = [1.0 / len(confidence_scores)] * len(confidence_scores)
        
        fused_scores = {}
        
        # Get all landmark types from all sources
        all_types = set()
        for conf in confidence_scores:
            all_types.update(conf.keys())
        
        for landmark_type in all_types:
            scores = []
            valid_weights = []
            
            for i, conf in enumerate(confidence_scores):
                if landmark_type in conf:
                    score = conf[landmark_type]
                    if isinstance(score, (int, float, np.ndarray)):
                        scores.append(score)
                        valid_weights.append(weights[i])
            
            if scores:
                if self.fusion_method == "product":
                    fused = self._product_fusion(scores, valid_weights)
                elif self.fusion_method == "average":
                    fused = self._average_fusion(scores, valid_weights)
                elif self.fusion_method == "minimum":
                    fused = self._minimum_fusion(scores)
                elif self.fusion_method == "maximum":
                    fused = self._maximum_fusion(scores)
                else:
                    fused = self._average_fusion(scores, valid_weights)
                
                fused_scores[landmark_type] = fused
        
        return fused_scores
    
    def _product_fusion(self, scores: List[Any], weights: List[float]) -> Any:
        """Product fusion of confidence scores."""
        if all(isinstance(score, (int, float)) for score in scores):
            # Scalar scores
            result = 1.0
            for score, weight in zip(scores, weights):
                result *= score ** weight
            return result
        else:
            # Array scores - element-wise product
            # Use the shape of the first array score
            first_array = next(s for s in scores if isinstance(s, np.ndarray))
            result = np.ones_like(first_array)
            for score, weight in zip(scores, weights):
                result *= score ** weight
            return result
    
    def _average_fusion(self, scores: List[Any], weights: List[float]) -> Any:
        """Weighted average fusion."""
        if all(isinstance(score, (int, float)) for score in scores):
            # Scalar scores
            total_weight = sum(weights)
            return sum(score * weight for score, weight in zip(scores, weights)) / total_weight
        else:
            # Array scores - weighted average
            first_array = next(s for s in scores if isinstance(s, np.ndarray))
            result = np.zeros_like(first_array)
            total_weight = sum(weights)
            for score, weight in zip(scores, weights):
                result += score * weight
            return result / total_weight
    
    def _minimum_fusion(self, scores: List[Any]) -> Any:
        """Minimum confidence fusion."""
        if all(isinstance(score, (int, float)) for score in scores):
            return min(scores)
        else:
            # Array scores - element-wise minimum
            first_array = next(s for s in scores if isinstance(s, np.ndarray))
            result = first_array.copy()
            for score in scores:
                if isinstance(score, np.ndarray):
                    result = np.minimum(result, score)
                else:
                    result = np.minimum(result, score)
            return result
    
    def _maximum_fusion(self, scores: List[Any]) -> Any:
        """Maximum confidence fusion."""
        if all(isinstance(score, (int, float)) for score in scores):
            return max(scores)
        else:
            # Array scores - element-wise maximum
            first_array = next(s for s in scores if isinstance(s, np.ndarray))
            result = first_array.copy()
            for score in scores:
               if isinstance(score, np.ndarray):
                   result = np.maximum(result, score)
               else:
                   result = np.maximum(result, score)
            return result
