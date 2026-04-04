"""
Pose estimation inference service.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from configs import get_config
from models.architectures.transformer_model import PoseTransformer
from core.data_models import PoseData, Keypoint

class PosePredictor:
    """Pose estimation inference engine."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = get_config()
        self.model_path = model_path or self.config.inference.pose.model_path
        self.device = self._setup_device()
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"Pose predictor initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup inference device based on hardware and config."""
        if self.config.inference.pose.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_model(self) -> PoseTransformer:
        """Load trained PoseTransformer weights from checkpoint."""
        if not Path(self.model_path).exists():
            # In development, we might not have the model yet
            logger.warning(f"Model checkpoint not found at {self.model_path}. creating an uninitialized model for testing.")
            return PoseTransformer(
                input_dim=51,
                hidden_dim=256,
                num_layers=4,
                num_heads=8,
                dropout=0.1
            ).to(self.device)
        
        # Load model configuration from checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint.get('config', {})
        
        model = PoseTransformer(
            input_dim=model_config.get('input_dim', 51),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 4),
            num_heads=model_config.get('num_heads', 8),
            dropout=model_config.get('dropout', 0.1)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess_poses(self, pose_data: List[PoseData]) -> torch.Tensor:
        """Flatten PoseData objects into model-ready tensors (Batch, Seq, Features)."""
        features = []
        
        for pose in pose_data:
            frame_features = []
            for kp in pose.keypoints:
                # Flat format: x, y, confidence
                frame_features.extend([kp.x, kp.y, kp.confidence])
            features.append(frame_features)
        
        # Convert to tensor and add batch dimension
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def postprocess_output(self, output: torch.Tensor, original_poses: List[PoseData]) -> List[PoseData]:
        """Convert refined model tensors back to canonical PoseData objects."""
        # Remove batch dimension
        output = output.squeeze(0).detach().cpu().numpy()
        
        refined_poses = []
        for i, original_pose in enumerate(original_poses):
            if i >= len(output):
                break
                
            refined_keypoints = []
            frame_output = output[i]
            
            for j, original_kp in enumerate(original_pose.keypoints):
                if j * 3 + 2 < len(frame_output):
                    refined_keypoints.append(Keypoint(
                        id=original_kp.id,
                        name=original_kp.name,
                        x=float(frame_output[j * 3]),
                        y=float(frame_output[j * 3 + 1]),
                        z=original_kp.z,
                        confidence=float(frame_output[j * 3 + 2]),
                        visible=original_kp.visible
                    ))
            
            # Map back to standardized model
            refined_poses.append(PoseData(
                version=original_pose.version,
                source_video=original_pose.source_video,
                frame_number=original_pose.frame_number,
                timestamp=original_pose.timestamp,
                camera_id=original_pose.camera_id,
                subject_id=original_pose.subject_id,
                keypoints=refined_keypoints,
                bounding_box=original_pose.bounding_box
            ))
        
        return refined_poses
    
    def predict(self, pose_sequence: List[PoseData]) -> List[PoseData]:
        """Refine a sequence of poses through the Transformer model."""
        if not pose_sequence:
            return []
        
        # Preprocessing stage
        input_tensor = self.preprocess_poses(pose_sequence)
        input_tensor = input_tensor.to(self.device)
        
        # Core model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocessing stage
        refined_poses = self.postprocess_output(output, pose_sequence)
        
        return refined_poses
    
    def predict_batch(self, batch_sequences: List[List[PoseData]]) -> List[List[PoseData]]:
        """Process multiple sequences concurrently."""
        # Simple iterative batch processing for now
        results = []
        for sequence in batch_sequences:
            results.append(self.predict(sequence))
        return results

# Singleton pose predictor initialization
try:
    pose_predictor = PosePredictor()
except Exception as e:
    logger.error(f"Failed to initialize global pose predictor: {e}")
    pose_predictor = None

def get_pose_predictor() -> Optional[PosePredictor]:
    """Get singleton pose predictor instance."""
    return pose_predictor
