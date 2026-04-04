"""
Training pipeline for pose estimation model.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Any, List
import json
from loguru import logger

from configs import get_config
from models.architectures.transformer_model import PoseTransformer
from models.utils.training_utils import ModelTrainer
from core.data_manager import data_manager

class PoseDataset(Dataset):
    """Dataset for pose sequence training."""
    
    def __init__(self, data_dir: str, split: str = "train", seq_length: int = 30):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_length = seq_length
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load processed pose data samples from the Data Layer."""
        samples = []
        pose_files = list((self.data_dir / "processed" / "poses").glob("*.json"))
        
        for pose_file in pose_files:
            try:
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                    samples.append(pose_data)
            except Exception as e:
                logger.warning(f"Failed to load {pose_file}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert keypoints to flat tensor (x, y, confidence)
        keypoints = []
        for kp in sample['keypoints']:
            keypoints.extend([kp['x'], kp['y'], kp['confidence']])
        
        features = torch.tensor(keypoints, dtype=torch.float32)
        
        return {
            'features': features,
            'source_video': sample['source_video'],
            'frame_number': sample['frame_number']
        }

def prepare_dataloaders() -> Dict[str, DataLoader]:
    """Prepare train, validation, and test dataloaders based on system config."""
    config = get_config()
    
    # Load dataset from the base path defined in the data configs
    full_dataset = PoseDataset(config.data.base_path)
    
    if len(full_dataset) < 10:
        logger.warning("Dataset too small for proper splitting. Using all for train/val.")
        return {'train': DataLoader(full_dataset, batch_size=1), 'val': DataLoader(full_dataset, batch_size=1), 'test': DataLoader(full_dataset, batch_size=1)}

    # Split dataset based on global constants
    train_size = int(config.data.train_split * len(full_dataset))
    val_size = int(config.data.val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config.training.pose_model.batch_size,
            shuffle=True,
            num_workers=0 # Disable workers for basic local run
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config.training.pose_model.batch_size,
            shuffle=False,
            num_workers=0
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=config.training.pose_model.batch_size,
            shuffle=False,
            num_workers=0
        )
    }
    
    return dataloaders

def train_pose_model():
    """Main training entry for the PoseTransformer."""
    logger.info("Starting pose model training lifecycle...")
    
    config = get_config()
    model_config = config.training.pose_model
    
    # Initialize architecture from configs
    model = PoseTransformer(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
        num_classes=None  # Mode: Refinement
    )
    
    # Prepare dataloaders from processed data layer
    dataloaders = prepare_dataloaders()
    
    # Initialize high-level trainer utility
    trainer = ModelTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config.training.general
    )
    
    # Start training fit cycle
    history = trainer.fit()
    
    # Persist the final weights to the checkpoints directory
    trainer.save_checkpoint("final_pose_model.pt")
    
    logger.success("Pose model training cycle successfully completed!")
    return history

if __name__ == "__main__":
    train_pose_model()
