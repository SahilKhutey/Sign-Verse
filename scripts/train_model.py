#!/usr/bin/env python3
"""
Model training script for SignVerse system.
Handles training of pose estimation and action recognition models.
"""
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from loguru import logger

from configs import get_config
from models.training.train_pose_model import train_pose_model
from models.training.train_action_model import train_action_model
from models.utils.experiment_tracker import ExperimentTracker

class ModelTrainer:
    """Orchestrates model training."""
    
    def __init__(self):
        self.config = get_config()
    
    def train_pose_model(self, experiment_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train pose estimation model."""
        try:
            # Setup experiment tracking
            tracker = ExperimentTracker(experiment_name)
            
            # Merge config overrides
            training_config = self.config.training.pose_model.copy()
            if config_overrides:
                training_config.update(config_overrides)
            
            # Log experiment configuration
            tracker.log_config({
                "model_type": "pose_estimation",
                "training_config": training_config,
                "start_time": datetime.now().isoformat()
            })
            
            # Train model
            logger.info(f"Starting pose model training: {experiment_name}")
            history = train_pose_model()
            
            # Save results
            results = {
                "experiment_name": experiment_name,
                "training_history": history,
                "final_metrics": history[-1] if history else {},
                "duration": (datetime.now() - datetime.fromisoformat(tracker.config['start_time'])).total_seconds()
            }
            
            tracker.save_artifact(results, "training_results.json")
            tracker.create_report()
            
            logger.success(f"Pose model training completed: {experiment_name}")
            return results
            
        except Exception as e:
            logger.error(f"Pose model training failed: {e}")
            raise
    
    def train_action_model(self, experiment_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train action recognition model."""
        # Similar implementation to train_pose_model
        pass
    
    def list_experiments(self) -> List[str]:
        """List all training experiments."""
        experiment_dir = Path(self.config.training.general.log_dir)
        if not experiment_dir.exists():
            return []
        
        return [d.name for d in experiment_dir.iterdir() if d.is_dir()]

def main():
    parser = argparse.ArgumentParser(description="SignVerse Model Trainer")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train pose model
    pose_parser = subparsers.add_parser('pose', help='Train pose estimation model')
    pose_parser.add_argument('--experiment', required=True, help='Experiment name')
    pose_parser.add_argument('--config', help='JSON string with config overrides')
    
    # Train action model
    action_parser = subparsers.add_parser('action', help='Train action recognition model')
    action_parser.add_argument('--experiment', required=True, help='Experiment name')
    action_parser.add_argument('--config', help='JSON string with config overrides')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List experiments')
    
    args = parser.parse_args()
    trainer = ModelTrainer()
    
    if args.command == 'pose':
        config_overrides = json.loads(args.config) if args.config else None
        results = trainer.train_pose_model(args.experiment, config_overrides)
        print(f"Training completed: {results}")
    
    elif args.command == 'action':
        config_overrides = json.loads(args.config) if args.config else None
        results = trainer.train_action_model(args.experiment, config_overrides)
        print(f"Training completed: {results}")
    
    elif args.command == 'list':
        experiments = trainer.list_experiments()
        print("Available experiments:")
        for exp in experiments:
            print(f"  - {exp}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
