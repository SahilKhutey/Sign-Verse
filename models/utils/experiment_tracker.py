"""
Experiment tracking and model management.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
from loguru import logger

from configs import get_config

class ExperimentTracker:
    """Tracks training experiments and model versions."""
    
    def __init__(self, experiment_name: str):
        self.config = get_config()
        self.experiment_name = experiment_name
        self.experiment_dir = Path(self.config.training.general.log_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.config_file = self.experiment_dir / "config.json"
        self.artifacts_dir = self.experiment_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> List[Dict[str, Any]]:
        """Load existing metrics if available."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metrics file: {e}")
                return []
        return []
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration and metadata."""
        config_data = config.copy()
        config_data['experiment_name'] = self.experiment_name
        config_data['start_time'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Logged experiment config: {self.config_file}")
    
    def log_metrics(self, metrics: Dict[str, Any], epoch: int):
        """Log training and validation metrics for an epoch."""
        metric_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics.append(metric_entry)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_artifact(self, artifact: Any, filename: str):
        """Save training artifact (model weights, plots, or configs)."""
        artifact_path = self.artifacts_dir / filename
        
        try:
            if isinstance(artifact, torch.nn.Module):
                torch.save(artifact.state_dict(), artifact_path)
            elif isinstance(artifact, dict) or isinstance(artifact, list):
                # If it's a serializable object but not a torch object
                torch.save(artifact, artifact_path)
            else:
                torch.save(artifact, artifact_path)
            logger.info(f"Saved artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to save artifact {filename}: {e}")
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best model checkpoint based on validation loss."""
        if not self.metrics:
            return None
            
        best_epoch = min(self.metrics, key=lambda x: x.get('val_loss', float('inf'))).get('epoch')
        checkpoints = list(self.artifacts_dir.glob(f"*epoch_{best_epoch}*.pt"))
        
        if not checkpoints:
            # Fallback to the latest checkpoint if specific epoch not found
            checkpoints = list(self.artifacts_dir.glob("*.pt"))
            if not checkpoints:
                return None
            return sorted(checkpoints)[-1]
            
        return checkpoints[0]
    
    def create_report(self):
        """Create a summary JSON report for the experiment."""
        if not self.metrics:
            logger.warning("No metrics logged for this experiment.")
            return {}

        report = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.metrics),
            'final_metrics': self.metrics[-1],
            'best_metrics': min(self.metrics, key=lambda x: x.get('val_loss', float('inf'))),
            'artifacts': [p.name for p in self.artifacts_dir.iterdir()]
        }
        
        report_path = self.experiment_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.success(f"Experiment report created: {report_path}")
        return report
