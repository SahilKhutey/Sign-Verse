"""
Model Registry — Centralized deployment management for SignVerse AI.
Tracks and versions all 5 core model layers.
"""

import os
import json
import shutil
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any


class ModelRegistry:
    """
    Manages model versioning, metadata, and deployment staging.
    """

    def __init__(self, registry_dir: str = "deployment/model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.manifest_path = os.path.join(registry_dir, "manifest.json")
        self._load_manifest()

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "last_updated": datetime.now().isoformat(),
                "models": {}
            }

    def _save_manifest(self):
        self.manifest["last_updated"] = datetime.now().isoformat()
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def register_model(self, 
                       model_type: str, 
                       source_path: str, 
                       version: str = "latest",
                       metadata: Dict[str, Any] = None):
        """
        Register a model checkpoint into the registry.
        
        Args:
            model_type: One of [gesture, foundation, sign_transformer, diffusion, llm]
            source_path: Path to the .pt checkpoint
            version: Version string (e.g., 'v1.0.0', 'prod', 'best')
            metadata: Additional performance metrics or training info
        """
        dest_dir = os.path.join(self.registry_dir, model_type, version)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, "model.pt")
        shutil.copy2(source_path, dest_path)
        
        # Save metadata
        meta_data = {
            "registered_at": datetime.now().isoformat(),
            "source": source_path,
            "version": version,
            **(metadata or {})
        }
        
        with open(os.path.join(dest_dir, "metadata.json"), "w") as f:
            json.dump(meta_data, f, indent=2)
            
        # Update manifest
        if model_type not in self.manifest["models"]:
            self.manifest["models"][model_type] = {}
        
        self.manifest["models"][model_type][version] = meta_data
        self._save_manifest()
        
        print(f"✓ Registered {model_type} ({version}) at {dest_path}")

    def get_model_path(self, model_type: str, version: str = "latest") -> Optional[str]:
        """Retrieve the file path for a registered model."""
        if version == "latest":
            # Find the most recently registered version
            versions = self.manifest["models"].get(model_type, {})
            if not versions:
                return None
            latest_v = max(versions.keys(), key=lambda v: versions[v]["registered_at"])
            return os.path.join(self.registry_dir, model_type, latest_v, "model.pt")
            
        path = os.path.join(self.registry_dir, model_type, version, "model.pt")
        return path if os.path.exists(path) else None

    def list_models(self):
        """Print a summary of the registry status."""
        print(f"\n{'='*60}")
        print(f"  SignVerse Model Registry")
        print(f"  Location: {self.registry_dir}")
        print(f"{'='*60}")
        
        for mtype, versions in self.manifest["models"].items():
            print(f"\n[{mtype.upper()}]")
            for v, meta in versions.items():
                print(f"  {v:10} | Registered: {meta['registered_at'][:19]} | Source: {os.path.basename(meta['source'])}")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    registry = ModelRegistry()
    registry.list_models()
