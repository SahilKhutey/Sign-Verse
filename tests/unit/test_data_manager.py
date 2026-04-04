"""
Unit tests for the SignVerse Storage Manager.
Validates tiered storage across File System, SQLite, and Parquet/HDF5 layers.
"""
import pytest
import os
import shutil
import tempfile
from pathlib import Path
from core.storage.storage_manager import StorageManager
from core.data_models.skeleton import SkeletonFrame, BodyJoints, Vector3D

class TestStorageManager:
    """Test suite for the unified StorageManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for storage tests."""
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp)

    @pytest.fixture
    def storage(self, temp_dir):
        """Initialize StorageManager with temporary paths and ensure cleanup."""
        db_path = os.path.join(temp_dir, "test_metadata.db")
        sm = StorageManager(base_path=temp_dir, database_url=f"sqlite:///{db_path}")
        yield sm
        sm.close()

    @pytest.fixture
    def sample_frame(self):
        """Create a sample SkeletonFrame for testing."""
        body = BodyJoints(
            nose=Vector3D(x=0.5, y=0.5, z=0.0, confidence=0.9),
            left_shoulder=Vector3D(x=0.4, y=0.6, z=0.2, confidence=0.8),
            right_shoulder=Vector3D(x=0.6, y=0.6, z=0.2, confidence=0.8)
        )
        return SkeletonFrame(
            person_id="P1",
            frame_id=101,
            timestamp=3.36,
            overall_confidence=0.85,
            body=body
        )

    def test_directory_structure(self, storage, temp_dir):
        """Verify that storage subdirectories are correctly initialized."""
        # StorageManager initializes FileStorage and HighPerformanceStorage
        # which create directories on demand or during init
        assert os.path.exists(temp_dir)
        # Check if subfolders were created by component managers
        assert os.path.exists(os.path.join(temp_dir, "raw"))
        assert os.path.exists(os.path.join(temp_dir, "processed"))

    def test_pose_archival_json(self, storage, sample_frame):
        """Verify that pose frames are correctly archived as JSON."""
        video_name = "test_video_01"
        results = storage.save_pose_frame(sample_frame, video_name)
        
        assert "file_system" in results
        json_path = results["file_system"]
        assert os.path.exists(json_path)
        assert json_path.endswith(".json")

    def test_metadata_indexing_sqlite(self, storage, sample_frame):
        """Verify that metadata is correctly indexed in the SQLite database."""
        video_name = "test_video_01"
        meta = {"path": "test.mp4", "fps": 30}
        results = storage.save_pose_frame(sample_frame, video_name, video_metadata=meta)
        
        assert "metadata_db" in results
        db_res = results["metadata_db"]
        assert db_res["video_id"] is not None
        assert db_res["frame_id"] is not None

    def test_hpc_export_parquet(self, storage, sample_frame, temp_dir):
        """Verify high-performance export to Parquet format."""
        video_name = "test_video_01"
        # 1. Save a frame to populate the temporal tracker
        storage.save_pose_frame(sample_frame, video_name)
        
        # 2. Export to Parquet
        parquet_path = storage.export_for_training(video_name, format="parquet")
        
        assert os.path.exists(parquet_path)
        assert parquet_path.endswith(".parquet")
        assert "processed" in parquet_path
