"""
Unit tests for SignVerse 3D Physical Intelligence Data Models.
"""
import pytest
import numpy as np
from pydantic import ValidationError
from core.data_models import (
    SkeletonFrame, 
    BodyJoints, 
    HandJoints, 
    Vector3D, 
    JointType,
    LandmarkType
)

class TestDataModels:
    """Test validation and integrity of 3D data models."""
    
    def test_vector3d_validation(self):
        """Test Vector3D position validation."""
        # Valid vector
        v = Vector3D(x=1.5, y=2.0, z=-0.5, confidence=0.9)
        assert v.x == 1.5
        assert v.confidence == 0.9
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            Vector3D(x=0, y=0, z=0, confidence=1.5)

    def test_body_joints_mapping(self):
        """Test BodyJoints naming and access."""
        joints = BodyJoints()
        # Ensure we have common joints listed in the class fields
        fields = joints.model_fields.keys()
        assert "nose" in fields
        assert "left_shoulder" in fields
        # MediaPipe pose has 33 points, but we only define a subset in BodyJoints currently
        assert len(fields) >= 15

    def test_skeleton_frame_composition(self):
        """Test the composition of a complete SkeletonFrame."""
        # Mock body joints data
        body = BodyJoints(
            nose=Vector3D(x=0.5, y=0.5, z=0.0, confidence=0.8),
            left_shoulder=Vector3D(x=0.4, y=0.6, z=0.0, confidence=0.9)
        )
        
        frame = SkeletonFrame(
            person_id="P1",
            frame_id=10,
            timestamp=0.33,
            body=body,
            left_hand=HandJoints(),
            right_hand=HandJoints()
        )
        
        assert frame.person_id == "P1"
        assert frame.body.nose.x == 0.5
        assert frame.timestamp == 0.33
        
    def test_serialization_integrity(self):
        """Test Pydantic JSON serialization/deserialization."""
        v = Vector3D(x=0.5, y=0.5, z=0.5, confidence=1.0)
        json_data = v.model_dump_json()
        v_loaded = Vector3D.model_validate_json(json_data)
        
        assert v_loaded.x == v.x
        assert v_loaded.y == v.y
        assert v_loaded.z == v.z
