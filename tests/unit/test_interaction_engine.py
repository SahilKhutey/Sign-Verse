"""
Unit tests for the SignVerse Spatial Interaction Mapper.
Validates Human-Object Interaction (HOI) detection and proximity mapping.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

# CRITICAL: Import mapper but mock its multi_detector dependency to prevent ultralytics load
with patch('core.perception.interaction_mapper.EntityDetection'):
    from core.perception.interaction_mapper import SpatialInteractionMapper, InteractionType
from core.data_models.skeleton import SkeletonFrame, BodyJoints, Vector3D, HandJoints

class TestInteractionMapper:
    """Test suite for the SpatialInteractionMapper."""
    
    @pytest.fixture
    def mapper(self):
        """Initialize the interaction mapper with standard thresholds."""
        return SpatialInteractionMapper(proximity_threshold=0.2, touch_threshold=0.05)

    @pytest.fixture
    def sample_entities(self):
        """Create sample entity detections using Mocks to avoid importing multi_detector."""
        bottle = Mock()
        bottle.class_id = 39
        bottle.class_name = "bottle"
        bottle.bbox = np.array([0.45, 0.45, 0.55, 0.55])
        bottle.confidence = 0.92
        bottle.entity_type = "object"
        
        phone = Mock()
        phone.class_id = 67
        phone.class_name = "cell phone"
        phone.bbox = np.array([0.8, 0.8, 0.9, 0.9])
        phone.confidence = 0.88
        phone.entity_type = "object"
        
        return [bottle, phone]

    @pytest.fixture
    def sample_skeleton(self):
        """Create a skeleton frame with one hand near the bottle."""
        right_hand = HandJoints(
            wrist=Vector3D(x=0.48, y=0.48, z=0.1, confidence=0.9)
        )
        body = BodyJoints(
            nose=Vector3D(x=0.5, y=0.2, z=0.1, confidence=0.95)
        )
        return SkeletonFrame(
            person_id="P1",
            frame_id=1,
            timestamp=0.0,
            body=body,
            right_hand=right_hand
        )

    def test_proximity_calculation(self, mapper, sample_skeleton, sample_entities):
        """Verify that proximity is correctly mapped for nearby objects."""
        # Update skeleton to be nearby but not touching
        sample_skeleton.right_hand.wrist.x = 0.6 # Centroid is 0.5, 0.5. Dist is ~0.1
        
        interactions = mapper.map_interactions(sample_skeleton, sample_entities)
        
        # Should detect proximity to bottle
        bottle_ints = [i for i in interactions if i.entity_class == "bottle"]
        assert len(bottle_ints) > 0
        assert bottle_ints[0].interaction_type == InteractionType.PROXIMITY

    def test_touch_detection(self, mapper, sample_skeleton, sample_entities):
        """Verify that touch is detected when a hand is very close to an object."""
        interactions = mapper.map_interactions(sample_skeleton, sample_entities)
        
        bottle_ints = [i for i in interactions if i.entity_class == "bottle"]
        assert len(bottle_ints) > 0
        assert bottle_ints[0].interaction_type == InteractionType.TOUCH

    def test_batch_mapping(self, mapper, sample_skeleton, sample_entities):
        """Verify batch interaction mapping for multiple persons."""
        results = mapper.batch_map([sample_skeleton], sample_entities)
        
        assert "P1" in results
        assert len(results["P1"]) > 0
        assert results["P1"][0].person_id == "P1"
