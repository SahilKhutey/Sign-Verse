import numpy as np
import os
import sys

# Add the parent directory to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotics.motion_encoder import MotionEncoder, ConfigProfile

def test_holistic_pipeline():
    print("--- Testing SignVerse Holistic Feature Pipeline ---")
    
    # 1. Initialize Encoder with Robotics Profile (144 pos)
    encoder = MotionEncoder(profile=ConfigProfile.ROBOTICS)
    raw_landmarks = np.full(1629, 0.5) 
    
    # Left Shoulder (11): (0, 0, 0)
    # Left Elbow (13): (0, 0.5, 0)
    # Left Wrist (15): (0, 1.0, 0)
    raw_landmarks[11*3 : 11*3+3] = [0, 0, 0]
    raw_landmarks[13*3 : 13*3+3] = [0, 0.5, 0]
    raw_landmarks[15*3 : 15*3+3] = [0, 1.0, 0]
    
    # 2. Encode Frame
    features = encoder.encode_frame(raw_landmarks)
    
    print(f"Profile: {encoder.profile.value}")
    print(f"Positions Length: {len(features['positions'])}") # Should be 144
    print(f"Distances Length: {len(features['distances'])}") # Should be 11
    print(f"Angles Length: {len(features['vector_angles'])}") # Should be 4
    
    assert len(features['positions']) == 144
    assert len(features['distances']) == 11
    assert len(features['vector_angles']) == 4
    
    # 3. Test Sign Language Profile (291 pos)
    encoder_sign = MotionEncoder(profile=ConfigProfile.SIGN_LANGUAGE)
    features_sign = encoder_sign.encode_frame(raw_landmarks)
    print(f"Profile: {encoder_sign.profile.value} | Positions: {len(features_sign['positions'])}")
    assert len(features_sign['positions']) == 291
    
    # 4. Test Basics Profile (132 pos)
    encoder_basics = MotionEncoder(profile=ConfigProfile.BASICS)
    features_basics = encoder_basics.encode_frame(raw_landmarks)
    print(f"Profile: {encoder_basics.profile.value} | Positions: {len(features_basics['positions'])}")
    assert len(features_basics['positions']) == 132
    
    print("\n✅ Holistic Feature Pipeline Verified Successfully!")

if __name__ == "__main__":
    test_holistic_pipeline()
