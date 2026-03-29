import numpy as np

def map_human_to_robot(human_pose):
    """
    Kinematic Mapping from Human Skeleton (MediaPipe Holistic) to Robot Joint space.
    
    This function extracts specific 3D landmarks from the unified Holistic array.
    Supports Body Pose (0-32), Face Mesh (33-500), and Hands (501+).
    
    Args:
        human_pose (np.ndarray): Flat array of 1629 values [x, y, z, ...].
    
    Returns:
        dict: Mapping of robot joint names to human coordinate arrays [x, y, z].
    """
    robot_joints = {}

    def get_landmark(idx):
        start = idx * 3
        # Ensure we don't go out of bounds (1629 values / 3 = 543 landmarks)
        if start + 3 > len(human_pose):
            return np.zeros(3)
        return human_pose[start : start + 3]

    # --- BODY POSE (MediaPipe Pose Indices) ---
    # 11: Left Shoulder, 13: Left Elbow, 15: Left Wrist
    # 12: Right Shoulder, 14: Right Elbow, 16: Right Wrist
    
    # LEFT ARM
    robot_joints["left_shoulder"] = get_landmark(11)
    robot_joints["left_elbow"] = get_landmark(13)
    robot_joints["left_wrist"] = get_landmark(15)

    # RIGHT ARM
    robot_joints["right_shoulder"] = get_landmark(12)
    robot_joints["right_elbow"] = get_landmark(14)
    robot_joints["right_wrist"] = get_landmark(16)

    # --- HAND ORIENTATION (Holistic Indices) ---
    # Left Hand Wrist: 501, Middle Finger Base: 510
    # Right Hand Wrist: 522, Middle Finger Base: 531
    
    robot_joints["left_hand_wrist"] = get_landmark(501)
    robot_joints["left_hand_index_base"] = get_landmark(506) # Index finger base for orientation
    
    robot_joints["right_hand_wrist"] = get_landmark(522)
    robot_joints["right_hand_index_base"] = get_landmark(527)

    # --- FACE (Optional ROI) ---
    # Nose tip: 33 (First face landmark)
    # Mouth: 99 + example offsets
    robot_joints["eye_center"] = get_landmark(33 + 168) # Landmark 168 is between eyes
    robot_joints["nose_tip"] = get_landmark(33 + 1)   # Landmark 1 is nose tip in mesh

    return robot_joints

if __name__ == "__main__":
    # Test with dummy data
    dummy_pose = np.zeros(1629)
    dummy_pose[11*3] = 0.5 # Left Shoulder X
    dummy_pose[501*3] = 0.8 # Left Hand Wrist X
    
    mapped = map_human_to_robot(dummy_pose)
    print("Mapped Robot Joints (Holistic):")
    for joint, pos in mapped.items():
        if np.any(pos):
            print(f"  {joint}: {pos}")
