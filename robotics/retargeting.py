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

def map_vibe_to_robot(vibe_joints):
    """
    Kinematic Mapping from VIBE 3D SMPL Joints [81, 3] to Robot Joint space.
    VIBE provides authoritative 3D (X, Y, Z) coordinates.

    Args:
        vibe_joints (np.ndarray): Array of shape [81, 3].
    
    Returns:
        dict: High-fidelity mapping of robot joint positions.
    """
    return robot_joints

def map_vibe_to_robot(vibe_joints):
    """
    Kinematic Mapping from VIBE 3D SMPL Joints [81, 3] to Robot Joint space.
    VIBE provides authoritative 3D (X, Y, Z) coordinates.

    Args:
        vibe_joints (np.ndarray): Array of shape [81, 3].
    
    Returns:
        dict: High-fidelity mapping of robot joint positions.
    """
    # Mapping for common SMPL-to-Robot joints
    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee, 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3
    # 13: L_Shoulder, 14: R_Shoulder, 16: L_Elbow, 17: R_Elbow, 18: L_Wrist, 19: R_Wrist
    
    robot_joints = {}
    
    # Body Core (3D stabilized)
    robot_joints["base_pelvis"] = vibe_joints[0]
    robot_joints["spine_center"] = vibe_joints[9]
    
    # Left Arm
    robot_joints["left_shoulder"] = vibe_joints[13]
    robot_joints["left_elbow"] = vibe_joints[16]
    robot_joints["left_wrist"] = vibe_joints[18]
    
    # Right Arm
    robot_joints["right_shoulder"] = vibe_joints[14]
    robot_joints["right_elbow"] = vibe_joints[17]
    robot_joints["right_wrist"] = vibe_joints[19]
    
    # Legs (Critical for mobile robotics)
    robot_joints["left_hip"] = vibe_joints[1]
    robot_joints["left_knee"] = vibe_joints[4]
    robot_joints["left_ankle"] = vibe_joints[7]
    
    robot_joints["right_hip"] = vibe_joints[2]
    robot_joints["right_knee"] = vibe_joints[5]
    robot_joints["right_ankle"] = vibe_joints[8]
    
    return robot_joints

def map_smplx_to_robot(pose_params, shape_params=None):
    """
    Parametric Mapping from SMPL-X Pose parameters [162,] to Robot Joint space.
    Converts relative 3D axis-angles into motor rotations.
    Standardized for 54-joint SMPL-X rig.
    """
    robot_joints = {}
    
    # 1. Torso & Arms
    # Standard SMPL-X joints (Body):
    # 0: Root, 1/2: Hips, 12: Neck, 13/14: Shoulders, 16/17: Elbows, 18/19: Wrists
    
    # Left Arm
    robot_joints["left_shoulder_rot"] = pose_params[13*3 : 13*3 + 3]
    robot_joints["left_elbow_rot"] = pose_params[16*3 : 16*3 + 3]
    robot_joints["left_wrist_rot"] = pose_params[18*3 : 18*3 + 3]
    
    # Right Arm
    robot_joints["right_shoulder_rot"] = pose_params[14*3 : 14*3 + 3]
    robot_joints["right_elbow_rot"] = pose_params[17*3 : 17*3 + 3]
    robot_joints["right_wrist_rot"] = pose_params[19*3 : 19*3 + 3]
    
    # 2. Hands (Expressive Mapping)
    # L_Hand Start: Joint 22. R_Hand Start: Joint 37.
    # Each hand has 15 joints (5 fingers * 3 joints).
    for i in range(15):
        l_idx = 22 + i
        r_idx = 37 + i
        robot_joints[f"left_hand_joint_{i}_rot"] = pose_params[l_idx*3 : l_idx*3 + 3]
        robot_joints[f"right_hand_joint_{i}_rot"] = pose_params[r_idx*3 : r_idx*3 + 3]
        
    # 3. Jaw (Sign Language Expression)
    # Joint 52
    robot_joints["jaw_rot"] = pose_params[52*3 : 52*3 + 3]
        
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

    # Test SMPL-X Mapping
    dummy_smplx = np.zeros(162)
    dummy_smplx[13*3] = 1.0 # Left Shoulder rotation
    mapped_smplx = map_smplx_to_robot(dummy_smplx)
    print(f"Mapped SMPL-X Left Shoulder Rotation: {mapped_smplx['left_shoulder_rot']}")
