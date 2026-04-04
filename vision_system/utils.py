import numpy as np

def normalize_pose(pose):
    """Normalize pose by centering on the first landmark and scaling by shoulder width."""
    if pose is None or len(pose) < 13:
        return pose
    
    # Center on first landmark (usually nose or pelvis depending on model)
    center = pose[0]
    pose = pose - center

    # Scale by shoulder width
    left_shoulder = pose[11]
    right_shoulder = pose[12]
    scale = np.linalg.norm(left_shoulder - right_shoulder)

    return pose / scale if scale > 1e-6 else pose
