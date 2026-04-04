import numpy as np

def hand_distances(hand):
    """Compute all pairwise distances between 21 hand landmarks (210 features)."""
    features = []
    if hand is None or len(hand) == 0:
        return np.zeros(210)
    for i in range(len(hand)):
        for j in range(i+1, len(hand)):
            features.append(np.linalg.norm(hand[i] - hand[j]))
    return np.array(features)

def body_orientation(pose):
    """Compute normalized shoulder vector."""
    if pose is None or len(pose) < 13:
        return np.zeros(3)
    left = pose[11]
    right = pose[12]
    direction = right - left
    norm = np.linalg.norm(direction)
    return direction / norm if norm != 0 else direction

def mouth_open(face):
    """Compute vertical distance between lips."""
    if face is None or len(face) < 15:
        return 0.0
    top = face[13]
    bottom = face[14]
    return np.linalg.norm(top - bottom)

def extract_features(data):
    """Combine all geometric features into a single vector (424 dimensions)."""
    lh = hand_distances(data.get("left_hand"))
    rh = hand_distances(data.get("right_hand"))
    body = body_orientation(data.get("pose"))
    mouth = np.array([mouth_open(data.get("face"))])
    return np.concatenate([lh, rh, body, mouth])
