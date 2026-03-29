import cv2
import numpy as np
import time
from tracker import FullBodyTracker
from features import extract_features
from temporal import TemporalEngine
from utils import normalize_pose

def main():
    cap = cv2.VideoCapture(0)
    tracker = FullBodyTracker()
    temporal = TemporalEngine()

    print("--- SignVerse Motion Intelligence Engine ---")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Track
        data = tracker.process(frame)

        # 2. Normalize Pose
        data["pose"] = normalize_pose(data.get("pose"))

        # 3. Extract Geometric Features (424-dim)
        geo_features = extract_features(data)

        # 4. Compute Temporal Features (424-dim)
        vel_features = temporal.compute(geo_features)

        # 5. Final Intelligence Vector (848-dim)
        intelligence_vector = np.concatenate([geo_features, vel_features])

        # Debug Overlay
        cv2.putText(frame, f"Features: {len(intelligence_vector)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("SignVerse Intelligence Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
