import cv2
import numpy as np
import os
import sys

# Add the parent directory to sys.path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_capture.pose_extractor import NeuralMotionCapture
from robotics.motion_encoder import MotionEncoder
from robotics.movement_db import MovementDatabase
from robotics.retargeting import map_human_to_robot
from robotics.policy_model import RobotPolicy

class RobotControlPipeline:
    """
    Advanced Orchestrator for SignVerse Robotics:
    Captures Holistic data, encodes into Motion Features, and performs AI inference.
    """

    def __init__(self, profile=ConfigProfile.ROBOTICS, db_path="movements_library"):
        self.capture = NeuralMotionCapture()
        self.encoder = MotionEncoder(profile=profile)
        self.db = MovementDatabase(path=db_path)
        
        # Calculate Input Dimension based on Profile and Features
        # positions * 2 (pos+vel) + 11 distances + 4 angles
        if profile == ConfigProfile.ROBOTICS:
            pos_dim = 144
        elif profile == ConfigProfile.SIGN_LANGUAGE:
            pos_dim = 291
        elif profile == ConfigProfile.BASICS:
            pos_dim = 132
        else: # FULL
            pos_dim = 1629
            
        self.input_dim = (pos_dim * 2) + 11 + 4
        self.output_dim = 18 
        
        self.policy = RobotPolicy(self.input_dim, self.output_dim)

    def run_recording_session(self, label="GRAB_OBJECT"):
        """Record movement sequence with advanced feature extraction."""
        cap = cv2.VideoCapture(0)
        print(f"Holistic Recording: {label} | Profile: {self.encoder.profile.value}")
        
        self.current_sequence = []
        self.encoder.reset()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            raw_pose = self.capture.extract_pose(frame)
            features = self.encoder.encode_frame(raw_pose)
            
            # Concatenate selected features for storage/learning
            # [Pos, Vel, Distances, Angles]
            vector = np.concatenate([
                features["positions"], 
                features["velocities"],
                features["distances"],
                features["vector_angles"]
            ])
            
            self.current_sequence.append(vector)
            
            # Rendering
            frame = self.capture.visualize(frame)
            cv2.putText(frame, f"REC: {label} ({len(self.current_sequence)} frames)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("SignVerse Holistic Recorder", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()
        if self.current_sequence:
            self.db.save_movement(self.current_sequence, label)

    def run_inference_live(self):
        """AI Inference using trained RobotPolicy with holistic features."""
        cap = cv2.VideoCapture(0)
        self.encoder.reset()
        self.policy.eval()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            raw_pose = self.capture.extract_pose(frame)
            features = self.encoder.encode_frame(raw_pose)
            
            vector = np.concatenate([
                features["positions"], features["velocities"],
                features["distances"], features["vector_angles"]
            ])
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(vector).unsqueeze(0)
                # Ensure input_tensor matches policy input_dim
                if input_tensor.shape[1] == self.input_dim:
                    robot_action = self.policy(input_tensor)
                    status = f"Action: {robot_action.numpy()[0][:3]}"
                else:
                    status = f"Dim Mismatch: {input_tensor.shape[1]} vs {self.input_dim}"
            
            frame = self.capture.visualize(frame)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("SignVerse Robotics AI", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
                
        cap.release()
        cv2.destroyAllWindows()
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import torch # Import here to ensure it's available for the class
    
    pipeline = RobotControlPipeline()
    
    # Example: Start a recording session
    # print("Starting Recording Session...")
    # pipeline.run_recording_session(label="GREETING")
    
    print("SignVerse Robotics Pipeline Initialized.")
    print("Ready for: Recording -> Training -> Execution")
    
    # Note: Policy Training would typically happen offline using MovementDatabase loads
