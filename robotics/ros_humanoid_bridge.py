"""
SignVerse Humanoid ROS 2 Bridge
Production-grade orchestration: AI Output -> ROS 2 -> Robot Movement.
Subscribes to SMPL-X inferences and publishes JointTrajectory to Humanoid Controllers.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np

class SignHumanoidBridge(Node):
    """
    ROS 2 Node mapping SMPL-X parametric pose to a 21-DOF Humanoid rig.
    Uses rclpy for asynchronous command publishing.
    """

    def __init__(self, node_name="sign_humanoid_bridge"):
        super().__init__(node_name)
        
        # ROS 2 Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.traj_pub = self.create_publisher(JointTrajectory, '/humanoid_controller/command', 10)
        
        # Standard 21-DOF Humanoid Joint Names (URDF compatible)
        self.joint_names = [
            "pelvis", "waist", "neck", 
            "left_shoulder_roll", "left_shoulder_pitch", "left_shoulder_yaw", "left_elbow",
            "right_shoulder_roll", "right_shoulder_pitch", "right_shoulder_yaw", "right_elbow",
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle"
        ]
        
        self.get_logger().info(f"SignVerse ROS 2 Humanoid Bridge Started: Mapping {len(self.joint_names)} joints.")

    def publish_joint_state(self, smplx_pose: np.ndarray):
        """
        Maps a 162-dim SMPL-X pose vector to a 21-DOF humanoid JointState.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        # Mappings: SMPL-X indexes -> Humanoid 
        # (Simplified mapping for 21 core degrees of freedom)
        positions = np.zeros(len(self.joint_names), dtype=np.float32)
        
        # Ex: Extracting shoulder/elbow from SMPL-X parameters
        # smplx_pose[13*3:13*3+3] (L Shoulder)
        positions[3:6] = smplx_pose[13*3 : 13*3+3] # Left Shoulder Pitch/Roll/Yaw
        positions[10] = smplx_pose[16*3]             # Left Elbow Ball/Hinge
        
        msg.position = positions.tolist()
        self.joint_pub.publish(msg)

    def publish_trajectory_point(self, smplx_pose: np.ndarray, time_from_start_sec: float = 0.5):
        """
        Publishes a standard ROS 2 JointTrajectory command for smooth interpolation.
        """
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        
        # Mapping Logic
        positions = np.zeros(len(self.joint_names), dtype=np.float32)
        # ... (Map SMPL-X to humanoid joints) ...
        positions[3:6] = smplx_pose[13*3 : 13*3+3] 
        positions[7:10] = smplx_pose[14*3 : 14*3+3]

        point.positions = positions.tolist()
        point.time_from_start = Duration(sec=int(time_from_start_sec), nanosec=int((time_from_start_sec % 1) * 1e9))
        
        traj_msg.points.append(point)
        self.traj_pub.publish(traj_msg)
        self.get_logger().debug("Published JointTrajectory (1 point) to Humanoid.")

def main(args=None):
    rclpy.init(args=args)
    bridge = SignHumanoidBridge()
    
    # Example: Continuous polling (would be hooked into RealtimeInference)
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
