---
description: Robotics Motion SOP — Mapping Sign Language to Robotic Actuators
---
# Robotics Motion Workflow

Guidelines for using SignVerse motion intelligence to drive robotic arms and hands.

## Prerequisites
- **ROS 2 (Humble/Iron)**
- **MoveIt 2** for IK (Inverse Kinematics)
- **Universal Robots (UR5/e)** or **Franka Emika**

## Workflow Steps

### 1. Motion Mapping
1.  **Skeleton Triangulation**: Use `vision_system/multiview_pipeline.py` for 3D pose extraction.
2.  **Universal Vector (848-dim)**: Extract the 424-dim geometric sub-vector.
3.  **IK Solver**: Map 3D wrist and joint positions to the robotic driver.

### 2. Physical Actuation
1.  **Velocity Control**: Use the 424-dim velocity sub-vector for smooth joint-speed planning.
2.  **Hand Mimicry**: Map hand-distances (210/hand) to robotic finger encoders.

### 3. Safety Protocols
1.  **Collision Avoidance**: Integrate MoveIt collision scenes.
2.  **latency-check**: Ensure end-to-end delay (Vision -> Robot) is < 150ms for safe human interaction.

## Universal Interface (848-dim)
Robotic drivers must consume the **848-dimensional "Motion Intelligence" vector**. This ensures that any model trained by the AI team (e.g., SignGPT) can directly drive the robot.
