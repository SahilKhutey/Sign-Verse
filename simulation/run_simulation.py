"""
Main simulation pipeline for converting poses to robotic animations.
"""
import bpy
import json
from pathlib import Path
from typing import Dict, Any
import argparse
from loguru import logger

from configs import get_config
from blender.rigging.skeleton_rig import SkeletonRig
from blender.animation.pose_to_animation import PoseAnimator
from blender.exporters.fbx_exporter import AnimationExporter
from physics.motion_constraints import MotionValidator

def run_simulation_pipeline(pose_data_path: str, output_dir: str):
    """Run the complete 3D simulation and robotic validation pipeline."""
    config = get_config()
    
    # Initialize workspace
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load standardized pose data from the Vision/AI layer
    with open(pose_data_path, 'r') as f:
        pose_sequence = json.load(f)
    
    # Orchestrate Blender scene
    # Note: In a headless environment, ensure blender was launched correctly.
    scene = bpy.context.scene
    
    # 1. Digital Twin Synthesis (Rigging)
    # create_armature is handled by BlenderManager but we'll use bpy directly for simplicity in the runner
    armature_data = bpy.data.armatures.new("RoboticCharacterData")
    armature = bpy.data.objects.new("RoboticCharacter", armature_data)
    scene.collection.objects.link(armature)
    
    rig = SkeletonRig(armature)
    rig.create_humanoid_skeleton(scale=1.0)
    rig.setup_inverse_kinematics()
    
    # Apply robotics-first constraints (Joint limits)
    rig.apply_robotic_constraints(config.robotic.joint_limits)
    
    # 2. Animation Retargeting
    animator = PoseAnimator(armature)
    animator.create_animation_from_poses(pose_sequence, fps=config.animation.fps)
    
    # Apply temporal smoothing to the resulting motion curves
    animator.smooth_animation(window_size=config.animation.smoothing_window)
    
    # 3. Motion Validation & Safety Governance
    validator = MotionValidator(config.robotic.joint_limits)
    is_valid = validator.validate_animation(armature)
    
    if not is_valid:
        logger.warning("Simulated motion exceeds physical joint limits. Applying safety corrections (Velocity Clamping).")
        # Governance: Force robotic velocity limits on the digital twin
        validator.apply_velocity_limits(armature, config.robotic.max_velocity)
    
    # 4. Result Persistence (Exporters)
    exporter = AnimationExporter(armature)
    
    # Primary Asset: FBX skeletal animation for visualization and Unity/Unreal integration
    fbx_path = output_path / "animation.fbx"
    exporter.export_fbx(str(fbx_path))
    
    # Secondary Asset: Robotic Joint Angles (Degrees) for direct serial/PLC communication
    joint_path = output_path / "joint_angles.json"
    exporter.export_robot_joint_data(str(joint_path))
    
    logger.success(f"SignVerse Simulation pipeline completed successfully. Artifacts available in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse Simulation Entry Point")
    parser.add_argument("--pose-data", required=True, help="Path to input PoseData JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated artifacts")
    
    args = parser.parse_args()
    
    try:
        run_simulation_pipeline(args.pose_data, args.output_dir)
    except Exception as e:
        logger.error(f"Simulation pipeline failed: {e}")
        raise
