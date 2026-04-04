"""
Animation exporter for robotic and digital assets.
"""
import bpy
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AnimationExporter:
    """Handles exporting Blender animations to FBX and robotic joint data."""
    
    def __init__(self, armature_obj: bpy.types.Object):
        self.armature = armature_obj
        
    def export_fbx(self, filepath: str):
        """Export the armature and animation to an FBX file."""
        # Ensure only the armature and its mesh are selected
        bpy.ops.object.select_all(action='DESELECT')
        self.armature.select_set(True)
        bpy.context.view_layer.objects.active = self.armature
        
        # Export FBX
        try:
            bpy.ops.export_scene.fbx(
                filepath=filepath,
                use_selection=True,
                apply_scale_options='FBX_SCALE_ALL',
                bake_anim=True,
                bake_anim_use_nla_strips=False,
                bake_anim_use_all_actions=False,
                mesh_smooth_type='FACE',
                add_leaf_bones=False
            )
            logger.info(f"FBX animation exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export FBX: {e}")

    def export_robot_joint_data(self, filepath: str):
        """Extract and export joint angles for every frame in the animation."""
        scene = bpy.context.scene
        start_frame = scene.frame_start
        end_frame = scene.frame_end
        
        joint_data = {
            "metadata": {
                "fps": scene.render.fps,
                "total_frames": end_frame - start_frame + 1,
                "armature": self.armature.name
            },
            "frames": []
        }
        
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            
            frame_entry = {
                "frame": frame,
                "joints": {}
            }
            
            for bone in self.armature.pose.bones:
                # Store rotation in Euler degrees for easier robotic mapping
                euler = bone.rotation_quaternion.to_euler()
                frame_entry["joints"][bone.name] = {
                    "x": round(euler.x * 180 / 3.14159, 3),
                    "y": round(euler.y * 180 / 3.14159, 3),
                    "z": round(euler.z * 180 / 3.14159, 3)
                }
            
            joint_data["frames"].append(frame_entry)
            
        with open(filepath, 'w') as f:
            json.dump(joint_data, f, indent=2)
            
        logger.info(f"Robotic joint data exported to {filepath}")
