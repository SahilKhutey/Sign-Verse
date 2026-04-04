"""
Blender integration core module.
Handles communication between Python and Blender.
"""
import bpy
import mathutils
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlenderManager:
    """Manages Blender scene and operations."""
    
    def __init__(self):
        self.scene = bpy.context.scene
        self.setup_scene()
    
    def setup_scene(self):
        """Setup default Blender scene configuration."""
        # Set render engine
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.samples = 128
        
        # Set animation FPS
        self.scene.render.fps = 30
        
        # Clear existing objects (optional)
        # bpy.ops.object.select_all(action='SELECT')
        # bpy.ops.object.delete()
        
        logger.info("Blender scene setup complete")
    
    def create_armature(self, name: str = "HumanoidRig") -> bpy.types.Object:
        """Create a new armature object."""
        # Create armature data
        armature_data = bpy.data.armatures.new(name)
        armature_obj = bpy.data.objects.new(name, armature_data)
        
        # Link to scene
        self.scene.collection.objects.link(armature_obj)
        
        # Enter edit mode to create bones
        self.scene.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        return armature_obj
    
    def load_pose_data(self, pose_data_path: str) -> List[Dict[str, Any]]:
        """Load pose data from JSON file."""
        with open(pose_data_path, 'r') as f:
            return json.load(f)
    
    def export_animation(self, filepath: str, format: str = "fbx"):
        """Export animation to specified format for cross-platform use."""
        if format.lower() == "fbx":
            bpy.ops.export_scene.fbx(
                filepath=filepath,
                use_selection=True,
                apply_scale_options='FBX_SCALE_ALL',
                bake_anim_use_nla_strips=False,
                bake_anim_use_all_actions=False,
                bake_anim_simplify_factor=0.0
            )
        elif format.lower() == "bvh":
            bpy.ops.export_anim.bvh(
                filepath=filepath,
                frame_step=1,
                root_transform_only=True
            )
        elif format.lower() == "glb":
            bpy.ops.export_scene.gltf(
                filepath=filepath,
                export_format='GLB',
                export_animations=True
            )
        
        logger.info(f"Exported animation to {filepath}")

# Global blender manager instance
# Note: In a headless environment, this initialization might fail unless bpy is available.
try:
    blender_manager = BlenderManager()
except (ImportError, AttributeError):
    logger.warning("Blender 'bpy' module not found. Skipping immediate manager initialization.")
    blender_manager = None
