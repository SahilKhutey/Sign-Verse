"""
Skeleton rigging system for Blender.
Creates and manages bone structures for humanoid and robotic characters.
"""
import bpy
import mathutils
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SkeletonRig:
    """Creates and manages skeleton rigs for animation."""
    
    def __init__(self, armature_obj: bpy.types.Object):
        self.armature = armature_obj
        self.bones = {}
        self.ik_chains = {}
    
    def create_humanoid_skeleton(self, scale: float = 1.0):
        """Create a standard humanoid skeleton."""
        # Enter edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = self.armature.data.edit_bones
        
        # Create bones in hierarchical order
        self._create_spine(edit_bones, scale)
        self._create_limbs(edit_bones, scale)
        self._create_head(edit_bones, scale)
        self._create_hands_feet(edit_bones, scale)
        
        logger.info("Humanoid skeleton creation initiated.")
    
    def _create_spine(self, edit_bones, scale: float):
        """Create spine bones."""
        # Hip (root)
        hip = edit_bones.new('hip')
        hip.head = (0, 0, 0)
        hip.tail = (0, 0, 0.1 * scale)
        self.bones['hip'] = hip
        
        # Spine
        spine = edit_bones.new('spine')
        spine.head = hip.tail
        spine.tail = (0, 0, 0.4 * scale)
        spine.parent = hip
        self.bones['spine'] = spine
        
        # Chest
        chest = edit_bones.new('chest')
        chest.head = spine.tail
        chest.tail = (0, 0, 0.7 * scale)
        chest.parent = spine
        self.bones['chest'] = chest
    
    def _create_limbs(self, edit_bones, scale: float):
        """Create arm and leg bones."""
        chest = self.bones['chest']
        
        # Left arm
        shoulder_l = edit_bones.new('shoulder.L')
        shoulder_l.head = chest.tail
        shoulder_l.tail = (-0.15 * scale, 0, 0.7 * scale)
        shoulder_l.parent = chest
        self.bones['shoulder.L'] = shoulder_l
        
        upper_arm_l = edit_bones.new('upper_arm.L')
        upper_arm_l.head = shoulder_l.tail
        upper_arm_l.tail = (-0.4 * scale, 0, 0.7 * scale)
        upper_arm_l.parent = shoulder_l
        self.bones['upper_arm.L'] = upper_arm_l
        
        # Right arm (mirror)
        shoulder_r = edit_bones.new('shoulder.R')
        shoulder_r.head = chest.tail
        shoulder_r.tail = (0.15 * scale, 0, 0.7 * scale)
        shoulder_r.parent = chest
        self.bones['shoulder.R'] = shoulder_r
        
        upper_arm_r = edit_bones.new('upper_arm.R')
        upper_arm_r.head = shoulder_r.tail
        upper_arm_r.tail = (0.4 * scale, 0, 0.7 * scale)
        upper_arm_r.parent = shoulder_r
        self.bones['upper_arm.R'] = upper_arm_r
        
        # Legs
        self._create_leg('L', edit_bones, scale, -0.1)
        self._create_leg('R', edit_bones, scale, 0.1)
    
    def _create_leg(self, side: str, edit_bones, scale: float, x_offset: float):
        """Create leg bones for given side."""
        hip = self.bones['hip']
        
        upper_leg = edit_bones.new(f'upper_leg.{side}')
        upper_leg.head = (x_offset * scale, 0, hip.tail[2])
        upper_leg.tail = (x_offset * scale, 0, -0.4 * scale)
        upper_leg.parent = hip
        self.bones[f'upper_leg.{side}'] = upper_leg
        
        lower_leg = edit_bones.new(f'lower_leg.{side}')
        lower_leg.head = upper_leg.tail
        lower_leg.tail = (x_offset * scale, 0, -0.8 * scale)
        lower_leg.parent = upper_leg
        self.bones[f'lower_leg.{side}'] = lower_leg
        
        foot = edit_bones.new(f'foot.{side}')
        foot.head = lower_leg.tail
        foot.tail = (x_offset * scale, 0.1 * scale, -0.8 * scale)
        foot.parent = lower_leg
        self.bones[f'foot.{side}'] = foot
    
    def _create_head(self, edit_bones, scale: float):
        """Create head and neck bones."""
        chest = self.bones['chest']
        
        neck = edit_bones.new('neck')
        neck.head = chest.tail
        neck.tail = (0, 0, 0.85 * scale)
        neck.parent = chest
        self.bones['neck'] = neck
        
        head = edit_bones.new('head')
        head.head = neck.tail
        head.tail = (0, 0, 1.1 * scale)
        head.parent = neck
        self.bones['head'] = head
    
    def _create_hands_feet(self, edit_bones, scale: float):
        """Placeholder for detailed hand and finger skeleton."""
        pass
    
    def setup_inverse_kinematics(self):
        """Setup inverse kinematics for limbs in Pose Mode."""
        bpy.ops.object.mode_set(mode='POSE')
        
        # Attempt to setup IK for arms and legs if bones exist
        for side in ['L', 'R']:
            try:
                self._setup_limb_ik(f'upper_arm.{side}', f'lower_arm.{side}', f'hand.{side}')
                self._setup_limb_ik(f'upper_leg.{side}', f'lower_leg.{side}', f'foot.{side}')
            except KeyError as e:
                logger.warning(f"Skipping IK setup for missing limb bone: {e}")
        
        logger.info("Inverse kinematics orchestration complete.")
    
    def _setup_limb_ik(self, upper_bone_name: str, lower_bone_name: str, end_bone_name: str):
        """Apply an IK constraint to the lower bone of a limb targeting the end bone."""
        if lower_bone_name in self.armature.pose.bones:
            lower_pbone = self.armature.pose.bones[lower_bone_name]
            
            # Add IK constraint
            ik_constraint = lower_pbone.constraints.new('IK')
            ik_constraint.target = self.armature
            ik_constraint.subtarget = end_bone_name
            ik_constraint.chain_count = 2
            logger.debug(f"IK setup on {lower_bone_name} targeting {end_bone_name}")
    
    def apply_robotic_constraints(self, joint_limits: Dict[str, Tuple[float, float]]):
        """Apply robotic rotation limits to specific pose bones."""
        bpy.ops.object.mode_set(mode='POSE')
        
        for bone_name, limits in joint_limits.items():
            if bone_name in self.armature.pose.bones:
                bone = self.armature.pose.bones[bone_name]
                
                # Add rotation limit constraint to simulate robotic joint caps
                limit_constraint = bone.constraints.new('LIMIT_ROTATION')
                limit_constraint.use_limit_x = True
                limit_constraint.use_limit_y = True
                limit_constraint.use_limit_z = True
                
                # Use simplified limits for demonstration (mapped to all axes)
                limit_constraint.min_x = limit_constraint.min_y = limit_constraint.min_z = limits[0]
                limit_constraint.max_x = limit_constraint.max_y = limit_constraint.max_z = limits[1]
                
        logger.info("Robotic joint constraints successfully applied.")
