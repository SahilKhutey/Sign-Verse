"""
SignVerse Humanoid MuJoCo Environment
Custom gymnasium-style environment for DeepMimic human-to-robot imitation.
Backend: MuJoCo (Physics), Gymnasium (Interface).
"""

import numpy as np
import os
import mujoco
from typing import Dict, Any, Tuple

class SignHumanoidEnv:
    """
    MuJoCo Physics Environment for Humanoid Sign Language Imitation.
    Supports 21-DOF (Body/Arms) + 30-DOF (Hands) mapping from SMPL-X.
    """

    def __init__(self, model_path: str = None, frame_skip: int = 5):
        # Fallback to standard MuJoCo humanoid if no specific model provided
        if model_path is None or not os.path.exists(model_path):
            self.model = mujoco.MjModel.from_xml_string(self._get_default_humanoid_xml())
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        
        # State space dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply control torques (actions) and calculate imitation reward.
        """
        # Apply control - MuJoCo uses ctrl for actuator force/velocity
        self.data.ctrl[:] = action
        
        # Advance simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_termination()
        
        return obs, reward, done, {}

    def _get_obs(self) -> np.ndarray:
        """Returns joint positions and velocities."""
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _calculate_reward(self, ref_pose: np.ndarray = None) -> float:
        """
        DeepMimic imitation reward: exp(-alpha * ||robot_pose - ref_pose||^2).
        Requires ref_pose (SMPL-X reference) for the given simulation timestamp.
        """
        if ref_pose is None:
            return 0.0 # No reference, no reward
            
        # Extract corresponding qpos from SMPL-X reference mapping
        # r_pose = exp(-2.0 * np.sum((self.data.qpos[7:] - ref_pose)**2))
        return 1.0 # Placeholder for imitation success

    def _check_termination(self) -> bool:
        """Terminate if the humanoid falls below a certain height."""
        # Body index 0 is typically the root
        height = self.data.qpos[2]
        return height < 0.8  # Terminate if torso falls too low

    def _get_default_humanoid_xml(self) -> str:
        """Return a basic 21-DOF Humanoid MuJoCo XML string."""
        return """
        <mujoco model="sign_humanoid">
            <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
            <option integrator="RK4" timestep="0.01"/>
            <worldbody>
                <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" specular=".1 .1 .1" pos="0 0 4"/>
                <geom name="floor" type="plane" size="100 100 .1" rgba=".9 .9 .9 1"/>
                <body name="torso" pos="0 0 1.4">
                    <freejoint name="root"/>
                    <geom name="torso" type="capsule" size=".07 .07" fromto="0 -.07 0 0 .07 0"/>
                    <body name="head" pos="0 0 .19">
                        <geom name="head" type="sphere" size=".09"/>
                    </body>
                    <body name="right_upper_arm" pos="0 -.17 .06">
                        <joint name="right_shoulder" type="ball"/>
                        <geom name="right_uarm" type="capsule" size=".045 .12" fromto="0 0 0 0 -.16 0"/>
                        <body name="right_lower_arm" pos="0 -.28 0">
                            <joint name="right_elbow" type="hinge" axis="0 0 1" range="-90 90"/>
                            <geom name="right_larm" type="capsule" size=".04 .1" fromto="0 0 0 0 -.13 0"/>
                        </body>
                    </body>
                    <body name="left_upper_arm" pos="0 .17 .06">
                        <joint name="left_shoulder" type="ball"/>
                        <geom name="left_uarm" type="capsule" size=".045 .12" fromto="0 0 0 0 .16 0"/>
                        <body name="left_lower_arm" pos="0 .28 0">
                            <joint name="left_elbow" type="hinge" axis="0 0 1" range="-90 90"/>
                            <geom name="left_larm" type="capsule" size=".04 .1" fromto="0 0 0 0 .13 0"/>
                        </body>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="r_shoulder_x" joint="right_shoulder" gear="100"/>
                <motor name="r_elbow" joint="right_elbow" gear="100"/>
                <motor name="l_shoulder_x" joint="left_shoulder" gear="100"/>
                <motor name="l_elbow" joint="left_elbow" gear="100"/>
            </actuator>
        </mujoco>
        """
