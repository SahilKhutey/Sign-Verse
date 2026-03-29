import cv2
import numpy as np

class Triangulator:
    """
    SignVerse Triangulator — 3D Reconstruction Engine.
    Computes 3D coordinates from multiple 2D camera projections.
    """
    def __init__(self, camera_configs):
        """
        camera_configs: List of dicts with { "K": (3,3), "R": (3,3), "T": (3,1) }
        """
        self.projections = []
        for cfg in camera_configs:
            K = np.array(cfg["K"])
            R = np.array(cfg["R"])
            T = np.array(cfg["T"])
            P = K @ np.hstack((R, T))
            self.projections.append(P)

    def triangulate(self, points_2d):
        """
        points_2d: List of (x, y) coordinates from each of the N cameras.
        Returns (3,) array of (x, y, z) in world space.
        Uses Direct Linear Transform (DLT) for N views.
        """
        if len(points_2d) < 2:
            return None # Minimum 2 views needed

        # Prepare A matrix for AX = 0
        A = []
        for i, pt in enumerate(points_2d):
            if pt is None: continue
            P = self.projections[i]
            x, y = pt
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        # Solve using SVD
        _, _, vh = np.linalg.svd(A)
        X = vh[-1, :]
        return X[:3] / X[3] # Homogeneous to 3D

    def triangulate_batch(self, batch_2d):
        """
        batch_2d: List (N_cameras) of List (M_joints) of (x, y).
        Returns (M_joints, 3) 3D skeleton.
        """
        n_joints = len(batch_2d[0])
        skeleton_3d = []
        for j in range(n_joints):
            pts = [batch_2d[c][j] if batch_2d[c][j] is not None else None for c in range(len(batch_2d))]
            res = self.triangulate(pts)
            skeleton_3d.append(res if res is not None else [0, 0, 0])
        return np.array(skeleton_3d)

    def get_projection_error(self, p_3d, points_2d):
        """Helper to verify reconstruction quality."""
        errors = []
        p_homog = np.append(p_3d, 1.0)
        for i, pt in enumerate(points_2d):
            if pt is None: continue
            proj = self.projections[i] @ p_homog
            proj = proj[:2] / proj[2]
            errors.append(np.linalg.norm(proj - pt))
        return np.mean(errors) if errors else 0.0
