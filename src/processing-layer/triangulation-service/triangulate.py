import numpy as np

class Triangulator:
    """
    Standard DLT (Direct Linear Transformation) Triangulator for 
    multi-camera 3D reconstruction in Sign-Verse.
    """
    @staticmethod
    def triangulate_points(projection_matrices, points_2d):
        """
        Triangulates a single 3D point from N cameras.
        - projection_matrices: List of (3x4) numpy arrays.
        - points_2d: List of (2,) numpy arrays (x, y) for each camera.
        """
        num_cameras = len(projection_matrices)
        A = np.zeros((2 * num_cameras, 4))
        
        for i in range(num_cameras):
            P = projection_matrices[i]
            x, y = points_2d[i]
            
            # Formulating DLT matrix A
            A[2*i, :] = x * P[2, :] - P[0, :]
            A[2*i + 1, :] = y * P[2, :] - P[1, :]
            
        # Solve Ax = 0 using SVD (Singular Value Decomposition)
        _, _, vh = np.linalg.svd(A)
        point_3d_hom = vh[-1, :]
        
        # Convert back from homogeneous coordinates
        point_3d = point_3d_hom[:3] / point_3d_hom[3]
        return point_3d

if __name__ == "__main__":
    # Test with simple projection matrices
    P1 = np.eye(3, 4)
    P2 = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]])
    
    p1 = np.array([0.5, 0.5])
    p2 = np.array([0.4, 0.5])
    
    triangulator = Triangulator()
    result = triangulator.triangulate_points([P1, P2], [p1, p2])
    print(f"Triangulated 3D point: {result}")
