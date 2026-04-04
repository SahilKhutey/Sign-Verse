"""
Monocular depth estimation for 3D spatial understanding in SignVerse OS.
Enables metric-relative distance calculations and environmental mapping.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import torch
import torchvision.transforms as transforms
from loguru import logger

class DepthEstimator:
    """Monocular depth estimation with support for multiple model fidelities (DPT/MiDaS)."""
    
    # Model presets for adaptive workloads
    MODEL_PRESETS = {
        "fast": "midas_v21_small",      # Efficient, suitable for mobile/real-time
        "accurate": "DPT_Large",        # State-of-the-art transformer, high fidelity
        "balanced": "midas_v21"         # Standard MiDaS
    }
    
    def __init__(self, model_type: str = "balanced", device: str = "auto"):
        """
        Initialize depth estimator.
        
        Args:
            model_type: Preset name ('fast', 'balanced', 'accurate') or raw model name
            device: Computation device ('cpu', 'cuda', or 'auto')
        """
        self.model_type = self.MODEL_PRESETS.get(model_type, model_type)
        self.device = self._setup_device(device)
        self.model = None
        self.transform = None
        self.initialized = False
        
        logger.info(f"Initializing depth estimator tier: {model_type}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device for high-throughput inference."""
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA GPU for depth acceleration")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for depth estimation (High latency expected)")
        else:
            device = torch.device(device)
        
        return device
    
    def initialize(self):
        """Initialize depth model from external repository or torch.hub."""
        try:
            if self.model_type.startswith("midas"):
                # Check if we can load from torch.hub as a fallback if local midas module is missing
                try:
                    from midas.model_loader import load_model
                    self.model, self.transform, _ = load_model(self.device, self.model_type)
                except (ImportError, ModuleNotFoundError):
                    logger.warning("Local 'midas' module not found. Falling back to torch.hub.")
                    # MiDaS v2.1 Small model via torch hub
                    model_path = "intel-isl/MiDaS"
                    self.model = torch.hub.load(model_path, self.model_type.replace("midas_", "").upper())
                    
                    # Prepare transforms for MiDaS
                    midas_transforms = torch.hub.load(model_path, "transforms")
                    if "small" in self.model_type:
                        self.transform = midas_transforms.small_transform
                    else:
                        self.transform = midas_transforms.dpt_transform
            else:
                raise ValueError(f"Unsupported depth model type: {self.model_type}")
            
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            logger.success(f"Successfully initialized depth estimator: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Critical failure during depth model initialization: {e}")
            # Do not raise here to allow pipeline to continue with zero-depth if needed
            self.initialized = False
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate 2D depth map from a single BGR image."""
        if not self.initialized:
            self.initialize()
            if not self.initialized:
                return np.zeros(frame.shape[:2], dtype=np.float32)
        
        try:
            # Preprocess image for the backend
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img).to(self.device)
            
            # Predict depth (Inverse depth typically)
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Rescale to original input resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
            # Normalize depth map to [0, 1] range for visualization and relative analysis
            # Note: Monocular depth is scale-ambiguous, hence the normalization
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation inference failed: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)
    
    def estimate_entity_depth(self, depth_map: np.ndarray, bbox: np.ndarray) -> float:
        """Estimate median depth for a specific entity bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure bbox is within depth map bounds
        h, w = depth_map.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        depth_roi = depth_map[y1:y2, x1:x2]
        
        if depth_roi.size == 0:
            return 0.0
        
        # Median is more robust to outlier background pixels within the ROI
        return float(np.median(depth_roi))
    
    def estimate_relative_depth(self, depth_map: np.ndarray, bbox1: np.ndarray, 
                              bbox2: np.ndarray) -> float:
        """Calculate relative depth between two entities."""
        depth1 = self.estimate_entity_depth(depth_map, bbox1)
        depth2 = self.estimate_entity_depth(depth_map, bbox2)
        return depth1 - depth2  # Positive if entity1 is closer to the camera
    
    def create_point_cloud(self, depth_map: np.ndarray, intrinsic_matrix: np.ndarray,
                          max_points: int = 10000) -> np.ndarray:
        """Synthesize a 3D point cloud from a depth map and camera intrinsics."""
        height, width = depth_map.shape
        
        # Grid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        z = depth_map.flatten()
        
        # Subsampling for performance optimization
        if len(z) > max_points:
            indices = np.random.choice(len(z), max_points, replace=False)
            u, v, z = u[indices], v[indices], z[indices]
        
        # Pin-hole Camera Backprojection
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.column_stack([x, y, z])
        return points
    
    def estimate_3d_position(self, depth_map: np.ndarray, intrinsic_matrix: np.ndarray,
                           bbox: np.ndarray) -> np.ndarray:
        """Locate an entity's centroid in 3D camera space."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Centroid of the bounding box
        u_center = (x1 + x2) / 2.0
        v_center = (y1 + y2) / 2.0
        
        # Representative depth at centroid
        depth = depth_map[int(v_center), int(u_center)] if 0 <= int(v_center) < depth_map.shape[0] and 0 <= int(u_center) < depth_map.shape[1] else 0.5
        
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        
        x = (u_center - cx) * depth / fx
        y = (v_center - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def calculate_spatial_relationships(self, depth_map: np.ndarray, 
                                      intrinsic_matrix: np.ndarray,
                                      entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map comprehensive 3D spatial relationships between all tracked entities."""
        relationships = {}
        positions = {}
        
        # Phase 1: Locational 3D mapping
        for entity in entities:
            ent_id = entity.get("id") or entity.get("track_id")
            pos = self.estimate_3d_position(depth_map, intrinsic_matrix, entity["bbox"])
            positions[ent_id] = pos
        
        # Phase 2: Relational calculation
        entity_ids = list(positions.keys())
        for i, id1 in enumerate(entity_ids):
            for j, id2 in enumerate(entity_ids):
                if i >= j: continue
                
                pos1, pos2 = positions[id1], positions[id2]
                
                # 3D Euclidean distance
                distance = np.linalg.norm(pos1 - pos2)
                
                # Relative vector
                vec = pos2 - pos1
                
                # Polar coordinates for behavioral diagnostics
                h_angle = np.arctan2(vec[0], vec[2])
                v_angle = np.arctan2(vec[1], vec[2])
                
                relationships[f"{id1}_{id2}"] = {
                    "distance": float(distance),
                    "relative_pos": vec.tolist(),
                    "bearing": float(h_angle),
                    "elevation": float(v_angle)
                }
        
        return relationships

class CameraCalibrator:
    """Manages multi-device camera projection and intrinsic parameters."""
    
    # Pre-calibrated intrinsic presets for various hardware profiles
    DEVICE_PRESETS = {
        "webcam_720p": np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ]),
        "mobile_4k": np.array([
            [3000, 0, 1920],
            [0, 3000, 1080],
            [0, 0, 1]
        ]),
        "fisheye_wide": np.array([
            [600, 0, 640],
            [0, 600, 480],
            [0, 0, 1]
        ])
    }
    
    def __init__(self, intrinsic_matrix: Optional[np.ndarray] = None, device_preset: Optional[str] = "webcam_720p"):
        """
        Initialize camera calibrator using a custom matrix or a device preset.
        
        Args:
            intrinsic_matrix: Pre-calibrated 3x3 intrinsic matrix (Overrides preset)
            device_preset: Key from DEVICE_PRESETS (e.g., 'webcam_720p', 'mobile_4k')
        """
        if intrinsic_matrix is not None:
            self.intrinsic_matrix = intrinsic_matrix
        else:
            self.intrinsic_matrix = self.DEVICE_PRESETS.get(device_preset, self.DEVICE_PRESETS["webcam_720p"])
        
        logger.info(f"Initialized CameraCalibrator with profile: {device_preset if intrinsic_matrix is None else 'custom'}")
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        return self.intrinsic_matrix
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Map 3D world/camera points back to 2D pixel space."""
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
            
        # P = K * [R|t] * P_world (Assuming [R|t] is Identity for camera space)
        # Here we just use K for back-projection from camera local 3D to 2D
        # x = (X * fx / Z) + cx
        # y = (Y * fy / Z) + cy
        res = []
        for p in points_3d:
            if p[2] == 0: 
                res.append([0, 0])
                continue
            u = (p[0] * self.intrinsic_matrix[0,0] / p[2]) + self.intrinsic_matrix[0,2]
            v = (p[1] * self.intrinsic_matrix[1,1] / p[2]) + self.intrinsic_matrix[1,2]
            res.append([u, v])
            
        return np.array(res)
    
    def backproject_2d_to_3d(self, points_2d: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Backproject 2D pixels to 3D camera space using depth maps."""
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, -1)
        if isinstance(depths, (int, float)):
            depths = np.array([depths] * len(points_2d))
            
        points_3d = []
        for i, (uv, z) in enumerate(zip(points_2d, depths)):
            x = (uv[0] - self.intrinsic_matrix[0,2]) * z / self.intrinsic_matrix[0,0]
            y = (uv[1] - self.intrinsic_matrix[1,2]) * z / self.intrinsic_matrix[1,1]
            points_3d.append([x, y, z])
            
        return np.array(points_3d)
