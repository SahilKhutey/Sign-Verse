export interface VideoUpload {
  id: string;
  filename: string;
  source: 'upload' | 'youtube' | 'external';
  upload_time: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  file_path?: string;
  message?: string;
}

export interface PoseEstimationJob {
  job_id: string;
  video_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  total_frames: number;
  processed_frames: number;
  estimated_time?: number;
  result_path?: string;
  error?: string;
}

export interface Keypoint3D {
  name: string; // Added for skeletal mapping (e.g., 'left_shoulder')
  x: number;
  y: number;
  z: number;
  confidence: number;
  visible: boolean;
}

export interface PoseFrame {
  frame_number: number;
  timestamp: number;
  keypoints: Keypoint3D[];
  bounding_box?: {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  };
}

export interface PoseResult {
  video_id: string;
  frames: PoseFrame[];
  model_name: string;
  processing_time: number;
  resolution?: [number, number];
}

export interface SimulationJob {
  simulation_id: string;
  status: string;
  output_formats: string[];
  file_paths: Record<string, string>;
  processing_time: number;
  constraints_violations?: number;
}

export interface SystemHealth {
  status: string;
  version: string;
  uptime: number;
  requests_processed: number;
  services: Record<string, string>;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}
