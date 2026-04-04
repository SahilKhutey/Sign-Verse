import axios from 'axios';
import {
  VideoUpload,
  PoseEstimationJob,
  PoseResult,
  SimulationJob,
  SystemHealth,
  ApiResponse
} from '@/types';

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('api_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message;
    return Promise.reject(new Error(message));
  }
);

export const apiClient = {
  // Health check
  health: (): Promise<SystemHealth> => api.get('/health'),

  // Video upload
  uploadVideo: (formData: FormData): Promise<VideoUpload> =>
    api.post('/upload/video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  listVideos: (source?: string, limit?: number, offset?: number): Promise<VideoUpload[]> =>
    api.get('/upload/videos', { params: { source, limit, offset } }),

  deleteVideo: (videoId: string, source: string): Promise<void> =>
    api.delete(`/upload/video/${videoId}`, { params: { source } }),

  // Pose estimation
  estimatePoses: (data: {
    video_id: string;
    model_name?: string;
    confidence_threshold?: number;
    include_3d?: boolean;
    smooth_output?: boolean;
  }): Promise<PoseEstimationJob> => api.post('/inference/pose', data),

  getPoseJobStatus: (jobId: string): Promise<PoseEstimationJob> =>
    api.get(`/inference/pose/${jobId}`),

  getPoseResult: (jobId: string): Promise<PoseResult> =>
    api.get(`/inference/pose/${jobId}/result`),

  cancelPoseJob: (jobId: string): Promise<void> =>
    api.delete(`/inference/pose/${jobId}`),

  // Simulation
  createSimulation: (data: {
    pose_data_id: string;
    character_type?: string;
    output_format?: string;
    include_constraints?: boolean;
    max_velocity?: number;
  }): Promise<SimulationJob> => api.post('/simulation', data),

  getSimulationStatus: (simulationId: string): Promise<SimulationJob> =>
    api.get(`/simulation/${simulationId}`),

  downloadResult: (filePath: string): Promise<Blob> =>
    api.get(`/simulation/download/${encodeURIComponent(filePath)}`, {
      responseType: 'blob',
    }),
};

export default apiClient;
