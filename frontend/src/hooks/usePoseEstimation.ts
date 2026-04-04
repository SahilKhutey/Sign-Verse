import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { PoseEstimationJob, PoseResult } from '@/types';

export const usePoseEstimation = () => {
  const queryClient = useQueryClient();

  const estimatePoses = useMutation({
    mutationFn: (data: {
      video_id: string;
      model_name?: string;
      confidence_threshold?: number;
      include_3d?: boolean;
      smooth_output?: boolean;
    }) => apiClient.estimatePoses(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['pose-jobs'] });
    },
  });

  const getJobStatus = (jobId: string) =>
    useQuery({
      queryKey: ['pose-job', jobId],
      queryFn: () => apiClient.getPoseJobStatus(jobId),
      enabled: !!jobId,
      refetchInterval: (query) =>
        query.state.data?.status === 'processing' ? 2000 : false,
    });

  const getResult = (jobId: string) =>
    useQuery({
      queryKey: ['pose-result', jobId],
      queryFn: () => apiClient.getPoseResult(jobId),
      enabled: !!jobId,
    });

  const cancelJob = useMutation({
    mutationFn: (jobId: string) => apiClient.cancelPoseJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pose-jobs'] });
    },
  });

  return {
    estimatePoses,
    getJobStatus,
    getResult,
    cancelJob,
  };
};

export const usePoseJobs = () => {
  return useQuery({
    queryKey: ['pose-jobs'],
    queryFn: async () => {
      // This would be replaced when we have a full list endpoint for all jobs
      // For now, we can list videos and infer jobs or just return empty
      return [];
    },
  });
};
