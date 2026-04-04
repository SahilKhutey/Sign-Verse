import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { SimulationJob } from '@/types';

export const useSimulation = () => {
  const queryClient = useQueryClient();

  const createSimulation = useMutation({
    mutationFn: (data: {
      pose_data_id: string;
      character_type?: string;
      output_format?: string;
      include_constraints?: boolean;
      max_velocity?: number;
    }) => apiClient.createSimulation(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['simulations'] });
    },
  });

  const getStatus = (simulationId: string) =>
    useQuery({
      queryKey: ['simulation', simulationId],
      queryFn: () => apiClient.getSimulationStatus(simulationId),
      enabled: !!simulationId,
      refetchInterval: (query) =>
        query.state.data?.status === 'processing' ? 2000 : false,
    });

  const downloadResult = useMutation({
    mutationFn: (filePath: string) => apiClient.downloadResult(filePath),
    onSuccess: (blob, filePath) => {
      // Logic for triggering download in browser
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filePath.split('/').pop() || 'result');
      document.body.appendChild(link);
      link.click();
      link.remove();
    },
  });

  return {
    createSimulation,
    getStatus,
    downloadResult,
  };
};

export const useSimulationsList = () => {
  return useQuery({
    queryKey: ['simulations'],
    queryFn: async () => {
      // For now, returning empty until a list endpoint is available
      return [];
    },
  });
};
