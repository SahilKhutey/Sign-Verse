import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { SystemHealth } from '@/types';

export const useHealth = () => {
  return useQuery({
    queryKey: ['system-health'],
    queryFn: () => apiClient.health(),
    refetchInterval: 60000, // Poll every minute
  });
};
