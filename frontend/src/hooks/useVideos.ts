import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { VideoUpload } from '@/types';

export const useVideos = () => {
  const queryClient = useQueryClient();

  const useListVideos = (source?: string, limit?: number, offset?: number) =>
    useQuery({
      queryKey: ['videos', source, limit, offset],
      queryFn: () => apiClient.listVideos(source, limit, offset),
    });

  const uploadVideo = useMutation({
    mutationFn: (formData: FormData) => apiClient.uploadVideo(formData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] });
    },
  });

  const deleteVideo = useMutation({
    mutationFn: ({ videoId, source }: { videoId: string; source: string }) =>
      apiClient.deleteVideo(videoId, source),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] });
    },
  });

  return {
    useListVideos,
    uploadVideo,
    deleteVideo,
  };
};
