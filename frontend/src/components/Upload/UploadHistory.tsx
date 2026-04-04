import React from 'react';
import { useVideos } from '@/hooks/useVideos';
import { ClockIcon, FileVideoIcon, AlertCircleIcon, ExternalLinkIcon } from 'lucide-react';
import { cn } from '@/utils/cn';

/**
 * UploadHistory Component
 * 
 * Displays a list of recently uploaded videos and their processing status.
 * Leverages the useVideos hook for data fetching.
 */
export const UploadHistory: React.FC = () => {
  const { useListVideos, deleteVideo } = useVideos();
  const { data: videos, isLoading, isError } = useListVideos();

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-12 bg-white dark:bg-gray-900 rounded-xl border border-gray-100 dark:border-gray-800">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mb-4"></div>
        <p className="text-sm text-gray-500 dark:text-gray-400">Loading your history...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex items-center p-4 bg-red-50 dark:bg-red-900/30 rounded-xl border border-red-100 dark:border-red-900/50">
        <AlertCircleIcon className="h-5 w-5 text-red-600 mr-2" />
        <p className="text-sm text-red-800 dark:text-red-200">Unable to load processing history.</p>
      </div>
    );
  }

  if (!videos || videos.length === 0) {
    return (
      <div className="text-center py-12 bg-white dark:bg-gray-900 rounded-xl border border-gray-100 dark:border-gray-800">
        <ClockIcon className="mx-auto h-12 w-12 text-gray-300 mb-4" />
        <p className="text-gray-500 dark:text-gray-400">No recent uploads found.</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden bg-white shadow sm:rounded-xl border border-gray-200 dark:bg-gray-900 dark:border-gray-800">
      <ul role="list" className="divide-y divide-gray-200 dark:divide-gray-800">
        {videos.map((video) => (
          <li key={video.id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center rounded-lg bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                <FileVideoIcon className="h-6 w-6" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-gray-900 truncate dark:text-white">
                  {video.filename}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Uploaded {new Date(video.upload_time).toLocaleDateString()} &mdash; Source: {video.source}
                </p>
              </div>
              <div>
                <span className={cn(
                  "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
                  video.status === 'completed' ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300" :
                  video.status === 'processing' ? "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300" :
                  video.status === 'failed' ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300" :
                  "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400"
                )}>
                  {video.status.toUpperCase()}
                </span>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};
