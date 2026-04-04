'use client'; // This is a client component

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { useAppStore } from '@/store/appStore';
import { 
  UploadIcon, 
  XIcon, 
  CheckCircleIcon, 
  VideoIcon, // Added missing VideoIcon to imports
  AlertCircleIcon // Added for error states
} from 'lucide-react';
import { cn } from '@/utils/cn';

export const VideoUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const { setUploading, setUploadProgress: setStoreProgress } = useAppStore();

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('source', 'upload');
      
      return apiClient.uploadVideo(formData);
    },
    onMutate: () => {
      setUploading(true);
      setUploadProgress(0);
    },
    onSuccess: () => {
      setUploadProgress(100);
      setTimeout(() => {
        setSelectedFile(null);
        setUploadProgress(0);
        setUploading(false);
      }, 2000);
    },
    onError: () => {
      setUploading(false);
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false,
  });

  const handleUpload = async () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
      
      // Simulate progress (this would ideally be based on axios upload progress)
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          const newProgress = Math.min(prev + 10, 90);
          setStoreProgress(newProgress);
          if (newProgress >= 90) clearInterval(interval);
          return newProgress;
        });
      }, 200);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    setUploading(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-4 sm:p-6 lg:p-8">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200',
          isDragActive
            ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20'
            : 'border-gray-300 hover:border-gray-400 dark:border-gray-700 dark:hover:border-gray-600',
          uploadMutation.isPending && 'opacity-50 cursor-not-allowed pointer-events-none'
        )}
      >
        <input {...getInputProps()} disabled={uploadMutation.isPending} />
        <UploadIcon className="mx-auto h-16 w-16 text-gray-400 mb-6" />
        <p className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          {isDragActive ? 'Drop video here' : 'Drag or click to upload video'}
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Supports MP4, AVI, MOV, MKV up to 100MB
        </p>
      </div>

      {/* Selected file & Progress */}
      {selectedFile && (
        <div className="mt-8 overflow-hidden bg-white shadow sm:rounded-lg dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
          <div className="px-4 py-5 sm:p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0 h-10 w-10 flex items-center justify-center rounded-full bg-indigo-100 text-indigo-600 dark:bg-indigo-900 dark:text-indigo-200">
                  <VideoIcon className="h-6 w-6" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-gray-900 truncate dark:text-white">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <button
                onClick={removeFile}
                disabled={uploadMutation.isPending}
                className="ml-4 flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-50 transition-colors"
                aria-label="Remove file"
              >
                <XIcon className="h-6 w-6" />
              </button>
            </div>

            {/* Progress Visualization */}
            {uploadMutation.isPending && (
              <div className="mt-6">
                <div className="relative pt-1">
                  <div className="flex mb-2 items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-indigo-600 bg-indigo-200 dark:bg-indigo-900 dark:text-indigo-200">
                        Uploading
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-semibold inline-block text-indigo-600 dark:text-indigo-400">
                        {uploadProgress}%
                      </span>
                    </div>
                  </div>
                  <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-indigo-200 dark:bg-indigo-900">
                    <div
                      style={{ width: `${uploadProgress}%` }}
                      className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-indigo-600 transition-all duration-300"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            {!uploadMutation.isPending && uploadProgress === 0 && (
              <div className="mt-6">
                <button
                  onClick={handleUpload}
                  className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
                >
                  <UploadIcon className="mr-2 h-4 w-4" />
                  Start Processing
                </button>
              </div>
            )}

            {/* Status Messages */}
            {uploadMutation.isSuccess && (
              <div className="mt-6 p-3 rounded-md bg-green-50 dark:bg-green-900/30 flex items-center">
                <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                <span className="text-sm font-medium text-green-800 dark:text-green-200">
                  Upload completed successfully! Redirecting...
                </span>
              </div>
            )}

            {uploadMutation.isError && (
              <div className="mt-6 p-3 rounded-md bg-red-50 dark:bg-red-900/30 flex items-center">
                <AlertCircleIcon className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-sm font-medium text-red-800 dark:text-red-200">
                  Upload failed: {uploadMutation.error instanceof Error ? uploadMutation.error.message : 'Unknown error'}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
