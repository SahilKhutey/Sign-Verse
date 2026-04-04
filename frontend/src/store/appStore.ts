import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface AppState {
  // UI state
  sidebarOpen: boolean;
  currentPage: string;
  theme: 'light' | 'dark';
  
  // Upload state
  uploadProgress: number;
  isUploading: boolean;
  
  // Selected items
  selectedVideo: string | null;
  selectedJob: string | null;
  
  // Actions
  toggleSidebar: () => void;
  setCurrentPage: (page: string) => void;
  toggleTheme: () => void;
  setUploadProgress: (progress: number) => void;
  setUploading: (uploading: boolean) => void;
  setSelectedVideo: (videoId: string | null) => void;
  setSelectedJob: (jobId: string | null) => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    (set) => ({
      // Initial state
      sidebarOpen: true,
      currentPage: 'dashboard',
      theme: 'light',
      uploadProgress: 0,
      isUploading: false,
      selectedVideo: null,
      selectedJob: null,
      
      // Actions
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })), // FIXED: sidebarOpen instead of sidebar
      setCurrentPage: (page) => set({ currentPage: page }),
      toggleTheme: () => set((state) => ({ theme: state.theme === 'light' ? 'dark' : 'light' })),
      setUploadProgress: (progress) => set({ uploadProgress: progress }),
      setUploading: (uploading) => set({ isUploading: uploading }),
      setSelectedVideo: (videoId) => set({ selectedVideo: videoId }),
      setSelectedJob: (jobId) => set({ selectedJob: jobId }),
    }),
    { name: 'AppStore' }
  )
);
