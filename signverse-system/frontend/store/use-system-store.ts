import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface SystemState {
  // UI State
  darkMode: boolean
  sidebarOpen: boolean
  
  // Processing State
  isProcessing: boolean
  fps: number
  latency: number
  memoryUsage: number
  cpuUsage: number
  socket: WebSocket | null
  
  // Visualization Toggles
  showPose: boolean
  showBoundingBoxes: boolean
  showDepth: boolean
  showInteractions: boolean
  isDemoMode: boolean
  
  // Actions
  setDarkMode: (darkMode: boolean) => void
  setSidebarOpen: (open: boolean) => void
  setDemoMode: (isDemoMode: boolean) => void
  startPipeline: () => void
  stopPipeline: () => void
  setFPS: (fps: number) => void
  setFps: (fps: number) => void
  setLatency: (latency: number) => void
  setMemoryUsage: (memory: number) => void
  setShowPose: (show: boolean) => void
  setShowBoundingBoxes: (show: boolean) => void
  setShowDepth: (show: boolean) => void
  setShowInteractions: (show: boolean) => void
  
  // Utility
  toggleDarkMode: () => void
  togglePose: () => void
  toggleBoundingBoxes: () => void
  toggleDepth: () => void
  toggleInteractions: () => void
}

export const useSystemStore = create<SystemState>()(
  persist(
    (set, get) => ({
      // Initial State
      darkMode: true,
      sidebarOpen: false,
      isProcessing: false,
      fps: 0,
      latency: 0,
      memoryUsage: 0,
      cpuUsage: 0,
      socket: null,
      showPose: true,
      showBoundingBoxes: true,
      showDepth: false,
      showInteractions: true,
      isDemoMode: false,
      
      // Actions
      setDarkMode: (darkMode: boolean) => set({ darkMode }),
      setSidebarOpen: (sidebarOpen: boolean) => set({ sidebarOpen }),
      setDemoMode: (isDemoMode: boolean) => set({ isDemoMode }),
      
      startPipeline: () => {
        const socket = new WebSocket("ws://localhost:8000/ws/stream")
        
        socket.onmessage = (event) => {
          const data = JSON.parse(event.data)
          if (data.type === "telemetry" || data.latency_ms !== undefined) {
            set({
              fps: data.fps ?? get().fps,
              latency: data.latency_ms ?? get().latency,
              memoryUsage: data.memory_usage ?? get().memoryUsage,
              cpuUsage: data.cpu_usage ?? get().cpuUsage,
            })
          }
        }

        socket.onopen = () => set({ isProcessing: true, socket })
        socket.onclose = () => set({ isProcessing: false, socket: null })
        socket.onerror = () => set({ isProcessing: false, socket: null })
      },

      stopPipeline: () => {
        const { socket } = get()
        if (socket) {
          socket.close()
        }
        set({ isProcessing: false, socket: null })
      },
      
      setFPS: (fps: number) => set({ fps }),
      setFps: (fps: number) => set({ fps }),
      setLatency: (latency: number) => set({ latency }),
      setMemoryUsage: (memoryUsage: number) => set({ memoryUsage }),
      
      setShowPose: (showPose: boolean) => set({ showPose }),
      setShowBoundingBoxes: (showBoundingBoxes: boolean) => set({ showBoundingBoxes }),
      setShowDepth: (showDepth: boolean) => set({ showDepth }),
      setShowInteractions: (showInteractions: boolean) => set({ showInteractions }),
      
      // Utility toggles
      toggleDarkMode: () => set((state: SystemState) => ({ darkMode: !state.darkMode })),
      togglePose: () => set((state: SystemState) => ({ showPose: !state.showPose })),
      toggleBoundingBoxes: () => set((state: SystemState) => ({ showBoundingBoxes: !state.showBoundingBoxes })),
      toggleDepth: () => set((state: SystemState) => ({ showDepth: !state.showDepth })),
      toggleInteractions: () => set((state: SystemState) => ({ showInteractions: !state.showInteractions })),
    }),
    {
      name: 'system-storage',
      partialize: (state: SystemState) => ({ 
        darkMode: state.darkMode,
        showPose: state.showPose,
        showBoundingBoxes: state.showBoundingBoxes,
        showDepth: state.showDepth,
        showInteractions: state.showInteractions,
      }),
    }
  )
)