import { create } from 'zustand'

export interface CameraConfig {
  id: string
  name: string
  source: string
  isActive: boolean
  position?: { x: number; y: number; z: number } // For 3D positioning
  rotation?: { x: number; y: number; z: number } // For 3D rotation
}

interface CameraState {
  cameras: Record<string, CameraConfig>
  activeCameraIds: string[]
  cameraLayout: 'grid' | 'focus' | 'custom'
  focusedCamera: string | null
  
  // Actions
  addCamera: (config: CameraConfig) => void
  removeCamera: (id: string) => void
  updateCamera: (id: string, updates: Partial<CameraConfig>) => void
  setCameraLayout: (layout: 'grid' | 'focus' | 'custom') => void
  setFocusedCamera: (id: string | null) => void
  toggleCameraActive: (id: string) => void
}

export const useCameraStore = create<CameraState>((set, get) => ({
  cameras: {
    'cam-1': {
      id: 'cam-1',
      name: 'Main Entrance',
      source: '/api/stream/cam-1',
      isActive: true,
      position: { x: 0, y: 2.5, z: 0 }
    },
    'cam-2': {
      id: 'cam-2',
      name: 'Hallway',
      source: '/api/stream/cam-2',
      isActive: true,
      position: { x: 5, y: 2.5, z: 0 }
    }
  },
  activeCameraIds: ['cam-1', 'cam-2'],
  cameraLayout: 'grid',
  focusedCamera: null,
  
  addCamera: (config) => set((state) => ({
    cameras: { ...state.cameras, [config.id]: config },
    activeCameraIds: [...state.activeCameraIds, config.id]
  })),
  
  removeCamera: (id) => set((state) => {
    const newCameras = { ...state.cameras }
    delete newCameras[id]
    return {
      cameras: newCameras,
      activeCameraIds: state.activeCameraIds.filter(camId => camId !== id)
    }
  }),
  
  updateCamera: (id, updates) => set((state) => ({
    cameras: {
      ...state.cameras,
      [id]: { ...state.cameras[id], ...updates }
    }
  })),
  
  setCameraLayout: (layout) => set({ cameraLayout: layout }),
  
  setFocusedCamera: (id) => set({ focusedCamera: id }),
  
  toggleCameraActive: (id) => set((state) => {
    const isActive = !state.cameras[id].isActive
    return {
      cameras: {
        ...state.cameras,
        [id]: { ...state.cameras[id], isActive }
      },
      activeCameraIds: isActive 
        ? [...state.activeCameraIds, id]
        : state.activeCameraIds.filter(camId => camId !== id)
    }
  })
}))
