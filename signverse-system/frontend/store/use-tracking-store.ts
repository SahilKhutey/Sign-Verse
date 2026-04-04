import { create } from 'zustand'

export interface BoundingBox {
  id: number
  x: number
  y: number
  width: number
  height: number
  label: string
  confidence: number
  color: string
  trackingId?: number // For persistent tracking across frames
  cameraId?: string // For multi-camera support
}

interface TrackingState {
  boundingBoxes: BoundingBox[]
  addBoundingBox: (box: BoundingBox) => void
  clearBoundingBoxes: () => void
  updateBoundingBox: (id: number, updates: Partial<BoundingBox>) => void
  // New methods for multi-camera support
  getBoundingBoxesByCamera: (cameraId: string) => BoundingBox[]
  setBoundingBoxesForCamera: (cameraId: string, boxes: BoundingBox[]) => void
}

export const useTrackingStore = create<TrackingState>((set, get) => ({
  boundingBoxes: [],
  addBoundingBox: (box) => set((state) => ({ 
    boundingBoxes: [...state.boundingBoxes, box] 
  })),
  clearBoundingBoxes: () => set({ boundingBoxes: [] }),
  updateBoundingBox: (id, updates) => set((state) => ({
    boundingBoxes: state.boundingBoxes.map(box => 
      box.id === id ? { ...box, ...updates } : box
    )
  })),
  getBoundingBoxesByCamera: (cameraId: string) => {
    // Filter boxes by camera ID
    return get().boundingBoxes.filter(box => box.cameraId === cameraId)
  },
  setBoundingBoxesForCamera: (cameraId: string, boxes: BoundingBox[]) => {
    // Remove existing boxes for this camera and add new ones
    const currentBoxes = get().boundingBoxes.filter(box => box.cameraId !== cameraId)
    const boxesWithCameraId = boxes.map(box => ({ ...box, cameraId }))
    set({ boundingBoxes: [...currentBoxes, ...boxesWithCameraId] })
  }
}))
