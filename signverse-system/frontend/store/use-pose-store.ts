import { create } from 'zustand'

export interface Keypoint {
  x: number
  y: number
  z?: number // For 3D poses
  score: number
  name: string
}

export interface Pose {
  id: number
  keypoints: Keypoint[]
  score: number
  trackingId?: number // For persistent tracking across frames
  cameraId?: string // For multi-camera support
}

interface PoseState {
  poses: Pose[]
  addPose: (pose: Pose) => void
  clearPoses: () => void
  // New methods for multi-camera support
  getPosesByCamera: (cameraId: string) => Pose[]
  setPosesForCamera: (cameraId: string, poses: Pose[]) => void
}

export const usePoseStore = create<PoseState>((set, get) => ({
  poses: [],
  addPose: (pose) => set((state) => ({ 
    poses: [...state.poses, pose] 
  })),
  clearPoses: () => set({ poses: [] }),
  getPosesByCamera: (cameraId: string) => {
    // Filter poses by camera ID
    return get().poses.filter(pose => pose.cameraId === cameraId)
  },
  setPosesForCamera: (cameraId: string, poses: Pose[]) => {
    // Remove existing poses for this camera and add new ones
    const currentPoses = get().poses.filter(pose => pose.cameraId !== cameraId)
    const posesWithCameraId = poses.map(pose => ({ ...pose, cameraId }))
    set({ poses: [...currentPoses, ...posesWithCameraId] })
  }
}))
