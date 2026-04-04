export interface Pose {
  keypoints: Array<{ x: number, y: number, z?: number, score?: number }>
  score: number
}

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  label?: string
  confidence?: number
}

export interface FrameData {
  id: string
  timestamp: number
  cameraId: string
  frame: string // Base64 encoded frame or URL
  poses: Pose[]
  boundingBoxes: BoundingBox[]
  interactions: Interaction[]
  events: Event[]
}

export interface Person {
  id: string
  trackingId: number
  position: { x: number; y: number; z?: number }
  velocity: { x: number; y: number; z?: number }
  pose: Pose
  appearance: {
    color: string
    height: number
    clothing: string[]
  }
}

export interface Object {
  id: string
  type: string
  position: { x: number; y: number; z?: number }
  dimensions: { width: number; height: number; depth?: number }
  confidence: number
}

export interface Interaction {
  id: string
  type: string
  personId: string
  objectId?: string
  startTime: number
  endTime?: number
  confidence: number
}

export interface Event {
  id: string
  type: string
  severity: 'low' | 'medium' | 'high'
  timestamp: number
  description: string
  cameraId: string
  data: any
}
