'use client'

import { useEffect, useRef } from 'react'
import { usePoseStore, Pose, Keypoint } from '@/store/use-pose-store'
import { useSystemStore } from '@/store/use-system-store'

interface PoseOverlayProps {
  canvasRef: React.RefObject<HTMLCanvasElement>
  cameraId?: string
}

// Define connections between keypoints for skeleton drawing
const POSE_CONNECTIONS = [
  // Face connections
  ['nose', 'left_eye'],
  ['nose', 'right_eye'],
  ['left_eye', 'left_ear'],
  ['right_eye', 'right_ear'],
  
  // Upper body connections
  ['nose', 'left_shoulder'],
  ['nose', 'right_shoulder'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  
  // Lower body connections
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
]

// Connection colors for different parts of the body
const CONNECTION_COLORS: Record<string, string> = {
  face: 'rgba(255, 107, 107, 0.8)',
  upper_body: 'rgba(78, 205, 196, 0.8)',
  lower_body: 'rgba(69, 183, 209, 0.8)',
  arms: 'rgba(249, 168, 38, 0.8)',
  legs: 'rgba(155, 89, 182, 0.8)'
}

export function PoseOverlay({ canvasRef, cameraId }: PoseOverlayProps) {
  const { poses, getPosesByCamera } = usePoseStore()
  const { isProcessing, showPose } = useSystemStore()
  const animationRef = useRef<number>()

  // Draw pose skeletons on the canvas
  const drawPoses = () => {
    const canvas = canvasRef.current
    if (!canvas || !isProcessing || !showPose) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // We don't clear the entire canvas here as bounding boxes might be drawn on the same canvas
    // ctx.clearRect(0, 0, canvas.width, canvas.height)

    const activePoses = cameraId ? getPosesByCamera(cameraId) : poses

    // Draw each pose
    activePoses.forEach((pose: Pose) => {
      const { keypoints, score, trackingId } = pose
      
      // Draw connections between keypoints
      POSE_CONNECTIONS.forEach(([startName, endName]) => {
        const startKp = keypoints.find(kp => kp.name === startName)
        const endKp = keypoints.find(kp => kp.name === endName)
        
        if (startKp && endKp && startKp.score > 0.2 && endKp.score > 0.2) {
          // Determine connection color based on body part
          let color
          if (startName.includes('eye') || startName.includes('ear') || startName.includes('nose')) {
            color = CONNECTION_COLORS.face
          } else if (startName.includes('shoulder') || startName.includes('elbow') || startName.includes('wrist')) {
            color = CONNECTION_COLORS.arms
          } else if (startName.includes('hip') || startName.includes('knee') || startName.includes('ankle')) {
            color = CONNECTION_COLORS.legs
          } else {
            color = CONNECTION_COLORS.upper_body
          }
          
          // Draw the connection line
          ctx.beginPath()
          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.moveTo(startKp.x, startKp.y)
          ctx.lineTo(endKp.x, endKp.y)
          ctx.stroke()
        }
      })

      // Draw keypoints
      keypoints.forEach((kp: Keypoint) => {
        if (kp.score > 0.2) { // Only draw if confidence is above threshold
          ctx.beginPath()
          ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI)
          
          // Color based on confidence
          const alpha = Math.min(1, kp.score * 2).toFixed(2)
          ctx.fillStyle = `rgba(34, 197, 94, ${alpha})` // Green with alpha
          ctx.fill()
          
          // Draw outline
          ctx.strokeStyle = '#166534'
          ctx.lineWidth = 1
          ctx.stroke()
        }
      })
      
      // Draw pose ID if available
      if (trackingId) {
        // Find a central point to display the ID (average of shoulder/hip)
        const neckIndices = ['left_shoulder', 'right_shoulder']
        const validNeck = keypoints.filter(kp => neckIndices.includes(kp.name) && kp.score > 0.5)
        
        if (validNeck.length > 0) {
          const centerX = validNeck.reduce((sum, kp) => sum + kp.x, 0) / validNeck.length
          const centerY = validNeck.reduce((sum, kp) => sum + kp.y, 0) / validNeck.length
          
          ctx.fillStyle = 'rgba(99, 102, 241, 0.9)'
          const idText = `Person ${trackingId}`
          ctx.font = 'bold 11px Inter, sans-serif'
          const textMetrics = ctx.measureText(idText)
          const textWidth = textMetrics.width
          const textHeight = 14
          
          ctx.fillRect(centerX - textWidth / 2 - 4, centerY - 24, textWidth + 8, textHeight + 4)
          ctx.fillStyle = '#FFFFFF'
          ctx.textAlign = 'center'
          ctx.fillText(idText, centerX, centerY - 13)
          ctx.textAlign = 'left' // Reset alignment
        }
      }
    })
  }

  // Animation loop for real-time updates
  useEffect(() => {
    const animate = () => {
      drawPoses()
      animationRef.current = requestAnimationFrame(animate)
    }

    if (isProcessing && showPose) {
      animationRef.current = requestAnimationFrame(animate)
    } else {
      // Clear pose overlays when not processing or hidden
      const canvas = canvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        // We don't clear the entire canvas here to avoid flickering with bounding boxes
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isProcessing, showPose, poses, canvasRef])

  return null
}
