'use client'

import { useEffect, useRef } from 'react'
import { useTrackingStore, BoundingBox } from '@/store/use-tracking-store'
import { useSystemStore } from '@/store/use-system-store'

interface BoundingBoxOverlayProps {
  canvasRef: React.RefObject<HTMLCanvasElement>
  cameraId?: string
}

export function BoundingBoxOverlay({ canvasRef, cameraId }: BoundingBoxOverlayProps) {
  const { boundingBoxes, getBoundingBoxesByCamera } = useTrackingStore()
  const { isProcessing, showBoundingBoxes } = useSystemStore()
  const animationRef = useRef<number>()

  // Draw bounding boxes on the canvas
  const drawBoundingBoxes = () => {
    const canvas = canvasRef.current
    if (!canvas || !isProcessing || !showBoundingBoxes) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Get boxes for this specific camera if ID is provided, otherwise use all
    const activeBoxes = cameraId ? getBoundingBoxesByCamera(cameraId) : boundingBoxes

    // Draw each bounding box
    activeBoxes.forEach((box: BoundingBox) => {
      const { x, y, width, height, label, confidence, color, trackingId } = box
      
      // Draw the bounding box
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, width, height)
      
      // Draw label background
      ctx.fillStyle = color
      const text = `${label} ${(confidence * 100).toFixed(1)}%`
      ctx.font = 'bold 12px Inter, sans-serif'
      const textMetrics = ctx.measureText(text)
      const textWidth = textMetrics.width
      const textHeight = 16
      
      // Top label background
      ctx.fillRect(x - 1, y - textHeight - 2, textWidth + 10, textHeight + 2)
      
      // Draw label text
      ctx.fillStyle = '#FFFFFF'
      ctx.fillText(text, x + 4, y - 4)
      
      // Draw tracking ID if available
      if (trackingId) {
        const idText = `ID: ${trackingId}`
        const idMetrics = ctx.measureText(idText)
        const idWidth = idMetrics.width
        
        ctx.fillStyle = 'rgba(0,0,0,0.5)'
        ctx.fillRect(x + width - idWidth - 8, y + 2, idWidth + 6, textHeight)
        ctx.fillStyle = '#FFFFFF'
        ctx.fillText(idText, x + width - idWidth - 5, y + 14)
      }
    })
  }

  // Animation loop for real-time updates
  useEffect(() => {
    const animate = () => {
      drawBoundingBoxes()
      animationRef.current = requestAnimationFrame(animate)
    }

    if (isProcessing && showBoundingBoxes) {
      animationRef.current = requestAnimationFrame(animate)
    } else {
      // Clear canvas when not processing or hidden
      const canvas = canvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        ctx?.clearRect(0, 0, canvas.width, canvas.height)
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isProcessing, showBoundingBoxes, boundingBoxes, canvasRef])

  return null
}
