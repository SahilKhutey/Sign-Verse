'use client'

import { useRef, useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Play, Pause, Square, Maximize } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useSystemStore } from '@/store/use-system-store'
import { useCameraStore } from '@/store/use-camera-store'
import { BoundingBoxOverlay } from '@/components/tracking/bounding-box'
import { PoseOverlay } from '@/components/pose/pose-overlay'

interface VideoPlayerProps {
  cameraId?: string
  isPlaying?: boolean
  onPlayPause?: () => void
  showControls?: boolean
  className?: string
}

export function VideoPlayer({ 
  cameraId, 
  isPlaying: externalIsPlaying, 
  onPlayPause: externalOnPlayPause, 
  showControls = true,
  className 
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [dimensions, setDimensions] = useState({ width: 1280, height: 720 })
  const [internalIsPlaying, setInternalIsPlaying] = useState(false)
  const { showPose, showBoundingBoxes, showDepth } = useSystemStore()
  const { cameras } = useCameraStore()
  
  const isPlaying = externalIsPlaying !== undefined ? externalIsPlaying : internalIsPlaying
  const onPlayPause = externalOnPlayPause || (() => setInternalIsPlaying(!internalIsPlaying))
  
  const camera = cameraId ? cameras[cameraId] : null
  const videoSource = camera?.source || "/api/video-feed"

  useEffect(() => {
    // Simulate video processing
    if (isPlaying && videoRef.current) {
      // This would connect to your actual video source
      videoRef.current.src = videoSource
      videoRef.current.play().catch(console.error)
    } else if (videoRef.current) {
      videoRef.current.pause()
    }
  }, [isPlaying, videoSource])

  const handleCanvasClick = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play()
        onPlayPause()
      } else {
        videoRef.current.pause()
        onPlayPause()
      }
    }
  }

  return (
    <div className={cn("bg-slate-900 rounded-xl border border-slate-800 overflow-hidden", className)}>
      {/* Video container */}
      <div className="relative w-full h-full bg-black">
        <video
          ref={videoRef}
          className="w-full h-full object-contain"
          muted
          loop
          onClick={handleCanvasClick}
        />
        
        {/* Overlay canvas for drawing */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          width={dimensions.width}
          height={dimensions.height}
        />
        
        {/* Computer vision overlays */}
        {showBoundingBoxes && <BoundingBoxOverlay canvasRef={canvasRef} />}
        {showPose && <PoseOverlay canvasRef={canvasRef} />}

        {/* Play/pause overlay */}
        {!isPlaying && showControls && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <Button
              size="lg"
              className="rounded-full w-16 h-16"
              onClick={onPlayPause}
            >
              <Play className="h-8 w-8" />
            </Button>
          </div>
        )}
      </div>

      {/* Controls */}
      {showControls && (
        <div className="p-4 bg-slate-800/50 border-t border-slate-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant={isPlaying ? "destructive" : "default"}
                onClick={onPlayPause}
                size="sm"
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Stop
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start
                  </>
                )}
              </Button>
            </div>

            <div className="flex items-center space-x-4 text-sm text-slate-400">
              <span className={cn(
                "w-2 h-2 rounded-full",
                isPlaying ? "bg-signverse-accent animate-pulse" : "bg-slate-600"
              )} />
              {isPlaying ? "Processing" : "Idle"}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}