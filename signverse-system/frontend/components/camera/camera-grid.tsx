'use client'

import { useCameraStore } from '@/store/use-camera-store'
import { VideoPlayer } from '@/components/video/video-player'
import { Button } from '@/components/ui/button'
import { LayoutGrid, Focus, Settings } from 'lucide-react'
import { cn } from '@/lib/utils'

export function CameraGrid() {
  const { cameras, activeCameraIds, cameraLayout, setCameraLayout, focusedCamera, setFocusedCamera } = useCameraStore()
  
  const activeCameras = activeCameraIds.map(id => cameras[id]).filter(Boolean)
  
  const getGridClass = () => {
    switch (activeCameras.length) {
      case 1: return 'grid-cols-1'
      case 2: return 'grid-cols-2'
      case 3: return 'grid-cols-2'
      case 4: return 'grid-cols-2 lg:grid-cols-2'
      default: return 'grid-cols-2 lg:grid-cols-3'
    }
  }
  
  if (cameraLayout === 'focus' && focusedCamera) {
    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Focused View: {cameras[focusedCamera]?.name}</h3>
          <Button variant="outline" onClick={() => setCameraLayout('grid')}>
            <LayoutGrid className="h-4 w-4 mr-2" />
            Show All
          </Button>
        </div>
        <VideoPlayer 
          cameraId={focusedCamera}
          showControls={true}
          className="h-[70vh]"
        />
      </div>
    )
  }
  
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Multi-Camera View</h3>
        <div className="flex space-x-2">
          <Button
            variant={cameraLayout === 'grid' ? 'secondary' : 'outline'}
            onClick={() => setCameraLayout('grid')}
            size="sm"
          >
            <LayoutGrid className="h-4 w-4" />
          </Button>
          <Button
            variant={cameraLayout === 'focus' ? 'secondary' : 'outline'}
            onClick={() => setCameraLayout('focus')}
            size="sm"
          >
            <Focus className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      <div className={cn('grid gap-4', getGridClass())}>
        {activeCameras.map((camera) => (
          <div key={camera.id} className="relative group">
            <VideoPlayer 
              cameraId={camera.id}
              showControls={false}
              className="h-48 lg:h-64"
            />
            <div className="absolute bottom-2 left-2 right-2 bg-black/70 text-white p-2 rounded text-sm">
              {camera.name}
            </div>
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  setFocusedCamera(camera.id)
                  setCameraLayout('focus')
                }}
              >
                <Focus className="h-3 w-3" />
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
