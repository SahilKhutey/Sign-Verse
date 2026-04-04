'use client'

import { CameraGrid } from "@/components/camera/camera-grid"
import { ControlPanel } from "@/components/control/control-panel"
import { StatsGrid } from "@/components/stats/stats-grid"
import { useSystemStore } from "@/store/use-system-store"

export default function MonitoringPage() {
  const { startPipeline, stopPipeline, isProcessing } = useSystemStore()
  
  const handleStartStop = () => {
    if (isProcessing) stopPipeline()
    else startPipeline()
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white tracking-tight">Real-Time Monitoring</h1>
        <p className="text-slate-400 mt-1.5 text-sm">Live perception system monitoring and control (Multi-Camera OS)</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <CameraGrid />
        </div>
        <div className="lg:col-span-1">
          <ControlPanel 
            isPlaying={isProcessing} 
            onPlayPause={handleStartStop} 
          />
        </div>
      </div>

      <StatsGrid />
    </div>
  )
}