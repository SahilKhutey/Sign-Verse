'use client'

import { useState } from 'react'
import { CameraGrid } from '@/components/camera/camera-grid'
import { MetricsDashboard } from '@/components/analytics/metrics-dashboard'
import { ReplaySystem } from '@/components/replay/replay-system'
import { SkeletonViewer } from '@/components/3d/skeleton-viewer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Video, BarChart3, History as HistoryIcon, Box, Download, Database, Youtube } from 'lucide-react'
import { YouTubeProcessor } from '@/components/youtube/youtube-processor'
import { DatasetBuilder } from '@/components/dataset/dataset-builder'
import { VideoSelector } from '@/components/video/video-selector'
import { cn } from '@/lib/utils'

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState('monitoring')

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 mb-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-signverse-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-signverse-primary"></span>
            </span>
            <span className="text-[10px] font-bold text-signverse-primary uppercase tracking-[0.2em]">System Live</span>
          </div>
          <h1 className="text-4xl font-extrabold text-white tracking-tight">AI Perception OS</h1>
          <p className="text-slate-400 mt-2 max-w-2xl text-sm leading-relaxed">
            Centralized multi-camera orchestration engine with real-time pose estimation, behavioral analytics, and historical playback.
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-xl py-4 -mx-4 px-4 border-b border-white/5 mb-6">
          <TabsList className="grid grid-cols-2 lg:grid-cols-6 w-full lg:w-fit gap-2">
            <TabsTrigger value="monitoring" className="flex items-center gap-2.5">
              <Video className="h-4 w-4" />
              Monitoring
            </TabsTrigger>
            <TabsTrigger value="youtube" className="flex items-center gap-2.5">
              <Download className="h-4 w-4" />
              YouTube
            </TabsTrigger>
            <TabsTrigger value="datasets" className="flex items-center gap-2.5 text-signverse-primary">
              <Database className="h-4 w-4" />
              Datasets
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2.5">
              <BarChart3 className="h-4 w-4" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="replay" className="flex items-center gap-2.5">
              <HistoryIcon className="h-4 w-4" />
              Replay
            </TabsTrigger>
            <TabsTrigger value="3d" className="flex items-center gap-2.5">
              <Box className="h-4 w-4" />
              3D Viewer
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="min-h-[600px]">
          <TabsContent value="monitoring" className="animate-in slide-in-from-left-2 duration-300">
            <CameraGrid />
          </TabsContent>

          <TabsContent value="youtube" className="animate-in slide-in-from-left-2 duration-300 space-y-8">
            <YouTubeProcessor />
            <div className="border-t border-slate-800 pt-8">
                <VideoSelector />
            </div>
          </TabsContent>

          <TabsContent value="datasets" className="animate-in slide-in-from-left-2 duration-300">
            <DatasetBuilder />
          </TabsContent>

          <TabsContent value="analytics" className="animate-in slide-in-from-left-2 duration-300">
            <MetricsDashboard />
          </TabsContent>

          <TabsContent value="replay" className="animate-in slide-in-from-left-2 duration-300">
            <ReplaySystem />
          </TabsContent>

          <TabsContent value="3d" className="animate-in slide-in-from-left-2 duration-300">
            <SkeletonViewer />
          </TabsContent>
        </div>
      </Tabs>
      
      {/* Dynamic Footer Status */}
      <div className="pt-8 border-t border-slate-800 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-slate-500 font-medium">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5"><div className="w-1 h-1 rounded-full bg-green-500"/> Foundation: FastAPI v2.4</span>
          <span className="flex items-center gap-1.5"><div className="w-1 h-1 rounded-full bg-blue-500"/> Inference: CUDA 12.1</span>
        </div>
        <span className="uppercase tracking-widest opacity-50">SignVerse Operating System v1.0.0-PRO</span>
      </div>
    </div>
  )
}