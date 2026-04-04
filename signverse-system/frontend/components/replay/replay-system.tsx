'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Play, Pause, SkipBack, SkipForward, Calendar, Clock } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { format } from 'date-fns'

// Mock replay data
const mockReplayData = {
  startTime: Date.now() - 24 * 60 * 60 * 1000, // 24 hours ago
  endTime: Date.now(),
  frames: Array.from({ length: 100 }, (_, i) => ({
    id: `frame-${i}`,
    timestamp: Date.now() - (100 - i) * 15 * 60 * 1000, // 15 min intervals
    cameraId: 'cam-1',
    people: Math.floor(Math.random() * 10) + 1,
    objects: Math.floor(Math.random() * 5) + 1,
    events: Math.random() > 0.7 ? [{ type: 'movement', severity: 'medium' }] : []
  }))
}

export function ReplaySystem() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const playbackInterval = useRef<NodeJS.Timeout>()

  const totalFrames = mockReplayData.frames.length
  const currentFrameData = mockReplayData.frames[currentFrame]

  useEffect(() => {
    if (isPlaying) {
      playbackInterval.current = setInterval(() => {
        setCurrentFrame(prev => {
          if (prev >= totalFrames - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 1000 / playbackSpeed)
    } else {
      clearInterval(playbackInterval.current)
    }

    return () => clearInterval(playbackInterval.current)
  }, [isPlaying, playbackSpeed, totalFrames])

  const handleSeek = (value: number[]) => {
    setCurrentFrame(value[0])
  }

  const formatTime = (timestamp: number) => {
    return format(new Date(timestamp), 'HH:mm:ss')
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Replay System</h2>
        <div className="flex items-center space-x-2 text-slate-400">
          <Calendar className="h-4 w-4" />
          <span>{format(new Date(currentFrameData.timestamp), 'MMM dd, yyyy')}</span>
          <Clock className="h-4 w-4 ml-2" />
          <span>{formatTime(currentFrameData.timestamp)}</span>
        </div>
      </div>

      {/* Video replay area */}
      <Card className="bg-slate-900 border-slate-800">
        <CardContent className="p-6">
          <div className="aspect-video bg-black rounded-lg flex items-center justify-center mb-4 relative overflow-hidden group">
            <div className="text-center text-slate-400 z-10">
              <div className="text-lg mb-2 font-medium text-white">Replay View</div>
              <div className="text-sm">Frame {currentFrame + 1} of {totalFrames}</div>
              <div className="text-sm mt-2">
                {currentFrameData.people} people, {currentFrameData.objects} objects
              </div>
              {currentFrameData.events.length > 0 && (
                <div className="text-red-400 text-sm mt-3 font-semibold px-4 py-1.5 bg-red-500/10 border border-red-500/20 rounded-full animate-pulse">
                  Event detected: {currentFrameData.events[0].type}
                </div>
              )}
            </div>
            {/* Background pattern */}
            <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#6366f1_1px,transparent_1px)] [background-size:20px_20px]" />
          </div>

          {/* Timeline slider */}
          <div className="space-y-4 pt-2">
            <Slider
              value={[currentFrame]}
              min={0}
              max={totalFrames - 1}
              step={1}
              onValueChange={handleSeek}
              className="w-full"
            />
            
            <div className="flex justify-between text-sm text-slate-500 font-medium font-mono">
              <span>{formatTime(mockReplayData.startTime)}</span>
              <span>{formatTime(mockReplayData.endTime)}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Controls */}
      <Card className="bg-slate-900 border-slate-800">
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center space-x-4">
              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentFrame(0)}
                className="border-slate-700 hover:bg-slate-800"
              >
                <SkipBack className="h-4 w-4" />
              </Button>
              
              <Button
                variant={isPlaying ? "destructive" : "default"}
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-28 shadow-lg shadow-primary/20"
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Play
                  </>
                )}
              </Button>
              
              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentFrame(totalFrames - 1)}
                className="border-slate-700 hover:bg-slate-800"
              >
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex items-center space-x-3">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Speed</span>
              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                className="bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-white text-sm focus:ring-2 focus:ring-signverse-primary transition-all outline-none"
              >
                <option value={0.5}>0.5x</option>
                <option value={1}>1.0x</option>
                <option value={2}>2.0x</option>
                <option value={4}>4.0x</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Event log */}
      <Card className="bg-slate-900 border-slate-800">
        <CardHeader className="border-b border-slate-800">
          <CardTitle className="text-white text-lg">System Event Log (Historical)</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="h-64 overflow-y-auto custom-scrollbar">
            {mockReplayData.frames
              .filter(frame => frame.events.length > 0)
              .map((frame, idx) => (
                <div 
                  key={frame.id} 
                  className={`p-4 border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors cursor-pointer ${currentFrame === idx ? 'bg-slate-800/40 border-l-2 border-l-red-500' : ''}`}
                  onClick={() => setCurrentFrame(mockReplayData.frames.indexOf(frame))}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-mono text-slate-300">{formatTime(frame.timestamp)}</span>
                    <span className="text-[10px] px-2 py-0.5 bg-red-500/20 text-red-400 rounded-md uppercase font-bold tracking-tight">
                      {frame.events[0].type}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500">
                    Camera: Main Entrance • Severity: {frame.events[0].severity} • Confidence: High
                  </div>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
