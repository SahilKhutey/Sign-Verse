'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { CheckCircle, XCircle, Download, Play, BarChart3, RefreshCw } from 'lucide-react'

interface VideoCandidate {
  id: string
  title: string
  url: string
  score: number
  human_score: number
  motion_score: number
  trackability_score: number
  relevance_score: number
  category: string
  status: 'accepted' | 'rejected' | 'pending'
  thumbnail?: string
  duration?: number
}

export function VideoSelector() {
  const [candidates, setCandidates] = useState<VideoCandidate[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadCandidates()
  }, [])

  const loadCandidates = async () => {
    setIsLoading(true)
    try {
      // Connecting to real API while maintaining requested data structure
      const response = await fetch('/api/v1/youtube/discovery/results')
      if (response.ok) {
        const data = await response.json()
        const mapped: VideoCandidate[] = (data.videos || []).map((v: any) => ({
            id: v.id,
            title: v.title,
            url: v.url,
            score: v.score || 0,
            human_score: v.metrics?.human_presence || 0,
            motion_score: v.metrics?.motion_quality || 0,
            trackability_score: v.metrics?.trackability || 0,
            relevance_score: v.metrics?.content_relevance || 0,
            category: v.category || 'unknown',
            status: v.status || (v.score > 0.7 ? 'accepted' : 'rejected'),
            duration: v.duration
        }))
        setCandidates(mapped)
      }
    } catch (error) {
      console.error('Error loading candidates:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleProcessVideo = async (url: string) => {
    try {
        await fetch('/api/v1/youtube/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        })
        console.log('Processing video initiated for:', url)
    } catch (err) {
        console.error('Processing failed:', err)
    }
  }

  if (isLoading) {
    return <div className="text-center py-8 text-slate-400">Loading video candidates...</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Video Selection Dashboard</h2>
        <Button onClick={loadCandidates} variant="outline" className="text-slate-400 border-slate-800 hover:bg-slate-800">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {candidates.map((candidate) => (
          <Card key={candidate.id} className="bg-slate-900 border-slate-800 overflow-hidden shadow-lg">
            <CardContent className="p-6">
              <div className="flex flex-col lg:flex-row gap-6">
                {/* Visual Preview / Placeholder Area */}
                <div className="lg:w-1/4 aspect-video bg-slate-950 rounded-lg flex items-center justify-center border border-slate-800 relative group overflow-hidden">
                  <div className="text-center text-slate-600 group-hover:text-signverse-primary transition-colors">
                    <Play className="h-10 w-10 mx-auto mb-2 opacity-50 group-hover:opacity-100 transition-opacity" />
                    <div className="text-[10px] uppercase font-bold tracking-widest">Preview Mode</div>
                    {candidate.duration && (
                      <div className="text-xs mt-1 font-mono text-slate-500">
                        {Math.floor(candidate.duration / 60)}:
                        {(candidate.duration % 60).toString().padStart(2, '0')}
                      </div>
                    )}
                  </div>
                  {/* Status Overlay */}
                  <div className={`absolute top-2 right-2 px-2 py-0.5 rounded text-[9px] font-black uppercase ${
                      candidate.status === 'accepted' ? 'bg-signverse-accent/20 text-signverse-accent' : 'bg-signverse-alert/20 text-signverse-alert'
                  }`}>
                    {candidate.status}
                  </div>
                </div>

                {/* Content & Metrics Area */}
                <div className="lg:w-3/4 space-y-4">
                  <div>
                    <h3 className="text-lg font-bold text-white mb-1 line-clamp-1">
                      {candidate.title}
                    </h3>
                    <div className="flex items-center space-x-4 text-xs text-slate-500 font-medium">
                      <span className="capitalize px-2 py-0.5 bg-slate-800 rounded">{candidate.category}</span>
                      <span className={`flex items-center gap-1.5 ${
                        candidate.status === 'accepted' ? 'text-signverse-accent' : 'text-signverse-alert'
                      }`}>
                        {candidate.status === 'accepted' ? (
                          <CheckCircle className="h-3.5 w-3.5" />
                        ) : (
                          <XCircle className="h-3.5 w-3.5" />
                        )}
                        <span className="uppercase tracking-tighter">{candidate.status}</span>
                      </span>
                    </div>
                  </div>

                  {/* High-Precision Metrics Grid */}
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-slate-800/50 rounded-xl border border-slate-800">
                      <div className="text-xl font-black text-white leading-none mb-1">{Math.round(candidate.score * 100)}%</div>
                      <div className="text-[9px] text-slate-500 uppercase tracking-widest font-black">Overall</div>
                    </div>
                    <div className="text-center p-3 bg-slate-800/50 rounded-xl border border-slate-800">
                        <div className="text-xl font-bold text-slate-300 leading-none mb-1">{Math.round(candidate.human_score * 100)}%</div>
                        <div className="text-[9px] text-slate-500 uppercase tracking-widest font-black">Human</div>
                    </div>
                    <div className="text-center p-3 bg-slate-800/50 rounded-xl border border-slate-800">
                        <div className="text-xl font-bold text-slate-300 leading-none mb-1">{Math.round(candidate.motion_score * 100)}%</div>
                        <div className="text-[9px] text-slate-500 uppercase tracking-widest font-black">Motion</div>
                    </div>
                    <div className="text-center p-3 bg-slate-800/50 rounded-xl border border-slate-800">
                        <div className="text-xl font-bold text-slate-300 leading-none mb-1">{Math.round(candidate.trackability_score * 100)}%</div>
                        <div className="text-[9px] text-slate-500 uppercase tracking-widest font-black">Track</div>
                    </div>
                  </div>

                  {/* Component Interactions */}
                  <div className="flex space-x-3 pt-2">
                    <Button 
                      onClick={() => handleProcessVideo(candidate.url)}
                      disabled={candidate.status !== 'accepted'}
                      className="bg-signverse-primary hover:bg-indigo-700 shadow-[0_0_15px_rgba(99,102,241,0.2)]"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Ingest Data
                    </Button>
                    <Button variant="outline" className="border-slate-800 hover:bg-slate-800 text-slate-400" asChild>
                      <a href={candidate.url} target="_blank" rel="noopener noreferrer">
                        <Play className="h-4 w-4 mr-2" />
                        View Source
                      </a>
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
