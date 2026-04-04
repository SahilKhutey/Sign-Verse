'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Download, Activity, CheckCircle, XCircle, Clock, Youtube } from 'lucide-react'

interface ProcessingJob {
  job_id: string
  status: 'processing' | 'completed' | 'failed' | 'queued'
  url: string
  progress?: number
  error?: string
  title?: string
  duration?: number
  frames_processed?: number
  persons_detected?: number
  events_count?: number
  intentions_count?: number
}

export function YouTubeProcessor() {
  const [url, setUrl] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [jobs, setJobs] = useState<ProcessingJob[]>([])
  const [currentJob, setCurrentJob] = useState<ProcessingJob | null>(null)

  const getYoutubeId = (url: string) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/
    const match = url.match(regExp)
    return (match && match[2].length === 11) ? match[2] : null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url) return

    setIsProcessing(true)
    
    try {
      const response = await fetch('/api/v1/youtube/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      })

      if (response.ok) {
        const data = await response.json()
        const newJob: ProcessingJob = {
          job_id: data.job_id,
          status: 'queued',
          url,
          progress: 0
        }
        
        setJobs(prev => [newJob, ...prev])
        setCurrentJob(newJob)
        
        // Start polling for status updates
        pollJobStatus(data.job_id)
      }
    } catch (error) {
      console.error('Error starting processing:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/v1/youtube/status/${jobId}`)
        if (response.ok) {
          const job = await response.json()
          
          setJobs(prev => prev.map(j => 
            j.job_id === jobId ? { ...j, ...job } : j
          ))
          
          setCurrentJob(prev => prev?.job_id === jobId ? { ...prev, ...job } : prev)
          
          if (job.status === 'completed' || job.status === 'failed') {
            clearInterval(interval)
            
            if (job.status === 'completed') {
              // Load full results
              const resultsResponse = await fetch(`/api/v1/youtube/results/${jobId}`)
              if (resultsResponse.ok) {
                const results = await resultsResponse.json()
                const fullData = { ...job, ...results }
                setJobs(prev => prev.map(j => 
                  j.job_id === jobId ? fullData : j
                ))
                setCurrentJob(prev => prev?.job_id === jobId ? fullData : prev)
              }
            }
          }
        }
      } catch (error) {
        console.error('Error polling job status:', error)
        clearInterval(interval)
      }
    }, 2000) // Poll every 2 seconds
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const videoId = currentJob ? getYoutubeId(currentJob.url) : null

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Download className="h-5 w-5" />
                YouTube Video Processing
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="youtube-url" className="block text-sm font-medium text-slate-300 mb-2">
                    YouTube URL
                  </label>
                  <Input
                    id="youtube-url"
                    type="url"
                    placeholder="https://www.youtube.com/watch?v=..."
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    className="bg-slate-800 border-slate-700 text-white"
                    required
                  />
                </div>
                
                <Button 
                  type="submit" 
                  disabled={isProcessing || !url}
                  className="w-full shadow-lg shadow-primary/20"
                >
                  {isProcessing ? (
                    <>
                      <Activity className="h-4 w-4 mr-2 animate-spin" />
                      Starting...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Process Video
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {currentJob && (
            <Card className="bg-slate-900 border-slate-800 overflow-hidden">
              <CardHeader className="border-b border-slate-800 flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium uppercase tracking-wider text-slate-400">Current Job Status</CardTitle>
                <div className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-tight ${
                  currentJob.status === 'completed' ? 'bg-signverse-accent/20 text-signverse-accent' :
                  currentJob.status === 'failed' ? 'bg-signverse-alert/20 text-signverse-alert' :
                  'bg-signverse-primary/20 text-signverse-primary'
                }`}>
                  {currentJob.status}
                </div>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-4">
                  {currentJob.title && (
                    <div className="flex justify-between items-start gap-4">
                      <span className="text-slate-400 text-sm">Title:</span>
                      <span className="text-white text-sm font-semibold truncate text-right flex-1" title={currentJob.title}>
                        {currentJob.title}
                      </span>
                    </div>
                  )}
                  
                  {currentJob.duration && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-sm">Duration:</span>
                      <span className="text-white font-mono text-sm">{formatDuration(currentJob.duration)}</span>
                    </div>
                  )}
                  
                  {currentJob.progress !== undefined && (currentJob.status === 'processing' || currentJob.status === 'queued') && (
                    <div className="space-y-3 pt-2">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-xs font-semibold uppercase">Processing Pipeline</span>
                        <span className="text-white text-xs font-bold">{Math.round(currentJob.progress * 100)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden ring-1 ring-slate-700/50">
                        <div 
                          className="bg-signverse-primary h-full rounded-full transition-all duration-500 ease-in-out glow-primary shadow-[0_0_8px_rgba(99,102,241,0.6)]"
                          style={{ width: `${currentJob.progress * 100}%` }}
                        />
                      </div>
                      <div className="text-[10px] text-slate-500 font-medium italic animate-pulse">
                        Analyzing frames, detecting poses, and mapping intentions...
                      </div>
                    </div>
                  )}
                  
                  {currentJob.status === 'completed' && (
                    <div className="grid grid-cols-2 gap-3 pt-2">
                      <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-800">
                        <div className="text-lg font-bold text-white leading-none mb-1">{currentJob.frames_processed}</div>
                        <div className="text-[10px] text-slate-500 uppercase font-black">Frames</div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-800">
                        <div className="text-lg font-bold text-white leading-none mb-1">{currentJob.persons_detected}</div>
                        <div className="text-[10px] text-slate-500 uppercase font-black">Persons</div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-800">
                        <div className="text-lg font-bold text-white leading-none mb-1">{currentJob.events_count}</div>
                        <div className="text-[10px] text-slate-500 uppercase font-black">Events</div>
                      </div>
                      <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-800">
                        <div className="text-lg font-bold text-white leading-none mb-1">{currentJob.intentions_count}</div>
                        <div className="text-[10px] text-slate-500 uppercase font-black">Inferences</div>
                      </div>
                    </div>
                  )}
                  
                  {currentJob.status === 'failed' && currentJob.error && (
                    <div className="p-4 bg-signverse-alert/10 rounded-lg border border-signverse-alert/20">
                      <div className="text-signverse-alert font-bold text-xs uppercase flex items-center gap-1.5 mb-1">
                        <XCircle className="h-3 w-3" />
                        Processing Error
                      </div>
                      <div className="text-xs text-slate-300 leading-relaxed">{currentJob.error}</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Card className="bg-slate-900 border-slate-800 h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Youtube className="h-5 w-5 text-red-500" />
                Real-time Preview
              </CardTitle>
            </CardHeader>
            <CardContent>
              {videoId ? (
                <div className="aspect-video bg-black rounded-xl overflow-hidden shadow-2xl ring-1 ring-white/5">
                  <iframe
                    width="100%"
                    height="100%"
                    src={`https://www.youtube.com/embed/${videoId}?autoplay=1&mute=1`}
                    title="YouTube video player"
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                    allowFullScreen
                  />
                </div>
              ) : (
                <div className="aspect-video bg-slate-800/20 rounded-xl border border-dashed border-slate-800 flex flex-col items-center justify-center text-slate-600 gap-4">
                  <Youtube className="h-12 w-12 opacity-20" />
                  <p className="text-xs font-medium uppercase tracking-widest">No Active Video Preview</p>
                </div>
              )}
              
              <div className="mt-6">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Processing History</h3>
                <div className="space-y-2 max-h-[300px] overflow-y-auto custom-scrollbar">
                  {jobs.length === 0 ? (
                    <div className="text-center py-8 text-slate-600 italic text-sm">No recent activity detected</div>
                  ) : (
                    jobs.map((job) => (
                      <div key={job.job_id} className="flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/70 border border-slate-800 rounded-lg transition-all group cursor-pointer" onClick={() => setCurrentJob(job)}>
                        <div className="flex items-center space-x-3 overflow-hidden">
                          {job.status === 'completed' ? (
                            <CheckCircle className="h-4 w-4 text-signverse-accent" />
                          ) : job.status === 'failed' ? (
                            <XCircle className="h-4 w-4 text-signverse-alert" />
                          ) : (
                            <Activity className="h-4 w-4 text-signverse-primary animate-spin" />
                          )}
                          <div className="min-w-0 flex-1">
                            <div className="text-xs font-bold text-slate-200 truncate pr-2 group-hover:text-white transition-colors">
                              {job.title || job.url}
                            </div>
                            <div className="text-[10px] text-slate-500 font-mono">
                              ID: {job.job_id.slice(0, 8)}
                            </div>
                          </div>
                        </div>
                        <Button variant="ghost" size="icon" className="h-6 w-6 text-slate-600 hover:text-white">
                          <Activity className="h-3 w-3" />
                        </Button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
