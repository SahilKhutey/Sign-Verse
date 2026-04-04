'use client'

import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Play, Database, CheckCircle, XCircle, BarChart3, Clock, Loader2, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DatasetBuildResult {
  dataset_id: string
  created_at: string
  metrics: {
    discovered: number
    metadata_passed: number
    visual_accepted: number
    pipeline_completed: number
  }
  categories: Record<string, number>
  average_quality_score: number
}

interface BuildJobStatus {
  job_id: string
  status: 'starting' | 'processing' | 'completed' | 'failed'
  progress: number
  result: DatasetBuildResult | null
  error: string | null
}

export function DatasetBuilder() {
  const [isBuilding, setIsBuilding] = useState(false)
  const [maxVideos, setMaxVideos] = useState(10)
  const [activeJob, setActiveJob] = useState<BuildJobStatus | null>(null)
  const [buildResult, setBuildResult] = useState<DatasetBuildResult | null>(null)
  const [buildHistory, setBuildHistory] = useState<DatasetBuildResult[]>([])
  
  const pollingRef = useRef<NodeJS.Timeout | null>(null)

  // 1. Initial Load: Fetch History
  useEffect(() => {
    fetchHistory()
    return () => stopPolling()
  }, [])

  const fetchHistory = async () => {
    try {
      const response = await fetch('/api/v1/datasets/list')
      if (response.ok) {
        const history = await response.json()
        // Sort by date descending
        const sorted = history.sort((a: any, b: any) => 
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        )
        setBuildHistory(sorted)
      }
    } catch (err) {
      console.error('Failed to fetch dataset history:', err)
    }
  }

  // 2. Start Build Job
  const handleBuildDataset = async () => {
    setIsBuilding(true)
    setBuildResult(null)
    setActiveJob({
        job_id: '',
        status: 'starting',
        progress: 0,
        result: null,
        error: null
    })
    
    try {
      const response = await fetch('/api/v1/datasets/build', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ max_videos: maxVideos }),
      })

      if (response.ok) {
        const { job_id } = await response.json()
        startPolling(job_id)
      } else {
        setIsBuilding(false)
        setActiveJob(null)
      }
    } catch (error) {
      console.error('Error starting dataset build:', error)
      setIsBuilding(false)
      setActiveJob(null)
    }
  }

  // 3. Polling Logic
  const startPolling = (jobId: string) => {
    stopPolling()
    pollingRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/datasets/status/${jobId}`)
        if (res.ok) {
          const status: BuildJobStatus = await res.json()
          setActiveJob(status)

          if (status.status === 'completed' && status.result) {
            stopPolling()
            setBuildResult(status.result)
            setIsBuilding(false)
            fetchHistory() // Refresh the history list
          } else if (status.status === 'failed') {
            stopPolling()
            setIsBuilding(false)
          }
        }
      } catch (err) {
        console.error('Polling failed:', err)
        stopPolling()
        setIsBuilding(false)
      }
    }, 2000)
  }

  const stopPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }

  return (
    <div className="space-y-6">
      {/* Configure Section */}
      <Card className="bg-slate-900 border-slate-800 shadow-xl overflow-hidden relative">
        <div className="absolute top-0 left-0 w-1 bg-signverse-primary h-full" />
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-white">
            <Database className="h-5 w-5 text-signverse-primary" />
            Automated Dataset Ingestion
          </CardTitle>
          <CardDescription>
            Bootstrap high-quality perception data by autonomously scanning YouTube categories.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-slate-300">
                Max Videos to Ingest
              </label>
              <span className="text-signverse-primary font-bold text-lg">{maxVideos}</span>
            </div>
            <Slider 
              value={[maxVideos]} 
              onValueChange={(v) => setMaxVideos(v[0])} 
              max={50} 
              min={1} 
              step={1}
              disabled={isBuilding}
            />
          </div>
            
          <Button 
            onClick={handleBuildDataset} 
            disabled={isBuilding}
            className={cn(
                "w-full h-12 transition-all duration-300",
                isBuilding ? "bg-slate-800" : "bg-signverse-primary hover:bg-indigo-700"
            )}
          >
            {isBuilding ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Building Dataset ({activeJob?.progress}%)...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Build New Dataset
              </>
            )}
          </Button>
          
          {activeJob?.status === 'failed' && (
              <div className="flex items-center gap-2 text-signverse-alert bg-red-950/20 p-3 rounded-lg border border-red-500/20">
                  <AlertCircle className="h-4 w-4" />
                  <span className="text-xs">Build failed: {activeJob.error || 'Unknown error'}</span>
              </div>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      {buildResult && (
        <Card className="bg-slate-900 border-slate-800 animate-in fade-in slide-in-from-bottom-4">
          <CardHeader>
            <CardTitle className="text-white flex justify-between items-center">
                <span>Result: {buildResult.dataset_id}</span>
                <CheckCircle className="h-5 w-5 text-signverse-accent" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50 text-center">
                <div className="text-2xl font-bold text-white">{buildResult.metrics.discovered}</div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Discovered</div>
              </div>
              <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50 text-center">
                <div className="text-2xl font-bold text-white">{buildResult.metrics.metadata_passed}</div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Pre-Filter</div>
              </div>
              <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50 text-center">
                <div className="text-2xl font-bold text-signverse-accent">{buildResult.metrics.visual_accepted}</div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Accepted</div>
              </div>
              <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50 text-center">
                <div className="text-2xl font-bold text-signverse-primary">{buildResult.metrics.pipeline_completed}</div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Processed</div>
              </div>
            </div>

            {Object.keys(buildResult.categories).length > 0 && (
              <div className="mt-8">
                <h4 className="text-sm font-medium text-slate-400 mb-4 flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" /> Category Distribution
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
                  {Object.entries(buildResult.categories).map(([category, count]) => (
                    <div key={category} className="flex justify-between items-center group">
                      <span className="text-slate-300 capitalize group-hover:text-white transition-colors">{category}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
                            <div 
                                className="h-full bg-signverse-primary" 
                                style={{ width: `${(count / buildResult.metrics.visual_accepted) * 100}%` }}
                            />
                        </div>
                        <span className="text-white font-mono text-sm">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* History Section */}
      {buildHistory.length > 0 && (
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white text-lg">Dataset Build History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {buildHistory.map((build) => (
                <div 
                    key={build.dataset_id} 
                    className="flex items-center justify-between p-4 bg-slate-800/30 hover:bg-slate-800/50 transition-colors rounded-xl border border-slate-800"
                >
                  <div className="flex items-center space-x-4">
                    <div className="p-2 bg-slate-900 rounded-lg">
                        <Database className="h-4 w-4 text-slate-500" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-white">{build.dataset_id}</div>
                      <div className="text-[10px] text-slate-500 flex items-center gap-1 uppercase tracking-tighter">
                        <Clock className="h-2 w-2" /> {new Date(build.created_at).toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-10">
                      <div className="flex flex-col items-center">
                          <span className="text-signverse-accent font-bold text-sm">{build.metrics.visual_accepted}</span>
                          <span className="text-[9px] text-slate-600 uppercase">Input</span>
                      </div>
                      <div className="flex flex-col items-center">
                          <span className="text-signverse-primary font-bold text-sm">{build.metrics.pipeline_completed}</span>
                          <span className="text-[9px] text-slate-600 uppercase">Ready</span>
                      </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
