"use client"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { useSystemStore } from "@/store/use-system-store"
import { formatFPS, formatMilliseconds, formatBytes } from "@/lib/utils"
interface ControlPanelProps { isPlaying: boolean; onPlayPause: () => void }
export function ControlPanel({ isPlaying }: ControlPanelProps) {
  const { showPose, showBoundingBoxes, showDepth, showInteractions,
          setShowPose, setShowBoundingBoxes, setShowDepth, setShowInteractions,
          fps, latency, memoryUsage } = useSystemStore()
  const toggles = [
    {id:"pose-toggle",        label:"Show Pose",      checked:showPose,          onChange:setShowPose},
    {id:"bbox-toggle",        label:"Bounding Boxes", checked:showBoundingBoxes, onChange:setShowBoundingBoxes},
    {id:"depth-toggle",       label:"Depth Map",      checked:showDepth,         onChange:setShowDepth},
    {id:"interaction-toggle", label:"Interactions",   checked:showInteractions,  onChange:setShowInteractions},
  ]
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 space-y-6 h-full">
      <h3 className="text-base font-semibold text-white">Perception Controls</h3>
      <div className="space-y-4">
        {toggles.map(({id, label, checked, onChange}) => (
          <div key={id} className="flex items-center justify-between">
            <Label htmlFor={id} className="text-sm text-slate-300 cursor-pointer">{label}</Label>
            <Switch id={id} checked={checked} onCheckedChange={onChange} />
          </div>
        ))}
      </div>
      <div className="pt-4 border-t border-slate-800">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Performance</h4>
        <div className="space-y-2.5 text-sm">
          <div className="flex justify-between"><span className="text-slate-500">FPS</span><span className="font-mono text-slate-200">{isPlaying ? formatFPS(fps) : "-"}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Latency</span><span className="font-mono text-slate-200">{isPlaying ? formatMilliseconds(latency) : "-"}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Memory</span><span className="font-mono text-slate-200">{isPlaying ? formatBytes(memoryUsage*1024*1024) : "-"}</span></div>
        </div>
      </div>
      <div className="pt-4 border-t border-slate-800">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Actions</h4>
        <div className="space-y-1">
          <button className="w-full text-left py-2 px-3 rounded-lg text-sm text-slate-300 hover:bg-slate-800 transition-colors">Capture Frame</button>
          <button className="w-full text-left py-2 px-3 rounded-lg text-sm text-slate-300 hover:bg-slate-800 transition-colors">Export Data</button>
          <button className="w-full text-left py-2 px-3 rounded-lg text-sm text-signverse-alert hover:bg-red-950/50 transition-colors font-medium">Emergency Stop</button>
        </div>
      </div>
    </div>
  )
}