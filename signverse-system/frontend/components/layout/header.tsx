"use client"
import { Menu, Bell, Settings, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useSystemStore } from "@/store/use-system-store"
import { cn } from "@/lib/utils"
interface HeaderProps { onMenuClick: () => void }
export function Header({ onMenuClick }: HeaderProps) {
  const { isProcessing, fps } = useSystemStore()
  return (
    <header className="flex items-center justify-between h-16 px-6 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-40">
      <div className="flex items-center space-x-4">
        <Button variant="ghost" size="icon" className="lg:hidden text-slate-400 hover:text-white" onClick={onMenuClick}>
          <Menu className="h-5 w-5" />
        </Button>
        <div className="flex items-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className={cn("w-2 h-2 rounded-full transition-colors", isProcessing ? "bg-signverse-accent animate-pulse" : "bg-slate-600")} />
            <span className="text-slate-300">{isProcessing ? "Processing" : "Idle"}</span>
          </div>
          {isProcessing && <div className="font-mono text-xs bg-slate-800 px-2 py-1 rounded-md text-slate-400">{fps.toFixed(1)} FPS</div>}
        </div>
      </div>
      <div className="flex items-center space-x-2">
        <Button variant="ghost" size="icon" className="text-slate-400 hover:text-white relative">
          <Bell className="h-5 w-5" />
          <span className="absolute top-2 right-2 w-1.5 h-1.5 bg-signverse-primary rounded-full" />
        </Button>
        <Button variant="ghost" size="icon" className="text-slate-400 hover:text-white"><Settings className="h-5 w-5" /></Button>
        <div className="w-8 h-8 bg-gradient-to-br from-signverse-primary to-violet-700 rounded-full flex items-center justify-center ml-1 cursor-pointer hover:opacity-90">
          <User className="h-4 w-4 text-white" />
        </div>
      </div>
    </header>
  )
}