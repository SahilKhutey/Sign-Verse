"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { LayoutDashboard,Video,Users,Activity,Database,Settings,Bot,Workflow,BarChart3,Layers,Clock,Youtube } from "lucide-react"
import { cn } from "@/lib/utils"
interface SidebarProps { open: boolean; onClose: () => void }
const NAV = [
  {name:"Dashboard",    href:"/",            icon:LayoutDashboard},
  {name:"Monitoring",   href:"/monitoring",  icon:Video},
  {name:"Tracking",     href:"/tracking",    icon:Users},
  {name:"Pose Analysis",href:"/pose",        icon:Activity},
  {name:"Interactions", href:"/interactions",icon:Workflow},
  {name:"Pipelines",    href:"/pipelines",   icon:Layers},
  {name:"Analytics",    href:"/analytics",   icon:BarChart3},
  {name:"Replay",       href:"/replay",      icon:Clock},
  {name:"YouTube",      href:"/youtube",     icon:Youtube},
  {name:"Datasets",     href:"/datasets",    icon:Database},
  {name:"Models",       href:"/models",      icon:Bot},
  {name:"Settings",     href:"/settings",    icon:Settings},
]
export function Sidebar({ open, onClose }: SidebarProps) {
  const pathname = usePathname()
  return (
    <>
      {open && <div className="fixed inset-0 z-50 bg-black/80 lg:hidden" onClick={onClose} />}
      <div className={cn("fixed inset-y-0 left-0 z-50 w-72 bg-slate-950 border-r border-slate-800 transform transition-transform duration-300 lg:translate-x-0 lg:static lg:z-auto", open?"translate-x-0":"-translate-x-full")}>
        <div className="flex items-center h-16 px-6 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-signverse-primary rounded-lg flex items-center justify-center glow-primary">
              <Activity className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-semibold text-white tracking-tight">SignVerse</span>
          </div>
        </div>
        <nav className="mt-8 px-4">
          <div className="space-y-1">
            {NAV.map(item => {
              const active = pathname === item.href
              return (
                <Link key={item.name} href={item.href} onClick={onClose}
                  className={cn("flex items-center px-4 py-3 rounded-lg text-sm font-medium transition-all",
                    active ? "bg-signverse-primary/10 text-signverse-primary border border-signverse-primary/20 shadow-lg shadow-signverse-primary/10"
                           : "text-slate-400 hover:text-white hover:bg-slate-800/70")}>
                  <item.icon className={cn("h-5 w-5 mr-3 flex-shrink-0", active && "text-signverse-primary")} />
                  {item.name}
                </Link>
              )
            })}
          </div>
        </nav>
        <div className="absolute bottom-4 left-4 right-4 p-4 bg-slate-900/80 rounded-xl border border-slate-800 glass">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">System Status</span>
            <div className="w-2 h-2 bg-signverse-accent rounded-full animate-pulse" />
          </div>
          <p className="text-xs text-slate-500">All systems operational</p>
        </div>
      </div>
    </>
  )
}