import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity, Cpu, MemoryStick, Clock } from "lucide-react"
import { useSystemStore } from "@/store/use-system-store"

export function StatsGrid() {
  const { fps, latency, memoryUsage, cpuUsage, isProcessing } = useSystemStore()

  const stats = [
    {
      title: "FPS",
      value: fps.toFixed(1),
      icon: Activity,
      color: "text-signverse-primary",
      suffix: " fps"
    },
    {
      title: "Latency",
      value: latency.toFixed(0),
      icon: Clock,
      color: "text-signverse-accent",
      suffix: " ms"
    },
    {
      title: "Memory",
      value: (memoryUsage / 1024 / 1024).toFixed(1),
      icon: MemoryStick,
      color: "text-purple-400",
      suffix: " MB"
    },
    {
      title: "CPU Usage",
      value: (cpuUsage ?? 0).toFixed(1),
      icon: Cpu,
      color: "text-orange-400",
      suffix: " %"
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {stats.map((stat, index) => (
        <Card key={index} className="bg-slate-900 border-slate-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-400">
              {stat.title}
            </CardTitle>
            <stat.icon className={`h-4 w-4 ${stat.color}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {stat.value}
              <span className="text-sm text-slate-400 ml-1">{stat.suffix}</span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
