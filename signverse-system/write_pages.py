import os

# Base path
B = "signverse-system/frontend"

def w(rel, lines):
    path = os.path.join(B, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(lines)
    print("[OK]", rel)

# StatsGrid component
w("components/stats/stats-grid.tsx", """import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity, Cpu, MemoryStick, Clock } from "lucide-react"
import { useSystemStore } from "@/store/use-system-store"

export function StatsGrid() {
  const { fps, latency, memoryUsage, isProcessing } = useSystemStore()

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
      value: isProcessing ? "24.5" : "0.0",
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
""")

# Tracking Page
w("app/tracking/page.tsx", """export default function TrackingPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Tracking</h1>
      <p className="text-slate-400">Tracking functionality coming soon...</p>
    </div>
  )
}
""")

# Pose Analysis Page
w("app/pose/page.tsx", """export default function PosePage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Pose Analysis</h1>
      <p className="text-slate-400">Pose analysis functionality coming soon...</p>
    </div>
  )
}
""")

# Interactions Page
w("app/interactions/page.tsx", """export default function InteractionsPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Interactions</h1>
      <p className="text-slate-400">Interactions functionality coming soon...</p>
    </div>
  )
}
""")

# Datasets Page
w("app/datasets/page.tsx", """export default function DatasetsPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Datasets</h1>
      <p className="text-slate-400">Datasets functionality coming soon...</p>
    </div>
  )
}
""")

# Models Page
w("app/models/page.tsx", """export default function ModelsPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Models</h1>
      <p className="text-slate-400">Models functionality coming soon...</p>
    </div>
  )
}
""")

# Pipelines Page
w("app/pipelines/page.tsx", """export default function PipelinesPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Pipelines</h1>
      <p className="text-slate-400">Pipelines functionality coming soon...</p>
    </div>
  )
}
""")

# Analytics Page
w("app/analytics/page.tsx", """export default function AnalyticsPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Analytics</h1>
      <p className="text-slate-400">Analytics functionality coming soon...</p>
    </div>
  )
}
""")

# Settings Page
w("app/settings/page.tsx", """export default function SettingsPage() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Settings</h1>
      <p className="text-slate-400">Settings functionality coming soon...</p>
    </div>
  )
}
""")

print("\\nSuccess.")
