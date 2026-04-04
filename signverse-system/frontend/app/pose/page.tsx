'use client'

import { SkeletonViewer } from "@/components/3d/skeleton-viewer"

export default function PosePage() {
  return (
    <div className="container mx-auto py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white tracking-tight">Pose Analysis</h1>
        <p className="text-slate-400 mt-1.5 text-sm">3D skeletal reconstruction and biomechanical tracking</p>
      </div>
      
      <SkeletonViewer />
    </div>
  )
}

