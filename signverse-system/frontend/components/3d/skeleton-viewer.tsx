'use client'

import { useRef, useEffect, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Sphere, Line } from '@react-three/drei'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Pause, RotateCcw } from 'lucide-react'

// Skeleton joint connections
const JOINT_CONNECTIONS = [
  // Spine
  [0, 1], [1, 2], [2, 3],
  // Left arm
  [2, 4], [4, 5], [5, 6],
  // Right arm
  [2, 7], [7, 8], [8, 9],
  // Left leg
  [0, 10], [10, 11], [11, 12],
  // Right leg
  [0, 13], [13, 14], [14, 15]
]

// Mock 3D pose data
const mockPoseData = [
  // Frame 0
  {
    joints: [
      { x: 0, y: 0, z: 0 },        // 0: Hip
      { x: 0, y: 0.3, z: 0 },      // 1: Spine1
      { x: 0, y: 0.6, z: 0 },      // 2: Spine2
      { x: 0, y: 0.9, z: 0 },      // 3: Head
      { x: -0.2, y: 0.6, z: 0 },   // 4: LeftShoulder
      { x: -0.4, y: 0.4, z: 0 },   // 5: LeftElbow
      { x: -0.6, y: 0.4, z: 0 },   // 6: LeftWrist
      { x: 0.2, y: 0.6, z: 0 },    // 7: RightShoulder
      { x: 0.4, y: 0.4, z: 0 },    // 8: RightElbow
      { x: 0.6, y: 0.4, z: 0 },    // 9: RightWrist
      { x: -0.1, y: 0, z: 0 },     // 10: LeftHip
      { x: -0.1, y: -0.4, z: 0 },  // 11: LeftKnee
      { x: -0.1, y: -0.8, z: 0 },  // 12: LeftAnkle
      { x: 0.1, y: 0, z: 0 },      // 13: RightHip
      { x: 0.1, y: -0.4, z: 0 },   // 14: RightKnee
      { x: 0.1, y: -0.8, z: 0 }    // 15: RightAnkle
    ]
  },
  // Add more frames for animation...
]

function SkeletonModel({ frame }: { frame: number }) {
  const joints = mockPoseData[frame % mockPoseData.length].joints
  
  return (
    <group>
      {/* Render joints */}
      {joints.map((joint, index) => (
        <Sphere key={index} position={[joint.x, joint.y, joint.z]} args={[0.03]}>
          <meshStandardMaterial color="#6366F1" />
        </Sphere>
      ))}
      
      {/* Render connections between joints */}
      {JOINT_CONNECTIONS.map(([start, end], index) => (
        <Line
          key={index}
          points={[
            [joints[start].x, joints[start].y, joints[start].z],
            [joints[end].x, joints[end].y, joints[end].z]
          ]}
          color="#22C55E"
          lineWidth={2}
        />
      ))}
    </group>
  )
}

function Scene({ frame }: { frame: number }) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <gridHelper args={[10, 10]} />
      <axesHelper args={[5]} />
      <SkeletonModel frame={frame} />
      <OrbitControls />
    </>
  )
}

export function SkeletonViewer() {
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const animationRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    if (isPlaying) {
      animationRef.current = setInterval(() => {
        setCurrentFrame(prev => (prev + 1) % 1) // Loop through animation (currently only 1 frame in mock)
      }, 100)
    } else {
      clearInterval(animationRef.current)
    }

    return () => clearInterval(animationRef.current)
  }, [isPlaying])

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">3D Skeleton Viewer</h2>
        <div className="flex space-x-2">
          <Button
            variant={isPlaying ? "destructive" : "default"}
            onClick={() => setIsPlaying(!isPlaying)}
            size="sm"
            className="shadow-lg shadow-primary/20"
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
            onClick={() => setCurrentFrame(0)}
            size="sm"
            className="border-slate-700 hover:bg-slate-800"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
        </div>
      </div>

      <Card className="bg-slate-950 border-slate-800 overflow-hidden relative">
        <CardContent className="p-0">
          <div className="h-[500px] w-full">
            <Canvas camera={{ position: [2, 2, 2], fov: 45 }}>
              <Scene frame={currentFrame} />
            </Canvas>
          </div>
          {/* Legend / Overlay */}
          <div className="absolute bottom-4 left-4 p-3 bg-slate-900/80 backdrop-blur-md rounded-lg border border-slate-800 text-[10px] text-slate-400 font-mono uppercase tracking-widest z-10">
            <div className="flex items-center mb-1">
              <span className="w-2 h-2 rounded-full bg-signverse-primary mr-2" />
              Joints (Spheres)
            </div>
            <div className="flex items-center">
              <span className="w-2 h-2 rounded-full bg-signverse-accent mr-2" />
              Connections (Lines)
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
              Joint Count
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white font-mono">16</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
              Current Frame
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white font-mono">{currentFrame}</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
              Tracking Quality
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-signverse-accent font-mono">92%</div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
