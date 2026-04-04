'use client'; // This is a client component

import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Line } from '@react-three/drei';
import { PoseResult, PoseFrame } from '@/types'; // Added PoseFrame to imports
import * as THREE from 'three';

interface PoseViewer3DProps {
  poseData: PoseResult;
  currentFrame: number;
  onFrameChange?: (frame: number) => void;
}

const BONE_CONNECTIONS = [
  // Left arm
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  // Right arm
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  // Body
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  // Left leg
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  // Right leg
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
];

const Joint: React.FC<{ position: [number, number, number]; confidence: number }> = ({
  position,
  confidence,
}) => (
  <Sphere args={[0.03]} position={position}>
    <meshStandardMaterial
      color={confidence > 0.7 ? '#00ff00' : confidence > 0.4 ? '#ffff00' : '#ff0000'}
      emissive={confidence > 0.7 ? '#00ff00' : confidence > 0.4 ? '#ffff00' : '#ff0000'}
      emissiveIntensity={0.5}
    />
  </Sphere>
);

const Bone: React.FC<{ start: [number, number, number]; end: [number, number, number] }> = ({
  start,
  end,
}) => {
  const points = useMemo(() => [new THREE.Vector3(...start), new THREE.Vector3(...end)], [start, end]);
  
  return (
    <Line points={points} color="white" lineWidth={1.5} transparent opacity={0.6} />
  );
};

const PoseScene: React.FC<{ frame: PoseFrame }> = ({ frame }) => {
  const keypointsMap = useMemo(() => {
    return frame.keypoints.reduce((acc, kp) => {
      acc[kp.name] = [kp.x, kp.y, kp.z] as [number, number, number];
      return acc;
    }, {} as Record<string, [number, number, number]>);
  }, [frame.keypoints]);

  return (
    <>
      {/* Joints */}
      {frame.keypoints.map((kp, index) => (
        <Joint
          key={`${kp.name}-${index}`}
          position={[kp.x, kp.y, kp.z]}
          confidence={kp.confidence}
        />
      ))}

      {/* Bones */}
      {BONE_CONNECTIONS.map(([start, end], index) => {
        const startPos = keypointsMap[start];
        const endPos = keypointsMap[end];
        
        if (!startPos || !endPos) return null;
        
        return (
          <Bone
            key={`bone-${index}`}
            start={startPos}
            end={endPos}
          />
        );
      })}
    </>
  );
};

export const PoseViewer3D: React.FC<PoseViewer3DProps> = ({
  poseData,
  currentFrame,
  onFrameChange,
}) => {
  const currentFrameData = poseData.frames[currentFrame];

  if (!currentFrameData) {
    return (
      <div className="w-full h-96 bg-gray-100 rounded-xl flex items-center justify-center dark:bg-gray-800 border-2 border-dashed border-gray-300 dark:border-gray-700">
        <p className="text-gray-500 italic">No pose data available for this frame</p>
      </div>
    );
  }

  return (
    <div className="relative w-full h-[400px] bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
      <Canvas
        camera={{ position: [0, 1.5, 3], fov: 60 }}
        shadows
        onCreated={({ gl }) => {
          gl.setClearColor('#020617'); // Tailwind Slate-950
        }}
      >
        <ambientLight intensity={0.7} />
        <pointLight position={[5, 5, 5]} intensity={1} castShadow />
        <spotLight position={[-5, 5, 5]} angle={0.15} penumbra={1} intensity={1} />
        
        <OrbitControls 
          enableDamping 
          dampingFactor={0.05}
          target={[0, 1, 0]} // Focus on the torso area roughly
        />
        
        <gridHelper args={[10, 20, '#1e293b', '#0f172a']} position={[0, -1, 0]} />
        <axesHelper args={[0.5]} />
        
        <PoseScene frame={currentFrameData} />
      </Canvas>

      {/* Frame controls overlay */}
      <div className="absolute bottom-6 left-6 right-6 bg-black/40 backdrop-blur-md p-4 rounded-lg border border-white/10">
        <div className="flex items-center space-x-4">
          <input
            type="range"
            min="0"
            max={poseData.frames.length - 1}
            value={currentFrame}
            onChange={(e) => onFrameChange?.(Number(e.target.value))}
            className="flex-1 accent-indigo-500"
          />
        </div>
        <div className="flex justify-between mt-2 px-1 text-[10px] font-mono tracking-wider text-slate-400 uppercase">
          <span>Frame: {currentFrame.toString().padStart(4, '0')}</span>
          <span>Total: {poseData.frames.length}</span>
        </div>
      </div>

      {/* Viewport Info */}
      <div className="absolute top-4 left-4 bg-black/20 text-indigo-400 text-[10px] px-2 py-1 rounded font-mono border border-indigo-500/20 backdrop-blur-sm">
        3D VIEWPORT: LIVE_RENDER
      </div>
    </div>
  );
};
