import { YouTubeProcessor } from "@/components/youtube/youtube-processor"
import { Metadata } from "next"

export const metadata: Metadata = {
  title: "YouTube Analysis | SignVerse AI",
  description: "Advanced computer vision pipeline for YouTube video analysis and behavioral inference.",
}

export default function YouTubePage() {
  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-10">
        <h1 className="text-4xl font-bold text-white tracking-tight leading-none mb-4">YouTube Analytics</h1>
        <p className="text-slate-400 max-w-2xl leading-relaxed font-medium">
          Leverage our complete computer vision pipeline to process and analyze YouTube videos. 
          Extract poses, detect interactions, and infer intentions directly from any URL.
        </p>
      </div>

      <YouTubeProcessor />
    </div>
  )
}
