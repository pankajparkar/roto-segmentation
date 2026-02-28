import { useParams } from 'react-router-dom'
import { Play, Pause, SkipBack, SkipForward, Download } from 'lucide-react'
import { useState } from 'react'

export function ShotViewer() {
  const { shotId } = useParams()
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(1001)

  // Placeholder - will be replaced with actual data
  const shot = {
    id: shotId,
    name: 'shot_001',
    frameStart: 1001,
    frameEnd: 1100,
    width: 1920,
    height: 1080,
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700 pb-4">
        <div>
          <h1 className="text-xl font-bold text-white">{shot.name}</h1>
          <p className="text-sm text-slate-400">
            {shot.width}x{shot.height} | Frames {shot.frameStart}-{shot.frameEnd}
          </p>
        </div>
        <div className="flex gap-2">
          <button className="flex items-center gap-2 rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600">
            <Download className="h-4 w-4" />
            Export
          </button>
        </div>
      </div>

      {/* Viewer */}
      <div className="flex flex-1 gap-4 pt-4">
        {/* Canvas area */}
        <div className="flex-1">
          <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black">
            {/* Placeholder for video/canvas */}
            <div className="flex h-full items-center justify-center text-slate-500">
              <p>Video viewer will be implemented here</p>
            </div>

            {/* Mask overlay will go here */}
          </div>

          {/* Playback controls */}
          <div className="mt-4 flex items-center justify-center gap-4">
            <button
              onClick={() => setCurrentFrame(shot.frameStart)}
              className="rounded-lg p-2 text-slate-400 hover:bg-slate-700 hover:text-white"
            >
              <SkipBack className="h-5 w-5" />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="rounded-lg bg-primary-600 p-3 text-white hover:bg-primary-700"
            >
              {isPlaying ? (
                <Pause className="h-5 w-5" />
              ) : (
                <Play className="h-5 w-5" />
              )}
            </button>
            <button
              onClick={() => setCurrentFrame(shot.frameEnd)}
              className="rounded-lg p-2 text-slate-400 hover:bg-slate-700 hover:text-white"
            >
              <SkipForward className="h-5 w-5" />
            </button>
          </div>

          {/* Timeline */}
          <div className="mt-4">
            <input
              type="range"
              min={shot.frameStart}
              max={shot.frameEnd}
              value={currentFrame}
              onChange={(e) => setCurrentFrame(Number(e.target.value))}
              className="w-full"
            />
            <div className="mt-1 flex justify-between text-sm text-slate-400">
              <span>Frame {currentFrame}</span>
              <span>
                {shot.frameStart} - {shot.frameEnd}
              </span>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="w-72 rounded-lg bg-slate-800 p-4">
          <h2 className="font-semibold text-white">Objects</h2>
          <p className="mt-2 text-sm text-slate-400">
            Click on the video to add segmentation points, or use the tools below.
          </p>

          {/* Segmentation tools will go here */}
          <div className="mt-4 space-y-2">
            <button className="w-full rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600">
              + Add Object (Click)
            </button>
            <button className="w-full rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600">
              + Add Object (Box)
            </button>
            <button className="w-full rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium text-white hover:bg-slate-600">
              + Add Object (Text)
            </button>
          </div>

          {/* Object list */}
          <div className="mt-6">
            <h3 className="text-sm font-medium text-slate-300">Detected Objects</h3>
            <div className="mt-2 text-sm text-slate-500">
              No objects segmented yet
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
