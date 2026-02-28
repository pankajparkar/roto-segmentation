import { useQuery } from '@tanstack/react-query'
import { FolderOpen, Film, Clock, CheckCircle } from 'lucide-react'
import { projectsApi } from '@/lib/api'

export function Dashboard() {
  const { data: projects, isLoading } = useQuery({
    queryKey: ['projects'],
    queryFn: () => projectsApi.list().then((res) => res.data),
  })

  const stats = [
    { name: 'Total Projects', value: projects?.length || 0, icon: FolderOpen },
    { name: 'Active Shots', value: 0, icon: Film },
    { name: 'Processing', value: 0, icon: Clock },
    { name: 'Completed Today', value: 0, icon: CheckCircle },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="mt-1 text-slate-400">
          Overview of your AI rotoscoping projects
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div
            key={stat.name}
            className="rounded-lg bg-slate-800 p-6"
          >
            <div className="flex items-center gap-4">
              <div className="rounded-lg bg-slate-700 p-3">
                <stat.icon className="h-6 w-6 text-primary-400" />
              </div>
              <div>
                <p className="text-sm text-slate-400">{stat.name}</p>
                <p className="text-2xl font-semibold text-white">{stat.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Projects */}
      <div>
        <h2 className="mb-4 text-lg font-semibold text-white">Recent Projects</h2>
        {isLoading ? (
          <div className="text-slate-400">Loading...</div>
        ) : projects?.length === 0 ? (
          <div className="rounded-lg border border-dashed border-slate-700 p-8 text-center">
            <FolderOpen className="mx-auto h-12 w-12 text-slate-600" />
            <h3 className="mt-4 text-sm font-medium text-white">No projects</h3>
            <p className="mt-1 text-sm text-slate-400">
              Get started by creating a new project.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {projects?.slice(0, 6).map((project: { id: string; name: string; shot_count: number }) => (
              <div
                key={project.id}
                className="rounded-lg bg-slate-800 p-4 hover:bg-slate-750 transition-colors cursor-pointer"
              >
                <h3 className="font-medium text-white">{project.name}</h3>
                <p className="mt-1 text-sm text-slate-400">
                  {project.shot_count} shots
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
