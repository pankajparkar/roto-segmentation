import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Projects } from './pages/Projects'
import { ShotViewer } from './pages/ShotViewer'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="projects" element={<Projects />} />
          <Route path="projects/:projectId" element={<Projects />} />
          <Route path="shots/:shotId" element={<ShotViewer />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
