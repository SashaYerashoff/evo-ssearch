import { useEffect, useRef, useState, useCallback } from 'react'
import type { AuthUser, Channel, ArchiveFilters } from './api/types'
import { login as apiLogin, me as apiMe, logout as apiLogout } from './api/auth'
import { getChannels } from './api/detections'
import { api } from './api/client'
import { TopBar } from './components/shell/TopBar'
import { LeftRail, type SectionId } from './components/shell/LeftRail'
import { AgentEar } from './components/shell/AgentEar'
import { AgentPanel, type AgentAction } from './components/shell/AgentPanel'
import { ArchiveScreen, type ArchiveTool } from './components/archive/ArchiveScreen'
import { MonitoringScreen, type MonitorAction } from './components/monitoring/MonitoringScreen'
import { VideoScreen } from './components/video/VideoScreen'

export type AgentDrive = AgentAction & { seq: number }

export interface StatusData {
  luxriot: boolean
  channels: number
  probes: number
  agent: 'idle' | 'working'
}

function LoginGate({ onDone }: { onDone: (u: AuthUser) => void }) {
  const [u, setU] = useState('admin')
  const [p, setP] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true); setErr('')
    try { onDone(await apiLogin(u, p)) }
    catch (ex: any) { setErr(ex?.message || 'Sign in failed') }
    finally { setBusy(false) }
  }
  return (
    <div className="gate">
      <form className="gate-card" onSubmit={submit}>
        <h1>EVA AI</h1>
        <div className="brand-sub">Command console · sign in</div>
        <input placeholder="Username" value={u} onChange={(e) => setU(e.target.value)} autoFocus />
        <input placeholder="Password" type="password" value={p} onChange={(e) => setP(e.target.value)} />
        <div className="gate-err">{err}</div>
        <button className="btn primary" disabled={busy} style={{ justifyContent: 'center' }}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </div>
  )
}

export default function App() {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [ready, setReady] = useState(false)
  const [channels, setChannels] = useState<Channel[]>([])
  const [status, setStatus] = useState<StatusData>({ luxriot: false, channels: 0, probes: 0, agent: 'idle' })
  const [section, setSection] = useState<SectionId>('archive')
  const [archiveTool, setArchiveTool] = useState<ArchiveTool>(null)
  const [agentOpen, setAgentOpen] = useState(false)
  const [agentFull, setAgentFull] = useState(false)
  const [railPinned, setRailPinned] = useState(false)
  const [noAnim, setNoAnim] = useState(false)
  const [drive, setDrive] = useState<AgentDrive | null>(null)
  const [archiveFilters, setArchiveFilters] = useState<ArchiveFilters | null>(null)
  const [monitorCmd, setMonitorCmd] = useState<{ seq: number; action: MonitorAction } | null>(null)
  const seqRef = useRef(0)
  const monSeqRef = useRef(0)

  // agent → console mirroring: route each agent action to the working area
  const handleAgentAction = useCallback((a: AgentAction) => {
    setSection('archive')
    setDrive({ ...a, seq: ++seqRef.current })
  }, [])

  useEffect(() => { apiMe().then((u) => { setUser(u); setReady(true) }) }, [])

  useEffect(() => {
    if (!user) return
    getChannels()
      .then((ch) => { setChannels(ch); setStatus((s) => ({ ...s, luxriot: true, channels: ch.length })) })
      .catch(() => setStatus((s) => ({ ...s, luxriot: false })))
    api.get('/probes/list')
      .then((r) => setStatus((s) => ({ ...s, probes: (r?.probes || []).length })))
      .catch(() => {})
  }, [user])

  if (!ready) return <div className="loading-state"><div className="spinner" /></div>
  if (!user) return <LoginGate onDone={setUser} />

  async function handleLogout() { await apiLogout(); setUser(null) }

  return (
    <div className={`shell ${noAnim ? 'no-anim' : ''}`}>
      <TopBar user={user} status={status} noAnim={noAnim} onToggleNoAnim={() => setNoAnim((v) => !v)} />
      <div className={`body-row ${railPinned ? 'pinned' : ''} ${agentOpen ? 'agent-open' : ''}`}>
        <LeftRail
          active={section}
          pinned={railPinned}
          onNavigate={setSection}
          onArchiveTool={(t) => { setSection('archive'); setArchiveTool(t) }}
          onMonitorAction={(a) => { setSection('monitoring'); setMonitorCmd({ seq: ++monSeqRef.current, action: a }) }}
          onTogglePin={() => setRailPinned((v) => !v)}
          onLogout={handleLogout}
        />
        <div className="center">
          {section === 'archive' && (
            <ArchiveScreen channels={channels} tool={archiveTool} drive={drive} noAnim={noAnim} onFilters={setArchiveFilters} onToolHandled={() => setArchiveTool(null)} />
          )}
          {section === 'monitoring' && <MonitoringScreen channels={channels} cmd={monitorCmd} onCmdHandled={() => setMonitorCmd(null)} />}
          {section === 'video' && <VideoScreen channels={channels} />}
          {section !== 'archive' && section !== 'monitoring' && section !== 'video' && (
            <div className="empty-state">
              <div style={{ fontSize: 15, color: 'var(--text-2)' }}>{section[0].toUpperCase() + section.slice(1)}</div>
              <div>This section is not part of the prototype yet.</div>
            </div>
          )}
        </div>
        <AgentEar open={agentOpen} onToggle={() => setAgentOpen((v) => !v)} />
        <AgentPanel
          open={agentOpen}
          full={agentFull}
          onClose={() => setAgentOpen(false)}
          onToggleFull={() => setAgentFull((v) => !v)}
          channels={channels}
          archiveFilters={archiveFilters}
          onAction={handleAgentAction}
        />
      </div>
    </div>
  )
}
