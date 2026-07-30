import { useEffect, useRef, useState, useCallback } from 'react'
import type { AuthUser, Channel, ArchiveFilters } from './api/types'
import { login as apiLogin, me as apiMe, logout as apiLogout } from './api/auth'
import {
  canOpenSettings,
  canViewSection,
  filterAllowedChannels,
  hasPermission,
  PERMISSION,
} from './api/access'
import { findParentAlert, getChannels } from './api/detections'
import type { Probe } from './api/probes'
import { api, API_FORBIDDEN_EVENT, AUTH_EXPIRED_EVENT } from './api/client'
import { TopBar } from './components/shell/TopBar'
import { LeftRail, SECTION_LABELS, type SectionId } from './components/shell/LeftRail'
import { AgentEar } from './components/shell/AgentEar'
import { AgentPanel, type AgentAction } from './components/shell/AgentPanel'
import { ArchiveScreen } from './components/archive/ArchiveScreen'
import { MonitoringScreen } from './components/monitoring/MonitoringScreen'
import { VideoScreen } from './components/video/VideoScreen'
import { SettingsModal } from './components/settings/SettingsModal'
import { HomeScreen } from './components/home/HomeScreen'
import { NeuralBackground } from './components/shell/NeuralBackground'
import { AppearanceModal } from './components/appearance/AppearanceModal'
import { useAppearance } from './appearance/AppearanceProvider'
import type { ConsoleUiEffect } from './ui-effects/consoleEffects'

export type AgentDrive = AgentAction & { seq: number }
export interface ConsoleDrive {
  effect: ConsoleUiEffect
  result: unknown
  seq: number
}

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
      <NeuralBackground />
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
  const {
    savedPreferences,
    isMotionReduced,
    commitPreferences,
  } = useAppearance()
  const [user, setUser] = useState<AuthUser | null>(null)
  const [ready, setReady] = useState(false)
  const [channels, setChannels] = useState<Channel[]>([])
  const [status, setStatus] = useState<StatusData>({ luxriot: false, channels: 0, probes: 0, agent: 'idle' })
  const [section, setSection] = useState<SectionId>('home')
  const [agentOpen, setAgentOpen] = useState(false)
  const [agentFull, setAgentFull] = useState(false)
  const [agentArchiveColumns, setAgentArchiveColumns] = useState(4)
  const [agentCommittedArchiveColumns, setAgentCommittedArchiveColumns] = useState(4)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [appearanceOpen, setAppearanceOpen] = useState(false)
  const [drive, setDrive] = useState<AgentDrive | null>(null)
  const [probeDrive, setProbeDrive] = useState<ConsoleDrive | null>(null)
  const [videoDrive, setVideoDrive] = useState<ConsoleDrive | null>(null)
  const [archiveFilters, setArchiveFilters] = useState<ArchiveFilters | null>(null)
  const [forbiddenNotice, setForbiddenNotice] = useState('')
  const [appVersion, setAppVersion] = useState('')
  const [serverStartedAtMs, setServerStartedAtMs] = useState<number | null>(null)
  const seqRef = useRef(0)
  const appliedEffectIds = useRef(new Set<string>())
  const visibleSections = (['home', 'archive', 'video', 'monitoring'] as SectionId[])
    .filter((candidate) => canViewSection(user, candidate))
  const settingsAllowed = canOpenSettings(user)

  // Trusted backend effects route completed domain reads/receipts into the console.
  const handleAgentUiEffects = useCallback((effects: ConsoleUiEffect[], result: unknown) => {
    for (const effect of effects) {
      if (appliedEffectIds.current.has(effect.effectId)) continue
      if (appliedEffectIds.current.size >= 512) appliedEffectIds.current.clear()
      appliedEffectIds.current.add(effect.effectId)
      const seq = ++seqRef.current
      if (effect.target === 'archive' && canViewSection(user, 'archive')) {
        setSection('archive')
        setDrive({
          name: effect.source.tool,
          args: effect.payload,
          done: true,
          result,
          seq,
        })
      } else if (effect.target === 'probes' && canViewSection(user, 'monitoring')) {
        setSection('monitoring')
        setProbeDrive({ effect, result, seq })
      } else if (effect.target === 'video' && canViewSection(user, 'video')) {
        setSection('video')
        setVideoDrive({ effect, result, seq })
      }
    }
  }, [user])
  const handleAgentBusy = useCallback((busy: boolean) => {
    const agent = busy ? 'working' : 'idle'
    setStatus((current) => current.agent === agent ? current : { ...current, agent })
  }, [])
  const handleOpenParentAlert = useCallback(async (probe: Probe) => {
    const parentAlertId = String(probe.parent_alert_id || '').trim()
    const channelId = Number(probe.channel_id)
    if (!parentAlertId || !Number.isInteger(channelId)) return
    try {
      const detection = await findParentAlert(
        parentAlertId,
        channelId,
        channels,
        probe.parent_alert_timestamp_ms,
      )
      if (!detection) {
        setForbiddenNotice('The parent VLM alert is no longer available in the archive.')
        window.setTimeout(() => setForbiddenNotice(''), 5_000)
        return
      }
      const timestamp = Number(probe.parent_alert_timestamp_ms || detection.tsMs || Date.now())
      setSection('archive')
      setDrive({
        name: 'get_detections',
        args: {
          channel_id: channelId,
          source: 'vlm_alert',
          since_ms: timestamp - 15 * 60_000,
          until_ms: timestamp + 15 * 60_000,
          open_detection_id: detection.id,
        },
        done: true,
        result: { detections: [detection.raw] },
        seq: ++seqRef.current,
      })
    } catch (exception: any) {
      setForbiddenNotice(exception?.message || 'Could not open the parent VLM alert.')
      window.setTimeout(() => setForbiddenNotice(''), 5_000)
    }
  }, [channels])

  const refreshChannels = useCallback(async () => {
    if (!user || !hasPermission(user, PERMISSION.streamsView)) {
      setChannels([])
      setStatus((s) => ({ ...s, luxriot: false, channels: 0 }))
      return
    }
    try {
      const ch = await getChannels()
      const allowed = filterAllowedChannels(user, ch)
      setChannels(allowed)
      setStatus((s) => ({ ...s, luxriot: true, channels: allowed.length }))
    } catch {
      setChannels([])
      setStatus((s) => ({ ...s, luxriot: false }))
    }
  }, [user])

  const refreshHealth = useCallback(async () => {
    try {
      const health = await api.get('/health')
      setAppVersion(String(health?.version || ''))
      const uptimeSec = Number(health?.uptime_sec)
      if (Number.isFinite(uptimeSec) && uptimeSec >= 0) {
        setServerStartedAtMs(Date.now() - uptimeSec * 1000)
      }
    } catch {
      // Keep the last known server epoch while the backend is temporarily unavailable.
    }
  }, [])

  useEffect(() => {
    const expired = () => {
      setUser(null)
      setChannels([])
      setSection('home')
      setSettingsOpen(false)
      setAgentOpen(false)
      setStatus({ luxriot: false, channels: 0, probes: 0, agent: 'idle' })
    }
    const forbidden = (event: Event) => {
      const detail = (event as CustomEvent)?.detail
      const payload = detail?.payload
      const message = payload && typeof payload === 'object' && typeof payload.error === 'string'
        ? payload.error
        : 'You do not have permission for that action.'
      setForbiddenNotice(message)
      window.setTimeout(() => setForbiddenNotice(''), 5000)
    }
    window.addEventListener(AUTH_EXPIRED_EVENT, expired)
    window.addEventListener(API_FORBIDDEN_EVENT, forbidden)
    return () => {
      window.removeEventListener(AUTH_EXPIRED_EVENT, expired)
      window.removeEventListener(API_FORBIDDEN_EVENT, forbidden)
    }
  }, [])
  useEffect(() => { apiMe().then((u) => { setUser(u); setReady(true) }) }, [])
  useEffect(() => {
    refreshHealth()
    const timer = window.setInterval(refreshHealth, 30_000)
    return () => window.clearInterval(timer)
  }, [refreshHealth])

  useEffect(() => {
    if (!user) return
    refreshChannels()
    if (hasPermission(user, PERMISSION.reportsView)) {
      api.get('/probes/list')
        .then((r) => setStatus((s) => ({ ...s, probes: (r?.probes || []).length })))
        .catch(() => {})
    } else {
      setStatus((s) => ({ ...s, probes: 0 }))
    }
  }, [user, refreshChannels])
  useEffect(() => {
    if (!user || !hasPermission(user, PERMISSION.streamsView)) return
    const timer = window.setInterval(refreshChannels, 30_000)
    return () => window.clearInterval(timer)
  }, [user, refreshChannels])

  useEffect(() => {
    if (!user) return
    if (!canViewSection(user, section)) setSection('home')
    if (!settingsAllowed) setSettingsOpen(false)
    if (!hasPermission(user, PERMISSION.agentUse)) setAgentOpen(false)
  }, [user, section, settingsAllowed])

  if (!ready) return <div className="loading-state"><div className="spinner" /></div>
  if (!user) return <LoginGate onDone={setUser} />

  async function handleLogout() { await apiLogout(); setUser(null) }
  const agentPresetGrid = agentOpen && !agentFull
  const noAnim = isMotionReduced

  return (
    <div className={`shell ${noAnim ? 'no-anim' : ''}`}>
      <NeuralBackground noAnim={noAnim} />
      <TopBar
        user={user}
        status={status}
        appVersion={appVersion}
        section={SECTION_LABELS[section]}
        noAnim={noAnim}
        canBenchmark={hasPermission(user, PERMISSION.diagnosticsView)}
        onToggleNoAnim={() => commitPreferences({
          ...savedPreferences,
          motion: noAnim ? 'full' : 'reduced',
        })}
        onAppearance={() => setAppearanceOpen(true)}
        onBrand={() => setSection('home')}
      />
      {forbiddenNotice && <div className="global-notice" role="alert">{forbiddenNotice}</div>}
      <div
        className={`body-row ${agentOpen ? 'agent-open' : ''} ${agentPresetGrid ? 'agent-preset-grid' : ''}`}
        style={agentPresetGrid
          ? ({
              '--archive-grid-columns': agentArchiveColumns,
              '--archive-results-columns': agentCommittedArchiveColumns,
            } as React.CSSProperties)
          : undefined}
      >
        <LeftRail
          active={section}
          visibleSections={visibleSections}
          showSettings={settingsAllowed}
          onNavigate={setSection}
          onSettings={() => setSettingsOpen(true)}
          onLogout={handleLogout}
        />
        <div className="center">
          <HomeScreen active={section === 'home'} serverStartedAtMs={serverStartedAtMs} />
          {section === 'archive' && (
            <ArchiveScreen
              channels={channels}
              drive={drive}
              noAnim={noAnim}
              canReportFeedback={hasPermission(user, PERMISSION.bookmarksCreate)}
              canExport={hasPermission(user, PERMISSION.dataExport)}
              onFilters={setArchiveFilters}
              onRefreshChannels={refreshChannels}
            />
          )}
          {section === 'monitoring' && (
            <MonitoringScreen
              channels={channels}
              drive={probeDrive}
              canOperate={hasPermission(user, PERMISSION.probesRun) && hasPermission(user, PERMISSION.captureManage)}
              canManage={hasPermission(user, PERMISSION.probesManage)}
              canCreateBookmarks={hasPermission(user, PERMISSION.bookmarksCreate)}
              onOpenParentAlert={handleOpenParentAlert}
            />
          )}
          {section === 'video' && (
            <VideoScreen
              channels={channels}
              drive={videoDrive}
              onReloadChannels={refreshChannels}
              canCapture={hasPermission(user, PERMISSION.captureManage)}
              canManagePrompts={hasPermission(user, PERMISSION.promptsManage)}
              canCreateBookmarks={hasPermission(user, PERMISSION.bookmarksCreate)}
              canExport={hasPermission(user, PERMISSION.dataExport)}
            />
          )}
          {section !== 'home' && section !== 'archive' && section !== 'monitoring' && section !== 'video' && (
            <div className="empty-state">
              <div style={{ fontSize: 15, color: 'var(--text-2)' }}>{section[0].toUpperCase() + section.slice(1)}</div>
              <div>This section is not part of the prototype yet.</div>
            </div>
          )}
        </div>
        {settingsOpen && (
          <SettingsModal
            user={user}
            channels={channels}
            onRefreshChannels={refreshChannels}
            onClose={() => setSettingsOpen(false)}
          />
        )}
        {appearanceOpen && <AppearanceModal onClose={() => setAppearanceOpen(false)} />}
        {hasPermission(user, PERMISSION.agentUse) && (
          <>
            <AgentEar open={agentOpen} onToggle={() => setAgentOpen((v) => !v)} />
            <AgentPanel
              open={agentOpen}
              full={agentFull}
              onClose={() => setAgentOpen(false)}
              onToggleFull={() => setAgentFull((v) => !v)}
              section={section}
              channels={channels}
              archiveFilters={archiveFilters}
              onUiEffects={handleAgentUiEffects}
              onBusyChange={handleAgentBusy}
              onLayoutPresetChange={setAgentArchiveColumns}
              onLayoutPresetCommit={setAgentCommittedArchiveColumns}
              canManageModels={hasPermission(user, PERMISSION.modelsManage)}
              canManageSkills={hasPermission(user, PERMISSION.promptsManage)}
            />
          </>
        )}
      </div>
    </div>
  )
}
