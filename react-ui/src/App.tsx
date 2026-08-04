import { useEffect, useRef, useState, useCallback } from 'react'
import type { AuthUser, Channel, ArchiveFilters, Detection } from './api/types'
import { login as apiLogin, me as apiMe, logout as apiLogout } from './api/auth'
import {
  canOpenSettings,
  canViewSection,
  filterAllowedChannels,
  hasPermission,
  PERMISSION,
} from './api/access'
import { findParentAlert, getChannelsStatus, normalizeDetection } from './api/detections'
import { deriveLuxriotLinkStatus, type LuxriotLinkState } from './api/luxriotStatus'
import type { Probe } from './api/probes'
import { api, API_FORBIDDEN_EVENT, AUTH_EXPIRED_EVENT } from './api/client'
import { TopBar } from './components/shell/TopBar'
import { StatusConsole } from './components/shell/StatusConsole'
import { LeftRail, SECTION_LABEL_KEYS, type SectionId } from './components/shell/LeftRail'
import { AgentEar } from './components/shell/AgentEar'
import { AgentPanel, type AgentAction } from './components/shell/AgentPanel'
import { ArchiveScreen } from './components/archive/ArchiveScreen'
import { InspectorModal } from './components/archive/InspectorModal'
import { MonitoringScreen } from './components/monitoring/MonitoringScreen'
import { VideoScreen } from './components/video/VideoScreen'
import { videoApi, type SummaryEntry } from './api/video'
import { SettingsModal } from './components/settings/SettingsModal'
import { HomeScreen } from './components/home/HomeScreen'
import { NeuralBackground } from './components/shell/NeuralBackground'
import { AppearanceModal } from './components/appearance/AppearanceModal'
import { useAppearance } from './appearance/AppearanceProvider'
import type { ConsoleUiEffect } from './ui-effects/consoleEffects'
import { useI18n } from './i18n/I18nProvider'

export type AgentDrive = AgentAction & { seq: number }
export interface ConsoleDrive {
  effect: ConsoleUiEffect
  result: unknown
  seq: number
}

export interface StatusData {
  luxriot: LuxriotLinkState
  luxriotDetail: string
  channels: number
  probes: number
  agent: 'idle' | 'working'
}

function LoginGate({ onDone }: { onDone: (u: AuthUser) => void }) {
  const { t } = useI18n()
  const [u, setU] = useState('admin')
  const [p, setP] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true); setErr('')
    try { onDone(await apiLogin(u, p)) }
    catch (ex: any) { setErr(ex?.message || t('auth.failed')) }
    finally { setBusy(false) }
  }
  return (
    <div className="gate">
      <NeuralBackground />
      <form className="gate-card" onSubmit={submit}>
        <h1>EVA AI</h1>
        <div className="brand-sub">{t('auth.command')}</div>
        <input placeholder={t('auth.username')} value={u} onChange={(e) => setU(e.target.value)} autoFocus />
        <input placeholder={t('auth.password')} type="password" value={p} onChange={(e) => setP(e.target.value)} />
        <div className="gate-err">{err}</div>
        <button className="btn primary" disabled={busy} style={{ justifyContent: 'center' }}>
          {busy ? t('auth.signingIn') : t('auth.signIn')}
        </button>
      </form>
    </div>
  )
}

export default function App() {
  const { isMotionReduced } = useAppearance()
  const { t } = useI18n()
  const [user, setUser] = useState<AuthUser | null>(null)
  const [ready, setReady] = useState(false)
  const [channels, setChannels] = useState<Channel[]>([])
  const [status, setStatus] = useState<StatusData>({
    luxriot: 'checking',
    luxriotDetail: 'Waiting for Luxriot status.',
    channels: 0,
    probes: 0,
    agent: 'idle',
  })
  const [section, setSection] = useState<SectionId>('home')
  const [agentOpen, setAgentOpen] = useState(false)
  const [agentFull, setAgentFull] = useState(false)
  const [agentArchiveColumns, setAgentArchiveColumns] = useState(4)
  const [agentCommittedArchiveColumns, setAgentCommittedArchiveColumns] = useState(4)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [appearanceOpen, setAppearanceOpen] = useState(false)
  const [drive, setDrive] = useState<AgentDrive | null>(null)
  const [summaryReview, setSummaryReview] = useState<Detection | null>(null)
  const [similarDrive, setSimilarDrive] = useState<{ detection: Detection; seq: number } | null>(null)
  const [probeDrive, setProbeDrive] = useState<ConsoleDrive | null>(null)
  const [videoDrive, setVideoDrive] = useState<ConsoleDrive | null>(null)
  const [archiveFilters, setArchiveFilters] = useState<ArchiveFilters | null>(null)
  const [forbiddenNotice, setForbiddenNotice] = useState('')
  const [appVersion, setAppVersion] = useState('')
  const [serverStartedAtMs, setServerStartedAtMs] = useState<number | null>(null)
  const seqRef = useRef(0)
  const summaryReviewOriginRef = useRef<HTMLElement | null>(null)
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
  const handleReviewVideoSummary = useCallback((entry: SummaryEntry) => {
    const detectionId = Number(entry.thumbnail_detection_id)
    const channelId = Number(entry.channel_id)
    if (!Number.isInteger(detectionId) || detectionId <= 0 || !Number.isInteger(channelId) || channelId <= 0) return
    const toMs = (value: unknown): number => {
      const number = Number(value)
      if (!Number.isFinite(number) || number <= 0) return 0
      return number > 1e12 ? number : number * 1000
    }
    const batchStartMs = toMs(entry.batch_start_ms ?? entry.window_start)
    const batchEndMs = toMs(entry.batch_end_ms ?? entry.window_end ?? entry.created_at)
    const timestampMs = batchEndMs || batchStartMs || Date.now()
    const raw = {
      id: detectionId,
      detection_id: detectionId,
      source: 'vlm_summary',
      source_label: 'Video description',
      channel_id: channelId,
      timestamp_ms: timestampMs,
      summary: String(entry.summary || ''),
      image_url: `/detections/thumbnail/${detectionId}`,
      payload: {
        source: 'vlm_summary',
        batch_id: String(entry.batch_id || ''),
        run_id: String(entry.run_id || ''),
        batch_start_ms: batchStartMs || undefined,
        batch_end_ms: batchEndMs || undefined,
        summary: String(entry.summary || ''),
        anchor_role: String(entry.thumbnail_role || 'sample'),
        snapshot_index: entry.thumbnail_snapshot_index,
        is_cover: entry.thumbnail_is_cover === true,
        cover_kind: String(entry.cover_kind || ''),
        cover_reason: String(entry.cover_reason || ''),
        cover_confidence: String(entry.cover_confidence || ''),
        selection_source: String(entry.thumbnail_selection_source || ''),
      },
    }
    summaryReviewOriginRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    const channelMap = new Map(channels.map((channel) => [channel.id, channel.title]))
    setSummaryReview(normalizeDetection(raw, channelMap))
  }, [channels])

  const closeSummaryReview = useCallback(() => {
    setSummaryReview(null)
    window.requestAnimationFrame(() => summaryReviewOriginRef.current?.focus({ preventScroll: true }))
  }, [])

  const findSimilarFromSummary = useCallback((detection: Detection) => {
    setSummaryReview(null)
    setSection('archive')
    setSimilarDrive({ detection, seq: ++seqRef.current })
  }, [])

  const refreshChannels = useCallback(async () => {
    if (!user || !hasPermission(user, PERMISSION.streamsView)) {
      setChannels([])
      setStatus((s) => ({
        ...s,
        luxriot: 'checking',
        luxriotDetail: 'Luxriot status is unavailable for the current role.',
        channels: 0,
      }))
      return
    }
    try {
      const [channelStatus, streams] = await Promise.all([
        getChannelsStatus(true),
        videoApi.streams().catch(() => null),
      ])
      const allowed = filterAllowedChannels(user, channelStatus.channels)
      const link = deriveLuxriotLinkStatus(
        channelStatus.channels,
        channelStatus.inventory,
        streams,
      )
      setChannels(allowed)
      setStatus((s) => ({
        ...s,
        luxriot: link.state,
        luxriotDetail: link.detail,
        channels: allowed.length,
      }))
    } catch (exception: any) {
      setChannels([])
      setStatus((s) => ({
        ...s,
        luxriot: 'offline',
        luxriotDetail: exception?.message || 'EVA could not query Luxriot.',
        channels: 0,
      }))
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
      setStatus({
        luxriot: 'checking',
        luxriotDetail: 'Waiting for Luxriot status.',
        channels: 0,
        probes: 0,
        agent: 'idle',
      })
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
        appVersion={appVersion}
        section={t(SECTION_LABEL_KEYS[section])}
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
              similarDrive={similarDrive}
              onSimilarDriveHandled={() => setSimilarDrive(null)}
              noAnim={noAnim}
              canReportFeedback={hasPermission(user, PERMISSION.bookmarksCreate)}
              canReportIncidents={hasPermission(user, PERMISSION.incidentsManage)}
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
              reviewOverlayOpen={!!summaryReview}
              onReloadChannels={refreshChannels}
              canCapture={hasPermission(user, PERMISSION.captureManage)}
              canManagePrompts={hasPermission(user, PERMISSION.promptsManage)}
              canCreateBookmarks={hasPermission(user, PERMISSION.bookmarksCreate)}
              canReportIncidents={hasPermission(user, PERMISSION.incidentsManage)}
              canExport={hasPermission(user, PERMISSION.dataExport)}
              onReviewSummary={handleReviewVideoSummary}
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
        {summaryReview && (
          <InspectorModal
            d={summaryReview}
            channels={channels}
            canReportFeedback={hasPermission(user, PERMISSION.bookmarksCreate)}
            canReportIncidents={hasPermission(user, PERMISSION.incidentsManage)}
            canExport={hasPermission(user, PERMISSION.dataExport)}
            onClose={closeSummaryReview}
            onFindSimilar={findSimilarFromSummary}
          />
        )}
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
              canManageSkills={hasPermission(user, PERMISSION.promptsManage)}
            />
          </>
        )}
      </div>
      <StatusConsole user={user} status={status} />
    </div>
  )
}
