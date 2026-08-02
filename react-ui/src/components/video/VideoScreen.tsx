import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  IconAlertTriangle,
  IconBookmark,
  IconChevronDown,
  IconChevronRight,
  IconCopy,
  IconDownload,
  IconFileDescription,
  IconSwitchHorizontal,
  IconVideoOff,
  IconX,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import type { ConsoleDrive } from '../../App'
import type { IncidentDraftInput } from '../../api/incidents'
import {
  attentionStreamUrl,
  buildIncidentDraftFromSummary,
  buildSummaryBookmarkInput,
  buildCaptureInput,
  fullLiveMediaUrl,
  mergeRuntime,
  recentFrameUrl,
  videoApi,
  type ChannelRuntime,
  type StreamsStatus,
  type SummaryEntry,
} from '../../api/video'
import type { DropOption } from '../shell/Dropdown'
import { renderMarkdown } from '../agent/markdown'
import { StreamControl, type VideoWorkspaceTab } from './StreamControl'
import { PromptSettingsModal } from './PromptSettingsModal'
import { IncidentModal } from '../incidents/IncidentModal'
import {
  SUMMARY_SEVERITIES,
  resolveSummaryResolution,
  splitSummaryMachineJson,
  summaryAlertCounts,
  summaryBurst,
  summaryEntryKey,
  summaryLevel,
  summaryPeriodBounds,
  summarySemanticStatus,
  type SummaryPeriod,
  type SummaryPeriodBounds,
  type SummaryResolution,
} from './summaryView'

function asTimestampMs(value: unknown): number | null {
  const number = Number(value)
  if (!Number.isFinite(number) || number <= 0) return null
  return number > 1e12 ? number : number * 1000
}

function fmtTimestamp(value: unknown): string {
  const ms = asTimestampMs(value)
  if (!ms) return '—'
  return new Date(ms).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function fmtAge(timestampMs: number): string {
  if (!timestampMs) return 'never'
  const seconds = Math.max(0, Math.round((Date.now() - timestampMs) / 1000))
  if (seconds < 2) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.floor(seconds / 60)}m ago`
}

function formatDatetimeLocal(timestampSec: number): string {
  const date = new Date(timestampSec * 1000)
  if (!Number.isFinite(date.getTime())) return ''
  const pad = (value: number) => String(value).padStart(2, '0')
  return [
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`,
    `${pad(date.getHours())}:${pad(date.getMinutes())}`,
  ].join('T')
}

function parseDatetimeLocal(value: string): number | undefined {
  const timestampMs = new Date(String(value || '')).getTime()
  return Number.isFinite(timestampMs) ? timestampMs / 1000 : undefined
}

function defaultCustomRange(): { inputs: { from: string; to: string }; bounds: SummaryPeriodBounds } {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).getTime() / 1000
  const bounds = { from_ts: yesterday, to_ts: today - 0.001 }
  return {
    inputs: { from: formatDatetimeLocal(yesterday), to: formatDatetimeLocal(today) },
    bounds,
  }
}

const REVIEW_CHANNEL_STORAGE_KEY = 'eva.video.reviewChannelId'
const SETTINGS_CHANNEL_STORAGE_KEY = 'eva.video.settingsChannelId'

function initialChannelId(channels: Channel[], storageKey: string): number | null {
  try {
    const stored = Number(window.localStorage.getItem(storageKey))
    if (Number.isInteger(stored) && channels.some((channel) => channel.id === stored)) return stored
  } catch {
    // Browser storage is an optional convenience; channel selection still works without it.
  }
  return channels[0]?.id ?? null
}

function summaryRange(entry: SummaryEntry): string {
  const start = asTimestampMs(entry.batch_start_ms ?? entry.window_start)
  const end = asTimestampMs(entry.batch_end_ms ?? entry.window_end)
  if (!start || !end || end <= start) return fmtTimestamp(entry.created_at ?? entry.window_end)
  const startDate = new Date(start)
  const endDate = new Date(end)
  const sameDay = startDate.toDateString() === endDate.toDateString()
  const startLabel = startDate.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
  const endLabel = endDate.toLocaleString([], sameDay
    ? { hour: '2-digit', minute: '2-digit' }
    : { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  return `${startLabel}–${endLabel}`
}

function copySummary(entry: SummaryEntry): void {
  void navigator.clipboard.writeText(String(entry.summary || '')).catch(() => {})
}

function exportSummary(entry: SummaryEntry, level: string): void {
  const text = String(entry.summary || '').trim()
  if (!text) return
  const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `eva-${level.toLowerCase()}-${entry.channel_id ?? 'channel'}-${Date.now()}.md`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function SummaryCard({
  entry,
  selectedDepth,
  collapsed,
  canCreateBookmarks,
  canReportIncidents,
  canExport,
  onToggle,
  onImage,
  onReportIncident,
}: {
  entry: SummaryEntry
  selectedDepth: string
  collapsed: boolean
  canCreateBookmarks: boolean
  canReportIncidents: boolean
  canExport: boolean
  onToggle: () => void
  onImage: (src: string, title: string) => void
  onReportIncident: (input: IncidentDraftInput) => void
}) {
  const [bookmarkState, setBookmarkState] = useState<'idle' | 'saving' | 'saved' | 'failed'>('idle')
  const level = summaryLevel(entry, selectedDepth)
  const alerts = summaryAlertCounts(entry)
  const burst = summaryBurst(entry)
  const semantic = summarySemanticStatus(entry)
  const parts = splitSummaryMachineJson(entry.summary)
  const thumbnailId = Number(entry.thumbnail_detection_id)
  const thumbnailSrc = Number.isInteger(thumbnailId) && thumbnailId > 0
    ? `/detections/thumbnail/${thumbnailId}`
    : ''
  const thumbnailRole = String(entry.thumbnail_role || 'sample').replace(/_/g, ' ')
  const coalesced = Number(entry.coalesced?.batches || 0)
  const itemCount = Number(entry.item_count || 0)
  const frameCount = Number(entry.frame_count || 0)
  const runCount = Array.isArray(entry.run_ids) ? entry.run_ids.length : 0
  const sourceTokens = Number(entry.source_tokens || 0)
  const contentStats = level === 'L0'
    ? [
        frameCount > 0 ? `${frameCount} frames` : '',
        entry.model ? String(entry.model) : '',
      ].filter(Boolean)
    : [
        itemCount > 0 ? `${itemCount} items` : '',
        frameCount > 0 ? `${frameCount} frames` : '',
        runCount > 0 ? `${runCount} runs` : '',
        sourceTokens > 0 ? `${sourceTokens} tok` : '',
      ].filter(Boolean)
  const bookmarkInput = level === 'L0' && !entry.coverage_gap
    ? buildSummaryBookmarkInput(entry)
    : null
  const incidentInput = !entry.coverage_gap ? buildIncidentDraftFromSummary(entry) : null

  async function bookmarkSummary() {
    if (!bookmarkInput) return
    setBookmarkState('saving')
    try {
      await videoApi.createBookmark(bookmarkInput)
      setBookmarkState('saved')
    } catch {
      setBookmarkState('failed')
    }
  }

  return (
    <article className={`vid-sum ${collapsed ? 'collapsed' : ''} ${entry.coverage_gap ? 'coverage-gap' : ''}`}>
      <div className="vid-sum-head">
        <button className="vid-sum-toggle" onClick={onToggle} aria-expanded={!collapsed}>
          {collapsed ? <IconChevronRight size={15} /> : <IconChevronDown size={15} />}
          <span className="vid-level">{level}</span>
          {semantic && (
            <span className={`vid-semantic ${semantic.tone}`} title={semantic.title}>{semantic.label}</span>
          )}
          <span className="vid-channel-pill">#{entry.channel_id ?? '?'}</span>
          <span className="vid-sum-ts">{summaryRange(entry)}</span>
          {contentStats.length > 0 && <span className="vid-sum-stats">{contentStats.join(' · ')}</span>}
          {coalesced > 1 && <span className="vid-meta-chip">coalesced ×{coalesced}</span>}
          {burst && (
            <span
              className="vid-meta-chip burst"
              title={`Motion above the measured channel norm${burst.snapshots.length ? ` · snapshots ${burst.snapshots.join(', ')}` : ''}`}
            >
              ⚡ burst ×{burst.count}{burst.maxActivity != null ? ` · ${burst.maxActivity.toFixed(1)}×` : ''}
            </span>
          )}
          {Number(entry.state_transition_total || 0) > 0 && (
            <span className="vid-meta-chip transition">{entry.state_transition_total} transitions</span>
          )}
          {entry.coverage_gap && <span className="vid-meta-chip gap">coverage gap</span>}
          {SUMMARY_SEVERITIES.filter((severity) => Number(alerts[severity] || 0) > 0).map((severity) => (
            <span key={severity} className={`vid-sev sev-${severity}`}>
              {severity} <strong>{alerts[severity]}</strong>
            </span>
          ))}
        </button>
        <div className="vid-sum-actions">
          <button className="btn compact" onClick={onToggle} aria-expanded={!collapsed}>
            {collapsed ? <IconChevronDown size={13} /> : <IconChevronRight size={13} />}
            {collapsed ? 'Expand' : 'Collapse'}
          </button>
          <button className="btn compact" onClick={() => copySummary(entry)} disabled={!entry.summary}>
            <IconCopy size={13} /> Copy
          </button>
          {canExport && (
            <button className="btn compact" onClick={() => exportSummary(entry, level)} disabled={!entry.summary}>
              <IconDownload size={13} /> Export
            </button>
          )}
          {canCreateBookmarks && level === 'L0' && !entry.coverage_gap && (
            <button
              className={`btn compact ${bookmarkState === 'saved' ? 'success' : ''}`}
              onClick={bookmarkSummary}
              disabled={!bookmarkInput || bookmarkState === 'saving'}
              title={bookmarkState === 'failed' ? 'Bookmark failed; click to retry' : 'Send this L0 event to Luxriot'}
            >
              <IconBookmark size={13} />
              {bookmarkState === 'saving'
                ? 'Saving…'
                : bookmarkState === 'saved'
                  ? 'Bookmarked'
                  : bookmarkState === 'failed'
                    ? 'Retry bookmark'
                    : 'Bookmark'}
            </button>
          )}
          {canReportIncidents && incidentInput && (
            <button className="btn compact" onClick={() => onReportIncident(incidentInput)}>
              <IconFileDescription size={13} /> Report incident
            </button>
          )}
        </div>
      </div>

      {!collapsed && (
        <div className={`vid-sum-content ${thumbnailSrc ? 'has-thumbnail' : ''}`}>
          {thumbnailSrc && (
            <button
              className="vid-sum-thumbnail"
              onClick={() => onImage(thumbnailSrc, `${level} · ${summaryRange(entry)}`)}
              title={String(entry.cover_reason || 'Open the representative VLM input')}
            >
              <img src={thumbnailSrc} alt="Representative VLM input" loading="lazy" />
              <span>
                VLM input · {thumbnailRole}
                {entry.thumbnail_snapshot_index != null ? ` · snapshot ${entry.thumbnail_snapshot_index}` : ''}
              </span>
            </button>
          )}
          <div className="vid-sum-copy">
            {entry.coverage_gap ? (
              <div className="vid-gap-copy">
                <IconAlertTriangle size={15} />
                No description exists for this window: {String(entry.gap_reason || 'dropped batch').replace(/_/g, ' ')}.
              </div>
            ) : (
              <>
                {parts.narrative
                  ? <div className="vid-sum-body md" dangerouslySetInnerHTML={{ __html: renderMarkdown(parts.narrative) }} />
                  : !parts.machineJson && <div className="vid-sum-body empty">No operator-facing narrative was returned.</div>}
                {parts.machineJson && (
                  <details className="vid-machine-state">
                    <summary>{parts.marker || 'BATCH_STATE_JSON'} · machine state</summary>
                    <pre>{parts.machineJson}</pre>
                  </details>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </article>
  )
}

export function VideoScreen({
  channels,
  drive,
  reviewOverlayOpen = false,
  onReloadChannels,
  canCapture,
  canManagePrompts,
  canCreateBookmarks,
  canReportIncidents,
  canExport,
  onReviewSummary,
}: {
  channels: Channel[]
  drive?: ConsoleDrive | null
  reviewOverlayOpen?: boolean
  onReloadChannels?: () => Promise<void> | void
  canCapture: boolean
  canManagePrompts: boolean
  canCreateBookmarks: boolean
  canReportIncidents: boolean
  canExport: boolean
  onReviewSummary?: (entry: SummaryEntry) => void
}) {
  const [activeTab, setActiveTab] = useState<VideoWorkspaceTab>('review')
  const [reviewChannelId, setReviewChannelId] = useState<number | null>(() => initialChannelId(channels, REVIEW_CHANNEL_STORAGE_KEY))
  const [settingsChannelId, setSettingsChannelId] = useState<number | null>(() => initialChannelId(channels, SETTINGS_CHANNEL_STORAGE_KEY))
  const [streams, setStreams] = useState<StreamsStatus>({})
  const [feed, setFeed] = useState<SummaryEntry[]>([])
  const [batch, setBatch] = useState('12')
  const [every, setEvery] = useState('5')
  const [model, setModel] = useState('auto')
  const [period, setPeriod] = useState<SummaryPeriod>('live')
  const [resolution, setResolution] = useState<SummaryResolution>('AUTO')
  const [customFrom, setCustomFrom] = useState('')
  const [customTo, setCustomTo] = useState('')
  const [customBounds, setCustomBounds] = useState<SummaryPeriodBounds>({})
  const [renderedDepth, setRenderedDepth] = useState<'L0' | 'L1' | 'L2' | 'L3'>('L0')
  const [live, setLive] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previewBust, setPreviewBust] = useState(1)
  const [previewError, setPreviewError] = useState(true)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewMode, setPreviewMode] = useState<'model' | 'live'>('model')
  const [previewReadyAt, setPreviewReadyAt] = useState(0)
  const [attentionMediaSrc, setAttentionMediaSrc] = useState<string | null>(null)
  const [fullLiveMedia, setFullLiveMedia] = useState<{
    kind: 'video' | 'mjpeg'
    src: string
  } | null>(null)
  const [promptOpen, setPromptOpen] = useState(false)
  const [reviewPreviewOpen, setReviewPreviewOpen] = useState(false)
  const [settingsDirty, setSettingsDirty] = useState(false)
  const [pendingSettingsSwitch, setPendingSettingsSwitch] = useState<{ channelId: number; openSettings: boolean } | null>(null)
  const [modelOptions, setModelOptions] = useState<DropOption[]>([{ value: 'auto', label: 'Auto (balance)' }])
  const [collapsedSummaries, setCollapsedSummaries] = useState<Set<string>>(new Set())
  const [summaryImage, setSummaryImage] = useState<{ src: string; title: string } | null>(null)
  const [incidentDraft, setIncidentDraft] = useState<IncidentDraftInput | null>(null)
  const feedRef = useRef<HTMLDivElement>(null)
  const reviewOverlayScrollRef = useRef<number | null>(null)
  const feedRequestRef = useRef(0)
  const modelPreviewRef = useRef<HTMLImageElement>(null)
  const livePreviewImageRef = useRef<HTMLImageElement>(null)
  const livePreviewVideoRef = useRef<HTMLVideoElement>(null)
  const hydratedSettingsChannelRef = useRef<number | null>(null)

  const releasePreviewMedia = useCallback(() => {
    const modelImage = modelPreviewRef.current
    const liveImage = livePreviewImageRef.current
    const liveVideo = livePreviewVideoRef.current
    if (modelImage) modelImage.removeAttribute('src')
    if (liveImage) liveImage.removeAttribute('src')
    if (liveVideo) {
      liveVideo.pause()
      liveVideo.removeAttribute('src')
      liveVideo.load()
    }
  }, [])

  const channelName = useCallback((id: number) => channels.find((c) => c.id === id)?.title, [channels])
  const runtime: ChannelRuntime[] = mergeRuntime(streams, channelName)
  const settingsRt = runtime.find((c) => c.channelId === settingsChannelId) || null
  const reviewRt = runtime.find((c) => c.channelId === reviewChannelId) || null
  const previewChannelId = reviewPreviewOpen
    ? reviewChannelId
    : activeTab === 'settings'
      ? settingsChannelId
      : null
  const previewRt = runtime.find((c) => c.channelId === previewChannelId) || null
  const capturing = !!settingsRt?.video?.running
  const previewCapturing = !!previewRt?.video?.running
  const noFrame = !reviewRt?.video?.running && !reviewRt?.probe
  const runtimeEvery = Number(settingsRt?.video?.interval_sec ?? every)
  const runtimeBatch = Number(settingsRt?.video?.batch_size ?? batch)
  const pendingFrames = Number(settingsRt?.video?.pending_frames || 0)
  const droppedFrames = Number(settingsRt?.video?.queue_dropped_batches ?? settingsRt?.video?.dropped_frames ?? 0)
  const summaryQueueDepth = Number(settingsRt?.video?.summary_queue_depth || 0)
  const selectorEnabled = settingsRt?.video?.capture_selector_enabled !== false
  const selectorBias = String(settingsRt?.video?.capture_selector_bias || 'auto')
  const selectorSource = String(settingsRt?.video?.capture_apex_last_selection?.selection_source || '').replace(/_/g, ' ')
  const selectorLabel = selectorEnabled
    ? `${selectorBias}${selectorSource ? ` · ${selectorSource}` : ''}`
    : 'off · midpoint'

  const loadStreams = useCallback(async () => {
    try { setStreams(await videoApi.streams()) } catch (e: any) { setError(e?.message || 'Streams failed') }
  }, [])

  const loadFeed = useCallback(async () => {
    if (reviewChannelId == null) return
    const requestId = ++feedRequestRef.current
    const bounds = summaryPeriodBounds(period, Date.now(), customBounds)
    const targetDepth = resolveSummaryResolution(resolution, period, bounds)
    const run = period === 'live' ? 'live' : 'all'
    try {
      let entries: SummaryEntry[] = []
      let displayDepth = targetDepth
      if (targetDepth === 'L0') {
        const response = await videoApi.session(reviewChannelId, { limit: 240, run, ...bounds })
        entries = response.logs || []
      } else {
        const response = await videoApi.rollups(reviewChannelId, {
          level_limit: 240,
          run,
          ...bounds,
          target_level: targetDepth,
        })
        entries = (response.levels as any)?.[targetDepth] || []
        if (!entries.length && resolution === 'AUTO' && period !== 'live') {
          const fallback = await videoApi.session(reviewChannelId, { limit: 240, run: 'all', ...bounds })
          entries = fallback.logs || []
          displayDepth = 'L0'
        }
      }
      if (requestId !== feedRequestRef.current) return
      setRenderedDepth(displayDepth)
      setFeed(entries.slice().sort((left, right) => (
        Number(left.created_at ?? left.window_start ?? 0)
        - Number(right.created_at ?? right.window_start ?? 0)
      )))
      setError(null)
    } catch (e: any) {
      if (requestId === feedRequestRef.current) setError(e?.message || 'Feed failed')
    }
  }, [reviewChannelId, customBounds, period, resolution])

  useEffect(() => { loadStreams() }, [loadStreams])
  useEffect(() => {
    if (!drive || drive.effect.target !== 'video') return
    const { action, payload } = drive.effect
    const nextChannel = Number(payload.channel_id)
    const validNextChannel = Number.isInteger(nextChannel) && channels.some((channel) => channel.id === nextChannel)
      ? nextChannel
      : null
    const promptAction = action === 'open_prompt_settings' || action === 'show_prompt_preview'
    if (promptAction) {
      if (validNextChannel != null) setSettingsChannelId(validNextChannel)
      setActiveTab('settings')
    } else {
      if (validNextChannel != null) setReviewChannelId(validNextChannel)
      setActiveTab('review')
    }
    const nextDepth = String(payload.depth || '').toUpperCase()
    if (['AUTO', 'L0', 'L1', 'L2', 'L3'].includes(nextDepth)) {
      setResolution(nextDepth as SummaryResolution)
    }
    const sinceMs = Number(payload.since_ms)
    const untilMs = Number(payload.until_ms)
    if (Number.isFinite(sinceMs) && Number.isFinite(untilMs) && untilMs >= sinceMs) {
      const bounds = { from_ts: sinceMs / 1000, to_ts: untilMs / 1000 }
      setCustomFrom(formatDatetimeLocal(bounds.from_ts))
      setCustomTo(formatDatetimeLocal(bounds.to_ts))
      setCustomBounds(bounds)
      setPeriod('custom')
      setLive(false)
    }
    if (action === 'open_prompt_settings' && canManagePrompts) setPromptOpen(true)
    if (action === 'show_channels' || action === 'show_restore_status') void loadStreams()
  }, [drive?.seq, channels, canManagePrompts, loadStreams])
  useEffect(() => {
    setReviewChannelId((current) => (
      current != null && channels.some((channel) => channel.id === current)
        ? current
        : (channels[0]?.id ?? null)
    ))
    setSettingsChannelId((current) => (
      current != null && channels.some((channel) => channel.id === current)
        ? current
        : (channels[0]?.id ?? null)
    ))
  }, [channels])
  useEffect(() => {
    if (reviewChannelId == null) return
    try { window.localStorage.setItem(REVIEW_CHANNEL_STORAGE_KEY, String(reviewChannelId)) } catch { /* optional */ }
  }, [reviewChannelId])
  useEffect(() => {
    feedRequestRef.current += 1
    setFeed([])
    setCollapsedSummaries(new Set())
  }, [reviewChannelId])
  useEffect(() => { if (activeTab === 'review') void loadFeed() }, [activeTab, loadFeed])
  useEffect(() => {
    if (settingsChannelId == null) return
    try { window.localStorage.setItem(SETTINGS_CHANNEL_STORAGE_KEY, String(settingsChannelId)) } catch { /* optional */ }
  }, [settingsChannelId])
  useEffect(() => {
    hydratedSettingsChannelRef.current = null
    setBatch('12')
    setEvery('5')
    setModel('auto')
    setSettingsDirty(false)
  }, [settingsChannelId])
  useEffect(() => {
    if (settingsChannelId == null || settingsRt?.channelId !== settingsChannelId) return
    if (hydratedSettingsChannelRef.current === settingsChannelId) return
    const nextBatch = Number(settingsRt.video?.batch_size)
    const nextEvery = Number(settingsRt.video?.interval_sec)
    setBatch(Number.isFinite(nextBatch) && nextBatch > 0 ? String(nextBatch) : '12')
    setEvery(Number.isFinite(nextEvery) && nextEvery > 0 ? String(nextEvery) : '5')
    setModel(String(settingsRt.video?.model || 'auto'))
    hydratedSettingsChannelRef.current = settingsChannelId
    setSettingsDirty(false)
  }, [settingsChannelId, settingsRt])

  // available VLM models for the "Live model" selector (matches the original /lm/models)
  useEffect(() => {
    videoApi.lmModels().then((cat) => {
      const autoVal = cat.auto_model_selector || 'auto'
      const opts: DropOption[] = [{ value: autoVal, label: cat.auto_model_label || 'Auto (balance)' }]
      for (const m of cat.models || []) if (m && m !== autoVal) opts.push({ value: m, label: m })
      setModelOptions(opts)
      setModel((cur) => (cur === 'auto' ? autoVal : cur))
    }).catch(() => {})
  }, [])

  // poll streams (runtime) every 4s
  useEffect(() => { const t = window.setInterval(loadStreams, 4000); return () => window.clearInterval(t) }, [loadStreams])
  // poll feed when live-following
  useEffect(() => {
    if (!live || activeTab !== 'review') return
    const t = window.setInterval(loadFeed, 3000); return () => window.clearInterval(t)
  }, [activeTab, live, loadFeed])

  useEffect(() => {
    const feedElement = feedRef.current
    if (reviewOverlayOpen) {
      if (reviewOverlayScrollRef.current === null && feedElement) {
        reviewOverlayScrollRef.current = feedElement.scrollTop
      }
      return
    }
    if (reviewOverlayScrollRef.current === null) return
    const savedScrollTop = reviewOverlayScrollRef.current
    reviewOverlayScrollRef.current = null
    window.requestAnimationFrame(() => {
      if (feedRef.current) feedRef.current.scrollTop = savedScrollTop
    })
  }, [reviewOverlayOpen])

  useEffect(() => {
    if (live && !reviewOverlayOpen) {
      feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
    }
  }, [feed, live, reviewOverlayOpen])
  // The model view is a cheap snapshot of EVA's existing attention ring.
  // Full live is a bounded second Luxriot stream and renews from its broker
  // lease rather than being torn down on every summary cadence tick.
  useEffect(() => {
    releasePreviewMedia()
    setPreviewError(true)
    setPreviewLoading(previewMode === 'live')
    setAttentionMediaSrc(null)
    setFullLiveMedia(null)
    setError((current) => current?.startsWith('Full live ') ? null : current)
    if (previewChannelId == null || previewMode !== 'model' || previewCapturing) return
    const ms = Math.max(3, Number(every) || 5) * 1000
    const t = window.setInterval(() => setPreviewBust((b) => b + 1), ms)
    return () => {
      window.clearInterval(t)
      releasePreviewMedia()
    }
  }, [every, previewCapturing, previewChannelId, previewMode, releasePreviewMedia])

  useEffect(() => {
    if (previewMode !== 'model' || previewChannelId == null || !previewCapturing) return
    const controller = new AbortController()
    let renewalTimer = 0
    const mediaUrl = attentionStreamUrl(previewChannelId, previewBust)
    void fetch(mediaUrl, {
      method: 'HEAD',
      cache: 'no-store',
      credentials: 'include',
      signal: controller.signal,
    }).then((response) => {
      const kind = String(response.headers.get('X-EVA-Media-Kind') || '').trim().toLowerCase()
      if (!response.ok || kind !== 'mjpeg' || response.headers.get('X-EVA-Attention-Preview') !== '1') {
        throw new Error('EVA attention preview is not ready')
      }
      setAttentionMediaSrc(mediaUrl)
      const rawRenewAfter = Number(response.headers.get('X-EVA-Media-Renew-After-Ms'))
      const renewAfter = Number.isFinite(rawRenewAfter) && rawRenewAfter > 0
        ? Math.max(750, Math.min(120_000, rawRenewAfter))
        : 20_000
      renewalTimer = window.setTimeout(() => setPreviewBust((value) => value + 1), renewAfter)
    }).catch(() => {
      if (controller.signal.aborted) return
      // Keep the last model-visible static frame on screen while the bounded
      // attention stream warms up, then retry without surfacing a false alarm.
      setAttentionMediaSrc(null)
      renewalTimer = window.setTimeout(() => setPreviewBust((value) => value + 1), 5_000)
    })
    return () => {
      controller.abort()
      if (renewalTimer) window.clearTimeout(renewalTimer)
      if (modelPreviewRef.current) modelPreviewRef.current.removeAttribute('src')
    }
  }, [previewBust, previewCapturing, previewChannelId, previewMode])

  useEffect(() => {
    if (previewMode !== 'live' || previewChannelId == null) return
    const controller = new AbortController()
    let renewalTimer = 0
    const mediaUrl = fullLiveMediaUrl(previewChannelId, previewBust)
    setPreviewLoading(true)
    setPreviewError(false)
    setFullLiveMedia(null)

    void fetch(mediaUrl, {
      method: 'HEAD',
      cache: 'no-store',
      credentials: 'include',
      signal: controller.signal,
    }).then((response) => {
      if (!response.ok) throw new Error(`Full live negotiation failed (${response.status})`)
      const mediaKind = String(response.headers.get('X-EVA-Media-Kind') || '').trim().toLowerCase()
      if (mediaKind !== 'video' && mediaKind !== 'mjpeg') {
        throw new Error(`Full live returned an unsupported transport (${mediaKind || 'unknown'})`)
      }
      setFullLiveMedia({ kind: mediaKind, src: mediaUrl })
      const rawRenewAfter = Number(response.headers.get('X-EVA-Media-Renew-After-Ms'))
      const renewAfter = Number.isFinite(rawRenewAfter) && rawRenewAfter > 0
        ? Math.max(750, Math.min(120_000, rawRenewAfter))
        : 20_000
      renewalTimer = window.setTimeout(() => setPreviewBust((value) => value + 1), renewAfter)
    }).catch((error) => {
      if (controller.signal.aborted) return
      setPreviewLoading(false)
      setPreviewError(true)
      const message = error instanceof Error ? error.message : 'Full live is unavailable'
      setError(message.startsWith('Full live ') ? message : `Full live is unavailable: ${message}`)
      renewalTimer = window.setTimeout(() => setPreviewBust((value) => value + 1), 5_000)
    })
    return () => {
      controller.abort()
      if (renewalTimer) window.clearTimeout(renewalTimer)
      const liveImage = livePreviewImageRef.current
      const liveVideo = livePreviewVideoRef.current
      if (liveImage) liveImage.removeAttribute('src')
      if (liveVideo) {
        liveVideo.pause()
        liveVideo.removeAttribute('src')
        liveVideo.load()
      }
    }
  }, [previewBust, previewChannelId, previewMode])

  const start = async () => {
    if (settingsChannelId == null) return
    setBusy(true); setError(null)
    try {
      const r = await videoApi.startCapture(buildCaptureInput(settingsChannelId, { batch, every, model }))
      if (!r.success) throw new Error(r.error || 'Start failed')
      hydratedSettingsChannelRef.current = settingsChannelId
      setSettingsDirty(false)
      await loadStreams()
    } catch (e: any) { setError(e?.message || 'Start failed') } finally { setBusy(false) }
  }
  const reloadChannels = async () => {
    await onReloadChannels?.()
    await loadStreams()
  }
  const stop = async () => { if (settingsChannelId == null) return; setBusy(true); try { await videoApi.stopCapture(settingsChannelId); await loadStreams() } catch (e: any) { setError(e?.message || 'Stop failed') } finally { setBusy(false) } }
  const flush = async () => {
    if (settingsChannelId == null) return; setBusy(true)
    try {
      const r = await videoApi.flushCapture(settingsChannelId)
      if (r.status?.logs?.length && settingsChannelId === reviewChannelId) void loadFeed()
    } catch (e: any) { setError(e?.message || 'Flush failed') } finally { setBusy(false) }
  }

  const previewSrc = previewChannelId != null ? recentFrameUrl(previewChannelId, previewBust) : ''
  const modelPreviewSrc = attentionMediaSrc || previewSrc
  const markPreviewReady = useCallback(() => {
    setPreviewLoading(false)
    setPreviewError(false)
    setPreviewReadyAt(Date.now())
    setError((current) => current?.startsWith('Full live ') ? null : current)
  }, [])
  const markPreviewFailed = useCallback(() => {
    setPreviewLoading(false)
    setPreviewError(true)
  }, [])
  const feedKeys = useMemo(
    () => feed.map((entry, index) => summaryEntryKey(entry, index)),
    [feed],
  )
  const collapseAll = useCallback(
    () => setCollapsedSummaries(new Set(feedKeys)),
    [feedKeys],
  )
  const expandAll = useCallback(() => setCollapsedSummaries(new Set()), [])
  const toggleSummary = useCallback((key: string) => {
    setCollapsedSummaries((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }, [])
  const selectPeriod = useCallback((nextPeriod: SummaryPeriod) => {
    if (nextPeriod === 'custom' && (!customFrom || !customTo)) {
      const initial = defaultCustomRange()
      setCustomFrom(initial.inputs.from)
      setCustomTo(initial.inputs.to)
      setCustomBounds(initial.bounds)
    }
    setPeriod(nextPeriod)
    setLive(nextPeriod === 'live')
  }, [customFrom, customTo])
  const applyCustomRange = useCallback(() => {
    const from_ts = parseDatetimeLocal(customFrom)
    const to_ts = parseDatetimeLocal(customTo)
    if (!Number.isFinite(from_ts) || !Number.isFinite(to_ts)) {
      setError('Choose both From and To values for the custom summary period.')
      return
    }
    setCustomBounds({ from_ts, to_ts })
    setLive(false)
  }, [customFrom, customTo])
  const toggleLive = useCallback(() => {
    if (period !== 'live') {
      setPeriod('live')
      setLive(true)
      return
    }
    setLive((current) => !current)
  }, [period])

  const selectWorkspaceTab = useCallback((nextTab: VideoWorkspaceTab) => {
    if (nextTab === 'settings') setReviewPreviewOpen(false)
    setActiveTab(nextTab)
  }, [])
  const requestSettingsChannel = useCallback((channelId: number, openSettings = false) => {
    if (channelId === settingsChannelId) {
      if (openSettings) {
        setReviewPreviewOpen(false)
        setActiveTab('settings')
      }
      return
    }
    if (settingsDirty) {
      setPendingSettingsSwitch({ channelId, openSettings })
      return
    }
    setSettingsChannelId(channelId)
    if (openSettings) {
      setReviewPreviewOpen(false)
      setActiveTab('settings')
    }
  }, [settingsChannelId, settingsDirty])
  const editReviewStream = useCallback(() => {
    if (reviewChannelId != null) requestSettingsChannel(reviewChannelId, true)
  }, [requestSettingsChannel, reviewChannelId])
  const updateBatch = useCallback((value: string) => {
    setBatch(value)
    setSettingsDirty(true)
  }, [])
  const updateEvery = useCallback((value: string) => {
    setEvery(value)
    setSettingsDirty(true)
  }, [])
  const updateModel = useCallback((value: string) => {
    setModel(value)
    setSettingsDirty(true)
  }, [])
  const discardSettingsDraft = useCallback(() => {
    const nextBatch = Number(settingsRt?.video?.batch_size)
    const nextEvery = Number(settingsRt?.video?.interval_sec)
    setBatch(Number.isFinite(nextBatch) && nextBatch > 0 ? String(nextBatch) : '12')
    setEvery(Number.isFinite(nextEvery) && nextEvery > 0 ? String(nextEvery) : '5')
    setModel(String(settingsRt?.video?.model || modelOptions[0]?.value || 'auto'))
    setSettingsDirty(false)
  }, [modelOptions, settingsRt])

  const previewCard = (
    <div className="vid-preview-card">
      <div className="vid-preview-head">
        <div>
          <div className="mon-panel-title">Preview</div>
          <div className="vid-preview-sub">
            {channelName(previewChannelId ?? -1) || 'No channel'} · {previewMode === 'model' ? 'EVA model view' : 'independent Luxriot live'}
          </div>
        </div>
        <div className="vid-preview-actions">
          <button
            className="mon-btn vid-preview-mode"
            onClick={() => {
              releasePreviewMedia()
              setPreviewMode((current) => current === 'model' ? 'live' : 'model')
            }}
            disabled={previewChannelId == null}
            title={previewMode === 'model'
              ? 'Open a second smooth Luxriot stream. This can add source load.'
              : 'Return to the frames already selected by EVA without another recorder stream.'}
          >
            <IconSwitchHorizontal size={13} />
            {previewMode === 'model' ? 'Full live' : 'Model view'}
          </button>
          {reviewPreviewOpen && (
            <button className="mon-icobtn" onClick={() => setReviewPreviewOpen(false)} aria-label="Close preview">
              <IconX size={15} />
            </button>
          )}
        </div>
      </div>
      <div className={`vid-viewport ${previewError ? 'err' : ''}`}>
        {previewMode === 'model' && modelPreviewSrc && (
          <img
            ref={modelPreviewRef}
            className={previewError ? 'preview-pending' : undefined}
            src={modelPreviewSrc}
            alt="EVA model-view preview"
            onLoad={markPreviewReady}
            onError={markPreviewFailed}
          />
        )}
        {previewMode === 'live' && fullLiveMedia?.kind === 'mjpeg' && (
          <img
            ref={livePreviewImageRef}
            className={previewError ? 'preview-pending' : undefined}
            src={fullLiveMedia.src}
            alt="Luxriot full live preview"
            onLoad={markPreviewReady}
            onError={markPreviewFailed}
          />
        )}
        {previewMode === 'live' && fullLiveMedia?.kind === 'video' && (
          <video
            ref={livePreviewVideoRef}
            src={fullLiveMedia.src}
            autoPlay
            muted
            controls
            playsInline
            preload="metadata"
            onCanPlay={markPreviewReady}
            onPlaying={markPreviewReady}
            onEnded={() => setPreviewBust((value) => value + 1)}
            onError={markPreviewFailed}
          />
        )}
        {previewLoading && <div className="vid-overlay loading">OPENING FULL LIVE…</div>}
        {previewError && !previewLoading && <div className="vid-overlay"><IconVideoOff size={20} /> PREVIEW UNAVAILABLE</div>}
      </div>
    </div>
  )

  return (
    <div className="vid-cols">
      <StreamControl
        channels={channels}
        activeTab={activeTab} onTab={selectWorkspaceTab}
        settingsChannelId={settingsChannelId} onSettingsChannel={(channelId) => requestSettingsChannel(channelId)}
        reviewChannelId={reviewChannelId} onReviewChannel={setReviewChannelId}
        onReload={reloadChannels}
        batch={batch} onBatch={updateBatch} every={every} onEvery={updateEvery} model={model} onModel={updateModel} modelOptions={modelOptions}
        canCapture={canCapture} canManagePrompts={canManagePrompts}
        capturing={capturing} busy={busy} onStart={start} onStop={stop} onFlush={flush}
        onPromptSettings={() => setPromptOpen(true)}
        period={period} onPeriod={selectPeriod} resolution={resolution} onResolution={setResolution}
        customFrom={customFrom} onCustomFrom={setCustomFrom} customTo={customTo} onCustomTo={setCustomTo}
        onApplyCustom={applyCustomRange}
        onRefreshFeed={loadFeed} live={live} onToggleLive={toggleLive}
        summaryCount={feed.length} onCollapseAll={collapseAll} onExpandAll={expandAll}
        onOpenPreview={() => setReviewPreviewOpen(true)}
        onEditReviewStream={editReviewStream}
        settingsDirty={settingsDirty}
        onDiscardSettings={discardSettingsDraft}
      />

      {activeTab === 'review' ? (
        <div className="vid-review-main">
          <div className="vid-feed-card vid-review-feed">
          <div className="vid-feed-heading">
            <div>
              <div className="mon-panel-title">Stream summaries</div>
              <div className="vid-feed-meta">
                {channelName(reviewChannelId ?? -1) || 'No channel'} · #{reviewChannelId ?? '—'} · {resolution === 'AUTO' ? `Auto → ${renderedDepth}` : renderedDepth} · {feed.length} summaries
                {live ? ' · following live' : ' · fixed view'}
              </div>
            </div>
          </div>
          {error && <div className="chat-error"><IconAlertTriangle size={14} /> {error}</div>}
          <div className="vid-feed" ref={feedRef}>
            {feed.length === 0 && (
              <div className="vid-feed-empty">
                {noFrame && <div className="vid-feed-note"><IconAlertTriangle size={16} /> No fresh EVA frame is available for this channel yet.</div>}
                <div className="empty-state">No summaries yet for this channel.</div>
              </div>
            )}
            {feed.map((entry, index) => {
              const key = feedKeys[index]
              return (
                <SummaryCard
                  key={key}
                  entry={entry}
                  selectedDepth={renderedDepth}
                  collapsed={collapsedSummaries.has(key)}
                  canCreateBookmarks={canCreateBookmarks}
                  canReportIncidents={canReportIncidents}
                  canExport={canExport}
                  onToggle={() => toggleSummary(key)}
                  onImage={(src, title) => {
                    if (onReviewSummary && Number(entry.thumbnail_detection_id) > 0) {
                      onReviewSummary(entry)
                    } else {
                      setSummaryImage({ src, title })
                    }
                  }}
                  onReportIncident={setIncidentDraft}
                />
              )
            })}
          </div>
          </div>
          {reviewPreviewOpen && <aside className="vid-review-preview-drawer">{previewCard}</aside>}
        </div>
      ) : (
        <div className="vid-settings-main">
          <section className="vid-settings-preview">
            {previewCard}
            <p>Model view shows EVA-selected input without opening another recorder stream. Full live is intended for short source verification.</p>
          </section>
          <section className="vid-selected-card vid-settings-status">
            <div className="vid-settings-status-head">
              <div>
                <div className="mon-panel-title">Runtime and attention</div>
                <div className="vid-preview-sub">Current state for the independently selected settings channel</div>
              </div>
              <span className="vid-sel-cur">{settingsChannelId != null ? `#${settingsChannelId}` : '—'}</span>
            </div>
            {error && <div className="chat-error"><IconAlertTriangle size={14} /> {error}</div>}
            <div className="vid-sel-body">
              <div className="vid-sel-name">{channelName(settingsChannelId ?? -1) || 'No channel selected'}</div>
              <div className="vid-sel-grid">
                <div><span>Channel</span><b>#{settingsChannelId ?? '—'}</b></div>
                <div><span>Summaries</span><b className={capturing ? 'good' : ''}>{capturing ? 'running' : 'idle'}</b></div>
                <div>
                  <span>Preview</span>
                  <b className={previewError ? 'bad' : 'good'}>
                    {previewLoading ? 'opening' : previewError ? 'failed' : previewMode === 'model' ? 'model view' : 'full live'}
                  </b>
                </div>
                <div><span>Cadence</span><b>{runtimeEvery > 0 ? `${(1 / runtimeEvery).toFixed(2)} fps · ${runtimeEvery}s` : '—'}</b></div>
                <div><span>Batch</span><b>{runtimeBatch || '—'}</b></div>
                <div><span>Draft</span><b className={settingsDirty ? 'bad' : 'good'}>{settingsDirty ? 'not applied' : 'in sync'}</b></div>
              </div>
              <div className="vid-sel-list">
                <div><span>Live model</span><b>{String(settingsRt?.video?.model || model || 'auto')}</b></div>
                <div><span>Frame selector</span><b>{selectorLabel}</b></div>
                <div>
                  <span>Summary queue</span>
                  <b>
                    {capturing
                      ? `${pendingFrames}/${runtimeBatch || '?'} frames${summaryQueueDepth ? ` · ${summaryQueueDepth} queued` : ''}${droppedFrames ? ` · ${droppedFrames} dropped` : ''}`
                      : 'idle'}
                  </b>
                </div>
                <div><span>Probe capture</span><b>{settingsRt?.probe?.running ? (settingsRt.probe.paused ? 'paused' : 'active') : 'idle'}</b></div>
                <div><span>Last preview</span><b>{previewLoading ? 'opening' : previewError ? 'never' : fmtAge(previewReadyAt)}</b></div>
                {settingsRt?.video?.frozen_signal && <div><span>Signal</span><b className="bad">frozen</b></div>}
                {settingsRt?.video?.last_error && <div><span>Last error</span><b className="bad">{String(settingsRt.video.last_error)}</b></div>}
              </div>
            </div>
          </section>
        </div>
      )}

      {promptOpen && canManagePrompts && settingsChannelId != null && (
        <PromptSettingsModal channelId={settingsChannelId} canCreateBookmarks={canCreateBookmarks} onClose={() => setPromptOpen(false)} />
      )}
      {pendingSettingsSwitch && (
        <div className="scrim" onClick={() => setPendingSettingsSwitch(null)}>
          <div className="modal usr-confirm" onClick={(event) => event.stopPropagation()}>
            <div className="usr-confirm-title"><IconAlertTriangle size={16} /> Unsaved stream settings</div>
            <p>
              Batch, cadence, or model changes for <b>{channelName(settingsChannelId ?? -1) || `#${settingsChannelId}`}</b> have not been applied.
              Discard them and open <b>{channelName(pendingSettingsSwitch.channelId) || `#${pendingSettingsSwitch.channelId}`}</b>?
            </p>
            <div className="usr-confirm-actions">
              <button className="mon-btn" autoFocus onClick={() => setPendingSettingsSwitch(null)}>Keep editing</button>
              <button
                className="mon-btn danger"
                onClick={() => {
                  const target = pendingSettingsSwitch
                  setPendingSettingsSwitch(null)
                  setSettingsChannelId(target.channelId)
                  if (target.openSettings) {
                    setReviewPreviewOpen(false)
                    setActiveTab('settings')
                  }
                }}
              >
                Discard &amp; switch
              </button>
            </div>
          </div>
        </div>
      )}
      {summaryImage && (
        <div className="inspect-zoom" onClick={() => setSummaryImage(null)}>
          <img src={summaryImage.src} alt={summaryImage.title} onClick={(event) => event.stopPropagation()} />
          <div className="vid-summary-lightbox-title">{summaryImage.title}</div>
          <button className="modal-close inspect-zoom-close" onClick={() => setSummaryImage(null)} aria-label="Close frame">
            <IconX size={22} />
          </button>
        </div>
      )}
      {incidentDraft && (
        <IncidentModal
          draftInput={incidentDraft}
          canExport={canExport}
          onClose={() => setIncidentDraft(null)}
        />
      )}
    </div>
  )
}
