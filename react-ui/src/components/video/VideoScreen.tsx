import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  IconAlertTriangle,
  IconChevronDown,
  IconChevronRight,
  IconCopy,
  IconDownload,
  IconVideoOff,
  IconX,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import type { ConsoleDrive } from '../../App'
import { buildCaptureInput, videoApi, recentFrameUrl, mergeRuntime, type StreamsStatus, type ChannelRuntime, type SummaryEntry } from '../../api/video'
import type { DropOption } from '../shell/Dropdown'
import { renderMarkdown } from '../agent/markdown'
import { StreamControl } from './StreamControl'
import { PromptSettingsModal } from './PromptSettingsModal'
import {
  SUMMARY_SEVERITIES,
  splitSummaryMachineJson,
  summaryAlertCounts,
  summaryBurst,
  summaryEntryKey,
  summaryLevel,
  summarySemanticStatus,
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
  canExport,
  onToggle,
  onImage,
}: {
  entry: SummaryEntry
  selectedDepth: string
  collapsed: boolean
  canExport: boolean
  onToggle: () => void
  onImage: (src: string, title: string) => void
}) {
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
  const isCover = entry.thumbnail_is_cover === true
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
          {isCover && <span className="vid-meta-chip cover" title={String(entry.cover_reason || 'Model-selected batch cover')}>cover</span>}
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
          <button className="btn compact" onClick={() => copySummary(entry)} disabled={!entry.summary}>
            <IconCopy size={13} /> Copy
          </button>
          {canExport && (
            <button className="btn compact" onClick={() => exportSummary(entry, level)} disabled={!entry.summary}>
              <IconDownload size={13} /> Export
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
                VLM input{isCover ? ' · cover' : ''} · {thumbnailRole}
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
  onReloadChannels,
  canCapture,
  canManagePrompts,
  canCreateBookmarks,
  canExport,
}: {
  channels: Channel[]
  drive?: ConsoleDrive | null
  onReloadChannels?: () => Promise<void> | void
  canCapture: boolean
  canManagePrompts: boolean
  canCreateBookmarks: boolean
  canExport: boolean
}) {
  const [channelId, setChannelId] = useState<number | null>(channels[0]?.id ?? null)
  const [streams, setStreams] = useState<StreamsStatus>({})
  const [feed, setFeed] = useState<SummaryEntry[]>([])
  const [batch, setBatch] = useState('12')
  const [every, setEvery] = useState('5')
  const [model, setModel] = useState('auto')
  const [prompt, setPrompt] = useState('')
  const [history, setHistory] = useState('6')
  const [depth, setDepth] = useState('L0')
  const [live, setLive] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previewBust, setPreviewBust] = useState(1)
  const [previewError, setPreviewError] = useState(true)
  const [promptOpen, setPromptOpen] = useState(false)
  const [selOpen, setSelOpen] = useState(false)  // Selected stream details — collapsed by default
  const [modelOptions, setModelOptions] = useState<DropOption[]>([{ value: 'auto', label: 'Auto (balance)' }])
  const [collapsedSummaries, setCollapsedSummaries] = useState<Set<string>>(new Set())
  const [summaryImage, setSummaryImage] = useState<{ src: string; title: string } | null>(null)
  const feedRef = useRef<HTMLDivElement>(null)

  const channelName = useCallback((id: number) => channels.find((c) => c.id === id)?.title, [channels])
  const runtime: ChannelRuntime[] = mergeRuntime(streams, channelName)
  const selRt = runtime.find((c) => c.channelId === channelId) || null
  const capturing = !!selRt?.video?.running
  const noFrame = !capturing && !selRt?.probe

  const loadStreams = useCallback(async () => {
    try { setStreams(await videoApi.streams()) } catch (e: any) { setError(e?.message || 'Streams failed') }
  }, [])

  const loadFeed = useCallback(async () => {
    if (channelId == null) return
    const from_ts = history !== '0' ? Math.floor(Date.now() / 1000 - Number(history) * 3600) : undefined
    try {
      let entries: SummaryEntry[] = []
      if (depth === 'L0') {
        const response = await videoApi.session(channelId, { limit: 240, from_ts })
        entries = response.logs || []
      } else {
        const response = await videoApi.rollups(channelId, { level_limit: 240, from_ts })
        entries = (response.levels as any)?.[depth] || []
      }
      setFeed(entries.slice().sort((left, right) => (
        Number(left.created_at ?? left.window_start ?? 0)
        - Number(right.created_at ?? right.window_start ?? 0)
      )))
    } catch (e: any) { setError(e?.message || 'Feed failed') }
  }, [channelId, depth, history])

  useEffect(() => { loadStreams() }, [loadStreams])
  useEffect(() => { loadFeed() }, [loadFeed])
  useEffect(() => {
    if (!drive || drive.effect.target !== 'video') return
    const { action, payload } = drive.effect
    const nextChannel = Number(payload.channel_id)
    if (Number.isInteger(nextChannel) && channels.some((channel) => channel.id === nextChannel)) {
      setChannelId(nextChannel)
    }
    const nextDepth = String(payload.depth || '').toUpperCase()
    if (['L0', 'L1', 'L2', 'L3'].includes(nextDepth)) setDepth(nextDepth)
    const sinceMs = Number(payload.since_ms)
    const untilMs = Number(payload.until_ms)
    if (Number.isFinite(sinceMs) && Number.isFinite(untilMs) && untilMs >= sinceMs) {
      const hours = Math.max(1, Math.ceil((untilMs - sinceMs) / 3_600_000))
      setHistory(String(hours))
      setLive(false)
    }
    if (action === 'open_prompt_settings' && canManagePrompts) setPromptOpen(true)
    if (action === 'show_channels' || action === 'show_restore_status') void loadStreams()
  }, [drive?.seq, channels, canManagePrompts, loadStreams])
  useEffect(() => {
    setChannelId((current) => (
      current != null && channels.some((channel) => channel.id === current)
        ? current
        : (channels[0]?.id ?? null)
    ))
  }, [channels])

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
    if (!live) return
    const t = window.setInterval(loadFeed, 3000); return () => window.clearInterval(t)
  }, [live, loadFeed])
  useEffect(() => {
    if (live) feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
  }, [feed, live])
  // refresh preview image on cadence
  useEffect(() => {
    setPreviewBust((b) => b + 1); setPreviewError(true)
    const ms = Math.max(3, Number(every) || 5) * 1000
    const t = window.setInterval(() => setPreviewBust((b) => b + 1), ms)
    return () => window.clearInterval(t)
  }, [channelId, every])

  const start = async () => {
    if (channelId == null) return
    setBusy(true); setError(null)
    try {
      const r = await videoApi.startCapture(buildCaptureInput(channelId, { batch, every, model, prompt }))
      if (!r.success) throw new Error(r.error || 'Start failed')
      await loadStreams(); loadFeed()
    } catch (e: any) { setError(e?.message || 'Start failed') } finally { setBusy(false) }
  }
  const reloadChannels = async () => {
    await onReloadChannels?.()
    await loadStreams()
  }
  const stop = async () => { if (channelId == null) return; setBusy(true); try { await videoApi.stopCapture(channelId); await loadStreams() } catch (e: any) { setError(e?.message || 'Stop failed') } finally { setBusy(false) } }
  const flush = async () => {
    if (channelId == null) return; setBusy(true)
    try { const r = await videoApi.flushCapture(channelId); if (r.status?.logs?.length) { loadFeed() } } catch (e: any) { setError(e?.message || 'Flush failed') } finally { setBusy(false) }
  }

  const previewSrc = channelId != null ? recentFrameUrl(channelId, previewBust) : ''
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

  return (
    <div className="vid-cols">
      <StreamControl
        channels={channels} channelId={channelId} onChannel={setChannelId} onReload={reloadChannels}
        batch={batch} onBatch={setBatch} every={every} onEvery={setEvery} model={model} onModel={setModel} modelOptions={modelOptions}
        prompt={prompt} onPrompt={setPrompt}
        canCapture={canCapture} canManagePrompts={canManagePrompts}
        capturing={capturing} busy={busy} onStart={start} onStop={stop} onFlush={flush}
        onPromptSettings={() => setPromptOpen(true)}
        history={history} onHistory={setHistory} depth={depth} onDepth={setDepth}
        onRefreshFeed={loadFeed} live={live} onToggleLive={() => setLive((v) => !v)}
        summaryCount={feed.length} onCollapseAll={collapseAll} onExpandAll={expandAll}
      />

      <div className="vid-main">
        <aside className="vid-side">
          <div className="vid-preview-card">
            <div className="mon-panel-title">Preview</div>
            <div className={`vid-viewport ${previewError ? 'err' : ''}`}>
              {previewSrc && <img className={previewError ? 'preview-pending' : undefined} src={previewSrc} alt="live preview"
                onLoad={() => setPreviewError(false)} onError={() => setPreviewError(true)} />}
              {previewError && <div className="vid-overlay"><IconVideoOff size={20} /> PREVIEW UNAVAILABLE</div>}
            </div>
          </div>

          <div className={`vid-selected-card ${selOpen ? 'open' : ''}`}>
            <button className="vid-sel-toggle" onClick={() => setSelOpen((v) => !v)} aria-expanded={selOpen}>
              <IconChevronRight size={15} className="vid-sel-chev" />
              <span className="mon-panel-title">Selected stream</span>
              <span className="vid-sel-cur">{channelId != null ? `#${channelId}` : '—'}</span>
            </button>
            {selOpen && (
              <div className="vid-sel-body">
                <div className="vid-sel-name">{channelName(channelId ?? -1) || 'No channel selected'}</div>
                <div className="vid-sel-grid">
                  <div><span>Channel</span><b>#{channelId ?? '—'}</b></div>
                  <div><span>Preview</span><b className={previewError ? 'bad' : 'good'}>{previewError ? 'failed' : 'live'}</b></div>
                  <div><span>Cadence</span><b>{(1 / Number(every || 5)).toFixed(2)} fps · {every}s</b></div>
                  <div><span>Batch</span><b>{batch}</b></div>
                </div>
                <div className="vid-sel-list">
                  <div><span>Live model</span><b>{model || 'auto'}</b></div>
                  <div><span>Summary queue</span><b>{capturing ? 'running' : 'idle'}</b></div>
                  <div><span>Probe capture</span><b>{selRt?.probe?.running ? (selRt.probe.paused ? 'paused' : 'active') : 'idle'}</b></div>
                  <div><span>Last preview</span><b>{previewError ? 'never' : 'just now'}</b></div>
                </div>
              </div>
            )}
          </div>
        </aside>

        <div className="vid-feed-card">
          <div className="vid-feed-heading">
            <div>
              <div className="mon-panel-title">VLM feed</div>
              <div className="vid-feed-meta">
                {channelName(channelId ?? -1) || 'No channel'} · {depth} · {feed.length} summaries
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
                  selectedDepth={depth}
                  collapsed={collapsedSummaries.has(key)}
                  canExport={canExport}
                  onToggle={() => toggleSummary(key)}
                  onImage={(src, title) => setSummaryImage({ src, title })}
                />
              )
            })}
          </div>
        </div>
      </div>

      {promptOpen && canManagePrompts && channelId != null && (
        <PromptSettingsModal channelId={channelId} canCreateBookmarks={canCreateBookmarks} onClose={() => setPromptOpen(false)} />
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
    </div>
  )
}
