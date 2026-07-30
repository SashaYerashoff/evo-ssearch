import { useEffect, useState, useCallback, useRef } from 'react'
import { IconVideoOff, IconAlertTriangle, IconChevronRight } from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import { buildCaptureInput, videoApi, recentFrameUrl, mergeRuntime, type StreamsStatus, type ChannelRuntime, type SummaryEntry } from '../../api/video'
import type { DropOption } from '../shell/Dropdown'
import { renderMarkdown } from '../agent/markdown'
import { StreamControl } from './StreamControl'
import { PromptSettingsModal } from './PromptSettingsModal'

const SEV_ORDER = ['critical', 'high', 'normal', 'low', 'info']

function fmtClock(sec?: number): string {
  if (!sec) return '—'
  return new Date(sec * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })
}

export function VideoScreen({
  channels,
  onReloadChannels,
  canCapture,
  canManagePrompts,
  canCreateBookmarks,
}: {
  channels: Channel[]
  onReloadChannels?: () => Promise<void> | void
  canCapture: boolean
  canManagePrompts: boolean
  canCreateBookmarks: boolean
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
      if (depth === 'L0') { const r = await videoApi.session(channelId, { limit: 60, from_ts }); entries = r.logs || [] }
      else { const r = await videoApi.rollups(channelId, { from_ts }); entries = (r.levels as any)?.[depth] || [] }
      setFeed(entries)
    } catch (e: any) { setError(e?.message || 'Feed failed') }
  }, [channelId, depth, history, channelName])

  useEffect(() => { loadStreams() }, [loadStreams])
  useEffect(() => { loadFeed() }, [loadFeed])
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
          <div className="mon-panel-title">VLM feed</div>
          {error && <div className="chat-error"><IconAlertTriangle size={14} /> {error}</div>}
          <div className="vid-feed">
            {feed.length === 0 && (
              <div className="vid-feed-empty">
                {noFrame && <div className="vid-feed-note"><IconAlertTriangle size={16} /> No fresh EVA frame is available for this channel yet.</div>}
                <div className="empty-state">No summaries yet for this channel.</div>
              </div>
            )}
            {feed.map((s, i) => {
              const chips = SEV_ORDER.filter((k) => (s.alert_counts?.[k] ?? 0) > 0)
              return (
                <div key={i} className="vid-sum">
                  <div className="vid-sum-head">
                    <span className="vid-sum-ts">{fmtClock(s.created_at ?? (s.window_end))}</span>
                    {s.model && <span className="vid-sum-model">{s.model}</span>}
                    {s.frame_count != null && <span className="vid-sum-frames">{s.frame_count} frames</span>}
                    {chips.map((k) => <span key={k} className={`vid-sev sev-${k}`}>{s.alert_counts![k]} {k}</span>)}
                  </div>
                  {s.summary && <div className="vid-sum-body md" dangerouslySetInnerHTML={{ __html: renderMarkdown(String(s.summary)) }} />}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {promptOpen && canManagePrompts && channelId != null && (
        <PromptSettingsModal channelId={channelId} canCreateBookmarks={canCreateBookmarks} onClose={() => setPromptOpen(false)} />
      )}
    </div>
  )
}
