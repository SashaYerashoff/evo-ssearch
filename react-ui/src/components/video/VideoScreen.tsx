import { useEffect, useState, useCallback, useRef } from 'react'
import { IconVideoOff, IconAlertTriangle } from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import { videoApi, recentFrameUrl, mergeRuntime, type StreamsStatus, type ChannelRuntime, type SummaryEntry } from '../../api/video'
import { renderMarkdown } from '../agent/markdown'
import { StreamControl } from './StreamControl'
import { ChannelRuntime as RuntimePanel } from './ChannelRuntime'
import { PromptSettingsModal } from './PromptSettingsModal'

const SEV_ORDER = ['critical', 'high', 'normal', 'low', 'info']

function fmtClock(sec?: number): string {
  if (!sec) return '—'
  return new Date(sec * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })
}

export function VideoScreen({ channels }: { channels: Channel[] }) {
  const [channelId, setChannelId] = useState<number | null>(channels[0]?.id ?? null)
  const [streams, setStreams] = useState<StreamsStatus>({})
  const [feed, setFeed] = useState<SummaryEntry[]>([])
  const [batch, setBatch] = useState('12')
  const [every, setEvery] = useState('5')
  const [model, setModel] = useState('auto')
  const [history, setHistory] = useState('6')
  const [depth, setDepth] = useState('L0')
  const [live, setLive] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [feedNote, setFeedNote] = useState('')
  const [previewBust, setPreviewBust] = useState(1)
  const [previewError, setPreviewError] = useState(true)
  const [promptOpen, setPromptOpen] = useState(false)

  const channelName = useCallback((id: number) => channels.find((c) => c.id === id)?.title, [channels])
  const runtime: ChannelRuntime[] = mergeRuntime(streams, channelName)
  const selRt = runtime.find((c) => c.channelId === channelId) || null
  const capturing = !!selRt?.video?.running

  const loadStreams = useCallback(async () => {
    try { setStreams(await videoApi.streams()) } catch (e: any) { setError(e?.message || 'Streams failed') }
  }, [])

  const loadFeed = useCallback(async () => {
    if (channelId == null) return
    const from_ts = history !== '0' ? Math.floor(Date.now() / 1000 - Number(history) * 3600) : undefined
    try {
      let entries: SummaryEntry[] = []
      if (depth === 'L0') { const r = await videoApi.session(channelId, 60); entries = r.logs || [] }
      else { const r = await videoApi.rollups(channelId, { from_ts }); entries = (r.levels as any)?.[depth] || [] }
      setFeed(entries)
      setFeedNote(`${entries.length} ${depth === 'L0' ? 'summaries' : 'rollups'} · ${channelName(channelId) || `ch ${channelId}`}`)
    } catch (e: any) { setError(e?.message || 'Feed failed') }
  }, [channelId, depth, history, channelName])

  useEffect(() => { loadStreams() }, [loadStreams])
  useEffect(() => { loadFeed() }, [loadFeed])

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
      const r = await videoApi.startCapture({ channel_id: channelId, batch_size: Number(batch), interval_sec: Number(every), model: model.trim() || undefined })
      if (!r.success) throw new Error(r.error || 'Start failed')
      await loadStreams(); loadFeed()
    } catch (e: any) { setError(e?.message || 'Start failed') } finally { setBusy(false) }
  }
  const stop = async () => { if (channelId == null) return; setBusy(true); try { await videoApi.stopCapture(channelId); await loadStreams() } catch (e: any) { setError(e?.message || 'Stop failed') } finally { setBusy(false) } }
  const flush = async () => {
    if (channelId == null) return; setBusy(true)
    try { const r = await videoApi.flushCapture(channelId); if (r.status?.logs?.length) { loadFeed() } } catch (e: any) { setError(e?.message || 'Flush failed') } finally { setBusy(false) }
  }

  const stopVideo = async (id: number) => { setBusy(true); try { const r = await videoApi.stopStream(id, 'video', false); if (r.streams || r.video_streams) setStreams(r); else await loadStreams() } finally { setBusy(false) } }
  const pauseProbes = async (id: number) => { setBusy(true); try { await videoApi.stopStream(id, 'analytics', true); await loadStreams() } finally { setBusy(false) } }
  const stopVideoAll = async () => { setBusy(true); try { await videoApi.stopAll({ stop_video: true, stop_analytics: false }); await loadStreams() } finally { setBusy(false) } }
  const pauseProbesAll = async () => { setBusy(true); try { await videoApi.stopAll({ stop_video: false, stop_analytics: true, pause_analytics: true }); await loadStreams() } finally { setBusy(false) } }
  const stopAll = async () => { setBusy(true); try { await videoApi.stopAll(); await loadStreams() } finally { setBusy(false) } }

  const previewSrc = channelId != null ? recentFrameUrl(channelId, previewBust) : ''

  return (
    <div className="vid-cols">
      <StreamControl
        channels={channels} channelId={channelId} onChannel={setChannelId} onReload={loadStreams}
        batch={batch} onBatch={setBatch} every={every} onEvery={setEvery} model={model} onModel={setModel}
        capturing={capturing} busy={busy} onStart={start} onStop={stop} onFlush={flush}
        onPromptSettings={() => setPromptOpen(true)}
        history={history} onHistory={setHistory} depth={depth} onDepth={setDepth}
        onRefreshFeed={loadFeed} live={live} onToggleLive={() => setLive((v) => !v)}
        note={!capturing && !selRt?.probe ? 'No fresh EVA frame is available for this channel yet.' : null}
      />

      <section className="vid-center">
        <div className="vid-preview-card">
          <div className={`vid-viewport ${previewError ? 'err' : ''}`}>
            {previewSrc && <img src={previewSrc} alt="live preview" onLoad={() => setPreviewError(false)} onError={() => setPreviewError(true)} />}
            {previewError && <div className="vid-overlay"><IconVideoOff size={22} /> PREVIEW UNAVAILABLE</div>}
          </div>
          <div className="vid-selected">
            <div className="mon-panel-title">Selected stream</div>
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
        </div>

        <div className="vid-feed-card">
          <div className="mon-panel-title">VLM feed</div>
          <div className="mon-panel-sub">Live summaries and drilled rollups for the selected channel context.</div>
          <div className="vid-feed-meta">{feedNote}</div>
          {error && <div className="chat-error"><IconAlertTriangle size={14} /> {error}</div>}
          <div className="vid-feed">
            {feed.length === 0 && <div className="empty-state" style={{ padding: 30 }}>No summaries yet for this channel.</div>}
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
      </section>

      <RuntimePanel
        runtime={runtime} busy={busy}
        onRefresh={loadStreams} onStopVideoAll={stopVideoAll} onPauseProbesAll={pauseProbesAll} onStopAll={stopAll}
        onStopVideo={stopVideo} onPauseProbes={pauseProbes}
        onViewSummaries={(id) => { setChannelId(id); loadFeed() }}
      />

      {promptOpen && channelId != null && <PromptSettingsModal channelId={channelId} onClose={() => setPromptOpen(false)} />}
    </div>
  )
}
