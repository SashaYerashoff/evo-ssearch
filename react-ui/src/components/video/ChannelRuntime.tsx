import { IconRefresh, IconPlayerStop, IconPlayerPause } from '@tabler/icons-react'
import type { ChannelRuntime as CR, Stream } from '../../api/video'

const fpsOf = (s?: Stream | null) => {
  if (!s) return null
  const iv = s.interval_sec
  if (iv && iv > 0) return (1 / iv).toFixed(2)
  return null
}

function videoState(v: Stream | null): string {
  if (!v || !v.running) return 'IDLE'
  const f = fpsOf(v)
  return f ? `ACTIVE · ${f} FPS` : 'ACTIVE'
}
function probeState(p: Stream | null): string {
  if (!p || !p.running) return 'IDLE'
  if (p.paused) return 'PAUSED'
  const f = fpsOf(p)
  const buf = p.pending_frames ?? p.recent_frame_count
  return `ACTIVE · ${f ?? '—'} FPS · ${buf ?? 0} BUFFERED`
}

export function ChannelRuntime({ runtime, busy, onRefresh, onStopVideoAll, onPauseProbesAll, onStopAll, onStopVideo, onPauseProbes, onViewSummaries }: {
  runtime: CR[]
  busy: boolean
  onRefresh: () => void
  onStopVideoAll: () => void
  onPauseProbesAll: () => void
  onStopAll: () => void
  onStopVideo: (channelId: number) => void
  onPauseProbes: (channelId: number) => void
  onViewSummaries: (channelId: number) => void
}) {
  return (
    <aside className="vid-runtime">
      <div className="mon-panel-title">Channel runtime</div>
      <div className="mon-panel-sub">Live channel states and stream lifecycle controls.</div>
      <div className="vid-runtime-toolbar">
        <button className="mon-btn" onClick={onRefresh} disabled={busy}><IconRefresh size={14} className={busy ? 'spin' : ''} /> Refresh</button>
        <button className="mon-btn" onClick={onStopVideoAll} disabled={busy}><IconPlayerStop size={14} /> Stop video</button>
        <button className="mon-btn" onClick={onPauseProbesAll} disabled={busy}><IconPlayerPause size={14} /> Pause probes</button>
      </div>

      <div className="vid-runtime-list">
        {runtime.length === 0 && <div className="ag-empty">No active channels.</div>}
        {runtime.map((c) => {
          const vRunning = !!c.video?.running
          const pActive = !!c.probe?.running && !c.probe?.paused
          return (
            <div key={c.channelId} className="vid-rt-card">
              <div className="vid-rt-head">
                <span className="vid-rt-name">{c.name || `Channel ${c.channelId}`}</span>
                <div className="vid-rt-tags">
                  <span className={`vid-tag ${vRunning ? 'on' : ''}`}>video {vRunning ? 'live' : 'idle'}</span>
                  <span className={`vid-tag ${pActive ? 'on' : c.probe?.paused ? 'warn' : ''}`}>probes {pActive ? 'active' : c.probe?.paused ? 'paused' : 'idle'}</span>
                </div>
              </div>
              <div className="vid-rt-row"><span>Video summaries</span><b>{videoState(c.video)}</b></div>
              <div className="vid-rt-row"><span>Probe capture</span><b>{probeState(c.probe)}</b></div>
              {c.probe?.last_error && <div className="vid-rt-err">{c.probe.last_error}</div>}
              <div className="vid-rt-actions">
                <button className="mon-btn" onClick={() => onViewSummaries(c.channelId)}>View summaries</button>
                {vRunning && <button className="mon-btn" onClick={() => onStopVideo(c.channelId)}>Stop video</button>}
                {pActive && <button className="mon-btn" onClick={() => onPauseProbes(c.channelId)}>Pause probes</button>}
              </div>
            </div>
          )
        })}
      </div>
      {runtime.length > 0 && <button className="mon-btn danger" onClick={onStopAll} disabled={busy} style={{ marginTop: 6 }}><IconPlayerStop size={14} /> Stop all</button>}
    </aside>
  )
}
