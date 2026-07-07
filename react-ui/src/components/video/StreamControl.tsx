import { IconReload, IconPlayerPlay, IconPlayerStop, IconDroplet, IconRoute, IconSettings } from '@tabler/icons-react'
import type { Channel } from '../../api/types'

const BATCHES = ['4', '8', '12', '16', '24', '32']
export const HISTORY = [
  { v: '6', label: 'Last 6 hours' }, { v: '24', label: 'Last day' }, { v: '72', label: 'Last 3 days' },
  { v: '168', label: 'Last week' }, { v: '720', label: 'Last month' }, { v: '0', label: 'All history' },
]
export const DEPTH = [
  { v: 'L0', label: 'Live' }, { v: 'L1', label: 'Minutes' }, { v: 'L2', label: 'Hours' }, { v: 'L3', label: 'Days' },
]

export function StreamControl(p: {
  channels: Channel[]
  channelId: number | null
  onChannel: (id: number) => void
  onReload: () => void
  batch: string; onBatch: (v: string) => void
  every: string; onEvery: (v: string) => void
  model: string; onModel: (v: string) => void
  capturing: boolean; busy: boolean
  onStart: () => void; onStop: () => void; onFlush: () => void
  onPromptSettings: () => void
  history: string; onHistory: (v: string) => void
  depth: string; onDepth: (v: string) => void
  onRefreshFeed: () => void
  live: boolean; onToggleLive: () => void
  note?: string | null
}) {
  const fps = Number(p.every) > 0 ? (1 / Number(p.every)).toFixed(2) : '—'
  return (
    <aside className="vid-control">
      <div className="mon-panel">
        <div className="mon-panel-title">Live stream control</div>
        <div className="mon-panel-sub">Choose a channel, configure cadence, and steer live summaries.</div>
        {p.note && <div className="vid-note">{p.note}</div>}

        <div className="wfield"><label>Channel</label>
          <div className="vid-row">
            <select value={p.channelId ?? ''} onChange={(e) => p.onChannel(Number(e.target.value))}>
              {p.channels.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
            </select>
            <button className="mon-icobtn" title="Reload channels" onClick={p.onReload}><IconReload size={15} /></button>
          </div>
        </div>
        <div className="wgrid">
          <div className="wfield"><label>Batch</label>
            <select value={p.batch} onChange={(e) => p.onBatch(e.target.value)}>{BATCHES.map((b) => <option key={b} value={b}>{b}</option>)}</select>
          </div>
          <div className="wfield"><label>Every (s)</label>
            <input type="number" min={0.2} max={300} step={0.1} value={p.every} onChange={(e) => p.onEvery(e.target.value)} />
          </div>
        </div>
        <div className="vid-pill">~{fps} fps · batch {p.batch} · 800px</div>
        <div className="wfield"><label>Live model</label>
          <input value={p.model} onChange={(e) => p.onModel(e.target.value)} placeholder="auto (balance)" />
        </div>
        <div className="vid-actions">
          {p.capturing
            ? <button className="mon-btn danger" disabled={p.busy} onClick={p.onStop}><IconPlayerStop size={15} /> Stop summaries</button>
            : <button className="mon-btn accent" disabled={p.busy} onClick={p.onStart}><IconPlayerPlay size={15} /> Start summaries</button>}
          <button className="mon-btn" disabled={p.busy || !p.capturing} onClick={p.onFlush}><IconDroplet size={15} /> Flush now</button>
          <button className="mon-btn" disabled title="Coming soon"><IconRoute size={15} /> Ground road mask</button>
          <button className="mon-btn" onClick={p.onPromptSettings}><IconSettings size={15} /> System prompt settings</button>
        </div>
      </div>

      <div className="mon-panel">
        <div className="mon-panel-title">Summary lens</div>
        <div className="mon-panel-sub">Choose what lands in the live text feed.</div>
        <div className="wfield"><label>History</label>
          <select value={p.history} onChange={(e) => p.onHistory(e.target.value)}>{HISTORY.map((h) => <option key={h.v} value={h.v}>{h.label}</option>)}</select>
        </div>
        <div className="wfield"><label>Depth</label>
          <select value={p.depth} onChange={(e) => p.onDepth(e.target.value)}>{DEPTH.map((d) => <option key={d.v} value={d.v}>{d.label}</option>)}</select>
        </div>
        <div className="vid-row">
          <button className="mon-btn" onClick={p.onRefreshFeed}><IconReload size={14} /> Refresh</button>
          <button className={`mon-btn ${p.live ? 'accent' : ''}`} onClick={p.onToggleLive}><IconPlayerPlay size={14} /> {p.live ? 'Live on' : 'Live off'}</button>
        </div>
      </div>
    </aside>
  )
}
