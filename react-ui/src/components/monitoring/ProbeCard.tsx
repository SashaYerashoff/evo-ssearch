import { IconTrash, IconPlayerPlay, IconPlayerStop, IconMaximize, IconRadar2 } from '@tabler/icons-react'
import type { Probe } from '../../api/probes'
import { hitImageSrc } from '../../api/probes'

function fmtTime(ms?: number | null): string {
  if (!ms) return '—'
  const d = new Date(Number(ms))
  return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}
const n3 = (v?: number | null) => (v == null ? '—' : Number(v).toFixed(3))

export function gateText(p: Probe): string {
  const g = p.bookmark_gate
  if (!p.bookmark) return 'Gate: bookmarks off'
  if (!g) return 'Gate: ready'
  const reason = String(g.reason || 'ready').replace(/_/g, ' ')
  const cd = g.remaining_sec ?? g.cooldown_sec
  return cd != null ? `Gate: ${reason} (${Number(cd).toFixed(1)}s)` : `Gate: ${reason}`
}

export function lastHit(p: Probe) { return p.last_hit || p.recent_hits?.[0] }

export type ProbeStatus = 'running' | 'paused' | 'idle' | 'disabled'

export function ProbeCard({ probe, status, selected, onSelect, onRun, onDelete }: {
  probe: Probe
  status: ProbeStatus
  selected: boolean
  onSelect: () => void
  onRun?: () => void
  onDelete?: () => void
}) {
  const hit = lastHit(probe)
  const src = hitImageSrc(hit)
  return (
    <button className={`probe-card ${selected ? 'sel' : ''} ${status}`} onClick={onSelect}>
      <div className="pc-head">
        <span className={`pc-badge ${status}`}>{status.toUpperCase()}</span>
        <div className="pc-actions">
          {onRun && <span className="pc-ico" title={status === 'running' ? 'Stop probe' : 'Start probe'}
            onClick={(e) => { e.stopPropagation(); onRun() }}>
            {status === 'running' ? <IconPlayerStop size={14} /> : <IconPlayerPlay size={14} />}
          </span>}
          <span className="pc-ico" title="Inspect" onClick={(e) => { e.stopPropagation(); onSelect() }}><IconMaximize size={14} /></span>
        </div>
      </div>
      <div className="pc-thumb">
        {src ? <img src={src} alt={probe.name || 'probe'} loading="lazy" /> : <IconRadar2 size={26} />}
      </div>
      <div className="pc-name">{probe.name || 'Untitled probe'}</div>
      <div className="pc-meta">Ch {probe.channel_id ?? '—'} · Last {fmtTime(hit?.timestamp_ms ?? hit?.recorded_at_ms)}</div>
      <div className="pc-scores">P: {n3(hit?.pos_score)} · N: {n3(hit?.neg_score)} · M: {n3(hit?.margin)}</div>
      <div className="pc-foot">
        <span className="pc-gate">{gateText(probe)}</span>
        {onDelete && <span className="pc-ico danger" title="Delete probe" onClick={(e) => { e.stopPropagation(); onDelete() }}><IconTrash size={14} /></span>}
      </div>
    </button>
  )
}
