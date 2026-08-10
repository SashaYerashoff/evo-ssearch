import { IconTrash, IconPlayerPlay, IconPlayerStop, IconMaximize, IconRadar2 } from '@tabler/icons-react'
import type { Probe } from '../../api/probes'
import { hitImageSrc } from '../../api/probes'
import {
  probeHitSeries,
  probeOrigin,
  probeTemporaryTtl,
} from './probeBoard'

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

export type ProbeStatus = 'running' | 'degraded' | 'paused' | 'idle' | 'disabled'

export const PROBE_ORIGIN_LABELS = {
  operator: { label: 'Operator', short: 'OP' },
  agent: { label: 'Agent', short: 'AI' },
  auto: { label: 'Background VLM', short: 'VLM' },
} as const

export function ProbeOriginBadge({ probe }: { probe: Probe }) {
  const origin = probeOrigin(probe)
  const view = PROBE_ORIGIN_LABELS[origin]
  return (
    <span className={`probe-origin ${origin}`} title={`Created by ${view.label}`}>
      <i aria-hidden="true" />
      {view.short}
    </span>
  )
}

export function ProbeSparkline({
  probe,
  compact = false,
}: {
  probe: Probe
  compact?: boolean
}) {
  const series = probeHitSeries(probe)
  const width = compact ? 96 : 180
  const height = compact ? 22 : 42
  if (!series.length) {
    return (
      <div className={`probe-spark empty ${compact ? 'compact' : ''}`} title="No probe hits recorded yet">
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          <line x1="0" y1={height - 2} x2={width} y2={height - 2} />
        </svg>
        {!compact && <span>no hits yet</span>}
      </div>
    )
  }
  const floor = Number(probe.pos_floor)
  const hasFloor = Number.isFinite(floor)
  const values = series.map((point) => point.score)
  const scale = hasFloor ? [...values, floor] : values
  const min = Math.min(...scale)
  const max = Math.max(...scale)
  const span = max - min > 1e-6 ? max - min : 1
  const stepX = series.length > 1 ? width / (series.length - 1) : width
  const toY = (value: number) => height - 2 - ((value - min) / span) * (height - 4)
  const points = series
    .map((point, index) => `${(index * stepX).toFixed(1)},${toY(point.score).toFixed(1)}`)
    .join(' ')
  const last = series[series.length - 1]
  return (
    <div
      className={`probe-spark ${hasFloor && last.score >= floor ? 'over' : ''} ${compact ? 'compact' : ''}`}
      title={`${series.length} hit${series.length === 1 ? '' : 's'} · last P ${last.score.toFixed(3)}${hasFloor ? ` · floor ${floor.toFixed(3)}` : ''}`}
    >
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
        {hasFloor && <line className="floor" x1="0" y1={toY(floor)} x2={width} y2={toY(floor)} />}
        <polyline className="signal" points={points} />
        <circle className="head" cx={(series.length - 1) * stepX} cy={toY(last.score)} r="2" />
      </svg>
    </div>
  )
}

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
  const ttl = probeTemporaryTtl(probe)
  return (
    <button className={`probe-card ${selected ? 'sel' : ''} ${status}`} onClick={onSelect}>
      <div className="pc-head">
        <div className="pc-tags">
          <span className={`pc-badge ${status}`}>{status.toUpperCase()}</span>
          <ProbeOriginBadge probe={probe} />
        </div>
        <div className="pc-actions">
          {onRun && <span className="pc-ico" title={status === 'disabled' ? 'Enable probe' : 'Disable probe'}
            onClick={(e) => { e.stopPropagation(); onRun() }}>
            {status === 'disabled' ? <IconPlayerPlay size={14} /> : <IconPlayerStop size={14} />}
          </span>}
          <span className="pc-ico" title="Inspect" onClick={(e) => { e.stopPropagation(); onSelect() }}><IconMaximize size={14} /></span>
        </div>
      </div>
      <div className="pc-thumb">
        {src
          ? <img src={src} alt={probe.name || 'probe'} loading="lazy" />
          : <><IconRadar2 className="pc-radar" size={20} /><ProbeSparkline probe={probe} /></>}
      </div>
      <div className="pc-name">{probe.name || 'Untitled probe'}</div>
      <div className="pc-meta">Ch {probe.channel_id ?? '—'} · Last {fmtTime(hit?.timestamp_ms ?? hit?.recorded_at_ms)}</div>
      <div className="pc-scores">P: {n3(hit?.pos_score)} · N: {n3(hit?.neg_score)} · M: {n3(hit?.margin)}</div>
      <div className="pc-foot">
        <span className="pc-gate">{gateText(probe)}</span>
        {ttl && <span className={`probe-ttl ${ttl.expired ? 'expired' : ''}`} title={ttl.title}>{ttl.text}</span>}
        {onDelete && <span className="pc-ico danger" title="Delete probe" onClick={(e) => { e.stopPropagation(); onDelete() }}><IconTrash size={14} /></span>}
      </div>
    </button>
  )
}
