import { IconTrash, IconPlayerPlay, IconPlayerStop, IconMaximize, IconRadar2 } from '@tabler/icons-react'
import type { Probe, ProbeLiveSignal } from '../../api/probes'
import { hitImageSrc } from '../../api/probes'
import {
  probeHitSeries,
  probeLiveSeries,
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
  history,
  compact = false,
}: {
  probe: Probe
  history?: ProbeLiveSignal[] | null
  compact?: boolean
}) {
  const liveSeries = probeLiveSeries(history)
  const series = liveSeries.length ? liveSeries : probeHitSeries(probe)
  const sourceLabel = liveSeries.length ? 'samples' : 'events'
  const width = compact ? 96 : 180
  const height = compact ? 22 : 42
  if (!series.length) {
    return (
      <div className={`probe-spark empty ${compact ? 'compact' : ''}`} title="No semantic samples recorded yet">
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          <line x1="0" y1={height - 2} x2={width} y2={height - 2} />
        </svg>
        {!compact && <span className="probe-spark-empty">no signal yet</span>}
      </div>
    )
  }
  const floor = Number(probe.pos_floor)
  const hasFloor = Number.isFinite(floor)
  const values = series.flatMap((point) => [point.posScore, point.negScore])
  const scale = hasFloor ? [...values, floor] : values
  const min = Math.min(...scale)
  const max = Math.max(...scale)
  const pad = Math.max(0.004, (max - min) * 0.1)
  const scaleMin = min - pad
  const scaleMax = max + pad
  const span = Math.max(0.008, scaleMax - scaleMin)
  const stepX = series.length > 1 ? width / (series.length - 1) : width
  const toY = (value: number) => height - 2 - ((value - scaleMin) / span) * (height - 4)
  const positivePoints = series
    .map((point, index) => `${(index * stepX).toFixed(1)},${toY(point.posScore).toFixed(1)}`)
    .join(' ')
  const negativePoints = series
    .map((point, index) => `${(index * stepX).toFixed(1)},${toY(point.negScore).toFixed(1)}`)
    .join(' ')
  const last = series[series.length - 1]
  return (
    <div
      className={`probe-spark ${hasFloor && last.posScore >= floor ? 'over' : ''} ${compact ? 'compact' : ''}`}
      title={`${series.length} ${sourceLabel} · P ${last.posScore.toFixed(3)} · N ${last.negScore.toFixed(3)} · M ${last.margin.toFixed(3)}${hasFloor ? ` · P floor ${floor.toFixed(3)}` : ''}`}
    >
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
        {hasFloor && <line className="floor" x1="0" y1={toY(floor)} x2={width} y2={toY(floor)} />}
        <polyline className="signal negative" points={negativePoints} />
        <polyline className="signal positive" points={positivePoints} />
        <circle className="head negative" cx={(series.length - 1) * stepX} cy={toY(last.negScore)} r="2" />
        <circle className="head positive" cx={(series.length - 1) * stepX} cy={toY(last.posScore)} r="2" />
      </svg>
      {!compact && (
        <div className="probe-spark-legend">
          <span className="positive">P</span>
          <span className="negative">N</span>
          <i>{liveSeries.length ? 'live' : 'events'}</i>
        </div>
      )}
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
          : <IconRadar2 className="pc-radar" size={20} />}
      </div>
      <div className="pc-pulse"><ProbeSparkline probe={probe} compact /></div>
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
