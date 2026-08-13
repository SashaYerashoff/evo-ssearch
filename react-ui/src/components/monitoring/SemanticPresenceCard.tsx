import { IconActivityHeartbeat, IconAlertTriangle } from '@tabler/icons-react'
import type {
  SemanticPresenceClass,
  SemanticPresencePoint,
  SemanticPresenceStatus,
} from '../../api/probes'

function n3(value?: number | null): string {
  return value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(3)
}

function presencePoints(history?: SemanticPresencePoint[]): string {
  if (!history?.length) return ''
  const width = 128
  const height = 28
  const values = history.flatMap((point) => [Number(point.score), Number(point.baseline)]).filter(Number.isFinite)
  if (!values.length) return ''
  const min = Math.min(...values)
  const max = Math.max(...values)
  const pad = Math.max(0.003, (max - min) * 0.12)
  const span = Math.max(0.006, max - min + pad * 2)
  const toY = (value: number) => height - 2 - ((value - min + pad) / span) * (height - 4)
  const step = history.length > 1 ? width / (history.length - 1) : width
  return history.map((point, index) => `${(index * step).toFixed(1)},${toY(Number(point.score)).toFixed(1)}`).join(' ')
}

function baselinePoints(history?: SemanticPresencePoint[]): string {
  if (!history?.length) return ''
  const width = 128
  const height = 28
  const values = history.flatMap((point) => [Number(point.score), Number(point.baseline)]).filter(Number.isFinite)
  if (!values.length) return ''
  const min = Math.min(...values)
  const max = Math.max(...values)
  const pad = Math.max(0.003, (max - min) * 0.12)
  const span = Math.max(0.006, max - min + pad * 2)
  const toY = (value: number) => height - 2 - ((value - min + pad) / span) * (height - 4)
  const step = history.length > 1 ? width / (history.length - 1) : width
  return history.map((point, index) => `${(index * step).toFixed(1)},${toY(Number(point.baseline)).toFixed(1)}`).join(' ')
}

function PresencePulse({ item }: { item: SemanticPresenceClass }) {
  const signal = presencePoints(item.history)
  const baseline = baselinePoints(item.history)
  return (
    <svg className="presence-pulse" viewBox="0 0 128 28" preserveAspectRatio="none" aria-hidden="true">
      {baseline && <polyline className="baseline" points={baseline} />}
      {signal && <polyline className="signal" points={signal} />}
      {!signal && <line className="empty" x1="0" y1="26" x2="128" y2="26" />}
    </svg>
  )
}

export function SemanticPresenceCard({
  presence,
  compact = false,
}: {
  presence?: SemanticPresenceStatus | null
  compact?: boolean
}) {
  if (!presence?.enabled) return null
  const classes = [...(presence.classes || [])]
    .sort((left, right) => Math.abs(Number(right.z || 0)) - Math.abs(Number(left.z || 0)))
    .slice(0, compact ? 3 : 10)
  const timestamp = Number(presence.timestamp_ms)
  const ageSeconds = Number.isFinite(timestamp) && timestamp > 0
    ? Math.max(0, (Date.now() - timestamp) / 1000)
    : null
  return (
    <section className={`semantic-presence-card ${compact ? 'compact' : ''}`} aria-label="Semantic presence">
      <div className="presence-card-head">
        <div>
          <span><IconActivityHeartbeat size={15} /> Semantic presence</span>
          <b>Pooled embedding pulse · attention only, not object detection</b>
        </div>
        <i>{presence.state || 'warming_up'}{ageSeconds != null ? ` · ${ageSeconds.toFixed(ageSeconds < 10 ? 1 : 0)}s` : ''}</i>
      </div>
      {presence.error && (
        <div className="presence-card-error"><IconAlertTriangle size={14} /> {presence.error}</div>
      )}
      <div className="presence-class-list">
        {classes.map((item) => (
          <div className={`presence-class ${item.state || 'warming_up'}`} key={item.key || item.label}>
            <div className="presence-class-label">
              <b>{item.label}</b>
              <span>{item.warmup ? `${item.samples || 0} warmup` : String(item.state || 'routine').replace(/_/g, ' ')}</span>
            </div>
            <PresencePulse item={item} />
            <div className="presence-class-values">
              <b>{n3(item.score)}</b>
              <span>base {n3(item.baseline)}</span>
              <em>{Number(item.delta) >= 0 ? '+' : ''}{n3(item.delta)}</em>
            </div>
          </div>
        ))}
        {!classes.length && <div className="presence-card-empty">Waiting for the first archived embedding.</div>}
      </div>
      {!compact && (
        <div className="presence-card-note">
          Baselines adapt per channel. Scores are semantic similarities, not probabilities or object counts.
        </div>
      )}
    </section>
  )
}
