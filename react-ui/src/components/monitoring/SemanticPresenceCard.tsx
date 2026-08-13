import { IconActivityHeartbeat, IconAlertTriangle } from '@tabler/icons-react'
import type {
  SemanticPresenceClass,
  SemanticPresencePoint,
  SemanticPresenceStatus,
} from '../../api/probes'
import {
  presenceClassKey,
  presenceDisplaySignal,
  presenceMatchesContext,
  presenceReaction,
  rankPresenceClasses,
} from './semanticPresenceView'

function n3(value?: number | null): string {
  return value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(3)
}

function signed3(value?: number | null): string {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  const number = Number(value)
  return `${number >= 0 ? '+' : ''}${number.toFixed(3)}`
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
  const view = presenceDisplaySignal(item)
  const signal = presencePoints(view.history)
  const baseline = baselinePoints(view.history)
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
  maxClasses,
  onInspect,
  busyKey,
  activeKey,
  contextTexts = [],
}: {
  presence?: SemanticPresenceStatus | null
  compact?: boolean
  maxClasses?: number
  onInspect?: (item: SemanticPresenceClass) => void
  busyKey?: string | null
  activeKey?: string | null
  contextTexts?: string[]
}) {
  if (!presence?.enabled) return null
  const classes = rankPresenceClasses([...(presence.classes || [])], contextTexts)
    .slice(0, maxClasses ?? (compact ? 3 : 10))
  const hasSpatial = classes.some((item) => presenceDisplaySignal(item).spatial)
  const timestamp = Number(presence.timestamp_ms)
  const ageSeconds = Number.isFinite(timestamp) && timestamp > 0
    ? Math.max(0, (Date.now() - timestamp) / 1000)
    : null
  return (
    <section className={`semantic-presence-card ${compact ? 'compact' : ''}`} aria-label="Semantic presence">
      <div className="presence-card-head">
        <div>
          <span><IconActivityHeartbeat size={15} /> Semantic presence</span>
          <b>{hasSpatial
            ? 'Same-forward patch response from each class baseline · spatial shadow, not object detection'
            : 'Response from each class baseline · attention only, not object detection'}</b>
        </div>
        <i>{presence.state || 'warming_up'}{ageSeconds != null ? ` · ${ageSeconds.toFixed(ageSeconds < 10 ? 1 : 0)}s` : ''}</i>
      </div>
      {presence.error && (
        <div className="presence-card-error"><IconAlertTriangle size={14} /> {presence.error}</div>
      )}
      <div className="presence-class-list">
        {classes.map((item) => {
          const key = presenceClassKey(item)
          const view = presenceDisplaySignal(item)
          const reaction = presenceReaction(item)
          const relevant = presenceMatchesContext(item, contextTexts)
          const responseLabel = view.warmup
            ? `${view.spatial ? 'spatial · ' : ''}${view.samples || 0} warmup`
            : reaction.current
              ? `${view.spatial ? 'spatial · ' : ''}responding ${reaction.direction === 'down' ? '↓' : '↑'}`
              : reaction.reacting
                ? `${view.spatial ? 'spatial · ' : ''}recent response ${reaction.direction === 'down' ? '↓' : '↑'}`
                : relevant
                  ? `${view.spatial ? 'spatial · ' : ''}probe context · baseline`
                  : `${view.spatial ? 'spatial · ' : ''}baseline`
          const content = <>
            <div className="presence-class-label">
              <b>{item.label}</b>
              <span>{busyKey === key
                ? 'mapping exact frame…'
                : onInspect
                  ? `${responseLabel} · inspect patches`
                  : responseLabel}</span>
            </div>
            <PresencePulse item={item} />
            <div className="presence-class-values">
              <b>{signed3(view.delta)}</b>
              <span>{view.spatial ? 'patch' : 'raw'} {n3(view.score)}</span>
              <em>base {n3(view.baseline)}</em>
            </div>
          </>
          return onInspect ? (
            <button
              type="button"
              className={`presence-class inspectable ${reaction.reacting ? 'reacting' : 'routine'} ${view.state || 'warming_up'} ${activeKey === key ? 'active' : ''}`}
              key={key}
              disabled={!!busyKey}
              onClick={() => onInspect(item)}
              title={`Inspect relative ${item.label} patch affinity on the exact scored frame`}
            >
              {content}
            </button>
          ) : (
            <div className={`presence-class ${reaction.reacting ? 'reacting' : 'routine'} ${view.state || 'warming_up'}`} key={key}>{content}</div>
          )
        })}
        {!classes.length && <div className="presence-card-empty">Waiting for the first archived embedding.</div>}
      </div>
      {!compact && (
        <div className="presence-card-note">
          Responding classes stay on top for about 12 samples; routine rows stay stable. Raw scores are prompt-specific and cannot be compared between labels.
        </div>
      )}
      {onInspect && (
        <div className="presence-card-note">
          Click a class for an on-demand patch hint. It is relative affinity, not a box detector.
        </div>
      )}
    </section>
  )
}
