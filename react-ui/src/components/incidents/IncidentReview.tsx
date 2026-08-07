import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  IconAlertTriangle,
  IconClock,
  IconEye,
  IconFileDescription,
  IconPhotoOff,
  IconRefresh,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import {
  incidentsApi,
  type IncidentReviewRecord,
  type IncidentReviewState,
} from '../../api/incidents'
import { Dropdown } from '../shell/Dropdown'
import { useI18n } from '../../i18n/I18nProvider'
import { IncidentModal } from './IncidentModal'

type IncidentPeriod = '24h' | '7d' | '30d' | 'all'

const PERIOD_SECONDS: Record<Exclude<IncidentPeriod, 'all'>, number> = {
  '24h': 24 * 60 * 60,
  '7d': 7 * 24 * 60 * 60,
  '30d': 30 * 24 * 60 * 60,
}

const REVIEW_ORDER: IncidentReviewState[] = ['active', 'needs_review', 'history']

function finiteNumber(value: unknown): number | null {
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? number : null
}

export function incidentReviewBounds(period: IncidentPeriod, nowMs = Date.now()): Record<string, number> {
  if (period === 'all') return {}
  return {
    from_ts: (nowMs - PERIOD_SECONDS[period] * 1000) / 1000,
    to_ts: nowMs / 1000,
  }
}

export function formatReviewDuration(value: unknown): string {
  const ms = finiteNumber(value)
  if (ms == null) return '—'
  const minutes = Math.floor(ms / 60_000)
  if (minutes < 1) return '<1m'
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  if (hours < 24) return remainingMinutes ? `${hours}h ${remainingMinutes}m` : `${hours}h`
  const days = Math.floor(hours / 24)
  const remainingHours = hours % 24
  return remainingHours ? `${days}d ${remainingHours}h` : `${days}d`
}

function formatTime(value: unknown, locale: string): string {
  const timestamp = Number(value)
  if (!Number.isFinite(timestamp) || timestamp <= 0) return 'unconfirmed'
  return new Date(timestamp).toLocaleString(locale, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function cleanState(value: unknown): string {
  return String(value || 'unknown').replace(/_/g, ' ')
}

function IncidentCard({
  incident,
  channels,
  locale,
  onOpen,
}: {
  incident: IncidentReviewRecord
  channels: Channel[]
  locale: string
  onOpen: () => void
}) {
  const channelNames = (incident.channels || []).map((channelId) => (
    channels.find((channel) => channel.id === Number(channelId))?.title || `#${channelId}`
  ))
  const detectionId = Number(incident.cover?.detection_id)
  const title = String(incident.title || 'Incident awaiting review')
  const summary = String(incident.summary || '').trim()
  const coverage = incident.coverage && typeof incident.coverage === 'object'
    ? incident.coverage as Record<string, unknown>
    : {}
  const coverageFraction = Number(coverage.covered_fraction_estimate)
  const follow = incident.follow && typeof incident.follow === 'object'
    ? incident.follow as Record<string, unknown>
    : {}

  return (
    <button className={`incident-review-card state-${incident.review_state}`} onClick={onOpen}>
      <div className="incident-review-cover">
        {Number.isInteger(detectionId) && detectionId > 0
          ? <img src={`/detections/thumbnail/${detectionId}`} alt="" loading="lazy" />
          : <div className="incident-review-no-cover"><IconPhotoOff size={21} /> No grounded cover</div>}
        <span className={`incident-review-severity severity-${String(incident.severity || 'info').toLowerCase()}`}>
          {cleanState(incident.severity)}
        </span>
        {follow.active === true && <span className="incident-review-follow"><IconEye size={12} /> follow</span>}
      </div>
      <div className="incident-review-card-body">
        <div className="incident-review-card-title">
          <strong>{title}</strong>
          <span>#{incident.incident_id}</span>
        </div>
        <p>{summary || 'No consolidated narrative yet. Open the report to inspect grounded evidence.'}</p>
        <div className="incident-review-axis" aria-label="Incident lifecycle">
          <span>{cleanState(incident.perception_state)}</span>
          <span>{cleanState(incident.risk_state)}</span>
          <span>{cleanState(incident.case_state)}</span>
          <span>{cleanState(incident.attention_state)}</span>
        </div>
        <dl className="incident-review-metrics">
          <div><dt>Observed</dt><dd>{formatReviewDuration(incident.observed_duration_ms)}</dd></div>
          <div><dt>Case age</dt><dd>{formatReviewDuration(incident.case_duration_ms)}</dd></div>
          <div><dt>Last evidence</dt><dd>{formatTime(incident.last_evidence_ms, locale)}</dd></div>
        </dl>
        <div className="incident-review-card-foot">
          <span>{channelNames.length ? channelNames.join(', ') : 'No channel'}</span>
          <span>{incident.evidence_count || 0} evidence</span>
          {incident.uncertainty_count > 0 && <span className="warning">{incident.uncertainty_count} uncertain</span>}
          {Number.isFinite(coverageFraction) && <span>{Math.round(coverageFraction * 100)}% coverage</span>}
        </div>
      </div>
    </button>
  )
}

export function IncidentReview({ channels, canExport, canManage }: { channels: Channel[]; canExport: boolean; canManage: boolean }) {
  const { locale, t } = useI18n()
  const [records, setRecords] = useState<IncidentReviewRecord[]>([])
  const [reviewState, setReviewState] = useState<IncidentReviewState>('active')
  const [channelId, setChannelId] = useState('all')
  const [period, setPeriod] = useState<IncidentPeriod>('30d')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const query: Record<string, unknown> = {
        ...incidentReviewBounds(period),
        limit: 500,
      }
      if (channelId !== 'all') query.channel_id = channelId
      const response = await incidentsApi.review(query)
      setRecords(Array.isArray(response.incidents) ? response.incidents : [])
    } catch (exception: any) {
      setError(exception?.message || 'Incident review is unavailable.')
    } finally {
      setLoading(false)
    }
  }, [channelId, period])

  useEffect(() => { void load() }, [load])

  const counts = useMemo(() => {
    const initial: Record<IncidentReviewState, number> = { active: 0, needs_review: 0, history: 0 }
    for (const record of records) {
      if (REVIEW_ORDER.includes(record.review_state)) initial[record.review_state] += 1
    }
    return initial
  }, [records])
  const visible = useMemo(
    () => records.filter((record) => record.review_state === reviewState),
    [records, reviewState],
  )
  const stateLabels: Record<IncidentReviewState, string> = {
    active: t('incident.active'),
    needs_review: t('incident.needsReview'),
    history: t('incident.history'),
  }

  return (
    <section className="incident-review-board">
      <header className="incident-review-toolbar">
        <div>
          <div className="mon-panel-title"><IconFileDescription size={14} /> {t('incident.review')}</div>
          <p>{t('incident.reviewHelp')}</p>
        </div>
        <div className="incident-review-filters">
          <label>
            {t('video.channel')}
            <Dropdown
              value={channelId}
              onChange={setChannelId}
              options={[
                { value: 'all', label: t('incident.allChannels') },
                ...channels.map((channel) => ({ value: String(channel.id), label: channel.title })),
              ]}
            />
          </label>
          <label>
            {t('video.period')}
            <Dropdown
              value={period}
              onChange={(value) => setPeriod(value as IncidentPeriod)}
              options={[
                { value: '24h', label: t('incident.last24h') },
                { value: '7d', label: t('period.last7d') },
                { value: '30d', label: t('period.last30d') },
                { value: 'all', label: t('incident.allTime') },
              ]}
            />
          </label>
          <button className="mon-btn" onClick={() => void load()} disabled={loading}>
            <IconRefresh size={14} /> {loading ? t('status.checking') : t('video.refresh')}
          </button>
        </div>
      </header>

      <nav className="incident-review-queues" aria-label="Incident queues">
        {REVIEW_ORDER.map((state) => (
          <button key={state} className={reviewState === state ? 'on' : ''} onClick={() => setReviewState(state)}>
            {stateLabels[state]} <b>{counts[state]}</b>
          </button>
        ))}
      </nav>

      {error && <div className="chat-error"><IconAlertTriangle size={14} /> {error}</div>}
      {!error && loading && records.length === 0 && (
        <div className="incident-review-empty"><IconClock size={22} /> {t('incident.loading')}</div>
      )}
      {!error && !loading && visible.length === 0 && (
        <div className="incident-review-empty"><IconFileDescription size={22} /> {t('incident.empty')}</div>
      )}
      {visible.length > 0 && (
        <div className="incident-review-grid">
          {visible.map((incident) => (
            <IncidentCard
              key={incident.incident_id}
              incident={incident}
              channels={channels}
              locale={locale}
              onOpen={() => setSelectedId(incident.incident_id)}
            />
          ))}
        </div>
      )}

      {selectedId && (
        <IncidentModal
          incidentIdValue={selectedId}
          canExport={canExport}
          canManage={canManage}
          onChanged={() => void load()}
          onClose={() => setSelectedId(null)}
        />
      )}
    </section>
  )
}
