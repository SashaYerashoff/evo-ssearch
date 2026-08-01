import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  IconAlertTriangle,
  IconClock,
  IconDownload,
  IconEye,
  IconFileDescription,
  IconPlayerStop,
  IconRefresh,
  IconX,
} from '@tabler/icons-react'
import {
  incidentExportUrl,
  incidentId,
  incidentsApi,
  type Incident,
  type IncidentDraftInput,
  type IncidentFollowMode,
  type IncidentTimelineEntry,
} from '../../api/incidents'
import {
  followExpiryMs,
  formatIncidentDuration,
  incidentChannels,
  incidentFollowState,
  incidentTimeline,
  incidentTimestampMs,
} from './incidentView'

function fmtTime(value: unknown): string {
  const timestamp = incidentTimestampMs(value)
  return timestamp
    ? new Date(timestamp).toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    : 'Unconfirmed'
}

function signalDigest(entry: IncidentTimelineEntry | undefined): string[] {
  if (!entry) return []
  const values: string[] = []
  for (const [key, label] of [['p', 'P'], ['n', 'N'], ['m', 'M'], ['motion', 'motion'], ['novelty', 'novelty']] as const) {
    const value = entry[key]
    if (typeof value === 'number' && Number.isFinite(value)) values.push(`${label} ${value.toFixed(2)}`)
  }
  return values
}

export function IncidentModal({
  draftInput,
  canExport,
  onClose,
}: {
  draftInput: IncidentDraftInput
  canExport: boolean
  onClose: () => void
}) {
  const [incident, setIncident] = useState<Incident | null>(null)
  const [loading, setLoading] = useState(true)
  const [busyAction, setBusyAction] = useState<'refresh' | 'follow' | 'stop' | null>(null)
  const [error, setError] = useState('')
  const [followMode, setFollowMode] = useState<IncidentFollowMode>('follow')
  const [ttlSeconds, setTtlSeconds] = useState(15 * 60)
  const [localExpiryMs, setLocalExpiryMs] = useState<number | null>(null)
  const [nowMs, setNowMs] = useState(Date.now())

  const id = incidentId(incident)
  const timeline = useMemo(() => incidentTimeline(incident), [incident])
  const channels = useMemo(() => incidentChannels(incident), [incident])
  const follow = incidentFollowState(incident)
  const expiryMs = followExpiryMs(follow) || localExpiryMs
  const followActive = (follow.active === true && (!expiryMs || expiryMs > nowMs))
    || (follow.active == null && !!follow.mode && (!expiryMs || expiryMs > nowMs))
    || (!!localExpiryMs && localExpiryMs > nowMs)
  const remainingMs = followActive && expiryMs ? Math.max(0, expiryMs - nowMs) : null

  const replaceIncident = useCallback((next: Incident) => {
    setIncident(next)
    const backendExpiry = followExpiryMs(incidentFollowState(next))
    if (backendExpiry) setLocalExpiryMs(null)
  }, [])

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError('')
    incidentsApi.draft(draftInput)
      .then((next) => { if (alive) replaceIncident(next) })
      .catch((exception: any) => {
        if (alive) setError(exception?.message || 'EVA could not draft this incident.')
      })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [draftInput.anchor_detection_id, draftInput.channel_id, draftInput.from_ts, draftInput.to_ts, replaceIncident])

  useEffect(() => {
    if (!followActive || !expiryMs) return
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [expiryMs, followActive])

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])

  async function refreshIncident(action: 'refresh' | 'follow' | 'stop' = 'refresh') {
    if (!id) return
    setBusyAction(action)
    setError('')
    try {
      replaceIncident(await incidentsApi.get(id))
    } catch (exception: any) {
      setError(exception?.message || 'Incident refresh failed.')
    } finally {
      setBusyAction(null)
    }
  }

  async function retryDraft() {
    setLoading(true)
    setError('')
    try {
      replaceIncident(await incidentsApi.draft(draftInput))
    } catch (exception: any) {
      setError(exception?.message || 'EVA could not draft this incident.')
    } finally {
      setLoading(false)
    }
  }

  async function startFollow() {
    if (!id) return
    setBusyAction('follow')
    setError('')
    try {
      const updated = await incidentsApi.follow(id, followMode, ttlSeconds)
      setLocalExpiryMs(Date.now() + ttlSeconds * 1000)
      if (updated) replaceIncident(updated)
      else replaceIncident(await incidentsApi.get(id))
    } catch (exception: any) {
      setError(exception?.message || 'EVA could not raise incident attention.')
    } finally {
      setBusyAction(null)
    }
  }

  async function stopFollow() {
    if (!id) return
    setBusyAction('stop')
    setError('')
    try {
      const updated = await incidentsApi.stopFollow(id)
      setLocalExpiryMs(null)
      if (updated) replaceIncident(updated)
      else replaceIncident(await incidentsApi.get(id))
    } catch (exception: any) {
      setError(exception?.message || 'EVA could not stop incident follow.')
    } finally {
      setBusyAction(null)
    }
  }

  const bounds = incident?.time_bounds || {}
  const title = String(incident?.title || 'Incident reconstruction')
  const narrative = String(incident?.summary || incident?.description || '').trim()
  const state = String(incident?.state || incident?.status || 'draft').replace(/_/g, ' ')
  const semanticKeys = Array.isArray(incident?.semantic_keys) ? incident.semantic_keys : []
  const coverage = incident?.coverage && typeof incident.coverage === 'object'
    ? incident.coverage as Record<string, unknown>
    : {}
  const evidenceCount = Array.isArray(incident?.evidence) ? incident.evidence.length : 0
  const uncertainties = Array.isArray(incident?.uncertainties) ? incident.uncertainties : []
  const qualia = incident?.qualia_digest && typeof incident.qualia_digest === 'object'
    ? incident.qualia_digest as Record<string, unknown>
    : {}
  const coverageFraction = Number(coverage.covered_fraction_estimate)
  const rawTimeline = incident?.timeline?.length
    ? incident.timeline
    : incident?.events?.length
      ? incident.events
      : incident?.qualia_timeline || []

  return (
    <div className="scrim incident-scrim" onClick={(event) => { event.stopPropagation(); onClose() }}>
      <section
        className="modal incident-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="incident-modal-title"
        aria-describedby="incident-modal-status"
        aria-busy={loading || busyAction != null}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="modal-head">
          <div>
            <div className="modal-title" id="incident-modal-title"><IconFileDescription size={16} /> Incident report</div>
            <div className="incident-subtitle">Drafted from stored evidence · current observations remain operator-reviewed</div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close incident report" autoFocus>
            <IconX size={18} />
          </button>
        </header>

        <div className="incident-body">
          <div id="incident-modal-status" className="incident-live-status" role="status" aria-live="polite">
            {loading && 'Finding neighboring evidence and reconstructing incident boundaries…'}
            {!loading && busyAction === 'refresh' && 'Refreshing incident…'}
            {!loading && busyAction === 'follow' && 'Raising attention for this incident…'}
            {!loading && busyAction === 'stop' && 'Returning incident to normal observation…'}
          </div>
          {error && <div className="chat-error incident-error" role="alert"><IconAlertTriangle size={15} /> {error}</div>}

          {!loading && incident && (
            <>
              <div className="incident-heading">
                <div>
                  <h2>{title}</h2>
                  <div className="incident-meta">
                    <span className="incident-state">{state}</span>
                    {id && <span>#{id}</span>}
                    {channels.length > 0 && <span>channels {channels.map((channel) => `#${channel}`).join(', ')}</span>}
                  </div>
                </div>
                <button className="btn compact" onClick={() => refreshIncident()} disabled={busyAction != null}>
                  <IconRefresh size={14} /> Refresh
                </button>
              </div>

              <dl className="incident-bounds">
                <div><dt>Possible start</dt><dd>{fmtTime(bounds.possible_start)}</dd></div>
                <div><dt>Observed start</dt><dd>{fmtTime(bounds.observed_start)}</dd></div>
                <div><dt>Observed end</dt><dd>{bounds.observed_end == null ? 'Still open or unconfirmed' : fmtTime(bounds.observed_end)}</dd></div>
              </dl>

              {narrative && <div className="incident-narrative">{narrative}</div>}
              {semanticKeys.length > 0 && (
                <div className="incident-keys" aria-label="Incident semantic keys">
                  {semanticKeys.map((key) => <span key={key}>{String(key).replace(/_/g, ' ')}</span>)}
                </div>
              )}

              <section className="incident-grounding" aria-label="Incident grounding and coverage">
                <div>
                  <span>Evidence refs</span>
                  <strong>{evidenceCount}</strong>
                </div>
                <div>
                  <span>Coverage</span>
                  <strong>
                    {String(coverage.status || 'unknown')}
                    {Number.isFinite(coverageFraction) ? ` · ${Math.round(coverageFraction * 100)}%` : ''}
                  </strong>
                </div>
                <div>
                  <span>Attention digest</span>
                  <strong>
                    {Number(qualia.probe_count || 0)} probes · {Number(qualia.motion_interval_count || 0)} motion intervals
                  </strong>
                </div>
              </section>
              {uncertainties.length > 0 && (
                <section className="incident-uncertainties" aria-label="Incident uncertainties">
                  <strong><IconAlertTriangle size={14} /> Operator review required</strong>
                  <ul>{uncertainties.map((item, index) => <li key={`${index}-${String(item)}`}>{String(item)}</li>)}</ul>
                </section>
              )}

              <section className="incident-timeline" aria-labelledby="incident-timeline-title">
                <h3 id="incident-timeline-title">Evidence timeline</h3>
                {timeline.length === 0 ? (
                  <div className="incident-empty">No grounded timeline entries were returned. The draft remains incomplete.</div>
                ) : (
                  <ol>
                    {timeline.map((entry, index) => {
                      const signals = signalDigest(rawTimeline[index])
                      return (
                        <li key={entry.key}>
                          <time>{entry.timestampMs ? fmtTime(entry.timestampMs) : 'Time uncertain'}</time>
                          <div><strong>{entry.label}</strong>{entry.confidence && <i>{entry.confidence}</i>}</div>
                          {entry.description && <p>{entry.description}</p>}
                          {signals.length > 0 && <small>{signals.join(' · ')}</small>}
                        </li>
                      )
                    })}
                  </ol>
                )}
              </section>

              <section className={`incident-follow ${followActive ? 'active' : ''}`} aria-labelledby="incident-follow-title">
                <div>
                  <h3 id="incident-follow-title"><IconEye size={16} /> Follow incident</h3>
                  <p>
                    Raise frame density and carry the compact incident history into VLM batches.
                    Semantic archive indexing continues independently.
                  </p>
                </div>
                {followActive ? (
                  <div className="incident-follow-active">
                    <span><IconClock size={14} /> {String(follow.mode || followMode)}{remainingMs != null ? ` · ${formatIncidentDuration(remainingMs)} left` : ' · active'}</span>
                    <button className="btn danger" onClick={stopFollow} disabled={busyAction != null}>
                      <IconPlayerStop size={14} /> {busyAction === 'stop' ? 'Stopping…' : 'Stop follow'}
                    </button>
                  </div>
                ) : (
                  <div className="incident-follow-controls">
                    <label>
                      Attention
                      <select value={followMode} onChange={(event) => setFollowMode(event.target.value as IncidentFollowMode)}>
                        <option value="follow">Follow</option>
                        <option value="critical">Critical</option>
                      </select>
                    </label>
                    <label>
                      Duration
                      <select value={ttlSeconds} onChange={(event) => setTtlSeconds(Number(event.target.value))}>
                        <option value={300}>5 minutes</option>
                        <option value={900}>15 minutes</option>
                        <option value={1800}>30 minutes</option>
                        <option value={3600}>1 hour</option>
                      </select>
                    </label>
                    <button className="btn primary" onClick={startFollow} disabled={busyAction != null || !id}>
                      <IconEye size={14} /> {busyAction === 'follow' ? 'Starting…' : 'Start follow'}
                    </button>
                  </div>
                )}
              </section>

              {id && canExport && (
                <footer className="incident-export">
                  <span>Export evidence-backed draft</span>
                  <a className="btn compact" href={incidentExportUrl(id, 'md')} download><IconDownload size={14} /> Markdown</a>
                  <a className="btn compact" href={incidentExportUrl(id, 'xml')} download><IconDownload size={14} /> XML</a>
                </footer>
              )}
            </>
          )}

          {!loading && !incident && (
            <div className="incident-empty">
              <p>No incident draft is available. The source evidence has not been changed.</p>
              <button className="btn" onClick={retryDraft}><IconRefresh size={14} /> Retry draft</button>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
