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
  type IncidentObservation,
  type IncidentReviewAction,
  type IncidentSeriesReviewAction,
  type IncidentTemporalContext,
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

function temporalDisposition(value: unknown): { label: string; description: string } {
  switch (String(value || '').trim().toLowerCase()) {
    case 'long_incident_candidate':
      return {
        label: 'Long incident candidate',
        description: 'Activity spans a 15-minute boundary and remains one operator-reviewed incident candidate.',
      }
    case 'continuing_incident':
      return {
        label: 'Continuing incident',
        description: 'No grounded return to routine has closed this episode yet.',
      }
    case 'short_incident':
      return {
        label: 'Short incident',
        description: 'A bounded episode was observed between routine intervals.',
      }
    default:
      return {
        label: 'Episode kept for review',
        description: 'The episode is preserved until routine boundaries and evidence are sufficient to classify it.',
      }
  }
}

export function IncidentModal({
  draftInput,
  incidentIdValue,
  canExport,
  canManage = false,
  onChanged,
  onClose,
}: {
  draftInput?: IncidentDraftInput
  incidentIdValue?: string
  canExport: boolean
  canManage?: boolean
  onChanged?: (incident: Incident) => void
  onClose: () => void
}) {
  const [incident, setIncident] = useState<Incident | null>(null)
  const [loading, setLoading] = useState(true)
  const [busyAction, setBusyAction] = useState<'refresh' | 'follow' | 'stop' | `review:${IncidentReviewAction}` | `series:${string}` | null>(null)
  const [error, setError] = useState('')
  const [followMode, setFollowMode] = useState<IncidentFollowMode>('follow')
  const [ttlSeconds, setTtlSeconds] = useState(15 * 60)
  const [localExpiryMs, setLocalExpiryMs] = useState<number | null>(null)
  const [nowMs, setNowMs] = useState(Date.now())
  const [observations, setObservations] = useState<IncidentObservation[]>([])
  const [observationTotal, setObservationTotal] = useState(0)
  const [observationsLoading, setObservationsLoading] = useState(false)
  const [observationError, setObservationError] = useState('')
  const [temporal, setTemporal] = useState<IncidentTemporalContext | null>(null)
  const [temporalLoading, setTemporalLoading] = useState(false)
  const [temporalError, setTemporalError] = useState('')
  const [reviewNote, setReviewNote] = useState('')

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

  const loadObservations = useCallback(async (incidentIdValue: string) => {
    if (!incidentIdValue) return
    setObservationsLoading(true)
    setObservationError('')
    try {
      const page = await incidentsApi.observations(incidentIdValue, { limit: 250 })
      setObservations(Array.isArray(page.observations) ? page.observations : [])
      setObservationTotal(Number(page.total || 0))
    } catch (exception: any) {
      setObservationError(exception?.message || 'Incident observation ledger is unavailable.')
    } finally {
      setObservationsLoading(false)
    }
  }, [])

  const loadTemporalContext = useCallback(async (incidentIdValue: string) => {
    if (!incidentIdValue) return
    setTemporalLoading(true)
    setTemporalError('')
    try {
      setTemporal(await incidentsApi.temporal(incidentIdValue))
    } catch (exception: any) {
      setTemporal(null)
      setTemporalError(exception?.message || 'Temporal incident memory is unavailable.')
    } finally {
      setTemporalLoading(false)
    }
  }, [])

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError('')
    const request = incidentIdValue
      ? incidentsApi.get(incidentIdValue)
      : draftInput
        ? incidentsApi.draft(draftInput)
        : Promise.reject(new Error('Incident source is missing.'))
    request
      .then((next) => { if (alive) replaceIncident(next) })
      .catch((exception: any) => {
        if (alive) setError(exception?.message || 'EVA could not load this incident.')
      })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [
    draftInput?.anchor_detection_id,
    draftInput?.channel_id,
    draftInput?.from_ts,
    draftInput?.to_ts,
    incidentIdValue,
    replaceIncident,
  ])

  useEffect(() => {
    if (!id) return
    void loadObservations(id)
    void loadTemporalContext(id)
  }, [id, loadObservations, loadTemporalContext])

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
      const updated = await incidentsApi.get(id)
      replaceIncident(updated)
      await Promise.all([loadObservations(id), loadTemporalContext(id)])
      onChanged?.(updated)
    } catch (exception: any) {
      setError(exception?.message || 'Incident refresh failed.')
    } finally {
      setBusyAction(null)
    }
  }

  async function retryLoad() {
    setLoading(true)
    setError('')
    try {
      if (incidentIdValue) replaceIncident(await incidentsApi.get(incidentIdValue))
      else if (draftInput) replaceIncident(await incidentsApi.draft(draftInput))
      else throw new Error('Incident source is missing.')
    } catch (exception: any) {
      setError(exception?.message || 'EVA could not load this incident.')
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
      const next = updated || await incidentsApi.get(id)
      replaceIncident(next)
      onChanged?.(next)
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
      const next = updated || await incidentsApi.get(id)
      replaceIncident(next)
      onChanged?.(next)
    } catch (exception: any) {
      setError(exception?.message || 'EVA could not stop incident follow.')
    } finally {
      setBusyAction(null)
    }
  }

  async function applyReview(action: IncidentReviewAction) {
    if (!id || !incident) return
    setBusyAction(`review:${action}`)
    setError('')
    try {
      const revision = Number(incident.revision)
      const updated = await incidentsApi.reviewIncident(id, {
        action,
        ...(Number.isInteger(revision) && revision > 0 ? { expected_revision: revision } : {}),
        ...(reviewNote.trim() ? { note: reviewNote.trim() } : {}),
      })
      setReviewNote('')
      replaceIncident(updated)
      await Promise.all([loadObservations(id), loadTemporalContext(id)])
      onChanged?.(updated)
    } catch (exception: any) {
      const message = exception?.message || 'Incident review could not be saved.'
      setError(message.includes('revision')
        ? 'This incident changed while it was open. Refresh it and review the latest state.'
        : message)
    } finally {
      setBusyAction(null)
    }
  }

  async function applySeriesReview(
    relationId: string,
    action: IncidentSeriesReviewAction,
  ) {
    if (!id || !relationId) return
    setBusyAction(`series:${relationId}:${action}`)
    setError('')
    try {
      const result = await incidentsApi.reviewSeries(
        id,
        relationId,
        action,
        reviewNote,
      )
      setTemporal(result.temporal)
      setReviewNote('')
    } catch (exception: any) {
      setError(exception?.message || 'Recurrence-series review could not be saved.')
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
  const synopsis = incident?.synopsis && typeof incident.synopsis === 'object'
    ? incident.synopsis as Record<string, unknown>
    : {}
  const homeostasis = incident?.homeostasis && typeof incident.homeostasis === 'object'
    ? incident.homeostasis as Record<string, unknown>
    : {}
  const followResult = incident?.follow_result && typeof incident.follow_result === 'object'
    ? incident.follow_result as Record<string, unknown>
    : {}
  const keyMoments = Array.isArray(incident?.key_moments) ? incident.key_moments.slice(0, 5) : []
  const outcome = String(synopsis.outcome || followResult.outcome || 'awaiting_review').replace(/_/g, ' ')
  const followHomeostasis = followResult.homeostasis && typeof followResult.homeostasis === 'object'
    ? followResult.homeostasis as Record<string, unknown>
    : {}
  const elevatedDurationMs = Number(homeostasis.elevated_duration_ms || 0)
  const settlingMs = Number(homeostasis.settling_ms || 0)
  const coverageFraction = Number(coverage.covered_fraction_estimate)
  const rawTimeline = incident?.timeline?.length
    ? incident.timeline
    : incident?.events?.length
      ? incident.events
      : incident?.qualia_timeline || []
  const lifecycle = [
    ['Perception', incident?.perception_state],
    ['Risk', incident?.risk_state],
    ['Case', incident?.case_state],
    ['Attention', incident?.attention_state],
  ]
  const visibleObservations = observations.slice(-8).reverse()
  const temporalEpisodes = temporal?.episodes || []
  const primaryEpisode = temporalEpisodes.find((episode) => episode.composition_parent)
    || temporalEpisodes.find((episode) => !episode.nested_context)
    || temporalEpisodes[0]
  const nestedEpisodes = temporalEpisodes
    .filter((episode) => episode !== primaryEpisode && episode.nested_context)
    .slice(0, 8)
  const disposition = temporalDisposition(primaryEpisode?.scale_disposition)
  const seriesLinks = temporal?.series_links || []
  const lifecycleHistory = temporal?.lifecycle_history || []
  const normalizedCaseState = String(incident?.case_state || 'candidate').toLowerCase()
  const historicalCase = ['closed', 'dismissed', 'false_positive'].includes(normalizedCaseState)

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
            <div className="incident-subtitle">
              {incidentIdValue ? 'Stored incident · current observations remain operator-reviewed' : 'Drafted from stored evidence · current observations remain operator-reviewed'}
            </div>
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
              <section className="incident-human-state" aria-label="Incident operator synopsis">
                <div><span>Outcome</span><strong>{outcome}</strong></div>
                <div><span>Confidence</span><strong>{String(synopsis.confidence || 'unknown')}</strong></div>
                <div><span>Coverage</span><strong>{String(coverage.status || 'unknown')}{Number.isFinite(coverageFraction) ? ` · ${Math.round(coverageFraction * 100)}%` : ''}</strong></div>
              </section>
              <section className="incident-homeostasis" aria-label="Homeostatic response">
                <div className="incident-section-title">Homeostatic response <small>attention signals, not visual proof</small></div>
                <dl>
                  <div><dt>Activity apex</dt><dd>{Number(homeostasis.activity_x_max || 0).toFixed(1)}×</dd></div>
                  <div><dt>Elevated</dt><dd>{elevatedDurationMs > 0 ? formatIncidentDuration(elevatedDurationMs) : '—'}</dd></div>
                  <div><dt>Settling</dt><dd>{settlingMs > 0 ? formatIncidentDuration(settlingMs) : '—'}</dd></div>
                  <div><dt>Bursts</dt><dd>{Number(homeostasis.burst_count || 0)}</dd></div>
                  <div><dt>Probe hits</dt><dd>{Number(homeostasis.probe_hits || 0)} / {Number(homeostasis.probe_samples || 0)}</dd></div>
                </dl>
              </section>
              {Object.keys(followResult).length > 0 && (
                <section className={`incident-follow-result outcome-${String(followResult.outcome || 'inconclusive')}`}>
                  <div className="incident-section-title">Last Follow result</div>
                  <strong>{String(followResult.outcome || 'inconclusive').replace(/_/g, ' ')}</strong>
                  <p>{String(followResult.description || '')}</p>
                  <small>
                    {Number(followResult.observation_count || 0)} L0 observations · {String(followResult.stop_reason || 'completed').replace(/_/g, ' ')}
                    {Number(followHomeostasis.sample_count || 0) > 0
                      ? ` · activity apex ${Number(followHomeostasis.activity_x_max || 0).toFixed(1)}× · ${Number(followHomeostasis.burst_count || 0)} bursts`
                      : ''}
                  </small>
                </section>
              )}
              {keyMoments.length > 0 && (
                <section className="incident-key-moments">
                  <div className="incident-section-title">Key moments</div>
                  <ol>
                    {keyMoments.map((moment, index) => (
                      <li key={`${String(moment.semantic_key || 'moment')}-${index}`}>
                        <time>{fmtTime(moment.timestamp_ms || moment.occurred_at_ms)}</time>
                        <span>{String(moment.label || moment.summary || moment.semantic_key || 'Observed transition')}</span>
                      </li>
                    ))}
                  </ol>
                </section>
              )}
              {(temporalLoading || temporalError || (temporal?.supported && primaryEpisode)) && (
                <section className="incident-temporal-memory" aria-label="Temporal incident memory">
                  <div className="incident-section-title">Temporal memory <small>routine-separated, operator-reviewed</small></div>
                  {temporalLoading && <p className="incident-temporal-status">Loading episode boundaries…</p>}
                  {temporalError && <p className="incident-temporal-status warning">{temporalError}</p>}
                  {!temporalLoading && !temporalError && primaryEpisode && (
                    <>
                      <div className="incident-temporal-disposition">
                        <strong>{disposition.label}</strong>
                        <span>{disposition.description}</span>
                        {primaryEpisode.semantic_key && <i>{primaryEpisode.semantic_key.replace(/_/g, ' ')}</i>}
                      </div>
                      {nestedEpisodes.length > 0 && (
                        <div className="incident-nested-episodes">
                          <span>Nested episode sequence</span>
                          <ol>
                            {nestedEpisodes.map((episode) => {
                              const nestedDisposition = temporalDisposition(episode.scale_disposition)
                              return (
                                <li key={episode.id || episode.episode_key}>
                                  <strong>{String(episode.semantic_key || 'Observed context').replace(/_/g, ' ')}</strong>
                                  <small>
                                    {fmtTime(episode.observed_start_ms || episode.possible_start_ms)}
                                    {' · '}{nestedDisposition.label}
                                    {episode.source_level ? ` · ${episode.source_level}` : ''}
                                  </small>
                                </li>
                              )
                            })}
                          </ol>
                          <small>Context is attached to the grounded parent; no incidents were merged automatically.</small>
                        </div>
                      )}
                      {seriesLinks.length > 0 && (
                        <div className="incident-series-links">
                          <span>Possible recurrence series</span>
                          <ul>
                            {seriesLinks.slice(0, 4).map((link) => (
                              <li key={link.relation_id}>
                                <strong>{link.direction === 'prior' ? 'Earlier' : 'Later'} incident #{link.related_incident_id.slice(0, 8)}</strong>
                                <span>{link.semantic_key.replace(/_/g, ' ')} · {formatIncidentDuration(link.gap_ms)} gap · {link.confidence} confidence</span>
                                {link.relation_state === 'candidate' ? (
                                  <div className="incident-series-review-actions">
                                    <small>Candidate only — incidents remain separate until operator review.</small>
                                    {canManage && (
                                      <span>
                                        <button
                                          className="btn compact"
                                          onClick={() => void applySeriesReview(link.relation_id, 'confirm')}
                                          disabled={busyAction != null}
                                        >Confirm series</button>
                                        <button
                                          className="btn compact"
                                          onClick={() => void applySeriesReview(link.relation_id, 'reject')}
                                          disabled={busyAction != null}
                                        >Reject link</button>
                                      </span>
                                    )}
                                  </div>
                                ) : (
                                  <small>Operator-confirmed recurrence series; incidents remain separate.</small>
                                )}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                </section>
              )}
              {semanticKeys.length > 0 && (
                <div className="incident-keys" aria-label="Incident semantic keys">
                  {semanticKeys.map((key) => <span key={key}>{String(key).replace(/_/g, ' ')}</span>)}
                </div>
              )}

              <details className="incident-technical">
                <summary>Technical evidence · {evidenceCount} references · {observationTotal} heartbeats</summary>
                <div className="incident-technical-body">
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
              <section className="incident-lifecycle" aria-label="Independent incident lifecycle axes">
                {lifecycle.map(([label, value]) => (
                  <div key={label}>
                    <span>{label}</span>
                    <strong>{String(value || 'unknown').replace(/_/g, ' ')}</strong>
                  </div>
                ))}
              </section>
              {lifecycleHistory.length > 0 && (
                <section className="incident-lifecycle-history" aria-labelledby="incident-lifecycle-history-title">
                  <div className="incident-observations-head">
                    <h3 id="incident-lifecycle-history-title">Lifecycle history</h3>
                    <span>{temporal?.transition_total || lifecycleHistory.length} immutable transitions</span>
                  </div>
                  <ol>
                    {lifecycleHistory.slice(-12).reverse().map((transition) => (
                      <li key={transition.id}>
                        <time>{fmtTime(transition.transitioned_at_ms)}</time>
                        <strong>{String(transition.axis || 'state').replace(/_/g, ' ')}</strong>
                        <span>
                          {String(transition.from_state || 'unset').replace(/_/g, ' ')} → {String(transition.to_state || 'unknown').replace(/_/g, ' ')}
                        </span>
                        <small>{String(transition.reason || transition.source_kind || '').replace(/_/g, ' ')}</small>
                      </li>
                    ))}
                  </ol>
                </section>
              )}
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

              <section className="incident-observations" aria-labelledby="incident-observations-title">
                <div className="incident-observations-head">
                  <h3 id="incident-observations-title">Observation ledger</h3>
                  <span>{observationTotal} immutable heartbeats</span>
                </div>
                {observationsLoading && <div className="incident-observation-status">Loading durable observations…</div>}
                {observationError && <div className="incident-observation-status warning">{observationError}</div>}
                {!observationsLoading && !observationError && visibleObservations.length === 0 && (
                  <div className="incident-observation-status">No L0 heartbeat has been appended yet.</div>
                )}
                {visibleObservations.length > 0 && (
                  <ol>
                    {visibleObservations.map((observation) => (
                      <li key={observation.id}>
                        <time>{fmtTime(observation.observed_at_ms)}</time>
                        <strong>{String(observation.source_kind || 'observation').replace(/_/g, ' ')}</strong>
                        <span>
                          {String(observation.payload?.association || observation.perception_state || 'unknown').replace(/_/g, ' ')} · channel #{observation.channel_id || '?'}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
              </section>
                </div>
              </details>

              {id && canManage && (
                <section className="incident-operator-review" aria-labelledby="incident-operator-review-title">
                  <div>
                    <h3 id="incident-operator-review-title">Operator review</h3>
                    <p>Lifecycle decisions are audited. They do not rewrite visual evidence or merge related incidents.</p>
                  </div>
                  <label>
                    Optional note
                    <textarea
                      value={reviewNote}
                      onChange={(event) => setReviewNote(event.target.value.slice(0, 1000))}
                      placeholder="Why this incident is being confirmed, closed, or dismissed…"
                      rows={2}
                    />
                  </label>
                  <div className="incident-review-actions">
                    {historicalCase ? (
                      <button className="btn primary" onClick={() => void applyReview('reopen')} disabled={busyAction != null}>
                        {busyAction === 'review:reopen' ? 'Reopening…' : 'Reopen incident'}
                      </button>
                    ) : (
                      <>
                        {['candidate', 'unknown'].includes(normalizedCaseState) && (
                          <button className="btn primary" onClick={() => void applyReview('confirm')} disabled={busyAction != null}>
                            {busyAction === 'review:confirm' ? 'Confirming…' : 'Confirm incident'}
                          </button>
                        )}
                        <button className="btn" onClick={() => void applyReview('resolve')} disabled={busyAction != null}>
                          {busyAction === 'review:resolve' ? 'Closing…' : 'Resolve & close'}
                        </button>
                        <button className="btn" onClick={() => void applyReview('dismiss')} disabled={busyAction != null}>
                          {busyAction === 'review:dismiss' ? 'Dismissing…' : 'Dismiss'}
                        </button>
                        <button className="btn danger" onClick={() => void applyReview('false_positive')} disabled={busyAction != null}>
                          {busyAction === 'review:false_positive' ? 'Saving…' : 'False positive'}
                        </button>
                      </>
                    )}
                  </div>
                </section>
              )}

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
                    <IconEye size={14} /> {busyAction === 'follow' ? 'Starting…' : (bounds.observed_end && Date.now() - Number(bounds.observed_end) > 120_000 ? 'Watch for recurrence' : 'Start follow')}
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
              <p>No incident report is available. The source evidence has not been changed.</p>
              <button className="btn" onClick={retryLoad}><IconRefresh size={14} /> Retry</button>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
