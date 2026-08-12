import { api } from './client'

export type IncidentFollowMode = 'follow' | 'critical'
export type IncidentExportFormat = 'md' | 'xml'
export type IncidentReviewAction = 'confirm' | 'resolve' | 'dismiss' | 'false_positive' | 'reopen'

export interface IncidentDraftInput {
  channel_id: number
  anchor_detection_id?: number
  from_ts?: number
  to_ts?: number
}

export interface IncidentFollowInput {
  mode: IncidentFollowMode
  ttl_seconds: number
}

export interface IncidentReviewInput {
  action: IncidentReviewAction
  expected_revision?: number
  note?: string
}

export interface IncidentTimelineEntry {
  key?: string
  semantic_key?: string
  label?: string
  description?: string
  summary?: string
  timestamp_ms?: number
  occurred_at_ms?: number
  timestamp?: number | string
  confidence?: number | string
  [key: string]: unknown
}

export interface IncidentFollowState {
  active?: boolean
  mode?: IncidentFollowMode | string
  ttl_seconds?: number
  started_at_ms?: number
  expires_at_ms?: number
  expires_at?: number | string
  [key: string]: unknown
}

export interface Incident {
  id?: string | number
  incident_id?: string | number
  state?: string
  perception_state?: string
  risk_state?: string
  case_state?: string
  attention_state?: string
  status?: string
  title?: string
  summary?: string
  description?: string
  channel_id?: number
  channels?: Array<number | string | { channel_id?: number; id?: number }>
  semantic_keys?: string[]
  timeline?: IncidentTimelineEntry[]
  events?: IncidentTimelineEntry[]
  qualia_timeline?: IncidentTimelineEntry[]
  time_bounds?: {
    possible_start?: number | string
    observed_start?: number | string
    observed_end?: number | string | null
    [key: string]: unknown
  }
  follow?: IncidentFollowState
  follow_policy?: IncidentFollowState
  synopsis?: Record<string, unknown>
  homeostasis?: Record<string, unknown>
  key_moments?: IncidentTimelineEntry[]
  follow_result?: Record<string, unknown>
  coverage_gaps?: unknown[]
  uncertainties?: unknown[]
  [key: string]: unknown
}

export interface IncidentObservation {
  id: string
  incident_id: string
  idempotency_key: string
  source_kind: string
  observed_at_ms: number
  channel_id?: number | null
  perception_state: string
  source_ref?: Record<string, unknown>
  payload?: Record<string, unknown>
}

export interface IncidentListEnvelope {
  incidents: Incident[]
  total: number
  limit: number
  offset: number
  attention?: Record<string, unknown>
}

export type IncidentReviewState = 'active' | 'needs_review' | 'history'

export interface IncidentReviewRecord extends Incident {
  incident_id: string
  review_state: IncidentReviewState
  severity: string
  source?: string
  priority?: 'operator_criterion' | 'safety' | 'context' | string
  channels: number[]
  possible_start_ms?: number | null
  observed_start_ms?: number | null
  observed_end_ms?: number | null
  possible_end_ms?: number | null
  last_evidence_ms?: number | null
  observed_duration_ms?: number | null
  case_duration_ms?: number | null
  evidence_count: number
  timeline_count: number
  uncertainty_count: number
  cover?: {
    detection_id?: number | null
    timestamp_ms?: number | null
    role?: string
  } | null
}

export interface IncidentReviewEnvelope extends Omit<IncidentListEnvelope, 'incidents'> {
  view?: 'review'
  incidents: IncidentReviewRecord[]
}

export interface IncidentObservationEnvelope {
  observations: IncidentObservation[]
  total: number
  limit: number
  offset: number
}

export interface IncidentTemporalEpisode {
  id: string
  episode_key: string
  perception_state: string
  semantic_key?: string | null
  entity_key?: string | null
  zone_key?: string | null
  possible_start_ms?: number | null
  observed_start_ms?: number | null
  observed_end_ms?: number | null
  possible_end_ms?: number | null
  scale_disposition: string
  operator_review_required: boolean
  nested_context: boolean
  composition_parent: boolean
  source_level?: string | null
  composition_id?: string | null
  automatic_merge: boolean
  evidence_count: number
}

export interface IncidentSeriesLink {
  relation_id: string
  relation_state: 'candidate' | 'confirmed' | string
  confidence: string
  related_incident_id: string
  direction: 'prior' | 'later' | string
  series_key: string
  semantic_key: string
  gap_ms: number
  automatic_merge: false
  operator_review_required: boolean
  rationale: string
}

export interface IncidentNestedLink {
  relation_id: string
  relation_state: 'candidate' | 'confirmed' | string
  confidence: string
  related_incident_id: string
  direction: 'child' | 'parent' | string
  title: string
  semantic_key: string
  possible_start_ms?: number | null
  possible_end_ms?: number | null
  scale_disposition: string
  presentation_scope: 'nested' | string
  automatic_merge: false
  operator_review_required: boolean
  rationale: string
}

export interface IncidentLifecycleTransition {
  id: string
  axis: 'perception' | 'risk' | 'case' | 'attention' | 'legacy' | string
  from_state?: string | null
  to_state: string
  incident_revision?: number | null
  transitioned_at_ms?: number | null
  reason: string
  source_kind: string
}

export interface IncidentTemporalContext {
  supported: boolean
  incident_id: string
  episodes: IncidentTemporalEpisode[]
  episode_total: number
  series_links: IncidentSeriesLink[]
  nested_incidents: IncidentNestedLink[]
  relation_total: number
  correction_count: number
  lifecycle_history: IncidentLifecycleTransition[]
  transition_total: number
}

export type IncidentSeriesReviewAction = 'confirm' | 'reject'

export interface IncidentSeriesReviewEnvelope {
  success: boolean
  relation: Record<string, unknown>
  temporal: IncidentTemporalContext
}

interface IncidentEnvelope {
  incident?: Incident
  [key: string]: unknown
}

function incidentFromResponse(response: IncidentEnvelope | Incident): Incident | null {
  if (!response || typeof response !== 'object') return null
  if ('incident' in response) {
    return response.incident && typeof response.incident === 'object'
      ? response.incident as Incident
      : null
  }
  return ('incident_id' in response || 'id' in response) ? response as Incident : null
}

function requireIncident(response: IncidentEnvelope | Incident): Incident {
  const candidate = incidentFromResponse(response)
  if (!candidate) throw new Error('Incident response did not contain an incident.')
  return candidate
}

export function incidentId(incident: Incident | null | undefined): string {
  const value = incident?.incident_id ?? incident?.id ?? ''
  return String(value || '').trim()
}

export function normalizeIncidentDraftInput(input: IncidentDraftInput): IncidentDraftInput {
  const channelId = Number(input.channel_id)
  if (!Number.isInteger(channelId) || channelId <= 0) {
    throw new Error('A valid incident channel is required.')
  }
  const anchor = Number(input.anchor_detection_id)
  const fromTs = Number(input.from_ts)
  const toTs = Number(input.to_ts)
  const normalized: IncidentDraftInput = { channel_id: channelId }
  if (Number.isInteger(anchor) && anchor > 0) normalized.anchor_detection_id = anchor
  if (Number.isFinite(fromTs) && fromTs > 0) normalized.from_ts = fromTs
  if (Number.isFinite(toTs) && toTs > 0) normalized.to_ts = toTs
  if (normalized.from_ts && normalized.to_ts && normalized.to_ts < normalized.from_ts) {
    throw new Error('Incident end time cannot precede its start time.')
  }
  return normalized
}

export function incidentExportUrl(id: string | number, format: IncidentExportFormat): string {
  const normalizedId = String(id || '').trim()
  if (!normalizedId) return ''
  return `/incidents/${encodeURIComponent(normalizedId)}/export?format=${format}`
}

export function incidentPath(id: string | number, suffix = ''): string {
  const normalizedId = String(id || '').trim()
  if (!normalizedId) throw new Error('A valid incident id is required.')
  return `/incidents/${encodeURIComponent(normalizedId)}${suffix}`
}

export function normalizeIncidentFollowInput(
  mode: IncidentFollowMode,
  ttlSeconds: number,
): IncidentFollowInput {
  if (!['follow', 'critical'].includes(mode)) throw new Error('Unsupported incident follow mode.')
  const ttl = Number(ttlSeconds)
  if (!Number.isFinite(ttl) || ttl <= 0) throw new Error('A valid incident follow TTL is required.')
  return { mode, ttl_seconds: Math.max(60, Math.round(ttl)) }
}

const inFlightDrafts = new Map<string, Promise<Incident>>()

function draftIncident(input: IncidentDraftInput): Promise<Incident> {
  const normalized = normalizeIncidentDraftInput(input)
  const key = JSON.stringify(normalized)
  const existing = inFlightDrafts.get(key)
  if (existing) return existing
  let request: Promise<Incident>
  request = api.postJson('/incidents/draft', normalized)
    .then(requireIncident)
    .finally(() => {
      if (inFlightDrafts.get(key) === request) inFlightDrafts.delete(key)
    })
  inFlightDrafts.set(key, request)
  return request
}

export const incidentsApi = {
  draft: draftIncident,
  get: async (id: string | number): Promise<Incident> =>
    requireIncident(await api.get(incidentPath(id))),
  list: async (query: Record<string, unknown> = {}): Promise<IncidentListEnvelope> =>
    api.get('/incidents', query),
  review: async (query: Record<string, unknown> = {}): Promise<IncidentReviewEnvelope> =>
    api.get('/incidents', { ...query, view: 'review' }),
  observations: async (
    id: string | number,
    query: Record<string, unknown> = {},
  ): Promise<IncidentObservationEnvelope> =>
    api.get(incidentPath(id, '/observations'), query),
  temporal: async (id: string | number): Promise<IncidentTemporalContext> =>
    api.get(incidentPath(id, '/temporal')),
  follow: async (
    id: string | number,
    mode: IncidentFollowMode,
    ttlSeconds: number,
  ): Promise<Incident | null> => incidentFromResponse(await api.postJson(
    incidentPath(id, '/follow'),
    normalizeIncidentFollowInput(mode, ttlSeconds),
  )),
  stopFollow: async (id: string | number): Promise<Incident | null> =>
    incidentFromResponse(await api.postJson(
      incidentPath(id, '/stop-follow'),
      {},
    )),
  reviewIncident: async (
    id: string | number,
    input: IncidentReviewInput,
  ): Promise<Incident> => requireIncident(await api.postJson(
    incidentPath(id, '/review'),
    input,
  )),
  reviewSeries: async (
    id: string | number,
    relationId: string,
    action: IncidentSeriesReviewAction,
    note = '',
  ): Promise<IncidentSeriesReviewEnvelope> => api.postJson(
    incidentPath(
      id,
      `/series/${encodeURIComponent(String(relationId || '').trim())}/review`,
    ),
    { action, ...(note.trim() ? { note: note.trim() } : {}) },
  ),
}
