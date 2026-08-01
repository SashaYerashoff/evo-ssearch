import { api } from './client'

export type IncidentFollowMode = 'follow' | 'critical'
export type IncidentExportFormat = 'md' | 'xml'

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
  coverage_gaps?: unknown[]
  uncertainties?: unknown[]
  [key: string]: unknown
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
}
