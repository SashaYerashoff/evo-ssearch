import type { Incident, IncidentFollowState, IncidentTimelineEntry } from '../../api/incidents'

export interface IncidentTimelineView {
  key: string
  label: string
  description: string
  timestampMs: number | null
  confidence: string
}

export function incidentTimestampMs(value: unknown): number | null {
  if (typeof value === 'string' && value.trim() && !/^\d+(?:\.\d+)?$/.test(value.trim())) {
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  const number = Number(value)
  if (!Number.isFinite(number) || number <= 0) return null
  return number > 1e12 ? number : number * 1000
}

function timelineTimestamp(entry: IncidentTimelineEntry): number | null {
  return incidentTimestampMs(entry.timestamp_ms ?? entry.occurred_at_ms ?? entry.timestamp)
}

export function incidentTimeline(incident: Incident | null | undefined): IncidentTimelineView[] {
  const source = incident?.timeline?.length
    ? incident.timeline
    : incident?.events?.length
      ? incident.events
      : incident?.qualia_timeline || []
  return source.map((entry, index) => ({
    key: String(entry.key || entry.semantic_key || `${timelineTimestamp(entry) || 'unknown'}-${index}`),
    label: String(entry.label || entry.semantic_key || entry.key || `Event ${index + 1}`).replace(/_/g, ' '),
    description: String(entry.description || entry.summary || ''),
    timestampMs: timelineTimestamp(entry),
    confidence: entry.confidence == null ? '' : String(entry.confidence),
  }))
}

export function incidentChannels(incident: Incident | null | undefined): string[] {
  const channels = Array.isArray(incident?.channels) ? incident.channels : []
  const values = channels.map((entry) => {
    if (entry && typeof entry === 'object') return entry.channel_id ?? entry.id
    return entry
  })
  if (incident?.channel_id != null) values.unshift(incident.channel_id)
  return [...new Set(values.map((value) => String(value || '').trim()).filter(Boolean))]
}

export function incidentFollowState(incident: Incident | null | undefined): IncidentFollowState {
  return incident?.follow || incident?.follow_policy || {}
}

export function followExpiryMs(follow: IncidentFollowState): number | null {
  return incidentTimestampMs(follow.expires_at_ms ?? follow.expires_at)
}

export function formatIncidentDuration(ms: number): string {
  const seconds = Math.max(0, Math.ceil(ms / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`
}
