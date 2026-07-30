import type { SummaryEntry } from '../../api/video'

export const SUMMARY_SEVERITIES = ['critical', 'high', 'normal', 'low', 'info'] as const
export type SummarySeverity = (typeof SUMMARY_SEVERITIES)[number]

export interface SummaryBurst {
  count: number
  maxActivity: number | null
  snapshots: string[]
}

export interface SummarySemanticStatus {
  label: string
  tone: 'ready' | 'pending' | 'queued' | 'degraded'
  title: string
}

export interface SummaryTextParts {
  narrative: string
  machineJson: string
  marker: string
}

export function summaryEntryKey(entry: SummaryEntry, index: number): string {
  const values = [
    entry.rollup_id,
    entry.batch_id,
    entry.run_id,
    entry.level,
    entry.batch_start_ms,
    entry.window_start,
    entry.created_at,
  ].map((value) => String(value ?? '').trim()).filter(Boolean)
  return values.length ? values.join(':') : `summary-${index}`
}

export function summaryAlertCounts(entry: SummaryEntry): Partial<Record<SummarySeverity, number>> {
  const counts: Partial<Record<SummarySeverity, number>> = {}
  for (const severity of SUMMARY_SEVERITIES) {
    const value = Number(entry.alert_counts?.[severity] || 0)
    if (Number.isFinite(value) && value > 0) counts[severity] = Math.floor(value)
  }
  if (!Object.keys(counts).length) {
    const total = Number(entry.alert_total || 0)
    const rawSeverity = String(entry.severity || '').trim().toLowerCase()
    const severity = SUMMARY_SEVERITIES.includes(rawSeverity as SummarySeverity)
      ? rawSeverity as SummarySeverity
      : 'normal'
    if (Number.isFinite(total) && total > 0) counts[severity] = Math.floor(total)
  }
  return counts
}

export function summaryBurst(entry: SummaryEntry): SummaryBurst | null {
  const seconds = Array.isArray(entry.vector_signal?.capture_attention?.seconds)
    ? entry.vector_signal.capture_attention.seconds
    : []
  const bursts = seconds.filter((value: any) => (
    value && typeof value === 'object' && String(value.mode || '').trim().toLowerCase() === 'burst'
  ))
  const role = String(entry.thumbnail_role || entry.anchor_role || '').trim().toLowerCase()
  if (!bursts.length && !role.includes('burst')) return null
  const activity = bursts
    .map((value: any) => Number(value.activity_x))
    .filter((value: number) => Number.isFinite(value) && value >= 0)
  const snapshots = bursts
    .map((value: any) => String(value.snapshot ?? '').trim())
    .filter(Boolean)
  return {
    count: Math.max(1, bursts.length),
    maxActivity: activity.length ? Math.max(...activity) : null,
    snapshots,
  }
}

export function summarySemanticStatus(entry: SummaryEntry): SummarySemanticStatus | null {
  const kind = String(entry.summary_kind || '').trim().toLowerCase()
  const generation = String(entry.generation_status || kind).trim().toLowerCase()
  if (!kind && !generation) return null
  const ready = ['llm', 'llm_cached', 'legacy_cached'].includes(kind)
  const legacy = kind === 'legacy_cached'
  const refreshPending = generation === 'refresh_pending' || entry.semantic_refresh_pending === true
  const pending = kind === 'pending_context' || generation === 'pending'
  const queued = kind === 'queued' || generation === 'queued' || generation === 'deferred'
  if (ready) {
    const label = legacy
      ? 'semantic · legacy'
      : refreshPending
        ? 'semantic · refreshing'
        : kind === 'llm_cached'
          ? 'semantic · cached'
          : 'semantic'
    return {
      label,
      tone: 'ready',
      title: refreshPending
        ? 'The previous semantic narrative remains visible while retained observations are consolidated.'
        : legacy
          ? 'Imported semantic history; current observations still override it.'
          : 'Completed semantic consolidation.',
    }
  }
  if (pending) {
    return {
      label: 'aggregation pending',
      tone: 'pending',
      title: 'The source window has not closed yet.',
    }
  }
  if (queued) {
    return {
      label: 'semantic queued',
      tone: 'queued',
      title: 'Background semantic aggregation is queued behind live descriptions.',
    }
  }
  return {
    label: 'semantic retry available',
    tone: 'degraded',
    title: String(entry.generation_error || 'The semantic pass did not complete; source observations remain available.'),
  }
}

export function summaryLevel(entry: SummaryEntry, selectedDepth: string): string {
  const level = String(entry.level || selectedDepth || 'L0').trim().toUpperCase()
  return ['L0', 'L1', 'L2', 'L3'].includes(level) ? level : 'L0'
}

export function splitSummaryMachineJson(value: unknown): SummaryTextParts {
  const text = String(value || '').trim()
  const marker = /\bBATCH[\s_-]*STATE[\s_-]*JSON\s*:?\s*/i.exec(text)
  if (!marker || marker.index == null) return { narrative: text, machineJson: '', marker: '' }
  const jsonStart = text.indexOf('{', marker.index + marker[0].length)
  if (jsonStart < 0) return { narrative: text, machineJson: '', marker: '' }
  return {
    narrative: text.slice(0, marker.index).trim(),
    machineJson: text.slice(jsonStart).trim(),
    marker: marker[0].replace(/\s+/g, ' ').trim(),
  }
}
