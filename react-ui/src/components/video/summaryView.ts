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

export interface SummaryNarrativeSections {
  scene: string
  episode: string
  alerts: string
  routine: string
  deviations: string
  memory: string
  other: string
  structured: boolean
}

export interface SummaryAlertItem {
  title: string
  description: string
  severity: SummarySeverity
  snapshotIndices: number[]
  timestampMs: number | null
}

export interface SummaryEvidenceMeta {
  selectedFrames: number
  frameBudget: number
  sourceFrames: number
  periodSeconds: number | null
}

export interface SummaryVisualCoverage {
  state: 'complete' | 'selected' | 'partial' | 'gap'
  selectedFrames: number
  sourceFrames: number
  reason: string
}

export type SummaryPeriod =
  | 'live'
  | 'today'
  | 'yesterday'
  | 'day_before_yesterday'
  | '7d'
  | '30d'
  | 'custom'

export type SummaryResolution = 'AUTO' | 'L0' | 'L1' | 'L2' | 'L3'

export interface SummaryPeriodBounds {
  from_ts?: number
  to_ts?: number
}

function localDayStart(nowMs: number, dayOffset: number): number {
  const now = new Date(nowMs)
  return new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate() + dayOffset,
  ).getTime() / 1000
}

export function summaryPeriodBounds(
  period: SummaryPeriod,
  nowMs = Date.now(),
  custom: SummaryPeriodBounds = {},
): SummaryPeriodBounds {
  const nowSec = nowMs / 1000
  if (period === 'live') return {}
  if (period === 'today') return { from_ts: localDayStart(nowMs, 0), to_ts: nowSec }
  if (period === 'yesterday') {
    return {
      from_ts: localDayStart(nowMs, -1),
      to_ts: localDayStart(nowMs, 0) - 0.001,
    }
  }
  if (period === 'day_before_yesterday') {
    return {
      from_ts: localDayStart(nowMs, -2),
      to_ts: localDayStart(nowMs, -1) - 0.001,
    }
  }
  if (period === '7d') return { from_ts: nowSec - 7 * 86400, to_ts: nowSec }
  if (period === '30d') return { from_ts: nowSec - 30 * 86400, to_ts: nowSec }
  const from = Number(custom.from_ts)
  const to = Number(custom.to_ts)
  if (!Number.isFinite(from) && !Number.isFinite(to)) return {}
  if (Number.isFinite(from) && Number.isFinite(to) && from > to) {
    return { from_ts: to, to_ts: from }
  }
  return {
    from_ts: Number.isFinite(from) ? from : undefined,
    to_ts: Number.isFinite(to) ? to : undefined,
  }
}

export function resolveSummaryResolution(
  resolution: SummaryResolution,
  period: SummaryPeriod,
  bounds: SummaryPeriodBounds = {},
): 'L0' | 'L1' | 'L2' | 'L3' {
  if (resolution !== 'AUTO') return resolution
  if (period === 'live') return 'L0'
  if (period === 'today' || period === 'yesterday' || period === 'day_before_yesterday') return 'L1'
  if (period === '7d') return 'L2'
  if (period === '30d') return 'L3'
  const duration = Number(bounds.to_ts) - Number(bounds.from_ts)
  if (!Number.isFinite(duration) || duration <= 0) return 'L3'
  if (duration <= 8 * 3600) return 'L0'
  if (duration <= 36 * 3600) return 'L1'
  if (duration <= 8 * 86400) return 'L2'
  return 'L3'
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
  const normalizeSeverity = (value: unknown): SummarySeverity => {
    const raw = String(value || '').trim().toLowerCase()
    const aliases: Record<string, SummarySeverity> = {
      information: 'info',
      informational: 'info',
      warn: 'low',
      warning: 'low',
      medium: 'normal',
      moderate: 'normal',
      danger: 'high',
      emergency: 'critical',
    }
    const normalized = aliases[raw] || raw
    return SUMMARY_SEVERITIES.includes(normalized as SummarySeverity)
      ? normalized as SummarySeverity
      : 'normal'
  }

  const fromRawCounts = (rawCounts: unknown): Partial<Record<SummarySeverity, number>> => {
    const counts: Partial<Record<SummarySeverity, number>> = {}
    if (!rawCounts || typeof rawCounts !== 'object' || Array.isArray(rawCounts)) return counts
    for (const [rawSeverity, rawValue] of Object.entries(rawCounts as Record<string, unknown>)) {
      const value = Number(rawValue || 0)
      if (!Number.isFinite(value) || value <= 0) continue
      const severity = normalizeSeverity(rawSeverity)
      counts[severity] = Number(counts[severity] || 0) + Math.floor(value)
    }
    return counts
  }

  const fromEvents = (rawEvents: unknown): Partial<Record<SummarySeverity, number>> => {
    const counts: Partial<Record<SummarySeverity, number>> = {}
    if (!Array.isArray(rawEvents)) return counts
    for (const event of rawEvents) {
      if (!event || typeof event !== 'object') continue
      const severity = normalizeSeverity((event as Record<string, unknown>).severity)
      counts[severity] = Number(counts[severity] || 0) + 1
    }
    return counts
  }

  const explicit = fromRawCounts(entry.alert_counts)
  if (Object.keys(explicit).length) return explicit

  const alertEvents = fromEvents(entry.alert_events)
  if (Object.keys(alertEvents).length) return alertEvents

  const batchState = entry.batch_state && typeof entry.batch_state === 'object'
    ? entry.batch_state as Record<string, unknown>
    : null
  const stateAlerts = fromEvents(batchState?.alerts)
  if (Object.keys(stateAlerts).length) return stateAlerts

  const parts = splitSummaryMachineJson(entry.summary)
  if (parts.machineJson) {
    try {
      const parsed = JSON.parse(parts.machineJson)
      const parsedAlerts = fromEvents(parsed?.alerts)
      if (Object.keys(parsedAlerts).length) return parsedAlerts
    } catch {
      // A malformed machine-state block remains visible to the operator, but
      // must not hide alert metadata supplied by the backend.
    }
  }

  const total = Number(entry.alert_total || 0)
  if (!Number.isFinite(total) || total <= 0) return {}
  return { [normalizeSeverity(entry.severity)]: Math.floor(total) }
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

function narrativeSectionKey(value: string): keyof Omit<SummaryNarrativeSections, 'structured'> | null {
  const key = value.trim().toLowerCase().replace(/[^a-z]+/g, ' ').trim()
  if (key === 'scene' || key === 'scene description') return 'scene'
  if (key === 'episode' || key === 'episode update' || key === 'activity description') return 'episode'
  if (key === 'alerts' || key === 'alert') return 'alerts'
  if (key === 'routine') return 'routine'
  if (key === 'deviation' || key === 'deviations') return 'deviations'
  if (key === 'worth to remember' || key === 'memory') return 'memory'
  if (key === 'routine and deviations') return 'routine'
  return null
}

function appendSection(current: string, value: string): string {
  const clean = value.trim()
  if (!clean) return current
  return current ? `${current}\n\n${clean}` : clean
}

export function summaryNarrativeSections(value: unknown): SummaryNarrativeSections {
  const text = String(value || '').trim()
  const result: SummaryNarrativeSections = {
    scene: '',
    episode: '',
    alerts: '',
    routine: '',
    deviations: '',
    memory: '',
    other: '',
    structured: false,
  }
  const headings = [...text.matchAll(/^#{1,6}\s+(.+?)\s*$/gm)]
  if (!headings.length) {
    result.other = text
    return result
  }
  const prefix = text.slice(0, headings[0].index).trim()
  if (prefix) result.other = prefix
  for (let index = 0; index < headings.length; index += 1) {
    const heading = headings[index]
    const start = Number(heading.index || 0) + heading[0].length
    const end = index + 1 < headings.length
      ? Number(headings[index + 1].index || text.length)
      : text.length
    const body = text.slice(start, end).trim()
    const key = narrativeSectionKey(heading[1])
    if (!key) {
      result.other = appendSection(result.other, `${heading[0]}\n${body}`)
      continue
    }
    result.structured = true
    if (heading[1].trim().toLowerCase().replace(/[^a-z]+/g, ' ').trim() === 'routine and deviations') {
      const deviationMarker = /(?:^|\n)\s*deviations?\s*:\s*/i.exec(body)
      if (deviationMarker?.index != null) {
        const routine = body.slice(0, deviationMarker.index).replace(/^\s*routine\s*:\s*/i, '').trim()
        const deviations = body.slice(deviationMarker.index + deviationMarker[0].length).trim()
        result.routine = appendSection(result.routine, routine)
        result.deviations = appendSection(result.deviations, deviations)
        continue
      }
    }
    result[key] = appendSection(result[key], body)
  }
  return result
}

function normalizeSummarySeverity(value: unknown): SummarySeverity {
  const raw = String(value || '').trim().toLowerCase()
  const aliases: Record<string, SummarySeverity> = {
    information: 'info',
    informational: 'info',
    warning: 'low',
    warn: 'low',
    medium: 'normal',
    moderate: 'normal',
    danger: 'high',
    emergency: 'critical',
  }
  const normalized = aliases[raw] || raw
  return SUMMARY_SEVERITIES.includes(normalized as SummarySeverity)
    ? normalized as SummarySeverity
    : 'normal'
}

export function summaryAlertItems(entry: SummaryEntry): SummaryAlertItem[] {
  const sources: unknown[] = [
    entry.alert_events,
    entry.batch_state && typeof entry.batch_state === 'object'
      ? (entry.batch_state as Record<string, unknown>).alerts
      : null,
  ]
  const parts = splitSummaryMachineJson(entry.summary)
  if (parts.machineJson) {
    try {
      sources.push(JSON.parse(parts.machineJson)?.alerts)
    } catch {
      // The visible narrative remains usable when legacy machine JSON is malformed.
    }
  }
  const selected = sources.find((source) => Array.isArray(source) && source.length > 0)
  if (!Array.isArray(selected)) return []
  const seen = new Set<string>()
  const alerts: SummaryAlertItem[] = []
  for (const raw of selected) {
    if (!raw || typeof raw !== 'object') continue
    const item = raw as Record<string, unknown>
    const title = String(item.title || item.label || 'Alert').trim() || 'Alert'
    const snapshotIndices = Array.isArray(item.snapshot_indices)
      ? item.snapshot_indices
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value > 0)
      : []
    const timestamp = Number(item.timestamp_ms)
    const timestampMs = Number.isFinite(timestamp) && timestamp > 0 ? timestamp : null
    const dedupe = `${title.toLowerCase()}|${snapshotIndices.join(',')}|${timestampMs || 0}`
    if (seen.has(dedupe)) continue
    seen.add(dedupe)
    alerts.push({
      title,
      description: String(item.description || item.summary || '').trim(),
      severity: normalizeSummarySeverity(item.severity),
      snapshotIndices,
      timestampMs,
    })
  }
  return alerts
}

export function summaryEvidenceMeta(entry: SummaryEntry): SummaryEvidenceMeta {
  const selected = Number(entry.selected_frame_count || entry.frame_count || 0)
  const source = Number(entry.source_frame_count || selected || 0)
  const configuredBudget = Number(entry.frame_selection?.frame_budget || 8)
  const frameBudget = Math.max(4, Math.min(8, Number.isFinite(configuredBudget) ? configuredBudget : 8))
  const start = Number(entry.batch_start_ms)
  const end = Number(entry.batch_end_ms)
  const periodSeconds = Number.isFinite(start) && Number.isFinite(end) && end >= start
    ? Math.round(((end - start) / 1000) * 10) / 10
    : null
  return {
    selectedFrames: Number.isFinite(selected) ? Math.max(0, Math.floor(selected)) : 0,
    frameBudget,
    sourceFrames: Number.isFinite(source) ? Math.max(0, Math.floor(source)) : 0,
    periodSeconds,
  }
}

export function summaryVisualCoverage(entry: SummaryEntry): SummaryVisualCoverage {
  const selected = Number(entry.selected_frame_count || entry.frame_count || 0)
  const source = Number(entry.source_frame_count || selected || 0)
  const selectedFrames = Number.isFinite(selected) ? Math.max(0, Math.floor(selected)) : 0
  const sourceFrames = Number.isFinite(source) ? Math.max(selectedFrames, Math.floor(source)) : selectedFrames
  const selection = entry.frame_selection && typeof entry.frame_selection === 'object'
    ? entry.frame_selection as Record<string, unknown>
    : {}
  const coverageStatus = String(selection.coverage_status || '').trim().toLowerCase()
  const uncoveredSalient = Number(
    selection.uncovered_salient_count
      ?? selection.omitted_salient_frame_count
      ?? 0,
  )

  if (entry.coverage_gap) {
    return {
      state: 'gap',
      selectedFrames,
      sourceFrames,
      reason: String(entry.gap_reason || 'The source or processing path contains an explicit coverage gap.'),
    }
  }
  if (
    ['partial', 'truncated', 'degraded'].includes(coverageStatus)
    || (Number.isFinite(uncoveredSalient) && uncoveredSalient > 0)
  ) {
    return {
      state: 'partial',
      selectedFrames,
      sourceFrames,
      reason: 'At least one salient source moment was not represented in the bounded VLM evidence packet.',
    }
  }
  if (sourceFrames > selectedFrames && selectedFrames > 0) {
    return {
      state: 'selected',
      selectedFrames,
      sourceFrames,
      reason: 'EVA attention-ranked the source observations into a bounded chronological VLM evidence packet.',
    }
  }
  return {
    state: 'complete',
    selectedFrames,
    sourceFrames,
    reason: 'All captured observations in this evidence window were sent to the VLM.',
  }
}
