import type { ArchiveFilters } from '../api/types'

export type ConsoleEffectTarget = 'archive' | 'probes' | 'video'

export interface ConsoleUiEffect {
  version: 1
  effectId: string
  target: ConsoleEffectTarget
  action: string
  source: {
    tool: string
    committed: boolean
  }
  payload: Record<string, unknown>
}

export interface AgentConsoleContext {
  version: 1
  section: 'home' | 'archive' | 'probes' | 'video'
  archive?: {
    channel_id?: number
    source?: 'semantic_snapshot' | 'probe' | 'vlm_summary' | 'vlm_alert'
    probe_id?: string
    since_ms?: number
    until_ms?: number
    sort_by?: 'similarity' | 'time'
    rows?: number
  }
}

const ACTIONS: Record<ConsoleEffectTarget, ReadonlySet<string>> = {
  archive: new Set(['show_results', 'open_review']),
  probes: new Set(['show_board', 'show_preview', 'refresh']),
  video: new Set([
    'open_prompt_settings',
    'show_prompt_preview',
    'show_period',
    'show_channels',
    'show_restore_status',
    'show_restore_preview',
  ]),
}
const ARCHIVE_SOURCES = new Set(['semantic_snapshot', 'probe', 'vlm_summary', 'vlm_alert'])

export function normalizeConsoleUiEffects(value: unknown): ConsoleUiEffect[] {
  if (!Array.isArray(value)) return []
  const effects: ConsoleUiEffect[] = []
  const seen = new Set<string>()
  for (const item of value.slice(0, 16)) {
    if (!item || typeof item !== 'object') continue
    const row = item as Record<string, unknown>
    if (Number(row.version) !== 1) continue
    const target = String(row.target || '') as ConsoleEffectTarget
    const action = String(row.action || '')
    const effectId = String(row.effect_id || '').slice(0, 128)
    const source = row.source && typeof row.source === 'object'
      ? row.source as Record<string, unknown>
      : {}
    const tool = String(source.tool || '').slice(0, 128)
    if (!ACTIONS[target]?.has(action) || !effectId || !tool || seen.has(effectId)) continue
    seen.add(effectId)
    effects.push({
      version: 1,
      effectId,
      target,
      action,
      source: { tool, committed: source.committed === true },
      payload: boundedPayload(row.payload),
    })
  }
  return effects
}

export function buildAgentConsoleContext(
  section: string,
  filters: ArchiveFilters | null | undefined,
  nowMs = Date.now(),
): AgentConsoleContext {
  const normalizedSection: AgentConsoleContext['section'] = section === 'monitoring'
    ? 'probes'
    : section === 'archive' || section === 'video' || section === 'home'
      ? section
      : 'home'
  const context: AgentConsoleContext = { version: 1, section: normalizedSection }
  if (normalizedSection !== 'archive' || !filters) return context

  const archive: NonNullable<AgentConsoleContext['archive']> = {}
  const channelId = positiveInt(filters.channelId)
  if (channelId != null) archive.channel_id = channelId
  if (ARCHIVE_SOURCES.has(String(filters.source || ''))) {
    archive.source = filters.source as NonNullable<typeof archive.source>
  }
  const probeId = String(filters.probeId || '').trim()
  if (archive.source === 'probe' && probeId) archive.probe_id = probeId.slice(0, 128)

  const explicitSince = nonnegativeInt(filters.sinceMs)
  const explicitUntil = nonnegativeInt(filters.untilMs)
  const until = explicitUntil ?? Math.max(0, Math.floor(nowMs))
  const hours = Number(filters.hours ?? '24')
  const since = explicitSince ?? (
    Number.isFinite(hours) && hours === 0
      ? 0
      : Number.isFinite(hours) && hours > 0
        ? Math.max(0, until - hours * 3_600_000)
        : null
  )
  if (since != null && since <= until) {
    archive.since_ms = Math.floor(since)
    archive.until_ms = Math.floor(until)
  }
  if (filters.sortBy === 'time' || filters.sortBy === 'similarity') {
    archive.sort_by = filters.sortBy
  }
  const rows = positiveInt(filters.rows)
  if (rows != null) archive.rows = Math.min(rows, 100)
  if (Object.keys(archive).length) context.archive = archive
  return context
}

function boundedPayload(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object') return {}
  const payload: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value as Record<string, unknown>).slice(0, 32)) {
    if (typeof item === 'string') payload[key.slice(0, 64)] = item.slice(0, 512)
    else if (typeof item === 'number' && Number.isFinite(item)) payload[key.slice(0, 64)] = item
    else if (typeof item === 'boolean') payload[key.slice(0, 64)] = item
    else if (Array.isArray(item)) {
      const bounded: Array<string | number> = []
      for (const entry of item.slice(0, 32)) {
        if (typeof entry === 'string') bounded.push(entry.slice(0, 128))
        else if (typeof entry === 'number' && Number.isFinite(entry)) bounded.push(entry)
      }
      payload[key.slice(0, 64)] = bounded
    }
  }
  return payload
}

function positiveInt(value: unknown): number | null {
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 && parsed <= 2_147_483_647 ? parsed : null
}

function nonnegativeInt(value: unknown): number | null {
  if (value == null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : null
}
