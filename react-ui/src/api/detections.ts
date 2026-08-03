import { api } from './client'
import type { Channel, Detection, ArchiveFilters } from './types'
import type { LuxriotInventoryStatus } from './luxriotStatus'

const SOURCE_LABELS: Record<string, string> = {
  semantic_snapshot: 'Continuous CLIP archive',
  probe: 'CLIP probe',
  vlm_summary: 'Video desc',
  vlm_alert: 'VLM alert',
}

export interface LuxriotChannelsStatus {
  channels: Channel[]
  inventory: LuxriotInventoryStatus | null
}

export async function getChannelsStatus(force = false): Promise<LuxriotChannelsStatus> {
  const res = await api.get('/luxriot/channels', force ? { force: '1' } : undefined)
  return {
    channels: (res?.channels || []).map((c: any) => ({
      id: c.id,
      title: c.title,
      guid: c.guid,
      server: c.server,
      source: c.source,
    })),
    inventory: res?.inventory && typeof res.inventory === 'object'
      ? res.inventory as LuxriotInventoryStatus
      : null,
  }
}

export async function getChannels(): Promise<Channel[]> {
  return (await getChannelsStatus()).channels
}

function num(v: any): number | null {
  if (v === null || v === undefined || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

function stableHash(value: string): string {
  let hash = 2166136261
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(36)
}

/** Normalize a /detections/list row OR a /detections/search_* row into one shape. */
export function normalizeDetection(raw: any, channels?: Map<number, string>): Detection {
  const id = raw.id ?? raw.detection_id ?? null
  const channelId = num(raw.channel_id)
  const tsMs = num(raw.recorded_at_ms) ?? num(raw.timestamp_ms)
  const source = String(raw.source || '')
  const fallbackIdentity = [
    raw.shard_key, raw.image_path, raw.path, raw.probe_id, channelId, tsMs,
    raw.filename, raw.anchor_frame_index, raw.frame_index,
  ].map((v) => String(v ?? '')).join('|')
  // match %: search rows carry `similarity`; list rows carry it inside payload.
  const sim =
    num(raw.similarity) ??
    num(raw?.payload?.context?.bookmark_gate?.similarity)
  const matchPct = sim != null ? Math.round(sim * 100) : null
  return {
    key: `${source}:${id ?? raw.shard_key ?? stableHash(fallbackIdentity)}:${tsMs ?? ''}`,
    id,
    channelId,
    channelTitle: channelId != null ? channels?.get(channelId) : undefined,
    probeId: raw.probe_id ?? null,
    probeName: raw.probe_name || raw.filename || (channelId != null ? `Channel ${channelId}` : 'Frame'),
    source,
    sourceLabel: raw.source_label || SOURCE_LABELS[source] || source || 'frame',
    severity: String(raw.severity || 'info'),
    posScore: num(raw.pos_score),
    negScore: num(raw.neg_score),
    margin: num(raw.margin),
    matchPct,
    tsMs,
    thumbnail: raw.thumbnail || null,
    imageRef: raw.image_path || raw.path || null,
    raw,
  }
}

function channelMap(channels: Channel[]): Map<number, string> {
  return new Map(channels.map((c) => [c.id, c.title]))
}

export interface ArchiveFilterPayload extends Record<string, string | number | string[] | undefined> {
  channel_id: string | undefined
  channel_ids: string[] | undefined
  source: string | undefined
  probe_id: string | undefined
  hours: number | undefined
  since_ms: string | undefined
  until_ms: string | undefined
}

/** The shared backend filter contract used by list, text search and image search. */
export function buildArchiveFilterPayload(f: ArchiveFilters): ArchiveFilterPayload {
  const customRange = !!(f.sinceMs || f.untilMs)
  const channelIds = Array.from(new Set(
    (f.channelIds?.length ? f.channelIds : (f.channelId ? [f.channelId] : []))
      .map((value) => String(value || '').trim())
      .filter((value) => /^\d+$/.test(value)),
  ))
  return {
    channel_id: channelIds.length === 1 ? channelIds[0] : undefined,
    channel_ids: channelIds.length > 1 ? channelIds : undefined,
    source: f.source,
    probe_id: f.source === 'probe' ? f.probeId : undefined,
    hours: customRange ? undefined : Number(f.hours ?? '24'),
    since_ms: f.sinceMs,
    until_ms: f.untilMs,
  }
}

export function buildArchiveListQuery(f: ArchiveFilters, offset = 0): Record<string, string | number | string[] | undefined> {
  return {
    ...buildArchiveFilterPayload(f),
    limit: Number(f.rows || 24),
    offset: Math.max(0, offset),
  }
}

export function buildArchiveSearchPayload(f: ArchiveFilters): Record<string, string | number | string[] | undefined> {
  return {
    ...buildArchiveFilterPayload(f),
    limit: Number(f.rows || 24),
    sort_by: f.sortBy || 'similarity',
  }
}

export async function listArchive(
  f: ArchiveFilters,
  channels: Channel[],
  offset = 0,
): Promise<{ items: Detection[]; total: number; hasMore: boolean; offset: number }> {
  const res = await api.get('/detections/list', buildArchiveListQuery(f, offset))
  const cmap = channelMap(channels)
  return {
    items: (res.detections || []).map((d: any) => normalizeDetection(d, cmap)),
    total: res.total ?? 0,
    hasMore: !!res.has_more,
    offset: Number(res.offset ?? offset),
  }
}

export async function searchText(query: string, f: ArchiveFilters, channels: Channel[]): Promise<Detection[]> {
  const res = await api.postJson('/detections/search_text', {
    query,
    ...buildArchiveSearchPayload(f),
  })
  const cmap = channelMap(channels)
  return (res.results || []).map((d: any) => normalizeDetection(d, cmap))
}

export interface ArchiveProbeOption {
  id: string
  name: string
  hitCount: number
}

export interface AlertFeedbackReason {
  code: string
  label: string
}

export interface AlertFeedback {
  id?: string | number
  detection_id?: number
  channel_id?: number
  reason_code: string
  reason_label?: string
  note?: string
  submitted_at_ms?: number
  updated_at_ms?: number
}

export interface AlertFeedbackResponse {
  feedback: AlertFeedback | null
  reason_options: AlertFeedbackReason[]
  success?: boolean
}

export async function getArchiveProbeOptions(f: ArchiveFilters): Promise<ArchiveProbeOption[]> {
  if (f.source !== 'probe') return []
  const res = await api.get('/detections/summary', {
    ...buildArchiveFilterPayload(f),
    limit: 300,
  })
  return (res.summary || []).flatMap((item: any) => {
    const id = String(item?.probe_id || '').trim()
    if (!id) return []
    return [{
      id,
      name: String(item?.probe_name || id),
      hitCount: Number(item?.hit_count || 0),
    }]
  })
}

export async function getAlertFeedback(detectionId: number): Promise<AlertFeedbackResponse> {
  return api.get(`/detections/${detectionId}/feedback`)
}

export async function saveAlertFeedback(
  detectionId: number,
  reasonCode: string,
  note: string,
): Promise<AlertFeedbackResponse> {
  return api.postJson(`/detections/${detectionId}/feedback`, {
    reason_code: reasonCode,
    note,
  })
}

export function falsePositiveExportUrl(
  format: 'md' | 'xml',
  channelId?: number | null,
): string {
  const params = new URLSearchParams({ format, hours: '24' })
  if (channelId != null && Number.isInteger(channelId) && channelId > 0) {
    params.set('channel_id', String(channelId))
  }
  return `/reports/false-positives/export?${params.toString()}`
}

export async function findParentAlert(
  parentAlertId: string,
  channelId: number,
  channels: Channel[],
  timestampMs?: number,
): Promise<Detection | null> {
  const response = await api.get('/detections/list', {
    channel_id: channelId,
    source: 'vlm_alert',
    parent_alert_id: parentAlertId,
    limit: 1,
    offset: 0,
  })
  const rows = response.detections || []
  if (rows.length) return normalizeDetection(rows[0], channelMap(channels))
  if (!Number.isFinite(Number(timestampMs))) return null
  const pad = 15 * 60_000
  const fallback = await api.get('/detections/list', {
    channel_id: channelId,
    source: 'vlm_alert',
    since_ms: Number(timestampMs) - pad,
    until_ms: Number(timestampMs) + pad,
    limit: 24,
    offset: 0,
  })
  return (fallback.detections || []).length
    ? normalizeDetection(fallback.detections[0], channelMap(channels))
    : null
}

export async function loadDetectionBatchFrames(
  detection: Detection,
  channels: Channel[],
): Promise<Detection[]> {
  if (!['vlm_summary', 'vlm_alert'].includes(detection.source)) return [detection]
  const payload = detection.raw?.payload || {}
  const batchId = String(payload.batch_id || detection.raw?.batch_id || '').trim()
  const batchStart = num(payload.batch_start_ms ?? detection.raw?.batch_start_ms)
  const batchEnd = num(payload.batch_end_ms ?? detection.raw?.batch_end_ms)
  if (detection.channelId == null || (!batchId && (batchStart == null || batchEnd == null))) {
    return [detection]
  }
  const response = await api.get('/detections/list', {
    channel_id: detection.channelId,
    source: 'vlm_summary',
    batch_id: batchId || undefined,
    since_ms: batchId ? undefined : Math.min(batchStart!, batchEnd!),
    until_ms: batchId ? undefined : Math.max(batchStart!, batchEnd!),
    limit: 120,
    offset: 0,
  })
  const normalized = (response.detections || [])
    .map((row: any) => normalizeDetection(row, channelMap(channels)))
    .filter((row: Detection) => !!detImageSrc(row))
  const unique = new Map<string, Detection>()
  for (const row of [detection, ...normalized]) unique.set(batchFrameIdentity(row), row)
  return [...unique.values()].sort((left, right) => (
    batchFrameNumber(left) - batchFrameNumber(right)
    || Number(left.tsMs || 0) - Number(right.tsMs || 0)
  ))
}

export function batchFrameNumber(detection: Detection): number {
  const payload = detection.raw?.payload || {}
  const value = num(
    payload.snapshot_index
      ?? payload.anchor_snapshot_index
      ?? payload.batch_position
      ?? payload.frame_index
      ?? payload.anchor_frame_index,
  )
  return value ?? Number.MAX_SAFE_INTEGER
}

function batchFrameIdentity(detection: Detection): string {
  const payload = detection.raw?.payload || {}
  return [
    payload.batch_id,
    payload.run_id,
    payload.snapshot_index ?? payload.anchor_snapshot_index ?? payload.frame_index,
    payload.frame_timestamp_ms ?? detection.tsMs,
    detection.source,
  ].map((value) => String(value ?? '')).join(':')
}

/** Describe a frame with the VLM. Returns the description text (`summary`). */
export async function describeFrame(d: Detection): Promise<string> {
  const form = new FormData()
  if (d.thumbnail) {
    const bin = atob(d.thumbnail)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    form.append('image', new Blob([bytes], { type: 'image/jpeg' }), 'frame.jpg')
    form.append('prompt', 'Describe this surveillance frame briefly and factually.')
    const res = await api.postForm('/describe_image', form)
    return res.summary || res.description || '(no description returned)'
  }
  if (d.imageRef) {
    const res = await api.postJson('/describe_image', {
      image_path: d.imageRef,
      prompt: 'Describe this surveillance frame briefly and factually.',
    })
    return res.summary || res.description || '(no description returned)'
  }
  throw new Error('No image available for this frame.')
}

export function thumbSrc(d: Detection): string | null {
  return d.thumbnail ? `data:image/jpeg;base64,${d.thumbnail}` : null
}

/** Best image source for a card: inline b64 → backend image_url → image_path URL → server thumbnail by id. */
export function detImageSrc(d: Detection): string | null {
  if (d.thumbnail) return `data:image/jpeg;base64,${d.thumbnail}`
  const url = d.raw?.image_url
  if (url) return String(url)
  if (d.imageRef && String(d.imageRef).startsWith('/')) return `/detections/image?image_path=${encodeURIComponent(String(d.imageRef))}`
  if (d.id != null) return `/detections/thumbnail/${d.id}`
  return null
}

/** Full-resolution Inspector source. Server paths are always routed through the guarded endpoint. */
export function fullDetectionImageSrc(d: Detection): string | null {
  if (d.imageRef) return `/detections/image?image_path=${encodeURIComponent(String(d.imageRef))}`
  const url = d.raw?.image_url
  if (url) return String(url)
  return detImageSrc(d)
}

/** Normalize an arbitrary agent tool_result into grid-ready detections. */
export function detectionsFromResult(result: any, channels: Channel[]): Detection[] {
  if (!result || typeof result !== 'object') return []
  let items: any[] = []
  for (const k of ['results', 'detections', 'frames', 'evidence_frames', 'matches', 'items', 'hits']) {
    if (Array.isArray(result[k]) && result[k].length) { items = result[k]; break }
  }
  const cmap = channelMap(channels)
  return items.map((d: any) => normalizeDetection(d, cmap))
}
