import { api } from './client'
import type { Channel, Detection, ArchiveFilters } from './types'

const SOURCE_LABELS: Record<string, string> = {
  probe: 'CLIP probe',
  vlm_summary: 'Video desc',
  vlm_alert: 'VLM alert',
}

export async function getChannels(): Promise<Channel[]> {
  const res = await api.get('/luxriot/channels')
  return (res?.channels || []).map((c: any) => ({
    id: c.id, title: c.title, guid: c.guid, server: c.server,
  }))
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

export interface ArchiveFilterPayload extends Record<string, string | number | undefined> {
  channel_id: string | undefined
  source: string | undefined
  probe_id: string | undefined
  hours: number | undefined
  since_ms: string | undefined
  until_ms: string | undefined
}

/** The shared backend filter contract used by list, text search and image search. */
export function buildArchiveFilterPayload(f: ArchiveFilters): ArchiveFilterPayload {
  const customRange = !!(f.sinceMs || f.untilMs)
  return {
    channel_id: f.channelId,
    source: f.source,
    probe_id: f.source === 'probe' ? f.probeId : undefined,
    hours: customRange ? undefined : Number(f.hours ?? '24'),
    since_ms: f.sinceMs,
    until_ms: f.untilMs,
  }
}

export function buildArchiveListQuery(f: ArchiveFilters, offset = 0): Record<string, string | number | undefined> {
  return {
    ...buildArchiveFilterPayload(f),
    limit: Number(f.rows || 24),
    offset: Math.max(0, offset),
  }
}

export function buildArchiveSearchPayload(f: ArchiveFilters): Record<string, string | number | undefined> {
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
