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
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

/** Normalize a /detections/list row OR a /detections/search_* row into one shape. */
export function normalizeDetection(raw: any, channels?: Map<number, string>): Detection {
  const id = raw.id ?? raw.detection_id ?? null
  const channelId = num(raw.channel_id)
  const tsMs = num(raw.recorded_at_ms) ?? num(raw.timestamp_ms)
  const source = String(raw.source || '')
  // match %: search rows carry `similarity`; list rows carry it inside payload.
  const sim =
    num(raw.similarity) ??
    num(raw?.payload?.context?.bookmark_gate?.similarity)
  const matchPct = sim != null ? Math.round(sim * 100) : null
  return {
    key: `${source}:${id ?? raw.shard_key ?? Math.random()}:${tsMs ?? ''}`,
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

export async function listArchive(f: ArchiveFilters, channels: Channel[]): Promise<{ items: Detection[]; total: number; hasMore: boolean }> {
  const res = await api.get('/detections/list', {
    channel_id: f.channelId,
    source: f.source,
    hours: f.sinceMs ? undefined : (f.hours ?? '24'),
    since_ms: f.sinceMs,
    until_ms: f.untilMs,
    limit: f.rows ?? '24',
  })
  const cmap = channelMap(channels)
  return {
    items: (res.detections || []).map((d: any) => normalizeDetection(d, cmap)),
    total: res.total ?? 0,
    hasMore: !!res.has_more,
  }
}

export async function searchText(query: string, f: ArchiveFilters, channels: Channel[]): Promise<Detection[]> {
  const res = await api.postJson('/detections/search_text', {
    query,
    channel_id: f.channelId || undefined,
    source: f.source || undefined,
    hours: f.hours ? Number(f.hours) : 24,
    limit: Number(f.rows || 24),
    sort_by: f.sortBy || 'similarity',
  })
  const cmap = channelMap(channels)
  return (res.results || []).map((d: any) => normalizeDetection(d, cmap))
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
