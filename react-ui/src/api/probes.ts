import { api } from './client'

// A single semantic-probe hit (frame that matched the active embedding space).
export interface ProbeHit {
  thumbnail?: string | null
  image_url?: string | null
  image_path?: string | null
  path?: string | null
  pos_score?: number | null
  neg_score?: number | null
  margin?: number | null
  similarity?: number | null
  timestamp_ms?: number | null
  recorded_at_ms?: number | null
  id?: number | string | null
}

// Bookmark gate state as returned by the backend.
export interface BookmarkGate {
  reason?: string
  cooldown_sec?: number
  remaining_sec?: number
  [k: string]: any
}

export interface ImageProbe { data?: string | null; name?: string; pos_floor?: number; enabled?: boolean }
// normalized region-of-interest rectangle (all values 0..1 of the frame)
export interface RoiNorm { x: number; y: number; w: number; h: number }
// pairs are stored/returned with positive/negative keys; positives[]/negatives[] are authoritative.
export interface TextPair { positive?: string; negative?: string }

// The persisted probe object (from /probes/list, /probes/save).
export interface Probe {
  id: string
  name?: string
  channel_id?: number
  enabled?: boolean
  pairs?: TextPair[]
  positives?: string[]
  negatives?: string[]
  pos_floor?: number
  margin?: number
  window_sec?: number      // archive query window in seconds
  top_k?: number
  fps?: number
  severity?: string
  bookmark?: boolean
  bookmark_cooldown_sec?: number
  bookmark_dedupe_window_sec?: number
  bookmark_gate?: BookmarkGate
  origin?: ProbeOrigin
  origin_meta?: { plan_id?: string; [k: string]: any }
  temporary?: boolean
  source?: string
  parent_alert_id?: string
  parent_alert_title?: string
  parent_alert_description?: string
  parent_alert_timestamp_ms?: number
  created_at_ms?: number
  expires_at_ms?: number
  image_probe?: ImageProbe | null
  roi_enabled?: boolean
  roi_norm?: RoiNorm | number[] | null
  recent_hits?: ProbeHit[]
  last_hit?: ProbeHit
  [k: string]: any
}

export type ProbeOrigin = 'operator' | 'agent' | 'auto'

export interface ProbeChannelGroup {
  id: string
  name: string
  channel_ids: number[]
  position?: number
  read_only?: boolean
  created_at_ms?: number
  updated_at_ms?: number
}

export interface ProbeListCounts {
  visible?: number
  persistent?: number
  temporary_active?: number
  temporary_expired_hidden?: number
  stored?: number
  by_origin?: Partial<Record<ProbeOrigin, number>>
}

export interface ProbeListResponse {
  probes: Probe[]
  channel_groups?: ProbeChannelGroup[]
  counts?: ProbeListCounts
  defaults?: ProbeThresholdDefaults
}

export interface ProbeThresholdDefaults {
  pos_floor: number
  margin: number
  embedding_backend?: string
  embedding_model?: string
  embedding_revision?: string
}

// Live capture status for a channel (GET /probes/status).
export interface ChannelStatus { channel_id: number; runtime_state?: string; last_snapshot_ms?: number; buffer_frames?: number }

export function probeRangeDurationMs(status: {
  time_range_ms?: number | [number, number] | null
  first_timestamp_ms?: number | null
  last_timestamp_ms?: number | null
}): number | null {
  const range = status?.time_range_ms
  if (Array.isArray(range) && range.length >= 2) {
    const first = Number(range[0])
    const last = Number(range[1])
    return Number.isFinite(first) && Number.isFinite(last) ? Math.max(0, last - first) : null
  }
  if (typeof range === 'number' && Number.isFinite(range)) return Math.max(0, range)
  if (status?.first_timestamp_ms == null || status?.last_timestamp_ms == null) return null
  const first = Number(status.first_timestamp_ms)
  const last = Number(status.last_timestamp_ms)
  return Number.isFinite(first) && Number.isFinite(last) ? Math.max(0, last - first) : null
}

export interface Benchmark {
  batch: number
  elapsed_sec: number
  approx_fps: number
  device: string
  backend: string
  model: string
  resolution: number
}

export interface ProbeRunResult {
  results: ProbeHit[]
  status?: string
  probe: Probe
  persisted_hits?: number
  bookmark_gate?: BookmarkGate
}

// Payload for creating/updating a probe (POST /probes/save).
export interface ProbeInput {
  id?: string
  name?: string
  channel_id?: number
  enabled?: boolean
  pairs?: TextPair[]
  positives?: string[]
  negatives?: string[]
  pos_floor?: number
  margin?: number
  window_sec?: number
  top_k?: number
  severity?: string
  bookmark?: boolean
  bookmark_cooldown_sec?: number
  bookmark_dedupe_window_sec?: number
  image_probe?: ImageProbe | null
  roi_enabled?: boolean
  roi_norm?: RoiNorm | null
}

export function authorizeProbeInput(input: ProbeInput, canCreateBookmarks: boolean): ProbeInput {
  const payload = { ...input }
  if (!canCreateBookmarks) {
    delete payload.bookmark
    delete payload.bookmark_cooldown_sec
    delete payload.bookmark_dedupe_window_sec
  }
  return payload
}

export function probeMutationRequiresBookmarkPermission(
  probe: Pick<Probe, 'bookmark'> | null | undefined,
  canCreateBookmarks: boolean,
): boolean {
  return !!probe?.bookmark && !canCreateBookmarks
}

// POST /probes/cast — copy the probe onto many channels at once.
export interface CastInput extends Omit<ProbeInput, 'id' | 'channel_id'> {
  channel_ids: number[]
  conflict: 'skip' | 'create' | 'update'
  copy_roi: boolean
}
export interface CastResult {
  success?: boolean
  error?: string
  counts?: { created: number; updated: number; skipped: number; failed: number }
  failed?: { channel_id: number; error: string }[]
}

export const probesApi = {
  list: (): Promise<ProbeListResponse> => api.get('/probes/list'),
  save: (p: ProbeInput): Promise<{ success: boolean; probe: Probe; error?: string }> => api.postJson('/probes/save', p),
  remove: (id: string): Promise<{ success: boolean; error?: string }> => api.postJson('/probes/delete', { id }),
  run: (id: string): Promise<ProbeRunResult> => api.postJson('/probes/run', { id }),
  bench: (batch = 16): Promise<Benchmark & { error?: string }> => api.get('/probes/bench', { batch: String(batch) }),
  status: (channelId: number): Promise<any> => api.get('/probes/status', { channel_id: String(channelId) }),
  cast: (payload: CastInput): Promise<CastResult> => api.postJson('/probes/cast', payload),
  startCapture: (channelId: number, fps?: number): Promise<any> => api.postJson('/probes/start_capture', { channel_id: channelId, fps }),
  stopCapture: (channelId: number): Promise<any> => api.postJson('/probes/stop_capture', { channel_id: channelId }),
  groups: (): Promise<{ groups: ProbeChannelGroup[] }> => api.get('/probes/channel_groups'),
  saveGroup: (group: {
    id?: string
    name: string
    channel_ids: number[]
    position?: number
  }): Promise<{ success: boolean; group: ProbeChannelGroup; groups: ProbeChannelGroup[]; error?: string }> =>
    api.postJson('/probes/channel_groups/save', group),
  deleteGroup: (id: string): Promise<{ success: boolean; groups: ProbeChannelGroup[]; error?: string }> =>
    api.postJson('/probes/channel_groups/delete', { id }),
}

// Best thumbnail source for a probe hit (mirrors detImageSrc).
export function hitImageSrc(h: ProbeHit | undefined | null): string | null {
  if (!h) return null
  if (h.thumbnail) return /^data:image\//i.test(h.thumbnail) ? h.thumbnail : `data:image/jpeg;base64,${h.thumbnail}`
  if (h.image_url) return String(h.image_url)
  if (h.image_path && String(h.image_path).startsWith('/')) return `/detections/image?image_path=${encodeURIComponent(String(h.image_path))}`
  if (h.id != null) return `/detections/thumbnail/${h.id}`
  return null
}
