import { api } from './client'

// A single CLIP-probe hit (frame that matched the probe).
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
  image_probe?: ImageProbe | null
  roi_enabled?: boolean
  roi_norm?: number[] | null
  recent_hits?: ProbeHit[]
  last_hit?: ProbeHit
  [k: string]: any
}

// Live capture status for a channel (GET /probes/status).
export interface ChannelStatus { channel_id: number; runtime_state?: string; last_snapshot_ms?: number; buffer_frames?: number }

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
}

export const probesApi = {
  list: (): Promise<{ probes: Probe[] }> => api.get('/probes/list'),
  save: (p: ProbeInput): Promise<{ success: boolean; probe: Probe; error?: string }> => api.postJson('/probes/save', p),
  remove: (id: string): Promise<{ success: boolean; error?: string }> => api.postJson('/probes/delete', { id }),
  run: (id: string): Promise<ProbeRunResult> => api.postJson('/probes/run', { id }),
  bench: (batch = 16): Promise<Benchmark & { error?: string }> => api.get('/probes/bench', { batch: String(batch) }),
  status: (channelId: number): Promise<any> => api.get('/probes/status', { channel_id: String(channelId) }),
  startCapture: (channelId: number, fps?: number): Promise<any> => api.postJson('/probes/start_capture', { channel_id: channelId, fps }),
  stopCapture: (channelId: number): Promise<any> => api.postJson('/probes/stop_capture', { channel_id: channelId }),
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
