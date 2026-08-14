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
  embedding_calibration_state?: 'calibrated' | 'recalibration_required' | 'embedding_space_mismatch' | string
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
  embedding_calibration_state?: string
}

// Live capture + semantic-scoring status for a channel (GET /probes/status).
// `live_signal` is deliberately pre-threshold: it lets an operator tune P/N/M
// without first manufacturing a hit.
export interface ProbeLiveSignal extends ProbeHit {
  threshold_state?: 'hit' | 'below_pos' | 'below_margin' | 'below_both' | string
  age_ms?: number
  stale?: boolean
  frame_url?: string
}
export interface SemanticPresencePoint {
  timestamp_ms: number
  score: number
  baseline: number
}
export interface SemanticPresenceClass {
  key: string
  label: string
  prompts?: string[]
  score?: number | null
  baseline?: number | null
  deviation?: number | null
  delta?: number | null
  z?: number | null
  state?: 'warming_up' | 'routine' | 'above_baseline' | 'below_baseline' | string
  warmup?: boolean
  samples?: number
  timestamp_ms?: number | null
  history?: SemanticPresencePoint[]
  spatial_score?: number | null
  spatial_baseline?: number | null
  spatial_deviation?: number | null
  spatial_delta?: number | null
  spatial_z?: number | null
  spatial_state?: 'warming_up' | 'routine' | 'above_baseline' | 'below_baseline' | string
  spatial_warmup?: boolean
  spatial_samples?: number
  spatial_timestamp_ms?: number | null
  spatial_history?: SemanticPresencePoint[]
  spatial_contrast?: number | null
  spatial_raw_score?: number | null
  spatial_score_semantics?: string
}
export interface SemanticPresenceStatus {
  enabled: boolean
  shadow?: boolean
  state?: 'warming_up' | 'ready' | 'degraded' | string
  channel_id?: number
  timestamp_ms?: number | null
  age_ms?: number | null
  semantics?: string
  spatial_semantics?: string
  spatial_state?: string
  error?: string | null
  classes?: SemanticPresenceClass[]
}
export interface PatchAttentionResult {
  channel_id: number
  timestamp_ms: number
  class_key: string
  label: string
  prompt?: string
  frame_url: string
  semantics: string
  backend?: string
  model?: string
  method?: string
  ephemeral?: boolean
  shadow?: boolean
  grid: { rows: number; cols: number }
  heatmap: number[]
  image?: { width: number; height: number }
  raw_range?: { p10?: number; p90?: number; contrast?: number }
  peak_cell?: { row: number; col: number }
  suggested_roi?: RoiNorm | null
  error?: string
  error_code?: string
  retry_after_ms?: number
}
export interface ChannelStatus {
  channel_id: number
  runtime_state?: 'running' | 'paused' | 'idle' | string
  semantic_state?: 'ready' | 'stale' | 'warming_up' | 'degraded' | string
  semantic_error?: string | null
  semantic_age_ms?: number | null
  semantic_stale_after_ms?: number | null
  semantic_stale?: boolean
  capture_error?: string | null
  last_snapshot_ms?: number
  buffer_frames?: number
  frames?: number
  time_range_ms?: number | [number, number] | null
  first_timestamp_ms?: number | null
  last_timestamp_ms?: number | null
  live_signal?: ProbeLiveSignal | null
  signal_history?: ProbeLiveSignal[]
  semantic_presence?: SemanticPresenceStatus | null
  embedding_backend?: string
  embedding_model?: string
  embedding_revision?: string
  embedding_calibration_state?: string
}

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
  iterations?: number
  requested_iterations?: number
  truncated?: boolean
  budget_ms?: number
  elapsed_sec: number
  approx_fps: number
  encoder_fps?: number
  effective_fps?: number
  average_compute_ms?: number
  average_lock_wait_ms?: number
  max_lock_wait_ms?: number
  warmup_ms?: number
  warmup_compute_ms?: number
  warmup_lock_wait_ms?: number
  samples?: { lock_wait_ms: number; compute_ms: number; total_ms: number }[]
  device: string
  device_name?: string
  cuda_visible_devices?: string
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
  status: (channelId: number, probeId?: string): Promise<ChannelStatus> => api.get('/probes/status', {
    channel_id: String(channelId),
    ...(probeId ? { probe_id: probeId } : {}),
  }),
  patchAttention: (
    channelId: number,
    timestampMs: number,
    classKey: string,
  ): Promise<PatchAttentionResult> => api.get('/probes/patch_attention', {
    channel_id: String(channelId),
    timestamp_ms: String(timestampMs),
    class_key: classKey,
  }),
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
