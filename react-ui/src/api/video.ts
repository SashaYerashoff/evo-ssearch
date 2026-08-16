import { api } from './client'
import type { IncidentDraftInput } from './incidents'

// One capture stream (video or analytics/probe) as returned inside /luxriot/streams.
export interface Stream {
  channel_id: number
  channel_name?: string
  capture_kind?: string       // 'video' | 'analytics'
  stream_type?: string
  running?: boolean
  paused?: boolean
  interval_sec?: number
  batch_size?: number
  model?: string | null
  model_selector?: string | null
  assigned_profile_id?: string | null
  routing_mode?: string | null
  routing_strategy?: string | null
  routing_reason?: string | null
  routing_capacity?: number | null
  pending_frames?: number
  max_buffer_frames?: number
  recent_frame_count?: number
  dropped_frames?: number
  frozen_signal?: boolean
  active_capture_source?: string | null
  summarization_enabled?: boolean
  last_error?: string | null
  last_snapshot_at?: number
  [k: string]: any
}

export interface StreamsStatus {
  video_streams?: Stream[]
  analytics_streams?: Stream[]
  capture_defaults?: { batch_size?: number; interval_sec?: number; allowed_batch_sizes?: number[] }
  capture_configurations?: Array<{
    channel_id: number
    enabled?: boolean
    batch_size?: number
    interval_sec?: number
    model?: string | null
    model_selector?: string | null
    assigned_profile_id?: string | null
    routing_mode?: string | null
    routing_strategy?: string | null
    routing_reason?: string | null
    routing_capacity?: number | null
    updated_at?: number | null
  }>
  desired_video_missing?: any[]
  paused_analytics_channels?: number[]
  running_total?: number
  [k: string]: any
}

// A summary/rollup entry from /luxriot/rollups levels or /luxriot/session logs.
export interface SummaryEntry {
  created_at?: number
  channel_id?: number
  summary?: string
  frame_count?: number
  model?: string
  batch_start_ms?: number
  batch_end_ms?: number
  window_start?: number
  window_end?: number
  run_id?: string
  severity?: string
  alert_counts?: Record<string, number>
  alert_total?: number
  level?: string
  rollup_id?: string
  batch_id?: string
  item_count?: number
  source_tokens?: number
  run_ids?: string[]
  summary_kind?: string
  generation_status?: string
  generation_error?: string
  semantic_refresh_pending?: boolean
  state_transition_total?: number
  coverage_gap?: boolean
  gap_reason?: string
  coalesced?: { batches?: number; omitted_frames?: number }
  thumbnail_detection_id?: number
  thumbnail_role?: string
  thumbnail_snapshot_index?: number
  thumbnail_selection_source?: string
  thumbnail_is_cover?: boolean
  cover_kind?: string
  cover_reason?: string
  cover_confidence?: string
  vector_signal?: {
    camera_scene?: {
      camera_motion?: 'steady' | 'pan' | 'tilt' | 'zoom' | 'preset_cut' | 'settling' | string
      scene_epoch?: number
      coverage_status?: string
      preset_id?: string
      preset_status?: string
      spatial_probes_enabled?: boolean
      [k: string]: any
    }
    capture_attention?: {
      seconds?: Array<{
        snapshot?: number | string
        mode?: string
        activity_x?: number
        [k: string]: any
      }>
      [k: string]: any
    }
    [k: string]: any
  }
  [k: string]: any
}

export interface SummaryBookmarkInput {
  channel_id: number
  title: string
  description: string
  severity: 'normal'
  state: 'new'
  timestamp_ms?: number
}

export function buildIncidentDraftFromSummary(entry: SummaryEntry): IncidentDraftInput | null {
  const channelId = Number(entry.channel_id)
  const anchorId = Number(entry.thumbnail_detection_id)
  if (!Number.isInteger(channelId) || channelId <= 0 || !Number.isInteger(anchorId) || anchorId <= 0) {
    return null
  }
  return {
    channel_id: channelId,
    anchor_detection_id: anchorId,
  }
}

export function buildSummaryBookmarkInput(entry: SummaryEntry): SummaryBookmarkInput | null {
  const channelId = Number(entry.channel_id)
  const summary = String(entry.summary || '').trim()
  if (!Number.isInteger(channelId) || channelId <= 0 || !summary) return null
  const firstLine = summary.split(/\r?\n/, 1)[0].trim() || `Channel ${channelId} summary`
  const shortTitle = firstLine.length > 80 ? `${firstLine.slice(0, 77)}...` : firstLine
  const createdAt = Number(entry.created_at)
  return {
    channel_id: channelId,
    title: `Live summary: ${shortTitle}`,
    description: summary.length > 2400 ? `${summary.slice(0, 2397)}...` : summary,
    severity: 'normal',
    state: 'new',
    timestamp_ms: Number.isFinite(createdAt) && createdAt > 0
      ? Math.round((createdAt > 1e12 ? createdAt : createdAt * 1000))
      : undefined,
  }
}

export interface RollupsResponse {
  channel_id?: number
  levels?: { L0?: SummaryEntry[]; L1?: SummaryEntry[]; L2?: SummaryEntry[]; L3?: SummaryEntry[] }
  runs?: any[]
  running?: boolean
  [k: string]: any
}

export interface CaptureInput {
  channel_id: number
  batch_size?: number
  interval_sec?: number
  model?: string
  prompt?: string
  system_prompt?: string
}

export function buildCaptureInput(
  channelId: number,
  values: { batch: string; every: string; model?: string },
): CaptureInput {
  const payload: CaptureInput = {
    channel_id: channelId,
    batch_size: Number(values.batch),
    interval_sec: Number(values.every),
  }
  const model = String(values.model || '').trim()
  if (model) payload.model = model
  return payload
}

export interface PromptSettings {
  channel_id?: number | null
  stream_system_prompt?: string
  alert_policy_prompt?: string
  json_alert_prompt?: string
  rollup_prompts?: { L1?: string; L2?: string; L3?: string }
  bookmark_enabled?: boolean
  bookmark_cooldown_sec?: number
  [k: string]: any
}

export function buildPromptSettingsPayload(
  settings: PromptSettings,
  channelId: number,
  canCreateBookmarks: boolean,
): PromptSettings {
  const payload = { ...settings, channel_id: channelId }
  if (!canCreateBookmarks) {
    delete payload.json_alert_prompt
    delete payload.bookmark_enabled
    delete payload.bookmark_cooldown_sec
  }
  return payload
}

export interface LmModelCatalog {
  models?: string[]
  default_model?: string
  auto_model_selector?: string
  auto_model_label?: string
  default_profile_id?: string
  profiles?: Array<{
    id?: string
    kind?: string
    model?: string
    selector?: string
    enabled?: boolean
    available?: boolean
    effective_capacity?: number
    routing_health?: string
    gpu?: string
  }>
  vlm_balancer?: {
    enabled?: boolean
    profile_ids?: string[]
    strategy?: string
  }
  error?: string | null
  [k: string]: any
}

export interface SessionQuery extends Record<string, unknown> {
  channel_id: string
  limit: string
  run?: string
  from_ts?: number
  to_ts?: number
}

export function buildSessionQuery(
  channelId: number,
  opts: { limit?: number; run?: string; from_ts?: number; to_ts?: number } = {},
): SessionQuery {
  return {
    channel_id: String(channelId),
    limit: String(opts.limit ?? 40),
    run: opts.run,
    from_ts: opts.from_ts,
    to_ts: opts.to_ts,
  }
}

export function buildSummaryFeedQuery(
  channelId: number,
  opts: { limit?: number; run?: string; from_ts?: number; to_ts?: number } = {},
): SessionQuery & { view: 'feed' } {
  return { ...buildSessionQuery(channelId, opts), view: 'feed' }
}

export const videoApi = {
  streams: (): Promise<StreamsStatus> => api.get('/luxriot/streams'),
  lmModels: (): Promise<LmModelCatalog> => api.get('/lm/models'),
  rollups: (
    channelId: number,
    opts: {
      level_limit?: number
      run?: string
      from_ts?: number
      to_ts?: number
      target_level?: 'L1' | 'L2' | 'L3'
    } = {},
  ): Promise<RollupsResponse> =>
    api.get('/luxriot/rollups', {
      channel_id: String(channelId),
      level_limit: String(opts.level_limit ?? 60),
      run: opts.run,
      from_ts: opts.from_ts,
      to_ts: opts.to_ts,
      target_level: opts.target_level,
    }),
  session: (channelId: number, opts: { limit?: number; run?: string; from_ts?: number; to_ts?: number } = {}): Promise<{ logs?: SummaryEntry[]; running?: boolean; [k: string]: any }> =>
    api.get('/luxriot/session', buildSummaryFeedQuery(channelId, opts)),
  startCapture: (b: CaptureInput): Promise<{ success: boolean; session?: any; error?: string }> => api.postJson('/luxriot/start_capture', b),
  stopCapture: (channelId: number): Promise<any> => api.postJson('/luxriot/stop_capture', { channel_id: channelId }),
  flushCapture: (channelId: number): Promise<{ success: boolean; status?: { logs?: SummaryEntry[] }; items?: number }> =>
    api.postJson('/luxriot/flush_capture', { channel_id: channelId }),
  stopStream: (channelId: number, streamType: 'both' | 'video' | 'analytics' = 'both', pauseAnalytics = true): Promise<any> =>
    api.postJson('/luxriot/streams/stop', { channel_id: channelId, stream_type: streamType, pause_analytics: pauseAnalytics }),
  stopAll: (opts: { stop_video?: boolean; stop_analytics?: boolean; pause_analytics?: boolean } = {}): Promise<any> =>
    api.postJson('/luxriot/streams/stop_all', { stop_video: opts.stop_video ?? true, stop_analytics: opts.stop_analytics ?? true, pause_analytics: opts.pause_analytics ?? true }),
  getPromptSettings: (channelId: number): Promise<PromptSettings> => api.get('/luxriot/prompt_settings', { channel_id: String(channelId) }),
  savePromptSettings: (b: PromptSettings): Promise<PromptSettings> => api.postJson('/luxriot/prompt_settings', b),
  createBookmark: (b: SummaryBookmarkInput): Promise<{ success?: boolean; [k: string]: any }> =>
    api.postJson('/luxriot/bookmark', b),
}

// Model-view preview from EVA's bounded attention-frame ring. It deliberately
// fails quickly when that ring is not fresh: a high-frequency UI refresh must
// never open another Luxriot recorder/snapshot request or compete with model
// capture. The UI keeps its last good decoded frame briefly while retrying.
export function recentFrameUrl(channelId: number, bust: number): string {
  return `/luxriot/recent_frame/${channelId}?stream=mainStream&fallback=0&mode=latest&max_age_sec=60&_=${bust}`
}

// Exact bounded MJPEG sequence of per-second EVA apex frames. Unlike full
// live, this reuses the attention ring and opens no additional Evo stream.
export function attentionStreamUrl(channelId: number, bust: number): string {
  return `/luxriot/attention_stream/${channelId}?max_age_sec=60&request=${bust}`
}

// Full live opens a separate bounded Luxriot media lease. It is intentionally
// opt-in because the model-view preview reuses EVA's existing capture ring.
export function fullLiveMediaUrl(channelId: number, bust: number): string {
  return `/luxriot/media/live/${channelId}?stream=mainStream&request=${bust}`
}

// Merge video + analytics streams into one per-channel runtime record.
export interface ChannelRuntime { channelId: number; name?: string; video: Stream | null; probe: Stream | null }
export function mergeRuntime(s: StreamsStatus, channelName: (id: number) => string | undefined): ChannelRuntime[] {
  const map = new Map<number, ChannelRuntime>()
  const ensure = (id: number) => {
    if (!map.has(id)) map.set(id, { channelId: id, name: channelName(id), video: null, probe: null })
    return map.get(id)!
  }
  for (const v of s.video_streams || []) ensure(v.channel_id).video = v
  for (const a of s.analytics_streams || []) ensure(a.channel_id).probe = a
  return Array.from(map.values()).sort((a, b) => a.channelId - b.channelId)
}

export interface CaptureSettings {
  batchSize: number
  intervalSec: number
  source: 'runtime' | 'saved' | 'server_default'
}

export interface CaptureRouting {
  selector: string
  assignedProfileId: string | null
  mode: 'auto' | 'manual' | 'legacy_pinned' | 'default'
  strategy: string | null
  reason: string | null
  capacity: number | null
  source: 'runtime' | 'saved' | 'server_default'
}

export function captureSettingsForChannel(
  status: StreamsStatus,
  channelId: number | null,
): CaptureSettings | null {
  if (channelId == null) return null
  const runtime = (status.video_streams || []).find((row) => Number(row.channel_id) === channelId)
  const saved = (status.capture_configurations || []).find((row) => Number(row.channel_id) === channelId)
  const defaults = status.capture_defaults || {}
  const candidates: Array<{ row: any; source: CaptureSettings['source'] }> = [
    { row: runtime, source: 'runtime' },
    { row: saved, source: 'saved' },
    { row: defaults, source: 'server_default' },
  ]
  for (const candidate of candidates) {
    const batchSize = Number(candidate.row?.batch_size)
    const intervalSec = Number(candidate.row?.interval_sec)
    if (Number.isFinite(batchSize) && batchSize > 0 && Number.isFinite(intervalSec) && intervalSec > 0) {
      return { batchSize, intervalSec, source: candidate.source }
    }
  }
  return null
}

export function captureRoutingForChannel(
  status: StreamsStatus,
  channelId: number | null,
  catalog: LmModelCatalog | null = null,
): CaptureRouting | null {
  if (channelId == null) return null
  const runtime = (status.video_streams || []).find((row) => Number(row.channel_id) === channelId)
  const saved = (status.capture_configurations || []).find((row) => Number(row.channel_id) === channelId)
  const candidates: Array<{ row: any; source: CaptureRouting['source'] }> = [
    { row: runtime, source: 'runtime' },
    { row: saved, source: 'saved' },
  ]
  for (const candidate of candidates) {
    if (!candidate.row) continue
    const explicitSelector = String(candidate.row.model_selector || '').trim()
    const routingMode = String(candidate.row.routing_mode || '').trim().toLowerCase()
    const assignedProfileId = String(
      candidate.row.assigned_profile_id || candidate.row.model || '',
    ).trim() || null
    if (explicitSelector || assignedProfileId || routingMode) {
      const autoSelector = String(catalog?.auto_model_selector || '__auto__')
      const isAuto = routingMode === 'auto' || explicitSelector === autoSelector
      const selector = explicitSelector || assignedProfileId || (
        isAuto ? autoSelector : String(catalog?.default_profile_id || catalog?.default_model || '')
      )
      return {
        selector,
        assignedProfileId,
        mode: isAuto
          ? 'auto'
          : explicitSelector
            ? 'manual'
            : assignedProfileId
              ? 'legacy_pinned'
              : 'default',
        strategy: String(candidate.row.routing_strategy || '').trim() || null,
        reason: String(candidate.row.routing_reason || '').trim() || null,
        capacity: Number.isFinite(Number(candidate.row.routing_capacity))
          ? Number(candidate.row.routing_capacity)
          : null,
        source: candidate.source,
      }
    }
  }
  const autoEnabled = Boolean(catalog?.vlm_balancer?.enabled)
  return {
    selector: autoEnabled
      ? String(catalog?.auto_model_selector || '__auto__')
      : String(catalog?.default_profile_id || catalog?.default_model || ''),
    assignedProfileId: null,
    mode: autoEnabled ? 'auto' : 'default',
    strategy: String(catalog?.vlm_balancer?.strategy || '').trim() || null,
    reason: null,
    capacity: null,
    source: 'server_default',
  }
}
