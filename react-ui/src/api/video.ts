import { api } from './client'

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
  values: { batch: string; every: string; model: string },
): CaptureInput {
  return {
    channel_id: channelId,
    batch_size: Number(values.batch),
    interval_sec: Number(values.every),
    model: values.model.trim() || undefined,
  }
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

// Model-view preview from EVA's bounded attention-frame ring. Dense Luxriot
// capture fills windows incrementally, so a five-second freshness limit can
// reject a healthy channel between apex admissions. Match the legacy
// operator preview contract: frames remain live evidence for at most 60 s.
export function recentFrameUrl(channelId: number, bust: number): string {
  return `/luxriot/recent_frame/${channelId}?stream=mainStream&fallback=snapshot&mode=latest&max_age_sec=60&_=${bust}`
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
