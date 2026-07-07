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
  [k: string]: any
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

export const videoApi = {
  streams: (): Promise<StreamsStatus> => api.get('/luxriot/streams'),
  rollups: (channelId: number, opts: { level_limit?: number; from_ts?: number; to_ts?: number } = {}): Promise<RollupsResponse> =>
    api.get('/luxriot/rollups', { channel_id: String(channelId), level_limit: String(opts.level_limit ?? 60), from_ts: opts.from_ts, to_ts: opts.to_ts }),
  session: (channelId: number, limit = 40): Promise<{ logs?: SummaryEntry[]; running?: boolean; [k: string]: any }> =>
    api.get('/luxriot/session', { channel_id: String(channelId), limit: String(limit) }),
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
}

// Live-preview image URL (EVA recent frame with snapshot fallback). Add a cache-buster on each poll.
export function recentFrameUrl(channelId: number, bust: number): string {
  return `/luxriot/recent_frame/${channelId}?stream=mainStream&fallback=snapshot&mode=latest&max_age_sec=5&_=${bust}`
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
