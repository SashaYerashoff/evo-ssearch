import type { Channel } from './types'
import type { Stream, StreamsStatus } from './video'

export type LuxriotLinkState = 'checking' | 'connected' | 'stale' | 'offline'

export interface LuxriotInventoryStatus {
  cached?: boolean
  stale?: boolean
  cache_age_sec?: number | null
  last_attempt_at?: number | null
  last_success_at?: number | null
  last_error?: string | null
  [key: string]: unknown
}

export interface LuxriotLinkStatus {
  state: LuxriotLinkState
  detail: string
}

function isLocalChannel(channel: Channel): boolean {
  const source = String(channel.source || '').toLowerCase()
  const guid = String(channel.guid || '').toLowerCase()
  const server = String(channel.server ?? '').toLowerCase()
  return source === 'local_v4l2' || guid.startsWith('local-v4l2:') || server.startsWith('local-')
}

function compactError(value: unknown): string {
  const text = String(value || '').trim().replace(/\s+/g, ' ')
  return text.length > 180 ? `${text.slice(0, 179)}…` : text
}

function ageLabel(timestampSec: unknown, nowSec: number): string {
  const timestamp = Number(timestampSec)
  if (!Number.isFinite(timestamp) || timestamp <= 0) return ''
  const age = Math.max(0, nowSec - timestamp)
  if (age < 5) return 'just now'
  if (age < 60) return `${Math.round(age)}s ago`
  if (age < 3600) return `${Math.round(age / 60)}m ago`
  return `${Math.round(age / 3600)}h ago`
}

function streamFailure(stream: Stream): string {
  if (stream.frozen_signal) return 'frozen signal'
  return compactError(
    stream.capture_last_error
    || stream.last_live_segment_error
    || stream.last_error,
  )
}

/**
 * Turn cached inventory plus live capture health into an operator-facing link state.
 * Local V4L2 channels are deliberately excluded: a healthy USB camera must not make
 * an unreachable Luxriot/Evo server look connected.
 */
export function deriveLuxriotLinkStatus(
  channels: Channel[],
  inventory: LuxriotInventoryStatus | null | undefined,
  streams: StreamsStatus | null | undefined,
  nowSec = Date.now() / 1000,
): LuxriotLinkStatus {
  if (!inventory) {
    return { state: 'checking', detail: 'Luxriot inventory status has not been received yet.' }
  }

  const upstreamChannelIds = new Set(
    channels.filter((channel) => !isLocalChannel(channel)).map((channel) => Number(channel.id)),
  )
  const upstreamStreams = (streams?.video_streams || []).filter((stream) => (
    upstreamChannelIds.has(Number(stream.channel_id)) && stream.running !== false
  ))
  const failedStreams = upstreamStreams.filter((stream) => Boolean(streamFailure(stream)))
  const healthyStreams = upstreamStreams.filter((stream) => !streamFailure(stream))
  const missingDesired = (streams?.desired_video_missing || []).filter((item: any) => (
    upstreamChannelIds.has(Number(item?.channel_id))
  ))

  const lastSuccess = ageLabel(inventory.last_success_at, nowSec)
  const inventoryError = compactError(inventory.last_error)
  if (inventory.stale === true || inventoryError) {
    if (healthyStreams.length) {
      const detail = [
        'Luxriot control inventory is stale; EVA is using the cached channel list.',
        `${healthyStreams.length} upstream capture signal${healthyStreams.length === 1 ? ' is' : 's are'} still active.`,
        lastSuccess ? `Last successful inventory contact: ${lastSuccess}.` : '',
        inventoryError,
      ].filter(Boolean).join(' ')
      return { state: 'stale', detail }
    }
    const detail = [
      'Luxriot is unreachable; EVA is showing a cached channel list.',
      lastSuccess ? `Last successful contact: ${lastSuccess}.` : '',
      inventoryError,
    ].filter(Boolean).join(' ')
    return { state: 'offline', detail }
  }

  if (inventory.cached !== true && !inventory.last_success_at) {
    return { state: 'checking', detail: 'Waiting for the first Luxriot inventory response.' }
  }

  if (failedStreams.length || missingDesired.length) {
    const firstFailure = failedStreams.length ? streamFailure(failedStreams[0]) : ''
    const affected = new Set([
      ...failedStreams.map((stream) => Number(stream.channel_id)),
      ...missingDesired.map((item: any) => Number(item?.channel_id)),
    ]).size
    const detail = [
      `Luxriot is reachable, but ${affected} configured signal${affected === 1 ? ' is' : 's are'} stale or unavailable.`,
      firstFailure,
    ].filter(Boolean).join(' ')
    return { state: 'stale', detail }
  }

  return {
    state: 'connected',
    detail: lastSuccess
      ? `Luxriot inventory is current. Last successful contact: ${lastSuccess}.`
      : 'Luxriot inventory is current.',
  }
}
