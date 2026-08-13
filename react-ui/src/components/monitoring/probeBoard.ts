import type {
  Probe,
  ProbeChannelGroup,
  ProbeHit,
  ProbeLiveSignal,
  ProbeOrigin,
} from '../../api/probes'
import type { Channel } from '../../api/types'
import type { ProbeStatus } from './ProbeCard'

export interface ProbeBoardFilters {
  origins: ReadonlySet<ProbeOrigin>
  states: ReadonlySet<ProbeStatus>
  query: string
}

export interface ProbeBoardChannel {
  channelId: number | null
  label: string
  probes: Probe[]
  runningCount: number
}

export interface ProbeBoardGroup {
  id: string
  name: string
  synthetic: boolean
  readOnly: boolean
  channels: ProbeBoardChannel[]
}

export function probeOrigin(probe: Probe): ProbeOrigin {
  const raw = String(probe.origin || '').trim().toLowerCase()
  if (raw === 'operator' || raw === 'agent' || raw === 'auto') return raw
  if (probe.temporary || probe.parent_alert_id) return 'auto'
  if (['vlm_alert', 'alert_probe', 'auto'].includes(String(probe.source || '').trim().toLowerCase())) {
    return 'auto'
  }
  return 'operator'
}

export interface ProbeSignalPoint {
  posScore: number
  negScore: number
  margin: number
  timestampMs: number
  thresholdState?: string
}

function normalizeProbeSignal(
  sample: ProbeHit | ProbeLiveSignal,
): ProbeSignalPoint | null {
  const posScore = Number(sample?.pos_score)
  const negScore = Number(sample?.neg_score)
  const margin = Number(sample?.margin)
  if (!Number.isFinite(posScore)) return null
  const timestampMs = Number(sample.timestamp_ms ?? sample.recorded_at_ms ?? 0)
  return {
    posScore,
    negScore: Number.isFinite(negScore) ? negScore : 0,
    margin: Number.isFinite(margin)
      ? margin
      : posScore - (Number.isFinite(negScore) ? negScore : 0),
    timestampMs: Number.isFinite(timestampMs) ? timestampMs : 0,
    ...('threshold_state' in sample && sample.threshold_state
      ? { thresholdState: String(sample.threshold_state) }
      : {}),
  }
}

export function probeHitSeries(probe: Probe): ProbeSignalPoint[] {
  const hits: ProbeHit[] = probe.recent_hits?.length
    ? probe.recent_hits
    : probe.last_hit
      ? [probe.last_hit]
      : []
  return hits
    .flatMap((hit) => {
      const point = normalizeProbeSignal(hit)
      return point ? [point] : []
    })
    .sort((left, right) => left.timestampMs - right.timestampMs)
    .slice(-24)
}

export function probeLiveSeries(
  history?: ProbeLiveSignal[] | null,
): ProbeSignalPoint[] {
  if (!Array.isArray(history)) return []
  return history
    .flatMap((sample) => {
      const point = normalizeProbeSignal(sample)
      return point ? [point] : []
    })
    .sort((left, right) => left.timestampMs - right.timestampMs)
    .slice(-60)
}

export function probeTemporaryTtl(
  probe: Probe,
  nowMs = Date.now(),
): { text: string; title: string; expired: boolean } | null {
  if (!probe.temporary) return null
  const expiresAt = Number(probe.expires_at_ms)
  if (!Number.isFinite(expiresAt)) {
    return {
      text: 'temporary',
      title: 'Temporary probe without a stored expiry',
      expired: false,
    }
  }
  const remainingMs = expiresAt - nowMs
  if (remainingMs <= 0) {
    return {
      text: 'expiring',
      title: 'Past its expiry; the next lifecycle sweep retires it',
      expired: true,
    }
  }
  const minutes = Math.floor(remainingMs / 60_000)
  const text = minutes >= 60
    ? `${Math.floor(minutes / 60)}h ${minutes % 60}m left`
    : minutes >= 1
      ? `${minutes}m left`
      : `${Math.max(1, Math.round(remainingMs / 1_000))}s left`
  return {
    text,
    title: `Retires at ${new Date(expiresAt).toLocaleString()}`,
    expired: false,
  }
}

export function probeMatchesFilters(
  probe: Probe,
  filters: ProbeBoardFilters,
  status: ProbeStatus,
  channelLabel: string,
): boolean {
  if (filters.origins.size && !filters.origins.has(probeOrigin(probe))) return false
  if (filters.states.size && !filters.states.has(status)) return false
  const query = filters.query.trim().toLowerCase()
  if (!query) return true
  const haystack = [
    probe.name,
    probe.id,
    probe.channel_id,
    channelLabel,
    probe.parent_alert_title,
    ...(probe.positives || []),
    ...(probe.negatives || []),
  ].join(' ').toLowerCase()
  return haystack.includes(query)
}

export function buildProbeBoardTree(
  probes: Probe[],
  groups: ProbeChannelGroup[],
  channels: Channel[],
  statusOf: (probe: Probe) => ProbeStatus,
): ProbeBoardGroup[] {
  const channelNames = new Map(channels.map((channel) => [channel.id, channel.title]))
  const groupIdByChannel = new Map<number, string>()
  for (const group of groups) {
    for (const channelId of group.channel_ids || []) groupIdByChannel.set(Number(channelId), group.id)
  }

  const probesByChannel = new Map<number | null, Probe[]>()
  for (const probe of probes) {
    const channelId = Number(probe.channel_id)
    const key = Number.isInteger(channelId) && channelId > 0 ? channelId : null
    probesByChannel.set(key, [...(probesByChannel.get(key) || []), probe])
  }

  const output = new Map<string, ProbeBoardGroup>()
  for (const group of groups) {
    output.set(group.id, {
      id: group.id,
      name: group.name || 'Group',
      synthetic: false,
      readOnly: group.read_only === true,
      channels: [],
    })
  }
  const ungrouped: ProbeBoardGroup = {
    id: '__ungrouped__',
    name: 'Ungrouped channels',
    synthetic: true,
    readOnly: true,
    channels: [],
  }

  for (const [channelId, channelProbes] of [...probesByChannel.entries()].sort(([left], [right]) => {
    if (left == null) return 1
    if (right == null) return -1
    return left - right
  })) {
    const target = channelId != null
      ? output.get(groupIdByChannel.get(channelId) || '') || ungrouped
      : ungrouped
    target.channels.push({
      channelId,
      label: channelId != null
        ? channelNames.get(channelId) || `Channel ${channelId}`
        : 'Unassigned channel',
      probes: channelProbes,
      runningCount: channelProbes.filter((probe) => statusOf(probe) === 'running').length,
    })
  }

  const ordered = groups
    .map((group) => output.get(group.id))
    .filter((group): group is ProbeBoardGroup => !!group && group.channels.length > 0)
  if (ungrouped.channels.length) ordered.push(ungrouped)
  return ordered
}
