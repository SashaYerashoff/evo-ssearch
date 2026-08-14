import type { SemanticPresenceClass } from '../../api/probes'

const DEFAULT_NOISE_FLOOR = 0.01
const REACTION_SIGMA = 3
export const PRESENCE_REACTION_WINDOW_SAMPLES = 12

export interface PresenceReaction {
  reacting: boolean
  current: boolean
  direction: 'up' | 'down' | null
  strength: number
  peakTimestampMs: number | null
}

export interface PresenceDisplaySignal {
  spatial: boolean
  score?: number | null
  baseline?: number | null
  deviation?: number | null
  delta?: number | null
  z?: number | null
  state?: string
  warmup?: boolean
  samples?: number
  timestamp_ms?: number | null
  history?: SemanticPresenceClass['history']
}

const PRESENCE_CONTEXT_ALIASES: Record<string, string[]> = {
  person: ['person', 'people', 'human', 'hand', 'finger', 'thumb', 'gesture', 'face', 'head', 'headphone', 'worker', 'pedestrian'],
  vehicle: ['vehicle', 'car', 'truck', 'bus', 'van', 'motorcycle', 'road', 'driver', 'drift', 'donut', 'skid', 'tire'],
  animal: ['animal', 'cat', 'dog', 'bird', 'pet'],
  smoke: ['smoke', 'smoky', 'haze', 'exhaust'],
  fire: ['fire', 'flame', 'burn', 'blaze'],
}

function contextTokens(texts: string[]): string[] {
  return texts
    .join(' ')
    .toLocaleLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean)
}

export function presenceMatchesContext(
  item: SemanticPresenceClass,
  texts: string[] = [],
): boolean {
  const key = presenceClassKey(item).toLocaleLowerCase()
  const aliases = PRESENCE_CONTEXT_ALIASES[key] || [key]
  const tokens = contextTokens(texts)
  return aliases.some((alias) => tokens.some((token) => (
    token === alias || (alias.length >= 4 && token.startsWith(alias))
  )))
}

function finite(value: unknown): number | null {
  if (value == null) return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function presenceClassKey(item: SemanticPresenceClass): string {
  return String(item.key || item.label || '').trim()
}

export function presenceDisplaySignal(
  item: SemanticPresenceClass,
): PresenceDisplaySignal {
  const pooled = Number(item.samples || 0) > 0
    && finite(item.score) != null
  if (pooled) {
    return {
      spatial: false,
      score: item.score,
      baseline: item.baseline,
      deviation: item.deviation,
      delta: item.delta,
      z: item.z,
      state: item.state,
      warmup: item.warmup,
      samples: item.samples,
      timestamp_ms: item.timestamp_ms,
      history: item.history,
    }
  }
  const spatial = Number(item.spatial_samples || 0) > 0
    && finite(item.spatial_score) != null
  return spatial
    ? {
        spatial: true,
        score: item.spatial_score,
        baseline: item.spatial_baseline,
        deviation: item.spatial_deviation,
        delta: item.spatial_delta,
        z: item.spatial_z,
        state: item.spatial_state,
        warmup: item.spatial_warmup,
        samples: item.spatial_samples,
        timestamp_ms: item.spatial_timestamp_ms,
        history: item.spatial_history,
      }
    : { spatial: false }
}

export function presenceSpatialSignal(
  item: SemanticPresenceClass,
): PresenceDisplaySignal | null {
  if (
    Number(item.spatial_samples || 0) <= 0
    || finite(item.spatial_score) == null
  ) return null
  return {
    spatial: true,
    score: item.spatial_score,
    baseline: item.spatial_baseline,
    deviation: item.spatial_deviation,
    delta: item.spatial_delta,
    z: item.spatial_z,
    state: item.spatial_state,
    warmup: item.spatial_warmup,
    samples: item.spatial_samples,
    timestamp_ms: item.spatial_timestamp_ms,
    history: item.spatial_history,
  }
}

/**
 * Detect a meaningful response against each class's own baseline.
 *
 * Absolute SigLIP similarities are prompt-specific and cannot be compared
 * between labels. A short peak hold keeps a real response above routine rows
 * without reordering the list on every one-Hz noise sample.
 */
export function presenceReaction(
  item: SemanticPresenceClass,
  windowSamples = PRESENCE_REACTION_WINDOW_SAMPLES,
): PresenceReaction {
  const signal = presenceDisplaySignal(item)
  if (signal.warmup) {
    return { reacting: false, current: false, direction: null, strength: 0, peakTimestampMs: null }
  }
  const deviation = Math.abs(finite(signal.deviation) ?? 0)
  const scale = Math.max(DEFAULT_NOISE_FLOOR, deviation)
  const history = (signal.history || []).slice(-Math.max(1, windowSamples))
  const candidates = history
    .map((point) => {
      const score = finite(point.score)
      const baseline = finite(point.baseline)
      const timestampMs = finite(point.timestamp_ms)
      if (score == null || baseline == null) return null
      const delta = score - baseline
      return {
        delta,
        strength: Math.abs(delta) / scale,
        timestampMs,
      }
    })
    .filter((value): value is { delta: number; strength: number; timestampMs: number | null } => value != null)
  const currentState = String(signal.state || '')
  const current = currentState === 'above_baseline' || currentState === 'below_baseline'
  const currentDelta = finite(signal.delta) ?? 0
  candidates.push({
    delta: currentDelta,
    strength: Math.abs(finite(signal.z) ?? (currentDelta / scale)),
    timestampMs: finite(signal.timestamp_ms),
  })
  const peak = candidates.reduce(
    (best, candidate) => candidate.strength > best.strength ? candidate : best,
    { delta: 0, strength: 0, timestampMs: null as number | null },
  )
  const reacting = current || peak.strength >= REACTION_SIGMA
  return {
    reacting,
    current,
    direction: reacting ? (peak.delta >= 0 ? 'up' : 'down') : null,
    strength: reacting ? peak.strength : 0,
    peakTimestampMs: reacting ? peak.timestampMs : null,
  }
}

export function rankPresenceClasses(
  values: SemanticPresenceClass[],
  contextTexts: string[] = [],
): SemanticPresenceClass[] {
  return values
    .map((item, index) => ({
      item,
      index,
      reaction: presenceReaction(item),
      relevant: presenceMatchesContext(item, contextTexts),
    }))
    .sort((left, right) => {
      if (left.reaction.reacting !== right.reaction.reacting) {
        return left.reaction.reacting ? -1 : 1
      }
      if (left.reaction.reacting && right.reaction.reacting) {
        if (left.relevant !== right.relevant) return left.relevant ? -1 : 1
        const strengthDifference = right.reaction.strength - left.reaction.strength
        if (Math.abs(strengthDifference) >= 0.25) return strengthDifference
      }
      if (!left.reaction.reacting && left.relevant !== right.relevant) {
        return left.relevant ? -1 : 1
      }
      // Routine noise and near-equal responses preserve the server's bounded
      // configured order instead of swapping rows on every poll. Classes named
      // by the selected probe remain visible, but are not called detections.
      return left.index - right.index
    })
    .map(({ item }) => item)
}
