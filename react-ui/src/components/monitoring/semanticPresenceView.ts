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

function finite(value: unknown): number | null {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function presenceClassKey(item: SemanticPresenceClass): string {
  return String(item.key || item.label || '').trim()
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
  if (item.warmup) {
    return { reacting: false, current: false, direction: null, strength: 0, peakTimestampMs: null }
  }
  const deviation = Math.abs(finite(item.deviation) ?? 0)
  const scale = Math.max(DEFAULT_NOISE_FLOOR, deviation)
  const history = (item.history || []).slice(-Math.max(1, windowSamples))
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
  const currentState = String(item.state || '')
  const current = currentState === 'above_baseline' || currentState === 'below_baseline'
  const currentDelta = finite(item.delta) ?? 0
  candidates.push({
    delta: currentDelta,
    strength: Math.abs(finite(item.z) ?? (currentDelta / scale)),
    timestampMs: finite(item.timestamp_ms),
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
): SemanticPresenceClass[] {
  return values
    .map((item, index) => ({ item, index, reaction: presenceReaction(item) }))
    .sort((left, right) => {
      if (left.reaction.reacting !== right.reaction.reacting) {
        return left.reaction.reacting ? -1 : 1
      }
      if (left.reaction.reacting && right.reaction.reacting) {
        const strengthDifference = right.reaction.strength - left.reaction.strength
        if (Math.abs(strengthDifference) >= 0.25) return strengthDifference
      }
      // Routine noise and near-equal responses preserve the server's bounded
      // configured order instead of swapping rows on every poll.
      return left.index - right.index
    })
    .map(({ item }) => item)
}

