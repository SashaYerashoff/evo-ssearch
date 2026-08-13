import { describe, expect, it } from 'vitest'
import type { SemanticPresenceClass } from '../../api/probes'
import { presenceReaction, rankPresenceClasses } from './semanticPresenceView'

function semanticClass(
  key: string,
  delta: number,
  historyDeltas: number[],
  state = 'routine',
): SemanticPresenceClass {
  const baseline = 0.06
  return {
    key,
    label: key,
    score: baseline + delta,
    baseline,
    delta,
    deviation: 0.002,
    z: delta / 0.01,
    state,
    warmup: false,
    samples: 60,
    timestamp_ms: 60_000,
    history: historyDeltas.map((value, index) => ({
      timestamp_ms: (index + 1) * 1_000,
      score: baseline + value,
      baseline,
    })),
  }
}

describe('semantic presence operator ranking', () => {
  it('keeps routine classes in configured order despite small z noise', () => {
    const person = semanticClass('person', -0.001, [-0.001, 0.001])
    const vehicle = semanticClass('vehicle', 0.004, [0.003, 0.004])
    const animal = semanticClass('animal', 0.002, [0.001, 0.002])

    expect(rankPresenceClasses([person, vehicle, animal]).map((item) => item.key)).toEqual([
      'person',
      'vehicle',
      'animal',
    ])
    expect(presenceReaction(vehicle).reacting).toBe(false)
  })

  it('moves a meaningful recent response above routine and holds it briefly', () => {
    const person = semanticClass('person', 0.001, [0.001, 0.002])
    const animal = semanticClass('animal', 0.002, [0.001, 0.041, 0.002])
    const vehicle = semanticClass('vehicle', 0.003, [0.002, 0.003])

    expect(rankPresenceClasses([person, vehicle, animal]).map((item) => item.key)).toEqual([
      'animal',
      'person',
      'vehicle',
    ])
    expect(presenceReaction(animal)).toMatchObject({ reacting: true, current: false, direction: 'up' })
  })

  it('drops an expired response instead of pinning it forever', () => {
    const oldResponse = semanticClass(
      'vehicle',
      0.001,
      [0.05, ...Array.from({ length: 12 }, () => 0.001)],
    )

    expect(presenceReaction(oldResponse).reacting).toBe(false)
  })
})
