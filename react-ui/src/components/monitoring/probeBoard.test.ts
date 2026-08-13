import { describe, expect, it } from 'vitest'
import {
  buildProbeBoardTree,
  probeHitSeries,
  probeLiveSeries,
  probeMatchesFilters,
  probeOrigin,
  probeTemporaryTtl,
} from './probeBoard'

describe('probe board', () => {
  it('backfills authorship without confusing alert lineage source', () => {
    expect(probeOrigin({ id: 'p1', temporary: true, source: 'vlm_alert' })).toBe('auto')
    expect(probeOrigin({ id: 'p2', origin: 'agent', source: 'vlm_alert' })).toBe('agent')
    expect(probeOrigin({ id: 'p3' })).toBe('operator')
  })

  it('groups channels once and leaves unassigned channels visible', () => {
    const tree = buildProbeBoardTree(
      [
        { id: 'p1', channel_id: 7 },
        { id: 'p2', channel_id: 8 },
      ],
      [{ id: 'g1', name: 'Gate', channel_ids: [7] }],
      [{ id: 7, title: 'Gate cam' }, { id: 8, title: 'Yard cam' }],
      () => 'idle',
    )
    expect(tree.map((group) => group.name)).toEqual(['Gate', 'Ungrouped channels'])
    expect(tree[0].channels[0].probes[0].id).toBe('p1')
    expect(tree[1].channels[0].probes[0].id).toBe('p2')
  })

  it('reports temporary expiry and filters against origin/state/query', () => {
    const probe = {
      id: 'p1',
      name: 'Person at gate',
      origin: 'auto' as const,
      temporary: true,
      expires_at_ms: 160_000,
    }
    expect(probeTemporaryTtl(probe, 100_000)?.text).toBe('1m left')
    expect(probeMatchesFilters(
      probe,
      {
        origins: new Set(['auto']),
        states: new Set(['running']),
        query: 'gate',
      },
      'running',
      'North gate',
    )).toBe(true)
  })

  it('keeps positive and negative values in event history', () => {
    const series = probeHitSeries({
      id: 'p1',
      recent_hits: [
        { timestamp_ms: 2_000, pos_score: 0.12, neg_score: 0.07, margin: 0.05 },
        { timestamp_ms: 1_000, pos_score: 0.03, neg_score: 0.08, margin: -0.05 },
      ],
    })
    expect(series.map((point) => point.timestampMs)).toEqual([1_000, 2_000])
    expect(series[0]).toMatchObject({ posScore: 0.03, negScore: 0.08, margin: -0.05 })
  })

  it('normalizes continuous pre-threshold samples independently from hits', () => {
    const series = probeLiveSeries([
      { timestamp_ms: 1_000, pos_score: 0.02, neg_score: 0.09, margin: -0.07, threshold_state: 'below_both' },
      { timestamp_ms: 2_000, pos_score: 0.14, neg_score: 0.08, margin: 0.06, threshold_state: 'hit' },
    ])
    expect(series).toHaveLength(2)
    expect(series[0].thresholdState).toBe('below_both')
    expect(series[1]).toMatchObject({ posScore: 0.14, negScore: 0.08, margin: 0.06 })
  })
})
