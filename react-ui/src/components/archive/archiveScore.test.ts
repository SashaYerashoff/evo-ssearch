import { describe, expect, it } from 'vitest'
import type { Detection } from '../../api/types'
import {
  archiveDetectionScore,
  archiveScoreRange,
  archiveScoreThreshold,
  passesArchiveScoreThreshold,
} from './archiveScore'

function detection(raw: any, matchPct: number | null = null): Detection {
  return {
    key: 'test', id: 1, channelId: 1, probeName: 'frame', source: 'probe',
    sourceLabel: 'probe', severity: 'info', tsMs: 1, raw, matchPct,
  }
}

describe('archive score normalization', () => {
  it('keeps raw 0..1 precision and accepts percentage-shaped scores', () => {
    expect(archiveDetectionScore(detection({ similarity: 0.7349 }))).toBe(0.7349)
    expect(archiveDetectionScore(detection({ score: 82.5 }))).toBe(0.825)
    expect(archiveDetectionScore(detection({ fusion: { clip_similarity: '71.2%' } }))).toBeCloseTo(0.712)
  })

  it('maps the slider over the returned score range', () => {
    const items = [detection({ similarity: 0.4 }), detection({ similarity: 0.8 })]
    const range = archiveScoreRange(items)
    expect(archiveScoreThreshold(range, 50)).toBeCloseTo(0.6)
    expect(items.filter((item) => passesArchiveScoreThreshold(item, 0.6))).toHaveLength(1)
  })

  it('shows all results at zero and when all scores are equal or absent', () => {
    const equal = archiveScoreRange([detection({ similarity: 0.7 }), detection({ score: 70 })])
    expect(archiveScoreThreshold(equal, 80)).toBe(0)
    expect(passesArchiveScoreThreshold(detection({}), 0)).toBe(true)
  })
})
