import type { Detection } from '../../api/types'

export interface ArchiveScoreRange {
  min: number | null
  max: number | null
  hasScores: boolean
  hasSpread: boolean
}

function finite(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const number = typeof value === 'string' ? Number.parseFloat(value) : Number(value)
  return Number.isFinite(number) ? number : null
}

function normalizeScore(value: unknown): number | null {
  const number = finite(value)
  if (number === null) return null
  if (number > 1 && number <= 100) return number / 100
  return Math.max(0, Math.min(1, number))
}

/** Preserve the backend's full score precision; matchPct is only a display fallback. */
export function archiveDetectionScore(detection: Detection): number | null {
  const raw = detection.raw || {}
  const candidates = [
    raw.similarity,
    raw.score,
    raw.match_score,
    raw.final_score,
    raw?.fusion?.clip_similarity,
    raw?.fusion?.dino_similarity,
    raw?.fusion?.score,
    raw?.fusion?.clip,
    raw?.fusion?.dino,
    raw?.payload?.similarity,
    raw?.payload?.score,
    raw?.payload?.context?.bookmark_gate?.similarity,
  ]
  for (const candidate of candidates) {
    const score = normalizeScore(candidate)
    if (score !== null) return score
  }
  return detection.matchPct == null ? null : normalizeScore(detection.matchPct)
}

export function archiveScoreRange(items: Detection[]): ArchiveScoreRange {
  const scores = items
    .map(archiveDetectionScore)
    .filter((score): score is number => score !== null)
  if (!scores.length) return { min: null, max: null, hasScores: false, hasSpread: false }
  const min = Math.min(...scores)
  const max = Math.max(...scores)
  return { min, max, hasScores: true, hasSpread: max - min > 0.000001 }
}

/** Slider position is relative to the score range returned by the current query. */
export function archiveScoreThreshold(range: ArchiveScoreRange, sliderPercent: number): number {
  if (sliderPercent <= 0 || !range.hasSpread || range.min === null || range.max === null) return 0
  return range.min + (range.max - range.min) * Math.max(0, Math.min(100, sliderPercent)) / 100
}

export function passesArchiveScoreThreshold(detection: Detection, threshold: number): boolean {
  if (threshold <= 0) return true
  const score = archiveDetectionScore(detection)
  return score !== null && score >= threshold
}

export function formatArchiveScore(score: number | null): string {
  return score === null ? 'n/a' : score.toFixed(3)
}
