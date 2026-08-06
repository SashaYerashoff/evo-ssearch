import { describe, expect, it } from 'vitest'
import { formatReviewDuration, incidentReviewBounds } from './IncidentReview'

describe('incident review helpers', () => {
  it('bounds review history in seconds for the HTTP API', () => {
    expect(incidentReviewBounds('24h', 172_800_000)).toEqual({
      from_ts: 86_400,
      to_ts: 172_800,
    })
    expect(incidentReviewBounds('all', 172_800_000)).toEqual({})
  })

  it('keeps long incident durations compact', () => {
    expect(formatReviewDuration(25_000)).toBe('<1m')
    expect(formatReviewDuration(65 * 60_000)).toBe('1h 5m')
    expect(formatReviewDuration(27 * 60 * 60_000)).toBe('1d 3h')
  })
})
