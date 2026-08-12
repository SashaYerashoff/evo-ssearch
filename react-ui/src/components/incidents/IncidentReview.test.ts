import { describe, expect, it } from 'vitest'
import { compareIncidentReviewPriority, formatReviewDuration, incidentReviewBounds } from './IncidentReview'

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

  it('puts configured operator criteria before generic context candidates', () => {
    const context = {
      incident_id: 'context',
      review_state: 'needs_review' as const,
      severity: 'high',
      priority: 'context',
      channels: [112],
      evidence_count: 1,
      timeline_count: 1,
      uncertainty_count: 1,
      last_evidence_ms: 2_000,
    }
    const configured = {
      ...context,
      incident_id: 'configured',
      severity: 'info',
      priority: 'operator_criterion',
      last_evidence_ms: 1_000,
    }

    expect([context, configured].sort(compareIncidentReviewPriority).map((item) => item.incident_id))
      .toEqual(['configured', 'context'])
  })
})
