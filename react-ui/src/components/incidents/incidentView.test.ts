import { describe, expect, it } from 'vitest'
import {
  followExpiryMs,
  formatIncidentDuration,
  incidentChannels,
  incidentTimeline,
  incidentTimestampMs,
} from './incidentView'

describe('incident view normalization', () => {
  it('orders the backend timeline without turning qualia into claims', () => {
    expect(incidentTimeline({
      timeline: [
        {
          semantic_key: 'risk_rising',
          timestamp_ms: 1_785_000_000_000,
          description: 'Two observed tracks converge.',
          confidence: 'medium',
        },
      ],
      qualia_timeline: [{ semantic_key: 'ignored_fallback', timestamp_ms: 1 }],
    })).toEqual([
      {
        key: 'risk_rising',
        label: 'risk rising',
        description: 'Two observed tracks converge.',
        timestampMs: 1_785_000_000_000,
        confidence: 'medium',
      },
    ])
  })

  it('normalizes channels, expiry and human durations', () => {
    expect(incidentChannels({ channel_id: 7, channels: [7, { channel_id: 8 }, '9'] })).toEqual(['7', '8', '9'])
    expect(incidentTimestampMs(1_785_000_000)).toBe(1_785_000_000_000)
    expect(followExpiryMs({ expires_at: '2026-08-01T12:00:00Z' })).toBe(Date.parse('2026-08-01T12:00:00Z'))
    expect(formatIncidentDuration(125_000)).toBe('2m 5s')
  })
})
