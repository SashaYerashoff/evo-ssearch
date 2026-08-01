import { describe, expect, it } from 'vitest'
import {
  resolveSummaryResolution,
  summaryAlertCounts,
  summaryBurst,
  summaryEntryKey,
  summaryPeriodBounds,
  summarySemanticStatus,
  splitSummaryMachineJson,
} from './summaryView'

describe('video summary view metadata', () => {
  it('keeps alert counts and falls back to alert_total', () => {
    expect(summaryAlertCounts({
      alert_counts: { critical: 1, low: 2 },
    })).toEqual({ critical: 1, low: 2 })
    expect(summaryAlertCounts({ alert_total: 3, severity: 'high' })).toEqual({ high: 3 })
  })

  it('restores alert badges from structured events when flat counts are absent', () => {
    expect(summaryAlertCounts({
      batch_state: {
        alerts: [
          { severity: 'high', title: 'Person entered' },
          { severity: 'warning', title: 'Vehicle stopped' },
        ],
      },
    })).toEqual({ high: 1, low: 1 })
    expect(summaryAlertCounts({
      summary: 'Episode update\nMovement detected.\n\nBATCH_STATE_JSON:\n'
        + '{"alerts":[{"severity":"critical"},{"severity":"critical"}]}',
    })).toEqual({ critical: 2 })
  })

  it('extracts homeostatic burst intensity and snapshots', () => {
    expect(summaryBurst({
      vector_signal: {
        capture_attention: {
          seconds: [
            { mode: 'quiet', activity_x: 0.2 },
            { mode: 'burst', activity_x: 8.1, snapshot: 4 },
            { mode: 'burst', activity_x: 12.4, snapshot: 6 },
          ],
        },
      },
    })).toEqual({
      count: 2,
      maxActivity: 12.4,
      snapshots: ['4', '6'],
    })
  })

  it('makes queued semantic work visible without alarm styling', () => {
    expect(summarySemanticStatus({
      summary_kind: 'queued',
      generation_status: 'deferred',
    })).toMatchObject({
      label: 'semantic queued',
      tone: 'queued',
    })
  })

  it('builds stable keys from batch and rollup identity', () => {
    expect(summaryEntryKey({ batch_id: 'batch-7', run_id: 'run-1' }, 3)).toBe('batch-7:run-1')
    expect(summaryEntryKey({}, 3)).toBe('summary-3')
  })

  it('separates the machine state from the operator narrative', () => {
    expect(splitSummaryMachineJson(
      'A cat jumps from the shelf.\n\nBATCH_STATE_JSON:\n{"events":[]}',
    )).toEqual({
      narrative: 'A cat jumps from the shelf.',
      machineJson: '{"events":[]}',
      marker: 'BATCH_STATE_JSON:',
    })
  })

  it('matches the legacy period windows in the browser timezone', () => {
    const now = new Date(2026, 6, 31, 12, 30, 0).getTime()
    expect(summaryPeriodBounds('live', now)).toEqual({})
    expect(summaryPeriodBounds('today', now)).toEqual({
      from_ts: new Date(2026, 6, 31).getTime() / 1000,
      to_ts: now / 1000,
    })
    expect(summaryPeriodBounds('yesterday', now)).toEqual({
      from_ts: new Date(2026, 6, 30).getTime() / 1000,
      to_ts: new Date(2026, 6, 31).getTime() / 1000 - 0.001,
    })
    expect(summaryPeriodBounds('custom', now, { from_ts: 20, to_ts: 10 })).toEqual({
      from_ts: 10,
      to_ts: 20,
    })
  })

  it('uses the legacy Auto resolution policy', () => {
    expect(resolveSummaryResolution('AUTO', 'live')).toBe('L0')
    expect(resolveSummaryResolution('AUTO', 'today')).toBe('L1')
    expect(resolveSummaryResolution('AUTO', '7d')).toBe('L2')
    expect(resolveSummaryResolution('AUTO', '30d')).toBe('L3')
    expect(resolveSummaryResolution('AUTO', 'custom', { from_ts: 0, to_ts: 6 * 3600 })).toBe('L0')
    expect(resolveSummaryResolution('L2', 'live')).toBe('L2')
  })
})
