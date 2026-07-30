import { describe, expect, it } from 'vitest'
import {
  summaryAlertCounts,
  summaryBurst,
  summaryEntryKey,
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
})
