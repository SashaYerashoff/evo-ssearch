import { describe, expect, it } from 'vitest'
import { archivePlaybackWindow } from '../../api/detections'
import type { Detection } from '../../api/types'

function detection(payload: Record<string, unknown>, tsMs = 120_000): Detection {
  return {
    key: 'vlm_alert:1',
    id: 1,
    channelId: 112,
    probeName: 'alert',
    source: 'vlm_alert',
    sourceLabel: 'VLM alert',
    severity: 'high',
    tsMs,
    raw: { payload },
  }
}

describe('archivePlaybackWindow', () => {
  it('centres recorder playback on the evidence timestamp within batch bounds', () => {
    expect(archivePlaybackWindow(detection({
      batch_start_ms: 100_000,
      batch_end_ms: 160_000,
      anchor_frame_timestamp_ms: 130_000,
    }))).toEqual({
      startMs: 125_000,
      durationSec: 15,
      evidenceMs: 130_000,
      batchStartMs: 100_000,
      batchEndMs: 160_000,
    })
  })

  it('does not seek before the observed batch at its leading edge', () => {
    expect(archivePlaybackWindow(detection({
      batch_start_ms: 100_000,
      batch_end_ms: 108_000,
      anchor_frame_timestamp_ms: 101_000,
    }))).toMatchObject({ startMs: 100_000, durationSec: 9, evidenceMs: 101_000 })
  })

  it('returns null without a channel or evidence timestamp', () => {
    expect(archivePlaybackWindow({ ...detection({}, 0), channelId: null })).toBeNull()
  })
})
