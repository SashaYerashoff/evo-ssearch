import { describe, expect, it } from 'vitest'
import { actionTables } from './ActionCard'

describe('agent action card structures', () => {
  it('renders channel coverage as bounded operator rows', () => {
    const tables = actionTables('list_video_summary_channels', {
      active_count: 1,
      error_count: 1,
      candidate_channels: [
        {
          channel_id: 112,
          title: 'Gate',
          running: true,
          coverage_status: 'covered',
          summary_count: 14,
          alert_total: 2,
          dropped_frames: 3,
        },
      ],
    })
    expect(tables[0].title).toBe('1 active · 1 problems')
    expect(tables[0].rows[0]).toEqual([
      '#112 Gate',
      'running · covered',
      '14 summaries · 2 alerts · 3 dropped',
    ])
  })

  it('keeps prompt layers distinct in tool output', () => {
    const tables = actionTables('get_prompt_settings', {
      scope: 'channel',
      channel_id: 7,
      stream_system_prompt: 'Describe current evidence.',
      alert_policy_prompt: 'Alert on smoke.',
      json_alert_prompt: 'Return BATCH_STATE_JSON.',
      prompt_health: { needs_migration: false },
    })
    expect(tables[0].rows.map((row) => row[0])).toEqual([
      'L0 live role',
      'Alert criteria',
      'BATCH_STATE_JSON',
    ])
    expect(tables[0].rows[1][1]).toBe('separate')
  })
})
