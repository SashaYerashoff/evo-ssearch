import { describe, expect, it } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { ActionCard, actionTables, deploymentInventoryChannels, deploymentScopeSelectionMessage } from './ActionCard'

describe('agent action card structures', () => {
  it('restores the complete authorized deployment picker after history reload', () => {
    const result = {
      available_channel_count: 4,
      available_channel_ids: [11, 12, 13, 14],
      available_channels: [
        { id: 11, title: 'Gate' },
        { id: 12, title: 'Beach' },
      ],
    }
    const channels = deploymentInventoryChannels(result, [
      { id: 11, title: 'Gate current' },
      { id: 12, title: 'Beach current' },
      { id: 13, title: 'Fairway' },
      { id: 14, title: 'Harbour wall' },
      { id: 99, title: 'Unauthorized for this inventory' },
    ])

    expect(channels.map((channel) => channel.id)).toEqual([11, 12, 13, 14])
    expect(channels.map((channel) => channel.title)).toEqual([
      'Gate', 'Beach', 'Fairway', 'Harbour wall',
    ])
  })

  it('requires closed maritime roles in the deployment inventory card', () => {
    const markup = renderToStaticMarkup(createElement(ActionCard, {
        action: {
          id: 1,
          name: 'start_deployment',
          result: {
            deployment_id: 'deploy-port-1',
            deployment_profile: 'maritime',
            stage: 'inventory',
            target_channel_count: 8,
            available_channel_ids: [41],
            available_channels: [{ id: 41, title: 'North gate' }],
          },
        },
        onThumb: () => undefined,
        onApply: () => undefined,
        onSend: () => undefined,
      }))

    expect(markup).toContain('North gate')
    expect(markup).toContain('A maritime role is required for every selected channel')
    expect(deploymentScopeSelectionMessage(
      { deployment_id: 'deploy-port-1', deployment_profile: 'maritime' },
      [41, 42],
      { 41: 'gates', 42: 'gates' },
      {
        41: { role: 'maritime_gate', location: 'Ventspils north gate' },
        42: { role: 'maritime_mixed_ptz', location: 'West coast' },
      },
    )).toBe(
      'Continue Protocol Deploy deploy-port-1. Select channels 41, 42; '
      + 'group gates: 41; group gates: 42; '
      + 'CH 41 role maritime_gate location "Ventspils north gate"; '
      + 'CH 42 role maritime_mixed_ptz location "West coast"',
    )
  })

  it('requires an explicit maritime starter-watch choice before scope review', () => {
    const markup = renderToStaticMarkup(createElement(ActionCard, {
      action: {
        id: 2,
        name: 'survey_deployment',
        result: {
          deployment_id: 'deploy-port-1',
          deployment_profile: 'maritime',
          stage: 'surveyed',
          selected_channel_ids: [41],
          groups: [{ name: 'gates', channel_ids: [41] }],
          surveys: [{ channel_id: 41, title: 'North gate', scene_fingerprint: 'VIEW: port gate' }],
        },
      },
      onThumb: () => undefined,
      onApply: () => undefined,
      onSend: () => undefined,
    }))

    expect(markup).toContain('Role-specific starter watches')
    expect(markup).toContain('Propose as non-bookmarking shadow probes')
    expect(markup).toContain('9B consolidation window')
    expect(markup).toContain('live L0 monitoring and alerts continue')
    expect(markup).toContain('Draft alerts for this scope')
  })

  it('returns a partial deployment to only the still-missing scope', () => {
    const markup = renderToStaticMarkup(createElement(ActionCard, {
      action: {
        id: 3,
        name: 'configure_deployment',
        result: {
          deployment_id: 'deploy-port-1',
          deployment_profile: 'maritime',
          starter_policy_mode: 'shadow',
          starter_policy_confirmed: true,
          stage: 'requirements_partial',
          selected_channel_ids: [41, 42],
          missing_requirement_channel_ids: [42],
          groups: [
            { name: 'north_gate', channel_ids: [41] },
            { name: 'west_coast', channel_ids: [42] },
          ],
          surveys: [
            { channel_id: 41, title: 'North gate', scene_fingerprint: 'VIEW: port gate' },
            { channel_id: 42, title: 'West coast', scene_fingerprint: 'VIEW: coastline' },
          ],
        },
      },
      onThumb: () => undefined,
      onApply: () => undefined,
      onSend: () => undefined,
    }))

    expect(markup).toContain('west_coast')
    expect(markup).toContain('West coast')
    expect(markup).not.toContain('North gate')
    expect(markup).toContain('value="shadow" selected=""')
  })

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

  it('surfaces one bounded archive vision batch with candidate count', () => {
    const tables = actionTables('describe_frame', {
      source: 'archive_candidate_batch',
      candidate_count: 8,
      parse_status: 'parsed',
      vision_checked: true,
      verdicts: [
        { detection_id: 71, verdict: 'match', visible_evidence: 'A person is visibly seated.' },
        { detection_id: 72, verdict: 'uncertain', visible_evidence: 'The chair is partially occluded.' },
      ],
    })

    expect(tables).toHaveLength(1)
    expect(tables[0].title).toBe('Bounded vision verification · 8 candidates · parsed')
    expect(tables[0].rows).toHaveLength(2)
    expect(tables[0].rows[0]).toEqual(['#71', 'match', 'A person is visibly seated.'])
  })

  it('keeps a failed bounded archive vision step visible', () => {
    const tables = actionTables('describe_frame', {
      source: 'archive_candidate_batch',
      candidate_count: 8,
      status: 'failed',
      error: 'Vision request failed',
    })

    expect(tables[0].title).toBe('Bounded vision verification · 8 candidates · failed')
    expect(tables[0].rows[0]).toEqual(['—', 'failed', 'Vision request failed'])
  })
})
