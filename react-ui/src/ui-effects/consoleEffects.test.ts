import { describe, expect, it } from 'vitest'
import { buildAgentConsoleContext, normalizeConsoleUiEffects } from './consoleEffects'

describe('console effects', () => {
  it('accepts only closed server-owned effect actions', () => {
    expect(normalizeConsoleUiEffects([
      {
        version: 1,
        effect_id: 'e1',
        target: 'archive',
        action: 'show_results',
        source: { tool: 'search_archive', committed: false },
        payload: { channel_id: 7, query: 'gate' },
      },
      {
        version: 1,
        effect_id: 'e2',
        target: 'settings',
        action: 'delete_everything',
        source: { tool: 'search_archive' },
      },
    ])).toEqual([{
      version: 1,
      effectId: 'e1',
      target: 'archive',
      action: 'show_results',
      source: { tool: 'search_archive', committed: false },
      payload: { channel_id: 7, query: 'gate' },
    }])
  })

  it('builds structured archive defaults without prompt prose', () => {
    expect(buildAgentConsoleContext('archive', {
      channelId: '7',
      source: 'probe',
      probeId: 'door',
      hours: '1',
      sortBy: 'time',
      rows: '24',
    }, 7_200_000)).toEqual({
      version: 1,
      section: 'archive',
      archive: {
        channel_id: 7,
        source: 'probe',
        probe_id: 'door',
        since_ms: 3_600_000,
        until_ms: 7_200_000,
        sort_by: 'time',
        rows: 24,
      },
    })
  })

  it('maps the monitoring route to the probes domain', () => {
    expect(buildAgentConsoleContext('monitoring', null)).toEqual({
      version: 1,
      section: 'probes',
    })
  })
})
