import { describe, expect, it } from 'vitest'
import { restoreAgentTranscript } from './agentTranscript'

describe('agent transcript restore', () => {
  it('reattaches persisted tool results to the assistant trace', () => {
    const restored = restoreAgentTranscript([
      { role: 'user', content: 'Find the cat' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{
          id: 'call-1',
          type: 'function',
          function: { name: 'search_archive', arguments: '{"query":"cat"}' },
        }],
      },
      {
        role: 'tool',
        content: '',
        tool_call_id: 'call-1',
        tool_name: 'search_archive',
        tool_result: '{"results":[{"id":7}]}',
      },
      { role: 'assistant', content: 'I found one match.' },
    ])

    expect(restored).toHaveLength(3)
    expect(restored[1].actions?.[0]).toMatchObject({
      name: 'search_archive',
      result: { results: [{ id: 7 }] },
    })
    expect(restored[2].text).toBe('I found one match.')
  })

  it('renders trusted apply receipts as applied actions', () => {
    const restored = restoreAgentTranscript([
      { role: 'assistant', content: 'Preview ready.' },
      {
        role: 'system',
        content: 'Trusted receipt',
        tool_name: 'action_receipt',
        tool_result: '{"tool":"update_probe","status":"applied","probe_id":"p1"}',
      },
    ])
    expect(restored[0].actions?.[0]).toMatchObject({
      name: 'update_probe',
      applied: true,
    })
  })
})
