import { describe, expect, it } from 'vitest'
import { describeVlmSampling, resolveVideoWorkspaceTab, visibleVideoWorkspaceTabs } from './StreamControl'

describe('video workspace feature visibility', () => {
  it('keeps the FiP incident tab available by default', () => {
    expect(visibleVideoWorkspaceTabs(true)).toEqual(['review', 'incidents', 'settings'])
    expect(resolveVideoWorkspaceTab('incidents', true)).toBe('incidents')
  })

  it('removes Incident Review and returns an open incident tab to review', () => {
    expect(visibleVideoWorkspaceTabs(false)).toEqual(['review', 'settings'])
    expect(resolveVideoWorkspaceTab('incidents', false)).toBe('review')
    expect(resolveVideoWorkspaceTab('settings', false)).toBe('settings')
  })
})

describe('VLM sampling contract', () => {
  it('shows complete default 8/2 coverage', () => {
    expect(describeVlmSampling('8', '2', 8)).toEqual({
      compressed: false,
      label: 'VLM sees all 8 frames · hard cap 8 · ~14s span',
    })
  })

  it('marks legacy wider batches as partial attention-selected coverage', () => {
    expect(describeVlmSampling('12', '2', 8)).toEqual({
      compressed: true,
      label: 'VLM sees 8 of 12 frames · attention-selected · partial coverage · hard cap 8 · ~22s span',
    })
  })

  it('does not confuse a smaller selector budget with the endpoint hard cap', () => {
    expect(describeVlmSampling('12', '2', 4, 8).label).toContain(
      'selection budget 4 · hard cap 8',
    )
  })
})
