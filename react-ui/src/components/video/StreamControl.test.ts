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
  it('shows a sparse chronological default 12/1 evidence packet', () => {
    expect(describeVlmSampling('12', '1', 8)).toEqual({
      compressed: true,
      label: 'VLM sees 1–8 useful images from 12 samples · no padding · chronological · partial coverage · hard cap 8 · seals by ~12s',
    })
  })

  it('does not promise to pad a smaller window', () => {
    expect(describeVlmSampling('8', '2', 8)).toEqual({
      compressed: false,
      label: 'VLM sees 1–8 useful images · no padding · chronological · hard cap 8 · seals by ~16s',
    })
  })

  it('marks legacy wider batches as partial attention-selected coverage', () => {
    expect(describeVlmSampling('12', '2', 8)).toEqual({
      compressed: true,
      label: 'VLM sees 1–8 useful images from 12 samples · no padding · chronological · partial coverage · hard cap 8 · seals by ~24s',
    })
  })

  it('does not confuse a smaller selector budget with the endpoint hard cap', () => {
    expect(describeVlmSampling('12', '2', 4, 8).label).toContain(
      'selection budget 4 · hard cap 8',
    )
  })
})
