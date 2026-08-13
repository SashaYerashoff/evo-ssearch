import { describe, expect, it } from 'vitest'
import { resolveVideoWorkspaceTab, visibleVideoWorkspaceTabs } from './StreamControl'

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
