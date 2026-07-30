import { describe, expect, it } from 'vitest'
import {
  agentDragGeometry,
  archiveColumnsForAgentWidth,
  agentWidthPresets,
  closestAgentWidthPresetIndex,
  nextAgentWidthPresetIndex,
} from './agentWidthPresets'

describe('agent width presets', () => {
  it('uses stable 4/3 Archive layouts on Full HD and never drops below three cards', () => {
    const presets = agentWidthPresets(1920)
    expect(presets.map((preset) => preset.archiveColumns)).toEqual([4, 3])
    expect(presets.map((preset) => preset.width)).toEqual([616, 924])
  })

  it('uses stable 5/4/3 Archive layouts on 2K', () => {
    const presets = agentWidthPresets(2560)
    expect(presets.map((preset) => preset.archiveColumns)).toEqual([5, 4, 3])
    expect(presets.map((preset) => preset.width)).toEqual([414, 829, 1244])
  })

  it('uses CSS viewport width so every preset fits without shrinking cards', () => {
    const presets = agentWidthPresets(2048)
    expect(presets[0].profile).toBe('full-hd')
    expect(presets.map((preset) => preset.archiveColumns)).toEqual([4, 3])
  })

  it('snaps a dragged width to the nearest preset', () => {
    const presets = agentWidthPresets(1920)
    expect(closestAgentWidthPresetIndex(800, presets)).toBe(1)
    expect(closestAgentWidthPresetIndex(1180, presets)).toBe(1)
  })

  it('moves complete cards away instead of resizing them during drag', () => {
    expect(archiveColumnsForAgentWidth(0, 1920)).toBe(6)
    expect(archiveColumnsForAgentWidth(616, 1920)).toBe(4)
    expect(archiveColumnsForAgentWidth(924, 1920)).toBe(3)
  })

  it('cycles presets in a loop', () => {
    expect(nextAgentWidthPresetIndex(0, 3)).toBe(1)
    expect(nextAgentWidthPresetIndex(1, 3)).toBe(2)
    expect(nextAgentWidthPresetIndex(2, 3)).toBe(0)
  })

  it('turns overdrag into an overlay without consuming more console width', () => {
    const presets = agentWidthPresets(1920)
    expect(agentDragGeometry(800, presets)).toEqual({ layoutWidth: 800, overlayWidth: 0 })
    expect(agentDragGeometry(1200, presets)).toEqual({ layoutWidth: 924, overlayWidth: 276 })
  })
})
