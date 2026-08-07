import { describe, expect, it } from 'vitest'
import {
  DEFAULT_APPEARANCE,
  THEME_PRESETS,
  contrastRatio,
  contrastText,
  mixHex,
  normalizeAppearance,
  normalizeSavedAppearancePresets,
  normalizeHex,
  resolveThemePalette,
  resolveThemeVariables,
} from './appearance'

describe('appearance preferences', () => {
  it('normalizes persisted values and rejects unknown keys', () => {
    expect(normalizeAppearance({
      preset: 'day-shift',
      shape: 'soft',
      density: 'compact',
      motion: 'reduced',
      overrides: {
        canvas: '#abc',
        accent: '#102030',
        text: 'red',
        unexpected: '#ffffff',
      },
    })).toEqual({
      version: 1,
      preset: 'day-shift',
      shape: 'soft',
      density: 'compact',
      motion: 'reduced',
      overrides: {
        canvas: '#aabbcc',
        accent: '#102030',
      },
    })
  })

  it('falls back safely when local data is corrupt', () => {
    expect(normalizeAppearance({ preset: 'unknown', overrides: [] })).toEqual(DEFAULT_APPEARANCE)
    expect(normalizeAppearance(null)).toEqual(DEFAULT_APPEARANCE)
  })

  it('bounds and normalizes named custom presets', () => {
    expect(normalizeSavedAppearancePresets([
      {
        id: 'night watch!',
        name: '  Night   watch  ',
        preferences: { ...DEFAULT_APPEARANCE, preset: 'amber-watch', overrides: { accent: '#abc' } },
      },
      { id: '', name: 'broken', preferences: {} },
    ])).toEqual([
      {
        id: 'nightwatch',
        name: 'Night watch',
        preferences: { ...DEFAULT_APPEARANCE, preset: 'amber-watch', overrides: { accent: '#aabbcc' } },
      },
    ])
  })

  it('derives a coherent palette from semantic overrides', () => {
    const preferences = normalizeAppearance({
      ...DEFAULT_APPEARANCE,
      overrides: { canvas: '#202020', surface: '#303030', accent: '#ffcc00' },
    })
    const palette = resolveThemePalette(preferences)
    const variables = resolveThemeVariables(preferences)

    expect(palette.canvas).toBe('#202020')
    expect(palette.surface).toBe('#303030')
    expect(variables['--accent']).toBe('#ffcc00')
    expect(variables['--primary-text']).toBe('#07120f')
    expect(variables['--panel']).toContain('rgba(48, 48, 48')
  })

  it('ships presets with readable text and operational accents', () => {
    for (const preset of THEME_PRESETS) {
      const palette = resolveThemePalette(normalizeAppearance({
        ...DEFAULT_APPEARANCE,
        preset: preset.id,
      }))
      expect(contrastRatio(palette.text, palette.canvas), preset.id).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(palette.text, palette.surface), preset.id).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(palette.accent, palette.surface), preset.id).toBeGreaterThanOrEqual(3)
      expect(contrastRatio(contrastText(palette.accent), palette.accent), preset.id).toBeGreaterThanOrEqual(4.5)
    }
  })
})

describe('color helpers', () => {
  it('accepts three and six digit hex only', () => {
    expect(normalizeHex('#AbC')).toBe('#aabbcc')
    expect(normalizeHex('#A0b1C2')).toBe('#a0b1c2')
    expect(normalizeHex('rgb(0,0,0)')).toBeNull()
  })

  it('mixes and selects readable foregrounds', () => {
    expect(mixHex('#000000', '#ffffff', 0.5)).toBe('#808080')
    expect(contrastText('#f5c542')).toBe('#07120f')
    expect(contrastText('#112233')).toBe('#ffffff')
    expect(contrastRatio('#ffffff', '#000000')).toBe(21)
    expect(contrastRatio('#777777', '#777777')).toBe(1)
  })
})
