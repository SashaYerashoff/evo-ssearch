export const APPEARANCE_STORAGE_KEY = 'eva.ui.appearance.v1'
export const CUSTOM_APPEARANCE_PRESETS_STORAGE_KEY = 'eva.ui.appearance.custom-presets.v1'

export type ThemePresetId = 'eva-deep' | 'graphite' | 'day-shift' | 'amber-watch'
export type ControlShape = 'precise' | 'balanced' | 'soft'
export type InterfaceDensity = 'compact' | 'comfortable'
export type InterfaceScale = 'normal' | 'large'
export type MotionPreference = 'system' | 'full' | 'reduced'
export type CustomColorKey = 'canvas' | 'surface' | 'control' | 'text' | 'accent'

export interface ThemePalette {
  scheme: 'dark' | 'light'
  canvasDeep: string
  canvas: string
  surface: string
  surfaceRaised: string
  control: string
  text: string
  textSecondary: string
  textMuted: string
  accent: string
  danger: string
  warning: string
  success: string
  info: string
}

export interface ThemePreset {
  id: ThemePresetId
  label: string
  description: string
  palette: ThemePalette
}

export interface AppearancePreferences {
  version: 1
  preset: ThemePresetId
  overrides: Partial<Record<CustomColorKey, string>>
  shape: ControlShape
  density: InterfaceDensity
  scale: InterfaceScale
  motion: MotionPreference
}

export interface SavedAppearancePreset {
  id: string
  name: string
  preferences: AppearancePreferences
}

export const THEME_PRESETS: readonly ThemePreset[] = [
  {
    id: 'eva-deep',
    label: 'EVA Deep',
    description: 'The original spatial command-console palette.',
    palette: {
      scheme: 'dark',
      canvasDeep: '#10162e',
      canvas: '#18203c',
      surface: '#1f2a4c',
      surfaceRaised: '#26335a',
      control: '#1d2743',
      text: '#eef2fc',
      textSecondary: '#b7c3e2',
      textMuted: '#8494b8',
      accent: '#38e0d4',
      danger: '#ff5d73',
      warning: '#ffb454',
      success: '#3ee6a4',
      info: '#6aa8ff',
    },
  },
  {
    id: 'graphite',
    label: 'Graphite',
    description: 'Neutral low-glare surfaces with a restrained blue signal.',
    palette: {
      scheme: 'dark',
      canvasDeep: '#0b0e12',
      canvas: '#14181e',
      surface: '#20262e',
      surfaceRaised: '#29313c',
      control: '#181d24',
      text: '#f1f4f7',
      textSecondary: '#c0c8d2',
      textMuted: '#8793a1',
      accent: '#79aaff',
      danger: '#ff6577',
      warning: '#f5b85b',
      success: '#47d99b',
      info: '#79aaff',
    },
  },
  {
    id: 'day-shift',
    label: 'Day Shift',
    description: 'A quiet light palette for bright control rooms.',
    palette: {
      scheme: 'light',
      canvasDeep: '#dfe6ef',
      canvas: '#e9eef5',
      surface: '#f8fafc',
      surfaceRaised: '#ffffff',
      control: '#edf2f7',
      text: '#17202e',
      textSecondary: '#3d4a5c',
      textMuted: '#6f7d90',
      accent: '#087f8c',
      danger: '#c9364d',
      warning: '#a55b08',
      success: '#167a55',
      info: '#2567bd',
    },
  },
  {
    id: 'amber-watch',
    label: 'Amber Watch',
    description: 'Warm highlights for long night shifts and low blue light.',
    palette: {
      scheme: 'dark',
      canvasDeep: '#14110d',
      canvas: '#211b14',
      surface: '#30271d',
      surfaceRaised: '#3a3024',
      control: '#271f17',
      text: '#f5eee3',
      textSecondary: '#d3c4b0',
      textMuted: '#9c8b75',
      accent: '#ffb454',
      danger: '#ff6673',
      warning: '#ffb454',
      success: '#61d89b',
      info: '#7db7e8',
    },
  },
] as const

export const DEFAULT_APPEARANCE: AppearancePreferences = {
  version: 1,
  preset: 'eva-deep',
  overrides: {},
  shape: 'balanced',
  density: 'comfortable',
  scale: 'normal',
  motion: 'system',
}

const PRESET_IDS = new Set(THEME_PRESETS.map((preset) => preset.id))
const SHAPES = new Set<ControlShape>(['precise', 'balanced', 'soft'])
const DENSITIES = new Set<InterfaceDensity>(['compact', 'comfortable'])
const SCALES = new Set<InterfaceScale>(['normal', 'large'])
const MOTION = new Set<MotionPreference>(['system', 'full', 'reduced'])
const CUSTOM_COLOR_KEYS: readonly CustomColorKey[] = ['canvas', 'surface', 'control', 'text', 'accent']

export function normalizeHex(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  const short = /^#([0-9a-f]{3})$/i.exec(trimmed)
  if (short) {
    const [r, g, b] = short[1].split('')
    return `#${r}${r}${g}${g}${b}${b}`.toLowerCase()
  }
  return /^#[0-9a-f]{6}$/i.test(trimmed) ? trimmed.toLowerCase() : null
}

export function normalizeAppearance(value: unknown): AppearancePreferences {
  if (!value || typeof value !== 'object') return { ...DEFAULT_APPEARANCE, overrides: {} }
  const candidate = value as Partial<AppearancePreferences>
  const overrides: AppearancePreferences['overrides'] = {}
  if (candidate.overrides && typeof candidate.overrides === 'object') {
    for (const key of CUSTOM_COLOR_KEYS) {
      const color = normalizeHex(candidate.overrides[key])
      if (color) overrides[key] = color
    }
  }
  return {
    version: 1,
    preset: PRESET_IDS.has(candidate.preset as ThemePresetId)
      ? candidate.preset as ThemePresetId
      : DEFAULT_APPEARANCE.preset,
    overrides,
    shape: SHAPES.has(candidate.shape as ControlShape)
      ? candidate.shape as ControlShape
      : DEFAULT_APPEARANCE.shape,
    density: DENSITIES.has(candidate.density as InterfaceDensity)
      ? candidate.density as InterfaceDensity
      : DEFAULT_APPEARANCE.density,
    scale: SCALES.has(candidate.scale as InterfaceScale)
      ? candidate.scale as InterfaceScale
      : DEFAULT_APPEARANCE.scale,
    motion: MOTION.has(candidate.motion as MotionPreference)
      ? candidate.motion as MotionPreference
      : DEFAULT_APPEARANCE.motion,
  }
}

export function normalizeSavedAppearancePresets(value: unknown): SavedAppearancePreset[] {
  if (!Array.isArray(value)) return []
  const seen = new Set<string>()
  const result: SavedAppearancePreset[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object') continue
    const candidate = item as Partial<SavedAppearancePreset>
    const name = String(candidate.name || '').replace(/\s+/g, ' ').trim().slice(0, 48)
    const id = String(candidate.id || '').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 80)
    if (!name || !id || seen.has(id)) continue
    seen.add(id)
    result.push({ id, name, preferences: normalizeAppearance(candidate.preferences) })
    if (result.length >= 12) break
  }
  return result
}

export function getThemePreset(id: ThemePresetId): ThemePreset {
  return THEME_PRESETS.find((preset) => preset.id === id) || THEME_PRESETS[0]
}

export function resolveThemePalette(preferences: AppearancePreferences): ThemePalette {
  const base = getThemePreset(preferences.preset).palette
  const canvas = preferences.overrides.canvas || base.canvas
  const surface = preferences.overrides.surface || base.surface
  const control = preferences.overrides.control || base.control
  const text = preferences.overrides.text || base.text
  const accent = preferences.overrides.accent || base.accent
  const pole = base.scheme === 'dark' ? '#000000' : '#ffffff'

  return {
    ...base,
    canvas,
    canvasDeep: preferences.overrides.canvas ? mixHex(canvas, pole, 0.2) : base.canvasDeep,
    surface,
    surfaceRaised: preferences.overrides.surface
      ? mixHex(surface, base.scheme === 'dark' ? text : pole, base.scheme === 'dark' ? 0.07 : 0.38)
      : base.surfaceRaised,
    control,
    text,
    textSecondary: preferences.overrides.text ? mixHex(text, canvas, 0.27) : base.textSecondary,
    textMuted: preferences.overrides.text ? mixHex(text, canvas, 0.48) : base.textMuted,
    accent,
  }
}

export function resolveThemeVariables(preferences: AppearancePreferences): Record<string, string> {
  const palette = resolveThemePalette(preferences)
  const dark = palette.scheme === 'dark'
  const accent2 = mixHex(palette.accent, dark ? '#ffffff' : '#000000', dark ? 0.13 : 0.12)
  const controlHover = mixHex(palette.control, palette.text, dark ? 0.08 : 0.045)
  const radius = preferences.shape === 'precise' ? 5 : preferences.shape === 'soft' ? 15 : 10
  const density = preferences.density === 'compact'
  const primaryText = contrastText(palette.accent)

  return {
    '--space-0': palette.canvasDeep,
    '--space-1': palette.canvas,
    '--panel': rgba(palette.surface, dark ? 0.92 : 0.96),
    '--panel-2': rgba(palette.surfaceRaised, dark ? 0.9 : 0.98),
    '--panel-solid': palette.surfaceRaised,
    '--void-tile': palette.control,
    '--surface': palette.surface,
    '--surface-raised': palette.surfaceRaised,
    '--control': palette.control,
    '--control-hover': controlHover,
    '--line': rgba(palette.text, dark ? 0.18 : 0.14),
    '--line-2': rgba(palette.text, dark ? 0.31 : 0.25),
    '--text': palette.text,
    '--text-2': palette.textSecondary,
    '--text-mut': palette.textMuted,
    '--accent': palette.accent,
    '--accent-2': accent2,
    '--accent-bg': rgba(palette.accent, dark ? 0.14 : 0.11),
    '--accent-line': rgba(palette.accent, dark ? 0.52 : 0.58),
    '--accent-soft': rgba(palette.accent, dark ? 0.08 : 0.07),
    '--primary-text': primaryText,
    '--danger': palette.danger,
    '--danger-bg': rgba(palette.danger, 0.15),
    '--warning': palette.warning,
    '--warning-bg': rgba(palette.warning, 0.15),
    '--success': palette.success,
    '--success-bg': rgba(palette.success, 0.15),
    '--info': palette.info,
    '--info-bg': rgba(palette.info, 0.14),
    '--radius': `${radius}px`,
    '--radius-lg': `${radius + 6}px`,
    '--control-radius': `${radius}px`,
    '--control-pad-y': density ? '7px' : '9px',
    '--control-pad-x': density ? '10px' : '12px',
    '--control-min-h': density ? '34px' : '38px',
    '--section-gap': density ? '10px' : '14px',
    '--backdrop-accent': rgba(palette.accent, dark ? 0.13 : 0.08),
    '--backdrop-info': rgba(palette.info, dark ? 0.14 : 0.08),
    '--backdrop-mid': rgba(palette.text, dark ? 0.06 : 0.025),
    '--topbar-start': rgba(palette.canvasDeep, dark ? 0.88 : 0.94),
    '--topbar-end': rgba(palette.canvas, dark ? 0.74 : 0.88),
    '--scrim': rgba(palette.canvasDeep, dark ? 0.78 : 0.55),
    '--elev': dark
      ? `0 14px 34px -16px ${rgba('#030714', 0.75)}, 0 2px 8px -4px ${rgba('#030714', 0.5)}`
      : `0 14px 34px -18px ${rgba('#526176', 0.34)}, 0 2px 8px -4px ${rgba('#526176', 0.24)}`,
    '--glow': `0 0 0 1px ${rgba(palette.accent, 0.34)}, 0 0 18px -2px ${rgba(palette.accent, 0.42)}`,
  }
}

export function hasCustomColors(preferences: AppearancePreferences): boolean {
  return Object.keys(preferences.overrides).length > 0
}

function hexToRgb(hex: string): [number, number, number] {
  const normalized = normalizeHex(hex) || '#000000'
  return [
    Number.parseInt(normalized.slice(1, 3), 16),
    Number.parseInt(normalized.slice(3, 5), 16),
    Number.parseInt(normalized.slice(5, 7), 16),
  ]
}

function componentToHex(value: number): string {
  return Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0')
}

export function mixHex(from: string, to: string, amount: number): string {
  const [fr, fg, fb] = hexToRgb(from)
  const [tr, tg, tb] = hexToRgb(to)
  const ratio = Math.max(0, Math.min(1, amount))
  return `#${componentToHex(fr + (tr - fr) * ratio)}${componentToHex(fg + (tg - fg) * ratio)}${componentToHex(fb + (tb - fb) * ratio)}`
}

export function rgba(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex)
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`
}

export function contrastText(background: string): '#07120f' | '#ffffff' {
  const dark = '#07120f'
  const light = '#ffffff'
  return contrastRatio(dark, background) >= contrastRatio(light, background) ? dark : light
}

export function contrastRatio(foreground: string, background: string): number {
  const lighter = Math.max(relativeLuminance(foreground), relativeLuminance(background))
  const darker = Math.min(relativeLuminance(foreground), relativeLuminance(background))
  return (lighter + 0.05) / (darker + 0.05)
}

function relativeLuminance(color: string): number {
  const [r, g, b] = hexToRgb(color).map((component) => {
    const normalized = component / 255
    return normalized <= 0.03928
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}
