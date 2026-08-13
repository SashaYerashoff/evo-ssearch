export const FEATURE_PREFERENCES_STORAGE_KEY = 'eva.ui.features.v1'

export interface FeaturePreferences {
  version: 1
  showIncidents: boolean
}

export const DEFAULT_FEATURE_PREFERENCES: FeaturePreferences = {
  version: 1,
  showIncidents: true,
}

export function normalizeFeaturePreferences(value: unknown): FeaturePreferences {
  if (!value || typeof value !== 'object') return { ...DEFAULT_FEATURE_PREFERENCES }
  const candidate = value as Partial<FeaturePreferences>
  return {
    version: 1,
    showIncidents: typeof candidate.showIncidents === 'boolean'
      ? candidate.showIncidents
      : DEFAULT_FEATURE_PREFERENCES.showIncidents,
  }
}

export function readFeaturePreferences(): FeaturePreferences {
  if (typeof window === 'undefined') return { ...DEFAULT_FEATURE_PREFERENCES }
  try {
    return normalizeFeaturePreferences(
      JSON.parse(window.localStorage.getItem(FEATURE_PREFERENCES_STORAGE_KEY) || 'null'),
    )
  } catch {
    return { ...DEFAULT_FEATURE_PREFERENCES }
  }
}

export function persistFeaturePreferences(value: FeaturePreferences): FeaturePreferences {
  const normalized = normalizeFeaturePreferences(value)
  try {
    window.localStorage.setItem(FEATURE_PREFERENCES_STORAGE_KEY, JSON.stringify(normalized))
  } catch {
    // Keep the session preference active when browser storage is unavailable.
  }
  return normalized
}
