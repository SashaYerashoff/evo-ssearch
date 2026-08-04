import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  APPEARANCE_STORAGE_KEY,
  DEFAULT_APPEARANCE,
  getThemePreset,
  normalizeAppearance,
  resolveThemeVariables,
  type AppearancePreferences,
} from './appearance'

interface AppearanceContextValue {
  savedPreferences: AppearancePreferences
  activePreferences: AppearancePreferences
  isMotionReduced: boolean
  previewPreferences: (preferences: AppearancePreferences) => void
  cancelPreview: () => void
  commitPreferences: (preferences: AppearancePreferences) => void
}

const AppearanceContext = createContext<AppearanceContextValue | null>(null)

function readPreferences(): AppearancePreferences {
  if (typeof window === 'undefined') return { ...DEFAULT_APPEARANCE, overrides: {} }
  try {
    return normalizeAppearance(JSON.parse(window.localStorage.getItem(APPEARANCE_STORAGE_KEY) || 'null'))
  } catch {
    return { ...DEFAULT_APPEARANCE, overrides: {} }
  }
}

export function AppearanceProvider({ children }: { children: ReactNode }) {
  const [savedPreferences, setSavedPreferences] = useState<AppearancePreferences>(readPreferences)
  const [preview, setPreview] = useState<AppearancePreferences | null>(null)
  const [systemReducedMotion, setSystemReducedMotion] = useState(false)
  const activePreferences = preview || savedPreferences
  const preset = getThemePreset(activePreferences.preset)
  const isMotionReduced = activePreferences.motion === 'reduced'
    || (activePreferences.motion === 'system' && systemReducedMotion)

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setSystemReducedMotion(media.matches)
    update()
    media.addEventListener?.('change', update)
    return () => media.removeEventListener?.('change', update)
  }, [])

  useLayoutEffect(() => {
    const root = document.documentElement
    const variables = resolveThemeVariables(activePreferences)
    Object.entries(variables).forEach(([key, value]) => root.style.setProperty(key, value))
    root.dataset.themePreset = activePreferences.preset
    root.dataset.colorScheme = preset.palette.scheme
    root.dataset.density = activePreferences.density
    root.dataset.interfaceScale = activePreferences.scale
    root.dataset.controlShape = activePreferences.shape
    root.style.colorScheme = preset.palette.scheme
    root.classList.toggle('anim-off', isMotionReduced)
  }, [activePreferences, isMotionReduced, preset.palette.scheme])

  const previewPreferences = useCallback((preferences: AppearancePreferences) => {
    setPreview(normalizeAppearance(preferences))
  }, [])

  const cancelPreview = useCallback(() => setPreview(null), [])

  const commitPreferences = useCallback((preferences: AppearancePreferences) => {
    const normalized = normalizeAppearance(preferences)
    setSavedPreferences(normalized)
    setPreview(null)
    try {
      window.localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify(normalized))
    } catch {
      // The active preference still works for this session in private/locked storage modes.
    }
  }, [])

  const value = useMemo<AppearanceContextValue>(() => ({
    savedPreferences,
    activePreferences,
    isMotionReduced,
    previewPreferences,
    cancelPreview,
    commitPreferences,
  }), [
    activePreferences,
    cancelPreview,
    commitPreferences,
    isMotionReduced,
    previewPreferences,
    savedPreferences,
  ])

  return <AppearanceContext.Provider value={value}>{children}</AppearanceContext.Provider>
}

export function useAppearance(): AppearanceContextValue {
  const context = useContext(AppearanceContext)
  if (!context) throw new Error('useAppearance must be used inside AppearanceProvider')
  return context
}
