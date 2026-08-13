import { describe, expect, it } from 'vitest'
import {
  DEFAULT_FEATURE_PREFERENCES,
  normalizeFeaturePreferences,
} from './featurePreferences'

describe('feature preferences', () => {
  it('keeps Incident Review visible for existing workstations by default', () => {
    expect(normalizeFeaturePreferences(null)).toEqual(DEFAULT_FEATURE_PREFERENCES)
    expect(normalizeFeaturePreferences({ version: 9 })).toEqual(DEFAULT_FEATURE_PREFERENCES)
  })

  it('accepts only an explicit boolean incident visibility preference', () => {
    expect(normalizeFeaturePreferences({ showIncidents: false }).showIncidents).toBe(false)
    expect(normalizeFeaturePreferences({ showIncidents: true }).showIncidents).toBe(true)
    expect(normalizeFeaturePreferences({ showIncidents: 'false' }).showIncidents).toBe(true)
  })
})
