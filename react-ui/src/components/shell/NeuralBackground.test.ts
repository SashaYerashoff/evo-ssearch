import { describe, expect, it } from 'vitest'
import { NEURAL_BACKGROUND_TIMING } from './NeuralBackground'

describe('neural background timing', () => {
  it('keeps the ambient canvas continuous without rendering every display frame', () => {
    expect(1000 / NEURAL_BACKGROUND_TIMING.paintIntervalMs).toBeGreaterThanOrEqual(30)
    expect(1000 / NEURAL_BACKGROUND_TIMING.paintIntervalMs).toBeLessThan(60)
  })

  it('refreshes the moving aurora often enough to avoid visible position jumps', () => {
    expect(1000 / NEURAL_BACKGROUND_TIMING.auroraRefreshMs).toBeGreaterThanOrEqual(15)
  })
})
