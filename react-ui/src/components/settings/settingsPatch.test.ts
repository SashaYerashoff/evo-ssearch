import { describe, expect, it } from 'vitest'

import { buildSettingsPatch } from './settingsPatch'

describe('settings patch', () => {
  it('keeps the running port when a transiently blank port accompanies a host edit', () => {
    expect(buildSettingsPatch(
      { host: '0.0.0.0', port: '' },
      ['host', 'port'],
      ['host', 'port'],
    )).toEqual({ host: '0.0.0.0' })
  })

  it('includes an explicit valid port and omits blank write-only secrets', () => {
    expect(buildSettingsPatch(
      { port: 5081, agentApiKey: '' },
      ['port', 'agentApiKey'],
      ['port', 'agentApiKey'],
    )).toEqual({ port: 5081 })
  })
})
