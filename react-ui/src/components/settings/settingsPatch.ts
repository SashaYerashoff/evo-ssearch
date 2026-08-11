import type { Settings } from '../../api/settings'

const WRITE_ONLY_KEYS = new Set(['luxriotPassword', 'vlmApiKey', 'agentApiKey'])

/** Build a surgical Settings PATCH without replaying blank transient controls. */
export function buildSettingsPatch(
  settings: Settings,
  dirtyKeys: Iterable<string>,
  writableKeys: ReadonlyArray<string>,
): Settings {
  const writable = new Set(writableKeys)
  const patch: Settings = {}
  for (const key of dirtyKeys) {
    if (!writable.has(key)) continue
    const value = settings[key]
    if (WRITE_ONLY_KEYS.has(key) && !value) continue
    if (key === 'port' && (value == null || String(value).trim() === '')) continue
    if (value !== undefined) patch[key] = value
  }
  return patch
}
