import { useEffect, useState } from 'react'
import { IconReload, IconDeviceFloppy } from '@tabler/icons-react'
import { settingsApi, type SettingsPrecedence } from '../../api/settings'

export function EnvTab() {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<{ msg: string; ok: boolean } | null>(null)
  const [envFile, setEnvFile] = useState('.env')
  const [precedence, setPrecedence] = useState<SettingsPrecedence | null>(null)

  const load = () => {
    setBusy(true)
    settingsApi.getEnv().then((r) => {
      setText(r.envText || '')
      setEnvFile(r.envFile || '.env')
      setPrecedence(r.precedence || null)
    }).catch(() => setStatus({ msg: 'Failed to load environment settings', ok: false })).finally(() => setBusy(false))
  }
  useEffect(() => { load() }, [])

  async function save() {
    setBusy(true); setStatus(null)
    try {
      const r = await settingsApi.saveEnv(text)
      if (!r.success) throw new Error(r.error || 'Save failed')
      setEnvFile(r.envFile || envFile)
      setPrecedence(r.precedence || precedence)
      const pending = r.pendingOrOverriddenKeys || []
      const sourceUnknown = r.precedence?.declared_file_matches_project === false
      const suffix = pending.length ? ` ${pending.length} value${pending.length === 1 ? '' : 's'} require restart.` : ''
      setStatus({ msg: `${r.message || `Saved ${r.count ?? 0} vars.`}${suffix}`, ok: !sourceUnknown })
    } catch (e: any) { setStatus({ msg: e?.message || 'Save failed', ok: false }) }
    finally { setBusy(false) }
  }

  return (
    <div className="set-section">
      <h3>Environment</h3>
      <p className="set-section-help">Edit <code>EVOSSEARCH_*</code> overrides saved to <code>{envFile}</code>. Secrets are masked and preserved unless you replace them. Restart the server to apply environment-backed changes.</p>
      {precedence?.declared_file_matches_project === false && (
        <div className="set-load-error">This file is not declared as the service configuration source. Verify the systemd EnvironmentFile before relying on a restart.</div>
      )}
      <textarea className="set-env" spellCheck={false} value={text} onChange={(e) => setText(e.target.value)} rows={18} placeholder="EVOSSEARCH_KEY=value" />
      <div className="set-env-actions">
        <button className="mon-btn" onClick={load} disabled={busy}><IconReload size={15} className={busy ? 'spin' : ''} /> Reload</button>
        <button className="mon-btn accent" onClick={save} disabled={busy}><IconDeviceFloppy size={15} /> Save .env</button>
        {status && <span className={`set-status ${status.ok ? 'ok' : 'err'}`}>{status.msg}</span>}
      </div>
    </div>
  )
}
