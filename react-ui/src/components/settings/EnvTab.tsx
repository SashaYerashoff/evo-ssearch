import { useEffect, useState } from 'react'
import { IconReload, IconDeviceFloppy } from '@tabler/icons-react'
import { settingsApi } from '../../api/settings'

export function EnvTab() {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<{ msg: string; ok: boolean } | null>(null)

  const load = () => {
    setBusy(true)
    settingsApi.getEnv().then((r) => setText(r.envText || '')).catch(() => setStatus({ msg: 'Failed to load .env', ok: false })).finally(() => setBusy(false))
  }
  useEffect(() => { load() }, [])

  async function save() {
    setBusy(true); setStatus(null)
    try {
      const r = await settingsApi.saveEnv(text)
      if (!r.success) throw new Error(r.error || 'Save failed')
      setStatus({ msg: r.message || `Saved ${r.count ?? 0} vars. Restart the server to apply.`, ok: true })
    } catch (e: any) { setStatus({ msg: e?.message || 'Save failed', ok: false }) }
    finally { setBusy(false) }
  }

  return (
    <div className="set-section">
      <h3>Environment</h3>
      <p className="set-section-help">Edit <code>EVOSSEARCH_*</code> overrides saved to <code>.env</code>. Secrets are masked (<code>***</code>) and preserved unless you replace them. Restart the server to apply all changes.</p>
      <textarea className="set-env" spellCheck={false} value={text} onChange={(e) => setText(e.target.value)} rows={18} placeholder="EVOSSEARCH_KEY=value" />
      <div className="set-env-actions">
        <button className="mon-btn" onClick={load} disabled={busy}><IconReload size={15} className={busy ? 'spin' : ''} /> Reload</button>
        <button className="mon-btn accent" onClick={save} disabled={busy}><IconDeviceFloppy size={15} /> Save .env</button>
        {status && <span className={`set-status ${status.ok ? 'ok' : 'err'}`}>{status.msg}</span>}
      </div>
    </div>
  )
}
