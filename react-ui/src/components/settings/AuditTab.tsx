import { useEffect, useState, useCallback } from 'react'
import { IconReload, IconAlertTriangle } from '@tabler/icons-react'
import { buildAuditQuery, settingsApi, type AuditEvent } from '../../api/settings'
import { Dropdown } from '../shell/Dropdown'

export function AuditTab() {
  const [events, setEvents] = useState<AuditEvent[]>([])
  const [busy, setBusy] = useState(false)
  const [denied, setDenied] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [appliedKey, setAppliedKey] = useState('')
  const [selected, setSelected] = useState<AuditEvent | null>(null)
  const [f, setF] = useState({ result: '', action: '', actor_user_id: '', channel_id: '', request_id: '', limit: '50' })

  const load = useCallback(async (append = false) => {
    if (append && !nextCursor) return
    setBusy(true); setDenied(false); setLoadError('')
    try {
      const params = buildAuditQuery(f, append ? nextCursor : null)
      const r = await settingsApi.audit(params)
      const page = r.events || []
      setEvents((current) => append ? [...current, ...page] : page)
      setNextCursor(r.nextCursor || null)
      setAppliedKey(JSON.stringify(f))
      if (!append) setSelected(null)
    } catch (e: any) {
      if (e?.status === 403) setDenied(true)
      else setLoadError(e?.message || 'Failed to load audit events')
    }
    finally { setBusy(false) }
  }, [f, nextCursor])

  useEffect(() => { load(false) }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (denied) return <div className="set-section"><h3>Audit</h3><div className="set-denied"><IconAlertTriangle size={15} /> You don't have permission to read the audit log (needs audit:view + all-channel scope).</div></div>

  const dirty = !!appliedKey && appliedKey !== JSON.stringify(f)

  return (
    <div className="set-section">
      <h3>Audit events</h3>
      <div className="set-audit-filters">
        <Dropdown value={f.result} onChange={(v) => setF({ ...f, result: v })}
          options={[{ value: '', label: 'Any result' }, { value: 'success', label: 'success' }, { value: 'failure', label: 'failure' }, { value: 'denied', label: 'denied' }]} />
        <input placeholder="action (auth.login)" value={f.action} onChange={(e) => setF({ ...f, action: e.target.value })} />
        <input placeholder="actor id" value={f.actor_user_id} onChange={(e) => setF({ ...f, actor_user_id: e.target.value })} />
        <input placeholder="channel" value={f.channel_id} onChange={(e) => setF({ ...f, channel_id: e.target.value })} />
        <input placeholder="request id" value={f.request_id} onChange={(e) => setF({ ...f, request_id: e.target.value })} />
        <Dropdown value={f.limit} onChange={(v) => setF({ ...f, limit: v })} options={['25', '50', '100'].map((l) => ({ value: l, label: l }))} />
        <button className="mon-btn" onClick={() => load(false)} disabled={busy}><IconReload size={14} className={busy ? 'spin' : ''} /> Apply</button>
      </div>
      {dirty && <div className="set-note">Filters changed — press Apply before loading another page.</div>}
      {loadError && <div className="set-denied"><IconAlertTriangle size={15} /> {loadError}</div>}
      {busy && events.length === 0 && <div className="loading-state"><div className="spinner" /><div>Loading audit events…</div></div>}
      <div className="set-audit-table">
        <div className="set-audit-row head"><span>Time</span><span>Action</span><span>Actor</span><span>Target</span><span>Result</span></div>
        {!busy && !loadError && events.length === 0 && <div className="set-note" style={{ padding: 12 }}>No events.</div>}
        {events.map((e, i) => (
          <button
            type="button"
            key={`${e.request_id || 'event'}:${e.timestamp || i}:${e.action || ''}`}
            className={`set-audit-row ${selected === e ? 'on' : ''}`}
            onClick={() => setSelected((current) => current === e ? null : e)}
            title="Open event details"
          >
            <span className="mono">{e.timestamp ? new Date(e.timestamp).toLocaleString() : '—'}</span>
            <span>{e.action}</span>
            <span className="mono">{(e.actor_user_id || '').slice(0, 8) || '—'}</span>
            <span>{e.target_type}{e.target_id ? ` #${String(e.target_id).slice(0, 8)}` : ''}</span>
            <span className={`set-audit-res ${e.result}`}>{e.result}</span>
          </button>
        ))}
      </div>
      <div className="set-audit-page">
        <span>{events.length} loaded</span>
        <button className="mon-btn" disabled={busy || dirty || !nextCursor} onClick={() => load(true)}>
          {busy ? <IconReload size={14} className="spin" /> : null} Next page
        </button>
      </div>
      {selected && (
        <div className="set-audit-details">
          <div className="set-audit-details-head">
            <b>{selected.action || 'Audit event'}</b>
            <button className="modal-close" onClick={() => setSelected(null)}>×</button>
          </div>
          <pre>{JSON.stringify({
            timestamp: selected.timestamp,
            actor_user_id: selected.actor_user_id,
            target_type: selected.target_type,
            target_id: selected.target_id,
            channel_id: selected.channel_id,
            request_id: selected.request_id,
            result: selected.result,
            details: selected.details,
          }, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
