import { useEffect, useState, useCallback, useRef } from 'react'
import { IconPlus, IconX } from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import { probesApi, type Probe, type ProbeInput } from '../../api/probes'
import { ProbeCard, type ProbeStatus } from './ProbeCard'
import { ProbeInspector } from './ProbeInspector'
import { ProbeSettingsModal } from './ProbeSettingsModal'

export type MonitorAction = 'refresh' | 'new'

function statusOf(p: Probe, runtime: Record<number, string>): ProbeStatus {
  if (p.enabled === false) return 'disabled'
  const rt = p.channel_id != null ? runtime[p.channel_id] : undefined
  if (rt === 'running') return 'running'
  if (rt === 'paused') return 'paused'
  return 'idle'
}

export function MonitoringScreen({ channels, cmd, onCmdHandled }: {
  channels: Channel[]
  cmd?: { seq: number; action: MonitorAction } | null
  onCmdHandled?: () => void
}) {
  const [probes, setProbes] = useState<Probe[]>([])
  const [runtime, setRuntime] = useState<Record<number, string>>({})
  const [inspectId, setInspectId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState<{ probe: Probe | null } | null>(null)
  const pollRef = useRef<number | undefined>(undefined)

  const refresh = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const { probes } = await probesApi.list()
      setProbes(probes || [])
    } catch (e: any) { setError(e?.message || 'Failed to load probes') }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // poll capture status for every channel that has a probe
  useEffect(() => {
    const chans = Array.from(new Set(probes.map((p) => p.channel_id).filter((c): c is number => c != null)))
    if (!chans.length) return
    let alive = true
    const tick = async () => {
      const results = await Promise.all(chans.map((c) => probesApi.status(c).then((s) => [c, s] as const).catch(() => [c, null] as const)))
      if (!alive) return
      setRuntime((prev) => {
        const next = { ...prev }
        for (const [c, s] of results) if (s && s.runtime_state) next[c] = String(s.runtime_state)
        return next
      })
    }
    tick()
    pollRef.current = window.setInterval(tick, 8000)
    return () => { alive = false; if (pollRef.current) window.clearInterval(pollRef.current) }
  }, [probes.map((p) => p.channel_id).join(',')]) // eslint-disable-line react-hooks/exhaustive-deps

  const inspected = probes.find((p) => p.id === inspectId) || null

  const runProbe = useCallback(async (id: string) => {
    setBusy(true); setError(null)
    try {
      const res = await probesApi.run(id)
      if (res.probe) setProbes((ps) => ps.map((p) => (p.id === id ? { ...p, ...res.probe } : p)))
    } catch (e: any) { setError(e?.message || 'Run failed') }
    finally { setBusy(false) }
  }, [])

  const deleteProbe = useCallback(async (id: string) => {
    setBusy(true)
    try { await probesApi.remove(id); setInspectId((cur) => (cur === id ? null : cur)); await refresh() }
    catch (e: any) { setError(e?.message || 'Delete failed') }
    finally { setBusy(false) }
  }, [refresh])

  const saveProbe = useCallback(async (input: ProbeInput) => {
    setBusy(true); setError(null)
    try {
      const res = await probesApi.save(input)
      if (res.error) throw new Error(res.error)
      setEditing(null)
      await refresh()
    } catch (e: any) { setError(e?.message || 'Save failed') }
    finally { setBusy(false) }
  }, [refresh])

  // actions driven from the left-rail Monitoring children
  useEffect(() => {
    if (!cmd) return
    if (cmd.action === 'refresh') refresh()
    else if (cmd.action === 'new') setEditing({ probe: null })
    onCmdHandled?.()
  }, [cmd?.seq]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="mon-cols">
      {/* center board */}
      <section className="mon-board">
        <div className="mon-panel-title">CLIP probe board</div>
        <div className="mon-panel-sub">Secondary visual-similarity signals for engineer tuning and agent-assisted checks.</div>
        {error && <div className="empty-state" style={{ color: 'var(--danger)', padding: 20 }}>{error}</div>}
        <div className="probe-grid">
          {probes.map((p) => (
            <ProbeCard key={p.id} probe={p} status={statusOf(p, runtime)} selected={p.id === inspectId}
              onSelect={() => setInspectId(p.id)} onRun={() => runProbe(p.id)} onDelete={() => deleteProbe(p.id)} />
          ))}
          <button className="probe-new" onClick={() => setEditing({ probe: null })}>
            <IconPlus size={22} /><span>New probe</span>
          </button>
        </div>
        {!loading && probes.length === 0 && !error && <div className="empty-state">No CLIP probes yet. Create one to start monitoring.</div>}
      </section>

      {/* inspector modal — opens on probe click */}
      {inspected && (
        <div className="scrim" onClick={() => setInspectId(null)}>
          <div className="modal mon-inspect-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close mon-inspect-close" onClick={() => setInspectId(null)}><IconX size={18} /></button>
            <ProbeInspector
              probe={inspected} status={statusOf(inspected, runtime)} busy={busy}
              onSettings={() => { setInspectId(null); setEditing({ probe: inspected }) }}
              onRun={() => runProbe(inspected.id)}
              onDelete={() => deleteProbe(inspected.id)}
            />
          </div>
        </div>
      )}

      {editing && (
        <ProbeSettingsModal probe={editing.probe} channels={channels} busy={busy} onClose={() => setEditing(null)} onSave={saveProbe} />
      )}
    </div>
  )
}
