import { useEffect, useState, useCallback, useRef } from 'react'
import { IconPlus, IconX, IconRefresh, IconRadar2, IconSearch } from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import { probeMutationRequiresBookmarkPermission, probesApi, type Probe, type ProbeInput } from '../../api/probes'
import { videoApi } from '../../api/video'
import { ToolTabs } from '../shell/ToolTabs'
import { Dropdown } from '../shell/Dropdown'
import { ProbeCard, type ProbeStatus } from './ProbeCard'
import { ProbeInspector } from './ProbeInspector'
import { ProbeSettingsModal } from './ProbeSettingsModal'

function statusOf(p: Probe, runtime: Record<number, string>): ProbeStatus {
  if (p.enabled === false) return 'disabled'
  const rt = p.channel_id != null ? runtime[p.channel_id] : undefined
  if (rt === 'running') return 'running'
  if (rt === 'paused') return 'paused'
  return 'idle'
}

export function MonitoringScreen({ channels, canOperate, canManage, canCreateBookmarks }: {
  channels: Channel[]
  canOperate: boolean
  canManage: boolean
  canCreateBookmarks: boolean
}) {
  const [probes, setProbes] = useState<Probe[]>([])
  const [runtime, setRuntime] = useState<Record<number, string>>({})
  const [inspectId, setInspectId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState<{ probe: Probe | null } | null>(null)
  const [chFilter, setChFilter] = useState('')   // '' = all channels
  const [query, setQuery] = useState('')          // probe search (name + prompts)
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

  // poll analytics-capture runtime for all channels (true probe running state)
  useEffect(() => {
    let alive = true
    const tick = async () => {
      const s = await videoApi.streams().catch(() => null)
      if (!alive || !s) return
      const next: Record<number, string> = {}
      for (const a of s.analytics_streams || []) {
        if (a.channel_id == null) continue
        next[a.channel_id] = a.paused ? 'paused' : a.running ? 'running' : 'idle'
      }
      setRuntime(next)
    }
    tick()
    pollRef.current = window.setInterval(tick, 5000)
    return () => { alive = false; if (pollRef.current) window.clearInterval(pollRef.current) }
  }, [])

  const inspected = probes.find((p) => p.id === inspectId) || null
  const inspectedBookmarkBlocked = canManage && probeMutationRequiresBookmarkPermission(inspected, canCreateBookmarks)

  // Run toggles the probe's analytics capture: start when idle, stop when running
  const toggleProbe = useCallback(async (p: Probe) => {
    const ch = p.channel_id
    if (ch == null) return
    setBusy(true); setError(null)
    try {
      const running = runtime[ch] === 'running'
      if (running) await probesApi.stopCapture(ch)
      else {
        await probesApi.startCapture(ch)
        probesApi.run(p.id).then((res) => {
          if (res.probe) setProbes((ps) => ps.map((x) => (x.id === p.id ? { ...x, ...res.probe } : x)))
        }).catch(() => {})
      }
      // reflect the new state immediately
      const s = await videoApi.streams().catch(() => null)
      if (s) {
        const next: Record<number, string> = {}
        for (const a of s.analytics_streams || []) {
          if (a.channel_id == null) continue
          next[a.channel_id] = a.paused ? 'paused' : a.running ? 'running' : 'idle'
        }
        setRuntime(next)
      }
    } catch (e: any) { setError(e?.message || 'Toggle failed') }
    finally { setBusy(false) }
  }, [runtime])

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

  // channel filter ('' = all) + text search over name and prompt texts
  const q = query.trim().toLowerCase()
  const shown = probes.filter((p) => {
    if (chFilter && String(p.channel_id) !== chFilter) return false
    if (!q) return true
    const hay = [p.name, ...(p.positives || []), ...(p.negatives || [])].join(' ').toLowerCase()
    return hay.includes(q)
  })
  const chTitle = chFilter ? (channels.find((c) => String(c.id) === chFilter)?.title || `ch ${chFilter}`) : 'All channels'
  const filtered = !!chFilter || !!q
  // creating while filtered → the editor opens preset to that channel
  const newProbe = () => setEditing({ probe: chFilter ? ({ channel_id: Number(chFilter) } as unknown as Probe) : null })

  return (
    <div className="mon-cols">
      {/* top tool tabs — same pattern as Archive/Video */}
      <ToolTabs
        tabs={[{
          id: 'probes', icon: <IconRadar2 size={13} />, label: 'CLIP probes',
          summary: `${filtered ? `${shown.length}/${probes.length}` : probes.length} probe${probes.length === 1 ? '' : 's'} · ${shown.filter((p) => p.channel_id != null && runtime[p.channel_id] === 'running').length} running · ${chTitle}`,
        }]}
        active="probes"
        onSelect={() => {}}
      >
        <div className="mon-toolbar-actions">
          <div className="mon-search" title="Search probes by name or prompt text">
            <IconSearch size={15} />
            <input placeholder="Search probes…" value={query} onChange={(e) => setQuery(e.target.value)} />
            {query && <button className="mon-search-clear" title="Clear" onClick={() => setQuery('')}><IconX size={13} /></button>}
          </div>
          <div className="mon-ch-filter" title="Show probes for one channel only">
            <Dropdown value={chFilter} onChange={setChFilter}
              options={[{ value: '', label: 'All channels' }, ...channels.map((c) => ({ value: String(c.id), label: c.title }))]} />
          </div>
          <button className="mon-btn" onClick={refresh} disabled={loading} title="Reload the probe list">
            <IconRefresh size={16} className={loading ? 'spin' : ''} /> Refresh list
          </button>
          {canManage && <button className="mon-btn accent" onClick={newProbe} title="Create a new CLIP probe">
            <IconPlus size={16} /> New CLIP probe
          </button>}
        </div>
      </ToolTabs>

      {/* center board */}
      <section className="mon-board">
        <div className="mon-panel-title">CLIP probe board</div>
        <div className="mon-panel-sub">Secondary visual-similarity signals for engineer tuning and agent-assisted checks.</div>
        {error && <div className="empty-state" style={{ color: 'var(--danger)', padding: 20 }}>{error}</div>}
        <div className="probe-grid">
          {shown.map((p) => (
            <ProbeCard key={p.id} probe={p} status={statusOf(p, runtime)} selected={p.id === inspectId}
              onSelect={() => setInspectId(p.id)}
              onRun={canOperate ? () => toggleProbe(p) : undefined}
              onDelete={canManage ? () => deleteProbe(p.id) : undefined} />
          ))}
          {canManage && <button className="probe-new" onClick={newProbe}>
            <IconPlus size={22} /><span>New probe</span>
          </button>}
        </div>
        {!loading && probes.length === 0 && !error && <div className="empty-state">No CLIP probes yet. Create one to start monitoring.</div>}
        {!loading && probes.length > 0 && shown.length === 0 && !error && (
          <div className="empty-state">
            {q ? <>No probes match “{query.trim()}”{chFilter ? ` on ${chTitle}` : ''}.</> : <>No probes on {chTitle}. Pick “All channels” or create one here.</>}
          </div>
        )}
      </section>

      {/* inspector modal — opens on probe click */}
      {inspected && (
        <div className="scrim" onClick={() => setInspectId(null)}>
          <div className="modal mon-inspect-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close mon-inspect-close" onClick={() => setInspectId(null)}><IconX size={18} /></button>
            <ProbeInspector
              probe={inspected} status={statusOf(inspected, runtime)} busy={busy}
              settingsBlockedReason={inspectedBookmarkBlocked
                ? 'This bookmarked probe requires bookmarks:create to edit.'
                : undefined}
              onSettings={canManage && !inspectedBookmarkBlocked
                ? () => { setInspectId(null); setEditing({ probe: inspected }) }
                : undefined}
              onRun={canOperate ? () => toggleProbe(inspected) : undefined}
              onDelete={canManage ? () => deleteProbe(inspected.id) : undefined}
            />
          </div>
        </div>
      )}

      {editing && canManage && (
        <ProbeSettingsModal
          probe={editing.probe}
          channels={channels}
          busy={busy}
          canControlCapture={canOperate}
          canCreateBookmarks={canCreateBookmarks}
          onClose={() => setEditing(null)}
          onSave={saveProbe}
          onCasted={refresh}
        />
      )}
    </div>
  )
}
