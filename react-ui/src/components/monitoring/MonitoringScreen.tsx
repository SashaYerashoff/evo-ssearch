import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  IconLayoutGrid,
  IconList,
  IconPlus,
  IconRadar2,
  IconRefresh,
  IconSearch,
  IconSettings,
  IconX,
} from '@tabler/icons-react'
import type { ConsoleDrive } from '../../App'
import {
  authorizeProbeInput,
  probeMutationRequiresBookmarkPermission,
  probesApi,
  type Probe,
  type ProbeChannelGroup,
  type ProbeInput,
  type ProbeListCounts,
  type ProbeOrigin,
  type ProbeThresholdDefaults,
} from '../../api/probes'
import type { Channel } from '../../api/types'
import { videoApi } from '../../api/video'
import { ToolTabs } from '../shell/ToolTabs'
import {
  ProbeCard,
  ProbeOriginBadge,
  ProbeSparkline,
  type ProbeStatus,
} from './ProbeCard'
import { ProbeGroupModal } from './ProbeGroupModal'
import { ProbeInspector } from './ProbeInspector'
import { ProbeSettingsModal } from './ProbeSettingsModal'
import {
  buildProbeBoardTree,
  probeMatchesFilters,
  probeOrigin,
  probeTemporaryTtl,
} from './probeBoard'

const VIEW_STORAGE_KEY = 'eva.probes.board.view.v1'
const COLLAPSED_STORAGE_KEY = 'eva.probes.board.collapsed.v1'
const ORIGINS: Array<{ value: ProbeOrigin; label: string }> = [
  { value: 'operator', label: 'Operator' },
  { value: 'agent', label: 'Agent' },
  { value: 'auto', label: 'Background VLM' },
]
const STATES: ProbeStatus[] = ['running', 'paused', 'idle', 'disabled']

function readBoardView(): 'grid' | 'list' {
  return window.localStorage.getItem(VIEW_STORAGE_KEY) === 'list' ? 'list' : 'grid'
}

function readCollapsed(): Set<string> {
  try {
    const value = JSON.parse(window.localStorage.getItem(COLLAPSED_STORAGE_KEY) || '[]')
    return new Set(Array.isArray(value) ? value.map(String) : [])
  } catch {
    return new Set()
  }
}

function statusOf(probe: Probe, runtime: Record<number, string>): ProbeStatus {
  if (probe.enabled === false) return 'disabled'
  const state = probe.channel_id != null ? runtime[probe.channel_id] : undefined
  if (state === 'running') return 'running'
  if (state === 'paused') return 'paused'
  return 'idle'
}

function probeInputForEnabled(
  probe: Probe,
  enabled: boolean,
  canCreateBookmarks: boolean,
): ProbeInput {
  return authorizeProbeInput({
    id: probe.id,
    name: probe.name,
    channel_id: probe.channel_id,
    enabled,
    pairs: probe.pairs,
    positives: probe.positives,
    negatives: probe.negatives,
    pos_floor: probe.pos_floor,
    margin: probe.margin,
    window_sec: probe.window_sec,
    top_k: probe.top_k,
    severity: probe.severity,
    bookmark: probe.bookmark,
    bookmark_cooldown_sec: probe.bookmark_cooldown_sec,
    bookmark_dedupe_window_sec: probe.bookmark_dedupe_window_sec,
    image_probe: probe.image_probe,
    roi_enabled: probe.roi_enabled,
    roi_norm: Array.isArray(probe.roi_norm)
      ? {
          x: Number(probe.roi_norm[0]),
          y: Number(probe.roi_norm[1]),
          w: Number(probe.roi_norm[2]),
          h: Number(probe.roi_norm[3]),
        }
      : probe.roi_norm,
  }, canCreateBookmarks)
}

export function MonitoringScreen({
  channels,
  drive,
  canOperate,
  canManage,
  canCreateBookmarks,
  onOpenParentAlert,
}: {
  channels: Channel[]
  drive?: ConsoleDrive | null
  canOperate: boolean
  canManage: boolean
  canCreateBookmarks: boolean
  onOpenParentAlert?: (probe: Probe) => void
}) {
  const [probes, setProbes] = useState<Probe[]>([])
  const [groups, setGroups] = useState<ProbeChannelGroup[]>([])
  const [counts, setCounts] = useState<ProbeListCounts>({})
  const [probeDefaults, setProbeDefaults] = useState<ProbeThresholdDefaults>({ pos_floor: 0.05, margin: 0.02 })
  const [runtime, setRuntime] = useState<Record<number, string>>({})
  const [inspectId, setInspectId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [editing, setEditing] = useState<{ probe: Probe | null } | null>(null)
  const [query, setQuery] = useState('')
  const [origins, setOrigins] = useState<Set<ProbeOrigin>>(new Set())
  const [states, setStates] = useState<Set<ProbeStatus>>(new Set())
  const [view, setView] = useState<'grid' | 'list'>(readBoardView)
  const [collapsed, setCollapsed] = useState<Set<string>>(readCollapsed)
  const [groupEditor, setGroupEditor] = useState<ProbeChannelGroup | null | undefined>()
  const [groupError, setGroupError] = useState<string | null>(null)
  const [, setClock] = useState(0)
  const pollRef = useRef<number | undefined>(undefined)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await probesApi.list()
      const next = response.probes || []
      setProbes(next)
      setGroups(response.channel_groups || [])
      setCounts(response.counts || {})
      if (response.defaults) setProbeDefaults(response.defaults)
      setInspectId((current) => (
        current && next.some((probe) => probe.id === current) ? current : null
      ))
    } catch (exception: any) {
      setError(exception?.message || 'Failed to load probes')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    const timer = window.setInterval(() => setClock((value) => value + 1), 1_000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    let alive = true
    const tick = async () => {
      const response = await videoApi.streams().catch(() => null)
      if (!alive || !response) return
      const next: Record<number, string> = {}
      for (const stream of response.analytics_streams || []) {
        if (stream.channel_id == null) continue
        next[stream.channel_id] = stream.paused ? 'paused' : stream.running ? 'running' : 'idle'
      }
      setRuntime(next)
    }
    void tick()
    pollRef.current = window.setInterval(tick, 5_000)
    return () => {
      alive = false
      if (pollRef.current) window.clearInterval(pollRef.current)
    }
  }, [])

  useEffect(() => {
    if (!drive || drive.effect.target !== 'probes') return
    const payload = drive.effect.payload
    if (drive.effect.action === 'refresh' || drive.effect.action === 'show_board') {
      void refresh().then(() => {
        const probeId = String(payload.probe_id || payload.id || '').trim()
        if (probeId) setInspectId(probeId)
      })
    }
  }, [drive?.seq, refresh])

  const channelNames = useMemo(
    () => new Map(channels.map((channel) => [channel.id, channel.title])),
    [channels],
  )
  const filtered = useMemo(() => probes.filter((probe) => probeMatchesFilters(
    probe,
    { origins, states, query },
    statusOf(probe, runtime),
    probe.channel_id != null ? channelNames.get(probe.channel_id) || '' : '',
  )), [probes, origins, states, query, runtime, channelNames])
  const tree = useMemo(
    () => buildProbeBoardTree(filtered, groups, channels, (probe) => statusOf(probe, runtime)),
    [filtered, groups, channels, runtime],
  )
  const inspected = probes.find((probe) => probe.id === inspectId) || null
  const inspectedBookmarkBlocked = canManage
    && probeMutationRequiresBookmarkPermission(inspected, canCreateBookmarks)
  const filtersActive = origins.size > 0 || states.size > 0 || !!query.trim()
  const runningCount = probes.filter((probe) => statusOf(probe, runtime) === 'running').length

  const persistView = (next: 'grid' | 'list') => {
    setView(next)
    window.localStorage.setItem(VIEW_STORAGE_KEY, next)
  }
  const toggleCollapsed = (groupId: string) => {
    setCollapsed((current) => {
      const next = new Set(current)
      if (next.has(groupId)) next.delete(groupId)
      else next.add(groupId)
      window.localStorage.setItem(COLLAPSED_STORAGE_KEY, JSON.stringify([...next]))
      return next
    })
  }
  const toggleFilter = <T extends string>(
    value: T,
    setter: React.Dispatch<React.SetStateAction<Set<T>>>,
  ) => setter((current) => {
    const next = new Set(current)
    if (next.has(value)) next.delete(value)
    else next.add(value)
    return next
  })

  const toggleProbeEnabled = useCallback(async (probe: Probe) => {
    setBusy(true)
    setError(null)
    try {
      const response = await probesApi.save(
        probeInputForEnabled(probe, probe.enabled === false, canCreateBookmarks),
      )
      if (response.error) throw new Error(response.error)
      await refresh()
    } catch (exception: any) {
      setError(exception?.message || 'Probe state change failed')
    } finally {
      setBusy(false)
    }
  }, [canCreateBookmarks, refresh])

  const toggleCapture = useCallback(async (probe: Probe) => {
    if (probe.channel_id == null) return
    setBusy(true)
    setError(null)
    try {
      if (runtime[probe.channel_id] === 'running') {
        await probesApi.stopCapture(probe.channel_id)
      } else {
        await probesApi.startCapture(probe.channel_id, probe.fps)
        await probesApi.run(probe.id).catch(() => null)
      }
      const response = await videoApi.streams().catch(() => null)
      if (response) {
        const next: Record<number, string> = {}
        for (const stream of response.analytics_streams || []) {
          if (stream.channel_id == null) continue
          next[stream.channel_id] = stream.paused ? 'paused' : stream.running ? 'running' : 'idle'
        }
        setRuntime(next)
      }
    } catch (exception: any) {
      setError(exception?.message || 'Capture toggle failed')
    } finally {
      setBusy(false)
    }
  }, [runtime])

  const deleteProbe = useCallback(async (id: string) => {
    setBusy(true)
    setError(null)
    try {
      await probesApi.remove(id)
      setInspectId((current) => current === id ? null : current)
      await refresh()
    } catch (exception: any) {
      setError(exception?.message || 'Delete failed')
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const saveProbe = useCallback(async (input: ProbeInput) => {
    setBusy(true)
    setError(null)
    try {
      const response = await probesApi.save(input)
      if (response.error) throw new Error(response.error)
      setEditing(null)
      await refresh()
    } catch (exception: any) {
      setError(exception?.message || 'Save failed')
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const saveGroup = useCallback(async (input: {
    id?: string
    name: string
    channel_ids: number[]
  }) => {
    setBusy(true)
    setGroupError(null)
    try {
      const response = await probesApi.saveGroup(input)
      if (response.error) throw new Error(response.error)
      setGroups(response.groups || [])
      setGroupEditor(undefined)
    } catch (exception: any) {
      setGroupError(exception?.message || 'Group save failed')
    } finally {
      setBusy(false)
    }
  }, [])

  const deleteGroup = useCallback(async (id: string) => {
    setBusy(true)
    setGroupError(null)
    try {
      const response = await probesApi.deleteGroup(id)
      if (response.error) throw new Error(response.error)
      setGroups(response.groups || [])
      setCollapsed((current) => {
        const next = new Set(current)
        next.delete(id)
        return next
      })
      setGroupEditor(undefined)
    } catch (exception: any) {
      setGroupError(exception?.message || 'Group delete failed')
    } finally {
      setBusy(false)
    }
  }, [])

  return (
    <div className="mon-cols probe-board-screen">
      <ToolTabs
        tabs={[{
          id: 'probes',
          icon: <IconRadar2 size={13} />,
          label: 'Probes',
          summary: `${filtersActive ? `${filtered.length}/${probes.length}` : probes.length} probes · ${runningCount} running`,
        }]}
        active="probes"
        onSelect={() => {}}
      >
        <div className="probe-board-toolbar">
          <div className="mon-search" title="Search names, prompts, channels and parent alerts">
            <IconSearch size={15} />
            <input
              placeholder="Search probes…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            {query && <button className="mon-search-clear" onClick={() => setQuery('')}><IconX size={13} /></button>}
          </div>
          <div className="probe-filter-set" aria-label="Filter by author">
            {ORIGINS.map((origin) => (
              <button
                key={origin.value}
                className={`probe-filter-chip origin-${origin.value} ${origins.has(origin.value) ? 'on' : ''}`}
                aria-pressed={origins.has(origin.value)}
                onClick={() => toggleFilter(origin.value, setOrigins)}
              >
                <i />{origin.label}<b>{counts.by_origin?.[origin.value] ?? probes.filter((probe) => probeOrigin(probe) === origin.value).length}</b>
              </button>
            ))}
          </div>
          <div className="probe-filter-set state-set" aria-label="Filter by state">
            {STATES.map((state) => (
              <button
                key={state}
                className={`probe-filter-chip ${states.has(state) ? 'on' : ''}`}
                aria-pressed={states.has(state)}
                onClick={() => toggleFilter(state, setStates)}
              >
                {state}
              </button>
            ))}
          </div>
          <div className="probe-view-toggle">
            <button className={view === 'grid' ? 'on' : ''} onClick={() => persistView('grid')} title="Card view"><IconLayoutGrid size={15} /></button>
            <button className={view === 'list' ? 'on' : ''} onClick={() => persistView('list')} title="List view"><IconList size={15} /></button>
          </div>
          {filtersActive && (
            <button className="mon-btn sm" onClick={() => { setOrigins(new Set()); setStates(new Set()); setQuery('') }}>
              Reset
            </button>
          )}
          <button className="mon-btn sm" onClick={refresh} disabled={loading}>
            <IconRefresh className={loading ? 'spin' : ''} size={15} /> Refresh
          </button>
          {canManage && (
            <>
              <button className="mon-btn sm" onClick={() => { setGroupError(null); setGroupEditor(null) }}>
                <IconSettings size={15} /> Groups
              </button>
              <button className="mon-btn accent" onClick={() => setEditing({ probe: null })}>
                <IconPlus size={16} /> New probe
              </button>
            </>
          )}
        </div>
      </ToolTabs>

      <section className="mon-board probe-board">
        <div className="probe-board-heading">
          <div>
            <div className="mon-panel-title">Probe board</div>
            <div className="mon-panel-sub">
              Group → channel → probe · operator, approved-agent and background-VLM lineage stay distinct.
            </div>
          </div>
          <div className="probe-board-count">
            {filtersActive ? `${filtered.length} of ${probes.length}` : `${probes.length}`} visible
            {counts.temporary_active ? ` · ${counts.temporary_active} temporary` : ''}
            {counts.temporary_expired_hidden ? ` · ${counts.temporary_expired_hidden} expired hidden` : ''}
          </div>
        </div>
        {error && <div className="chat-error"><IconRadar2 size={14} /> {error}</div>}
        {!loading && probes.length === 0 && !error && (
          <div className="empty-state">No probes yet. Create one or let a VLM alert raise a temporary follow-up.</div>
        )}
        {!loading && probes.length > 0 && filtered.length === 0 && !error && (
          <div className="empty-state">No probe matches the current filters.</div>
        )}

        <div className={`probe-groups ${view === 'list' ? 'list' : 'grid'}`}>
          {tree.map((group) => {
            const isCollapsed = collapsed.has(group.id)
            const probeCount = group.channels.reduce((total, channel) => total + channel.probes.length, 0)
            const groupRunning = group.channels.reduce((total, channel) => total + channel.runningCount, 0)
            const storedGroup = groups.find((candidate) => candidate.id === group.id)
            return (
              <section key={group.id} className={`probe-board-group ${group.synthetic ? 'synthetic' : ''} ${isCollapsed ? 'collapsed' : ''}`}>
                <div className="probe-board-group-head">
                  <button
                    className="probe-group-toggle"
                    aria-expanded={!isCollapsed}
                    onClick={() => toggleCollapsed(group.id)}
                  >
                    <span className="probe-group-chevron">›</span>
                    <b>{group.name}</b>
                  </button>
                  <span>{group.channels.length} ch · {probeCount} probes · {groupRunning} running</span>
                  {canManage && storedGroup && !storedGroup.read_only && (
                    <button className="pc-ico" title="Edit group" onClick={() => { setGroupError(null); setGroupEditor(storedGroup) }}>
                      <IconSettings size={14} />
                    </button>
                  )}
                </div>
                {!isCollapsed && (
                  <div className="probe-board-group-body">
                    {group.channels.map((channel) => (
                      <section key={channel.channelId ?? 'none'} className="probe-board-channel">
                        <div className="probe-board-channel-head">
                          <div>
                            <span>Channel {channel.channelId ?? '—'}</span>
                            <b>{channel.label}</b>
                          </div>
                          <span>{channel.probes.length} probes · {channel.runningCount} running</span>
                        </div>
                        {view === 'grid' ? (
                          <div className="probe-grid">
                            {channel.probes.map((probe) => (
                              <ProbeCard
                                key={probe.id}
                                probe={probe}
                                status={statusOf(probe, runtime)}
                                selected={probe.id === inspectId}
                                onSelect={() => setInspectId(probe.id)}
                                onRun={canManage ? () => toggleProbeEnabled(probe) : undefined}
                                onDelete={canManage ? () => deleteProbe(probe.id) : undefined}
                              />
                            ))}
                          </div>
                        ) : (
                          <div className="probe-row-list">
                            {channel.probes.map((probe) => {
                              const status = statusOf(probe, runtime)
                              const hit = probe.last_hit || probe.recent_hits?.[0]
                              const ttl = probeTemporaryTtl(probe)
                              return (
                                <button
                                  key={probe.id}
                                  className={`probe-row ${probe.id === inspectId ? 'selected' : ''}`}
                                  onClick={() => setInspectId(probe.id)}
                                >
                                  <span className={`pc-badge ${status}`}>{status}</span>
                                  <ProbeOriginBadge probe={probe} />
                                  <b>{probe.name || 'Untitled probe'}</b>
                                  <span>Ch {probe.channel_id ?? '—'}</span>
                                  <ProbeSparkline probe={probe} compact />
                                  <code>P {hit?.pos_score != null ? Number(hit.pos_score).toFixed(2) : '—'} · M {hit?.margin != null ? Number(hit.margin).toFixed(2) : '—'}</code>
                                  {ttl ? <span className={`probe-ttl ${ttl.expired ? 'expired' : ''}`} title={ttl.title}>{ttl.text}</span> : <span />}
                                </button>
                              )
                            })}
                          </div>
                        )}
                      </section>
                    ))}
                  </div>
                )}
              </section>
            )
          })}
        </div>
      </section>

      {inspected && (
        <div className="scrim" onClick={() => setInspectId(null)}>
          <div className="modal mon-inspect-modal probe-inspect-modal" onClick={(event) => event.stopPropagation()}>
            <button className="modal-close mon-inspect-close" onClick={() => setInspectId(null)}><IconX size={18} /></button>
            <ProbeInspector
              probe={inspected}
              status={statusOf(inspected, runtime)}
              busy={busy}
              settingsBlockedReason={inspectedBookmarkBlocked
                ? 'This bookmarked probe requires bookmarks:create to edit.'
                : undefined}
              onSettings={canManage && !inspectedBookmarkBlocked
                ? () => { setInspectId(null); setEditing({ probe: inspected }) }
                : undefined}
              onRun={canOperate ? () => toggleCapture(inspected) : undefined}
              onDelete={canManage ? () => deleteProbe(inspected.id) : undefined}
              onOpenParentAlert={onOpenParentAlert}
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
          defaults={probeDefaults}
          onClose={() => setEditing(null)}
          onSave={saveProbe}
          onCasted={refresh}
        />
      )}

      {groupEditor !== undefined && canManage && (
        <ProbeGroupModal
          group={groupEditor}
          groups={groups}
          channels={channels}
          busy={busy}
          error={groupError}
          onClose={() => setGroupEditor(undefined)}
          onSave={saveGroup}
          onDelete={deleteGroup}
        />
      )}
    </div>
  )
}
