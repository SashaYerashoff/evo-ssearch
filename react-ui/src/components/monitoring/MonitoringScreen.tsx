import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  IconAdjustments,
  IconFilter,
  IconFilterOff,
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
  type ChannelStatus,
} from '../../api/probes'
import type { Channel } from '../../api/types'
import { videoApi } from '../../api/video'
import { ToolTabs } from '../shell/ToolTabs'
import { IcoBtn } from '../shell/IcoBtn'
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
const STATES: ProbeStatus[] = ['running', 'degraded', 'paused', 'idle', 'disabled']

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

function statusOf(
  probe: Probe,
  runtime: Record<number, string>,
  semanticRuntime: Record<number, ChannelStatus> = {},
): ProbeStatus {
  if (probe.enabled === false) return 'disabled'
  if (probe.embedding_calibration_state && probe.embedding_calibration_state !== 'calibrated') return 'degraded'
  const semantic = probe.channel_id != null ? semanticRuntime[probe.channel_id] : undefined
  if (semantic?.semantic_state === 'degraded') return 'degraded'
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
  navigation,
  channels,
  drive,
  canOperate,
  canManage,
  canCreateBookmarks,
  onOpenParentAlert,
}: {
  navigation?: ReactNode
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
  const [semanticRuntime, setSemanticRuntime] = useState<Record<number, ChannelStatus>>({})
  const [inspectedRuntime, setInspectedRuntime] = useState<ChannelStatus | null>(null)
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
    let alive = true
    let timer: number | null = null
    // The modal owns the live channel while it is open. Avoid duplicate status
    // requests from the obscured board competing with its operator feedback.
    if (editing) return () => { alive = false }
    const channelIds = [...new Set(
      probes
        .map((probe) => probe.channel_id)
        .filter((channelId): channelId is number => channelId != null),
    )]
    if (!channelIds.length) {
      setSemanticRuntime({})
      return () => { alive = false }
    }
    const tick = async () => {
      const values = await Promise.all(channelIds.map(async (channelId) => {
        const status = await probesApi.status(channelId).catch(() => null)
        return [channelId, status] as const
      }))
      if (!alive) return
      const next: Record<number, ChannelStatus> = {}
      for (const [channelId, status] of values) if (status) next[channelId] = status
      setSemanticRuntime(next)
      timer = window.setTimeout(tick, 5_000)
    }
    void tick()
    return () => {
      alive = false
      if (timer != null) window.clearTimeout(timer)
    }
  }, [editing, probes])

  useEffect(() => {
    let alive = true
    let timer: number | null = null
    if (editing) return () => { alive = false }
    const probe = probes.find((candidate) => candidate.id === inspectId)
    if (!probe || probe.channel_id == null) {
      setInspectedRuntime(null)
      return () => { alive = false }
    }
    const tick = async () => {
      const status = await probesApi.status(probe.channel_id!, probe.id).catch(() => null)
      if (alive) setInspectedRuntime(status)
      if (alive) timer = window.setTimeout(tick, 1_200)
    }
    void tick()
    return () => {
      alive = false
      if (timer != null) window.clearTimeout(timer)
    }
  }, [editing, inspectId, probes])

  useEffect(() => {
    let alive = true
    let timer: number | null = null
    if (editing) return () => { alive = false }
    const tick = async () => {
      const response = await videoApi.streams().catch(() => null)
      if (alive && response) {
        const next: Record<number, string> = {}
        for (const stream of response.analytics_streams || []) {
          if (stream.channel_id == null) continue
          next[stream.channel_id] = stream.paused ? 'paused' : stream.running ? 'running' : 'idle'
        }
        setRuntime(next)
      }
      if (alive) timer = window.setTimeout(tick, 5_000)
    }
    void tick()
    return () => {
      alive = false
      if (timer != null) window.clearTimeout(timer)
    }
  }, [editing])

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
    statusOf(probe, runtime, semanticRuntime),
    probe.channel_id != null ? channelNames.get(probe.channel_id) || '' : '',
  )), [probes, origins, states, query, runtime, semanticRuntime, channelNames])
  const tree = useMemo(
    () => buildProbeBoardTree(filtered, groups, channels, (probe) => statusOf(probe, runtime, semanticRuntime)),
    [filtered, groups, channels, runtime, semanticRuntime],
  )
  const inspected = probes.find((probe) => probe.id === inspectId) || null
  const inspectedBookmarkBlocked = canManage
    && probeMutationRequiresBookmarkPermission(inspected, canCreateBookmarks)
  const filtersActive = origins.size > 0 || states.size > 0 || !!query.trim()
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

  const saveProbe = useCallback(async (input: ProbeInput): Promise<Probe | null> => {
    setBusy(true)
    setError(null)
    try {
      const response = await probesApi.save(input)
      if (response.error) throw new Error(response.error)
      await refresh()
      setEditing({ probe: response.probe })
      return response.probe
    } catch (exception: any) {
      setError(exception?.message || 'Save failed')
      return null
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
        }]}
        active="probes"
        onSelect={() => {}}
        leading={navigation}
        reserveLeading
        hideTabs
      >
        {/* Two blocks, same split as the Archive console: CONTROLS acts on the
            board (search, create, view mode, refresh), FILTERS narrows what it
            shows. Each is its own labelled card so the roles stay distinct. */}
        <div className="probe-board-toolbar" role="group" aria-label="Probe tools">
          <div className="probe-controls">
            <span className="atp-glabel is-icon-only" title="Controls" aria-label="Controls"><IconAdjustments size={14} /></span>
            <div className="probe-controls-row">
              <div className="mon-search" title="Search names, prompts, channels and parent alerts">
                <IconSearch size={15} />
                <input
                  placeholder="Search probes…"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
                {query && <button className="mon-search-clear" onClick={() => setQuery('')}><IconX size={13} /></button>}
              </div>
              {canManage && (
                <button className="mon-btn accent probe-primary" onClick={() => setEditing({ probe: null })}>
                  <IconPlus size={16} /> New probe
                </button>
              )}
            </div>
          </div>

          <div className="probe-filters" role="group" aria-label="Probe filters">
            <span className="atp-glabel is-icon-only" title="Filters" aria-label="Filters"><IconFilter size={14} /></span>
            <div className="probe-filters-row">
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
              {/* Actions unpacked from the dropdown into an icon rail on the right
                  edge of the FILTERS block — the label lives in the tooltip. */}
              <div className="probe-tb-actions">
                <IcoBtn title="Refresh probes" onClick={refresh} disabled={loading}>
                  <IconRefresh className={loading ? 'spin' : ''} size={16} />
                </IcoBtn>
                <IcoBtn title="Card view" onClick={() => persistView('grid')} active={view === 'grid'}>
                  <IconLayoutGrid size={16} />
                </IcoBtn>
                <IcoBtn title="List view" onClick={() => persistView('list')} active={view === 'list'}>
                  <IconList size={16} />
                </IcoBtn>
                {filtersActive && (
                  <IcoBtn title="Reset filters" onClick={() => { setOrigins(new Set()); setStates(new Set()); setQuery('') }}>
                    <IconFilterOff size={16} />
                  </IcoBtn>
                )}
                {canManage && (
                  <IcoBtn title="Manage probe groups" onClick={() => { setGroupError(null); setGroupEditor(null) }}>
                    <IconSettings size={16} />
                  </IcoBtn>
                )}
              </div>
            </div>
          </div>
        </div>
      </ToolTabs>

      <section className="mon-board probe-board">
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
                        </div>
                        {view === 'grid' ? (
                          <div className="probe-grid">
                            {channel.probes.map((probe) => (
                              <ProbeCard
                                key={probe.id}
                                probe={probe}
                                status={statusOf(probe, runtime, semanticRuntime)}
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
                              const status = statusOf(probe, runtime, semanticRuntime)
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
              status={statusOf(inspected, runtime, semanticRuntime)}
              runtime={inspectedRuntime}
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
