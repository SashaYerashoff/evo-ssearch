import { useEffect, useState, useCallback, type ReactNode } from 'react'
import { IconPlus, IconAlertTriangle, IconDeviceFloppy, IconLogout2, IconX } from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import { parseChannelSelection, toggleChannelSelection, unknownChannelIds } from '../../api/access'
import { settingsApi, type AuthSessionRow, type AuthUserRow } from '../../api/settings'

interface Draft { user_id?: string; username: string; display_name: string; password: string; roles: string[]; allowed: string; is_active: boolean }
const EMPTY: Draft = { username: '', display_name: '', password: '', roles: [], allowed: '*', is_active: true }

function fromRow(u: AuthUserRow): Draft {
  const allowed = u.allowed_channel_ids === '*' || !u.allowed_channel_ids ? '*'
    : Array.isArray(u.allowed_channel_ids) ? u.allowed_channel_ids.join(',') : String(u.allowed_channel_ids)
  return { user_id: u.user_id, username: u.username, display_name: u.display_name || '', password: '', roles: u.roles || [], allowed, is_active: u.is_active !== false }
}
function parseAllowed(s: string): number[] | '*' {
  return parseChannelSelection(s)
}
function selectedChannels(s: string): Set<number> {
  if (s.trim() === '*') return new Set()
  return new Set(parseAllowed(s) as number[])
}
function serializeChannels(ids: Set<number>): string {
  return [...ids].sort((a, b) => a - b).join(',')
}
const sameDraft = (a: Draft, b: Draft) =>
  a.username === b.username && a.display_name === b.display_name && a.password === b.password &&
  a.allowed === b.allowed && a.is_active === b.is_active && a.roles.slice().sort().join(',') === b.roles.slice().sort().join(',')

function Choice({
  checked,
  onChange,
  children,
  disabled = false,
  className = '',
  title,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  children: ReactNode
  disabled?: boolean
  className?: string
  title?: string
}) {
  return (
    <label className={`usr-choice ${checked ? 'on' : ''} ${disabled ? 'disabled' : ''} ${className}`} title={title}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <span className="usr-checkmark" aria-hidden="true" />
      <span className="usr-choice-label">{children}</span>
    </label>
  )
}

export function UsersTab({
  currentUserId,
  currentSessionId,
  channels,
  onRefreshChannels,
}: {
  currentUserId: string
  currentSessionId?: string
  channels: Channel[]
  onRefreshChannels?: () => Promise<void> | void
}) {
  const [users, setUsers] = useState<AuthUserRow[]>([])
  const [roles, setRoles] = useState<string[]>([])
  const [denied, setDenied] = useState(false)
  const [d, setD] = useState<Draft>(EMPTY)
  const [mode, setMode] = useState<'idle' | 'new' | 'edit'>('idle')
  const [busy, setBusy] = useState(false)
  const [activeOnly, setActiveOnly] = useState(false)
  const [sessions, setSessions] = useState<AuthSessionRow[]>([])
  const [sessionsActiveOnly, setSessionsActiveOnly] = useState(true)
  const [sessionsBusy, setSessionsBusy] = useState(false)
  const [status, setStatus] = useState<{ msg: string; ok: boolean } | null>(null)

  // backend returns camelCase (id, displayName, allowedChannelIds, isActive) — normalize to our row shape
  const norm = (r: any): AuthUserRow => ({
    ...r,
    user_id: r.user_id ?? r.id,
    display_name: r.display_name ?? r.displayName,
    allowed_channel_ids: r.allowed_channel_ids ?? r.allowedChannelIds,
    is_active: r.is_active ?? r.isActive,
  })
  const load = useCallback(async () => {
    try {
      const r = await settingsApi.users(true)
      setUsers((r.users || []).map(norm))
    } catch (e: any) { if (e?.status === 403) setDenied(true) }
    settingsApi.roles().then((r) => setRoles((r.roles || []).map((x) => x.name))).catch(() => {})
  }, [])
  useEffect(() => { load() }, [load])
  const loadSessions = useCallback(async (userId: string, activeOnly = sessionsActiveOnly) => {
    setSessionsBusy(true)
    try {
      const response = await settingsApi.sessions(userId, activeOnly)
      setSessions(response.sessions || [])
    } catch {
      setSessions([])
    } finally {
      setSessionsBusy(false)
    }
  }, [sessionsActiveOnly])
  useEffect(() => {
    if (d.user_id) loadSessions(d.user_id)
    else setSessions([])
  }, [d.user_id, sessionsActiveOnly]) // eslint-disable-line react-hooks/exhaustive-deps

  const selected = users.find((u) => u.user_id === d.user_id) || null
  const baseline = mode === 'edit' && selected ? fromRow(selected) : EMPTY
  const dirty = mode !== 'idle' && !sameDraft(d, baseline)

  // when leaving a dirty editor, ask the user instead of silently discarding
  const [confirmLeave, setConfirmLeave] = useState<(() => void) | null>(null)
  const guard = (action: () => void) => { if (dirty) setConfirmLeave(() => action); else action() }
  function startNew() { guard(() => { setD(EMPTY); setMode('new'); setStatus(null) }) }
  function selectUser(u: AuthUserRow) { guard(() => { setD(fromRow(u)); setMode('edit'); setStatus(null) }) }
  function closeEditor() { guard(() => { setD(EMPTY); setMode('idle'); setStatus(null) }) }

  async function save() {
    setBusy(true); setStatus(null)
    try {
      const allowed = parseAllowed(d.allowed)
      if (allowed !== '*') {
        const unknown = unknownChannelIds(allowed, channels)
        if (unknown.length) throw new Error(`Unknown or unavailable channel IDs: ${unknown.join(', ')}`)
      }
      if (d.user_id) {
        const body: any = { display_name: d.display_name, roles: d.roles, allowed_channel_ids: allowed, is_active: d.is_active }
        if (d.password) body.password = d.password
        const r = await settingsApi.updateUser(d.user_id, body)
        if (!r.success) throw new Error(r.error || 'Update failed')
        setD({ ...d, password: '' })
      } else {
        if (!d.username.trim() || !d.password) throw new Error('Username and password required')
        const r = await settingsApi.createUser({ username: d.username.trim(), password: d.password, displayName: d.display_name, roles: d.roles, allowedChannelIds: allowed, isActive: d.is_active })
        if (!r.success) throw new Error(r.error || 'Create failed')
        setD(EMPTY); setMode('idle')
      }
      setStatus({ msg: 'Saved', ok: true }); await load()
    } catch (e: any) { setStatus({ msg: e?.message || 'Failed', ok: false }) }
    finally { setBusy(false) }
  }
  async function revoke() {
    if (!d.user_id) return
    setBusy(true)
    try { const r = await settingsApi.revokeSessions(d.user_id); setStatus({ msg: `Revoked ${r.revoked_count ?? 0} sessions`, ok: true }); await load() }
    catch { setStatus({ msg: 'Revoke failed', ok: false }) } finally { setBusy(false) }
  }
  async function revokeOne(sessionId: string) {
    if (!d.user_id) return
    setSessionsBusy(true); setStatus(null)
    try {
      await settingsApi.revokeSession(sessionId)
      setStatus({ msg: 'Session revoked', ok: true })
      await loadSessions(d.user_id)
    } catch (e: any) {
      setStatus({ msg: e?.message || 'Session revoke failed', ok: false })
    } finally {
      setSessionsBusy(false)
    }
  }
  const toggleRole = (r: string) => setD((x) => ({ ...x, roles: x.roles.includes(r) ? x.roles.filter((y) => y !== r) : [...x.roles, r] }))
  const allChannels = d.allowed.trim() === '*'
  const chosenChannels = selectedChannels(d.allowed)
  const toggleChannel = (id: number) => setD((current) => {
    const selected = toggleChannelSelection(parseAllowed(current.allowed), id, channels)
    return { ...current, allowed: serializeChannels(new Set(selected)) }
  })
  const shownUsers = activeOnly ? users.filter((user) => user.is_active !== false) : users

  if (denied) return <div className="set-section"><h3>Users</h3><div className="set-denied"><IconAlertTriangle size={15} /> You don't have permission to manage users (needs users:manage).</div></div>

  return (
    <div className="set-section">
      <h3>Users</h3>

      <div className="usr-list">
        <div className="usr-list-toolbar">
          <button className="usr-new" onClick={startNew}><IconPlus size={16} /> New user</button>
          <label className="usr-active-only">
            <input type="checkbox" checked={activeOnly} onChange={(event) => setActiveOnly(event.target.checked)} />
            <span className="usr-switch" aria-hidden="true" />
            <span>Active only</span>
          </label>
        </div>
        <div className="usr-cards">
          {mode === 'new' && (
            <div className="usr-card draft on">
              <span className="usr-card-name">{d.username.trim() || 'New user'}{dirty && <i className="usr-dot" title="Unsaved changes" />}</span>
              <span className="usr-card-roles">{d.roles.join(', ') || 'unsaved draft'}</span>
            </div>
          )}
          {shownUsers.map((u) => (
            <button key={u.user_id} className={`usr-card ${mode === 'edit' && d.user_id === u.user_id ? 'on' : ''}`} onClick={() => selectUser(u)}>
              <span className="usr-card-name">{u.username}{u.is_active === false && <em> · off</em>}
                {mode === 'edit' && d.user_id === u.user_id && dirty && <i className="usr-dot" title="Unsaved changes" />}</span>
              <span className="usr-card-roles">{(u.roles || []).join(', ') || 'no roles'}</span>
            </button>
          ))}
          {shownUsers.length === 0 && <div className="set-note">{activeOnly ? 'No active users.' : 'No users.'}</div>}
        </div>
      </div>

      {mode === 'idle' && (
        <div className="set-note" style={{ paddingTop: 4 }}>
          Select a user to edit, or press <b>New user</b> to create one.
          {status && <span className={`set-status ${status.ok ? 'ok' : 'err'}`} style={{ marginLeft: 10 }}>{status.msg}</span>}
        </div>
      )}

      {mode !== 'idle' && (
      <div className="usr-editor">
        <div className="usr-editor-head">
          <span className="usr-editor-title">{d.user_id ? `Edit · ${d.username}` : 'New user'}</span>
          {dirty && <span className="usr-dirty">Unsaved changes</span>}
          <button className="modal-close usr-editor-close" title="Close editor" onClick={closeEditor}><IconX size={15} /></button>
        </div>
        <div className="usr-grid">
          <div className="set-row"><span className="set-label">Username</span>
            <input value={d.username} disabled={!!d.user_id} onChange={(e) => setD({ ...d, username: e.target.value })} /></div>
          <div className="set-row"><span className="set-label">Display name</span>
            <input value={d.display_name} onChange={(e) => setD({ ...d, display_name: e.target.value })} /></div>
          <div className="set-row"><span className="set-label">{d.user_id ? 'Reset password (blank = keep)' : 'Password'}</span>
            <input type="password" autoComplete="new-password" value={d.password} onChange={(e) => setD({ ...d, password: e.target.value })} /></div>
        </div>

        <div className="set-row">
          <span className="set-label">Allowed channels ({channels.length} available)</span>
          <div className="usr-channel-tools">
            <button type="button" className="mon-btn sm" onClick={() => setD({ ...d, allowed: '*' })}>All</button>
            <button type="button" className="mon-btn sm" onClick={() => setD({ ...d, allowed: '' })}>None</button>
            <button type="button" className="mon-btn sm" disabled={busy} onClick={() => onRefreshChannels?.()}>Refresh</button>
            <span className="set-note">{allChannels ? 'all channels' : `${chosenChannels.size} selected`}</span>
          </div>
          <div className="usr-channel-picker">
            {channels.map((channel) => {
              const checked = allChannels || chosenChannels.has(channel.id)
              return (
              <label key={channel.id} className={`usr-channel-option ${checked ? 'on' : ''}`} title={channel.title}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleChannel(channel.id)}
                />
                <span className="usr-checkmark" aria-hidden="true" />
                <span className="usr-channel-copy">
                  <span className="usr-channel-title">{channel.title}</span>
                  <span className="usr-channel-id">Channel #{channel.id}</span>
                </span>
              </label>
              )
            })}
            {channels.length === 0 && <span className="set-note">No accessible channels.</span>}
          </div>
        </div>

        <div className="set-row"><span className="set-label">Roles</span>
          <div className="set-roles">
            {roles.length === 0 && <span className="set-note">no roles loaded</span>}
            {roles.map((r) => (
              <Choice key={r} checked={d.roles.includes(r)} onChange={() => toggleRole(r)} className="usr-role-choice">
                {r}
              </Choice>
            ))}
          </div>
        </div>

        <Choice
          checked={d.is_active}
          disabled={d.user_id === currentUserId}
          title={d.user_id === currentUserId ? 'You cannot disable your current account.' : undefined}
          onChange={(checked) => setD({ ...d, is_active: checked })}
          className="usr-account-active"
        >
          Active account
        </Choice>

        {selected && (
          <div className="set-sessions">
            <div className="set-sessions-head">
              <div className="set-label">Sessions: {sessions.length}</div>
              <Choice checked={sessionsActiveOnly} onChange={setSessionsActiveOnly} className="usr-session-filter">
                Active only
              </Choice>
              <button className="mon-btn sm" disabled={sessionsBusy} onClick={() => loadSessions(selected.user_id)}>
                Refresh
              </button>
            </div>
            {sessionsBusy && sessions.length === 0 && <div className="set-note">Loading sessions…</div>}
            {!sessionsBusy && sessions.length === 0 && <div className="set-note">No sessions.</div>}
            {sessions.map((session) => (
              <div key={session.id} className="set-session-row">
                <div>
                  <div className="mono">
                    {session.id.slice(0, 12)} · {session.clientIp || 'IP unavailable'}
                    {session.id === currentSessionId && ' · current session'}
                  </div>
                  <div className="set-note">
                    Last seen {session.lastSeenAt ? new Date(session.lastSeenAt).toLocaleString() : 'unknown'}
                    {session.expiresAt ? ` · expires ${new Date(session.expiresAt).toLocaleString()}` : ''}
                    {session.revokedAt ? ` · revoked ${new Date(session.revokedAt).toLocaleString()}` : ''}
                  </div>
                  {session.userAgent && <div className="set-note" title={session.userAgent}>{session.userAgent}</div>}
                </div>
                {!session.revokedAt && (
                  <button
                    className="mon-btn sm danger"
                    disabled={sessionsBusy || session.id === currentSessionId}
                    title={session.id === currentSessionId ? 'The current session cannot be revoked here.' : undefined}
                    onClick={() => revokeOne(session.id)}
                  >
                    Revoke
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="set-user-actions">
          <button className="mon-btn accent" disabled={busy || !dirty} onClick={save}><IconDeviceFloppy size={15} /> Save user</button>
          {d.user_id && (
            <button
              className="mon-btn"
              disabled={busy || d.user_id === currentUserId}
              title={d.user_id === currentUserId ? 'Bulk revoke is disabled for the current account to protect this session.' : undefined}
              onClick={revoke}
            >
              <IconLogout2 size={15} /> Revoke sessions
            </button>
          )}
          {status && <span className={`set-status ${status.ok ? 'ok' : 'err'}`}>{status.msg}</span>}
        </div>
      </div>
      )}

      {confirmLeave && (
        <div className="scrim" onClick={() => setConfirmLeave(null)}>
          <div className="modal usr-confirm" onClick={(e) => e.stopPropagation()}>
            <div className="usr-confirm-title"><IconAlertTriangle size={16} /> Unsaved changes</div>
            <p>You have unsaved changes in <b>{d.username.trim() || 'New user'}</b>. Leave and discard them?</p>
            <div className="usr-confirm-actions">
              <button className="mon-btn" autoFocus onClick={() => setConfirmLeave(null)}>Stay here</button>
              <button className="mon-btn danger" onClick={() => { confirmLeave(); setConfirmLeave(null) }}>Discard &amp; leave</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
