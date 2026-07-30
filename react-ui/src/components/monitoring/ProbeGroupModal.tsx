import { useMemo, useState } from 'react'
import { IconAlertTriangle, IconDeviceFloppy, IconTrash, IconX } from '@tabler/icons-react'
import type { ProbeChannelGroup } from '../../api/probes'
import type { Channel } from '../../api/types'

export function ProbeGroupModal({
  group,
  groups,
  channels,
  busy,
  error,
  onClose,
  onSave,
  onDelete,
}: {
  group: ProbeChannelGroup | null
  groups: ProbeChannelGroup[]
  channels: Channel[]
  busy: boolean
  error?: string | null
  onClose: () => void
  onSave: (input: { id?: string; name: string; channel_ids: number[] }) => void
  onDelete: (id: string) => void
}) {
  const [name, setName] = useState(group?.name || '')
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(group?.channel_ids || []),
  )
  const [deleteArmed, setDeleteArmed] = useState(false)
  const owners = useMemo(() => {
    const result = new Map<number, string>()
    for (const candidate of groups) {
      if (candidate.id === group?.id) continue
      for (const channelId of candidate.channel_ids || []) {
        result.set(Number(channelId), candidate.name)
      }
    }
    return result
  }, [group?.id, groups])

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal probe-group-modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">{group ? 'Edit channel group' : 'New channel group'}</div>
            <div className="brand-sub">A channel selected here moves out of its previous group.</div>
          </div>
          <button className="modal-close" onClick={onClose}><IconX size={18} /></button>
        </div>
        <div className="modal-body probe-group-body">
          {group?.read_only && (
            <div className="set-denied">
              <IconAlertTriangle size={15} />
              This group contains channels outside your scope and is read-only.
            </div>
          )}
          <div className="wfield">
            <label>Group name</label>
            <input
              value={name}
              maxLength={80}
              autoFocus
              disabled={group?.read_only || busy}
              onChange={(event) => setName(event.target.value)}
              placeholder="Perimeter, Berth 3, Office…"
            />
          </div>
          <div className="probe-group-selection-head">
            <span>Channels</span>
            <b>{selected.size} selected</b>
          </div>
          <div className="probe-group-channel-list">
            {channels.map((channel) => {
              const owner = owners.get(channel.id)
              return (
                <label key={channel.id} className="probe-group-channel">
                  <input
                    type="checkbox"
                    checked={selected.has(channel.id)}
                    disabled={group?.read_only || busy}
                    onChange={(event) => setSelected((current) => {
                      const next = new Set(current)
                      if (event.target.checked) next.add(channel.id)
                      else next.delete(channel.id)
                      return next
                    })}
                  />
                  <span>Ch {channel.id} · {channel.title}</span>
                  {owner && <small title={`Selecting moves this channel out of ${owner}`}>in {owner}</small>}
                </label>
              )
            })}
          </div>
          {error && <div className="chat-error"><IconAlertTriangle size={14} /> {error}</div>}
        </div>
        <div className="probe-footer">
          <div>
            {group && !group.read_only && (
              <button
                className={`mon-btn danger ${deleteArmed ? 'armed' : ''}`}
                disabled={busy}
                onClick={() => {
                  if (!deleteArmed) setDeleteArmed(true)
                  else onDelete(group.id)
                }}
              >
                <IconTrash size={15} /> {deleteArmed ? 'Confirm delete group' : 'Delete group'}
              </button>
            )}
          </div>
          <div className="probe-footer-actions">
            <button className="mon-btn" onClick={onClose}>Cancel</button>
            <button
              className="mon-btn accent"
              disabled={busy || group?.read_only || !name.trim()}
              onClick={() => onSave({
                id: group?.id,
                name: name.trim(),
                channel_ids: [...selected],
              })}
            >
              <IconDeviceFloppy size={15} /> {busy ? 'Saving…' : 'Save group'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
