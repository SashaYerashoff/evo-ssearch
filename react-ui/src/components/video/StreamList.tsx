import { useMemo, useState } from 'react'
import { IconChevronRight, IconPlayerPlay, IconSearch, IconVideo, IconX } from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import type { ChannelRuntime } from '../../api/video'

export type StreamState = 'describing' | 'capturing' | 'paused' | 'idle'

/** What the operator needs to know at a glance: is this camera being described right now. */
export function streamState(runtime: ChannelRuntime | undefined): StreamState {
  const video = runtime?.video
  if (!video || !video.running) return 'idle'
  if (video.paused) return 'paused'
  return video.summarization_enabled === false ? 'capturing' : 'describing'
}

/** The list is the section's home screen, so the operator lands on it every time
   they close a channel. A filter that reset on every return would have to be
   re-ticked constantly, so it outlives the mount. */
const ONLY_LIVE_STORAGE_KEY = 'eva.video.streamList.onlyLive.v1'

function readOnlyLive(): boolean {
  try { return window.localStorage.getItem(ONLY_LIVE_STORAGE_KEY) === '1' } catch { return false }
}

const STATE_LABEL: Record<StreamState, string> = {
  describing: 'Live description',
  capturing: 'Capturing · no description',
  paused: 'Paused',
  idle: 'Not running',
}

export function StreamList({ channels, runtime, activeChannelId, onOpen }: {
  channels: Channel[]
  runtime: ChannelRuntime[]
  activeChannelId?: number | null
  onOpen: (channelId: number) => void
}) {
  const byChannel = new Map(runtime.map((entry) => [entry.channelId, entry]))
  const [query, setQuery] = useState('')
  const [onlyLive, setOnlyLive] = useState(readOnlyLive)

  function changeOnlyLive(next: boolean) {
    setOnlyLive(next)
    try {
      if (next) window.localStorage.setItem(ONLY_LIVE_STORAGE_KEY, '1')
      else window.localStorage.removeItem(ONLY_LIVE_STORAGE_KEY)
    } catch { /* private mode */ }
  }

  // Filter on the live list, so rows keep updating their state while a query is
  // active instead of freezing on a snapshot.
  const needle = query.trim().toLowerCase()
  const matches = useMemo(() => channels.filter((channel) => {
    if (onlyLive && streamState(byChannel.get(channel.id)) !== 'describing') return false
    if (!needle) return true
    return channel.title.toLowerCase().includes(needle) || String(channel.id).includes(needle)
  }), [channels, needle, onlyLive, runtime])   // eslint-disable-line react-hooks/exhaustive-deps
  const filtered = Boolean(needle) || onlyLive

  if (!channels.length) {
    return <div className="empty-state"><IconVideo size={30} /><div>No channels available.</div></div>
  }

  return (
    <div className="stream-list-wrap">
      <div className="stream-search">
        <div className="stream-search-field">
          <IconSearch size={15} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search channels — name or number…"
            aria-label="Search channels"
          />
          {query && (
            <button type="button" className="stream-search-clear" onClick={() => setQuery('')}
              title="Clear search" aria-label="Clear search">
              <IconX size={15} />
            </button>
          )}
        </div>
        <label className="stream-filter" title="Show only channels being described right now">
          <input type="checkbox" checked={onlyLive} onChange={(event) => changeOnlyLive(event.target.checked)} />
          <span>Show only live on</span>
        </label>

        <span className="stream-search-count">
          <strong>{filtered ? `${matches.length}/${channels.length}` : channels.length}</strong>
          <span>channels</span>
        </span>
      </div>

      {matches.length === 0 && (
        <div className="empty-state">
          {onlyLive && !needle ? 'No channel is being described right now.' : `No channel matches “${query}”.`}
        </div>
      )}

    <div className="stream-list" role="list" aria-label="Channels">
      {matches.map((channel) => {
        const entry = byChannel.get(channel.id)
        const state = streamState(entry)
        const video = entry?.video
        const facts = [
          video?.interval_sec ? `every ${video.interval_sec}s` : '',
          video?.batch_size ? `${video.batch_size}-frame batch` : '',
          video?.model || '',
        ].filter(Boolean)
        return (
          <button
            key={channel.id}
            type="button"
            role="listitem"
            className={`stream-row ${state} ${activeChannelId === channel.id ? 'is-active' : ''}`}
            onClick={() => onOpen(channel.id)}
            title={`Open summaries for ${channel.title}`}
          >
            <span className={`stream-dot ${state}`} aria-hidden="true" />
            <span className="stream-row-main">
              <span className="stream-row-name">{channel.title}</span>
              <span className="stream-row-facts">
                {`ch ${channel.id}`}{facts.length ? ` · ${facts.join(' · ')}` : ''}
              </span>
            </span>
            <span className={`stream-row-state ${state}`}>
              {state === 'describing' && <IconPlayerPlay size={13} />}
              {STATE_LABEL[state]}
            </span>
            <IconChevronRight className="stream-row-go" size={16} />
          </button>
        )
      })}
    </div>
    </div>
  )
}
