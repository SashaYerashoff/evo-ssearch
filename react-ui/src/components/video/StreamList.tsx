import { IconChevronRight, IconPlayerPlay, IconVideo } from '@tabler/icons-react'
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

  if (!channels.length) {
    return <div className="empty-state"><IconVideo size={30} /><div>No channels available.</div></div>
  }

  return (
    <div className="stream-list" role="list" aria-label="Channels">
      {channels.map((channel) => {
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
  )
}
