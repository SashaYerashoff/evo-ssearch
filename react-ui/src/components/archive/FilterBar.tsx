import { useEffect, useMemo, useRef, useState } from 'react'
import {
  IconArrowsSort,
  IconCalendarEvent,
  IconCheck,
  IconChevronDown,
  IconClock,
  IconDownload,
  IconFilter,
  IconRefresh,
  IconSearch,
  IconVideo,
} from '@tabler/icons-react'
import type { Channel, ArchiveFilters } from '../../api/types'
import { Dropdown } from '../shell/Dropdown'
import { ToolbarActionMenu } from '../shell/ToolbarActionMenu'
import { DateRangeModal } from './DateRangeModal'
import type { ArchiveProbeOption } from '../../api/detections'

const SOURCES = [
  { v: '', label: 'All evidence' },
  { v: 'semantic_snapshot', label: 'Continuous semantic archive' },
  { v: 'vlm_summary', label: 'Video descriptions' },
  { v: 'vlm_alert', label: 'VLM alerts' },
  { v: 'probe', label: 'Semantic probes' },
]
export const TIMES = [
  { v: '1', label: 'Last 1h' }, { v: '6', label: 'Last 6h' }, { v: '24', label: 'Last 24h' },
  { v: '72', label: 'Last 3d' }, { v: '168', label: 'Last 7d' }, { v: '0', label: 'All time' },
]
function rangeLabel(f: ArchiveFilters): string {
  const fmt = (ms?: string) => (ms ? new Date(Number(ms)).toLocaleDateString([], { month: 'short', day: 'numeric' }) : '…')
  return `${fmt(f.sinceMs)} → ${fmt(f.untilMs)}`
}

function ChannelPicker({
  channels,
  selected,
  onChange,
}: {
  channels: Channel[]
  selected: string[]
  onChange: (values: string[]) => void
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const selectedSet = useMemo(() => new Set(selected), [selected])
  const selectedChannels = channels.filter((channel) => selectedSet.has(String(channel.id)))
  const filtered = channels.filter((channel) => {
    const needle = query.trim().toLocaleLowerCase()
    return !needle || `${channel.title} #${channel.id}`.toLocaleLowerCase().includes(needle)
  })
  const summary = selectedChannels.length === 0
    ? 'All streams'
    : selectedChannels.length === 1
      ? selectedChannels[0].title
      : `${selectedChannels.length} streams`

  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  function toggle(channelId: string) {
    const next = new Set(selected)
    if (next.has(channelId)) next.delete(channelId)
    else next.add(channelId)
    onChange(channels.map((channel) => String(channel.id)).filter((id) => next.has(id)))
  }

  return (
    <div className={`archive-channel-picker ${open ? 'open' : ''}`} ref={rootRef}>
      <button
        type="button"
        className="archive-channel-picker-toggle"
        title={selectedChannels.length > 1 ? selectedChannels.map((channel) => channel.title).join(', ') : summary}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <IconVideo size={15} />
        <span>{summary}</span>
        <IconChevronDown size={13} />
      </button>
      {open && (
        <div className="archive-channel-picker-pop" role="dialog" aria-label="Select archive streams">
          <div className="archive-channel-picker-tools">
            <label>
              <IconSearch size={14} />
              <input
                type="search"
                value={query}
                autoFocus
                placeholder="Filter streams…"
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
            <button type="button" className="btn compact" onClick={() => onChange([])}>Use all</button>
          </div>
          <button
            type="button"
            className={`archive-channel-choice all ${selected.length === 0 ? 'on' : ''}`}
            onClick={() => onChange([])}
          >
            <span className="archive-channel-check">{selected.length === 0 && <IconCheck size={14} />}</span>
            <span><b>All current and future streams</b><small>Dynamic scope</small></span>
          </button>
          <div className="archive-channel-choice-list">
            {filtered.map((channel) => {
              const id = String(channel.id)
              const active = selectedSet.has(id)
              return (
                <button
                  type="button"
                  key={id}
                  className={`archive-channel-choice ${active ? 'on' : ''}`}
                  onClick={() => toggle(id)}
                >
                  <span className="archive-channel-check">{active && <IconCheck size={14} />}</span>
                  <span><b>{channel.title}</b><small>#{id}</small></span>
                </button>
              )
            })}
            {!filtered.length && <div className="archive-channel-empty">No matching streams</div>}
          </div>
          <div className="archive-channel-picker-foot">
            <span>{selected.length ? `${selected.length} selected` : 'Using all streams'}</span>
            <button type="button" className="btn primary compact" onClick={() => setOpen(false)}>Done</button>
          </div>
        </div>
      )}
    </div>
  )
}

export function FilterBar({
  filters, channels, probes, probesLoading, onChange, onLoad, onRefresh, loading,
}: {
  filters: ArchiveFilters
  channels: Channel[]
  probes: ArchiveProbeOption[]
  probesLoading: boolean
  onChange: (f: Partial<ArchiveFilters>) => void
  onLoad: () => void
  onRefresh: () => void
  loading: boolean
}) {
  const [timeOpen, setTimeOpen] = useState(false)
  const [rangeOpen, setRangeOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const custom = !!(filters.sinceMs || filters.untilMs)
  const timeLabel = custom ? rangeLabel(filters) : (TIMES.find((t) => t.v === (filters.hours || '24'))?.label || 'Last 24h')
  const selectedChannels = filters.channelIds?.length
    ? filters.channelIds
    : (filters.channelId ? [filters.channelId] : [])

  useEffect(() => {
    if (!timeOpen) return
    const onDown = (e: MouseEvent) => { if (!menuRef.current?.contains(e.target as Node)) setTimeOpen(false) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [timeOpen])

  return (
    <div className="filter-block">
      <span className="atp-glabel"><IconFilter size={13} /> Filters</span>
      <div className="filter-bar">
      <ChannelPicker
        channels={channels}
        selected={selectedChannels}
        onChange={(values) => onChange({
          channelIds: values.length ? values : undefined,
          channelId: values.length === 1 ? values[0] : undefined,
        })}
      />
      <Dropdown variant="chip" icon={<IconFilter size={15} />} value={filters.source || ''} onChange={(v) => onChange({ source: v, probeId: v === 'probe' ? filters.probeId : undefined })}
        options={SOURCES.map((s) => ({ value: s.v, label: s.label }))} />
      {filters.source === 'probe' && (
        <Dropdown
          variant="chip"
          icon={<IconFilter size={15} />}
          value={filters.probeId || ''}
          disabled={probesLoading}
          title={probesLoading ? 'Loading archived probes' : 'Filter archived semantic probe hits'}
          onChange={(v) => onChange({ probeId: v })}
          options={[
            { value: '', label: probesLoading ? 'Loading probes…' : 'All semantic probes' },
            ...probes.map((p) => ({ value: p.id, label: `${p.name} (${p.hitCount})` })),
          ]}
        />
      )}

      <div className="qf qf-split">
        <div className="qf-menu" ref={menuRef}>
          <button type="button" className={`qf-seg ${custom ? 'custom' : ''}`} onClick={() => setTimeOpen((v) => !v)}>
            <IconClock size={15} /> {timeLabel} <IconChevronDown size={13} />
          </button>
          {timeOpen && (
            <div className="qf-pop">
              {TIMES.map((t) => (
                <button key={t.v} type="button" className={`qf-opt ${!custom && (filters.hours || '24') === t.v ? 'on' : ''}`}
                  onClick={() => { onChange({ hours: t.v, sinceMs: undefined, untilMs: undefined }); setTimeOpen(false) }}>{t.label}</button>
              ))}
              <div className="qf-sep" />
              <button type="button" className={`qf-opt pick ${custom ? 'on' : ''}`} onClick={() => {
                setTimeOpen(false)
                setRangeOpen(true)
              }}>
                <IconCalendarEvent size={14} /> Custom range…
              </button>
            </div>
          )}
        </div>
        <span className="qf-divider" />
        <button type="button" className={`qf-seg qf-icon ${custom ? 'on' : ''}`} onClick={() => setRangeOpen((v) => !v)} title="Pick date range">
          <IconCalendarEvent size={15} />
        </button>
        {rangeOpen && (
          <DateRangeModal
            sinceMs={filters.sinceMs} untilMs={filters.untilMs}
            onApply={(since, until) => onChange({ sinceMs: since, untilMs: until })}
            onClear={() => onChange({ sinceMs: undefined, untilMs: undefined })}
            onClose={() => setRangeOpen(false)}
          />
        )}
      </div>

      <Dropdown variant="chip" icon={<IconArrowsSort size={15} />} value={filters.sortBy || 'similarity'} onChange={(v) => onChange({ sortBy: v })}
        options={[{ value: 'similarity', label: 'Similarity' }, { value: 'time', label: 'Newest' }]} />

      <div className="filter-bar-actions">
        <ToolbarActionMenu actions={[{
          id: 'refresh',
          label: 'Refresh filters',
          icon: <IconRefresh className={loading || probesLoading ? 'spin' : ''} size={15} />,
          onSelect: onRefresh,
          disabled: loading || probesLoading,
        }]} />
        <button className="btn primary" type="button" onClick={() => onLoad()} disabled={loading}>
          {loading ? <IconRefresh size={15} className="spin" /> : <IconDownload size={15} />}
          Load archive
        </button>
      </div>
      </div>
    </div>
  )
}
