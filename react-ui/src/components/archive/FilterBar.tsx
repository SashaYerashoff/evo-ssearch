import { useEffect, useRef, useState } from 'react'
import { IconVideo, IconFilter, IconClock, IconArrowsSort, IconDownload, IconRefresh, IconChevronDown, IconCalendarEvent } from '@tabler/icons-react'
import type { Channel, ArchiveFilters } from '../../api/types'
import { Dropdown } from '../shell/Dropdown'
import { DateRangeModal } from './DateRangeModal'
import type { ArchiveProbeOption } from '../../api/detections'

const SOURCES = [
  { v: '', label: 'All evidence' },
  { v: 'vlm_summary', label: 'Video descriptions' },
  { v: 'vlm_alert', label: 'VLM alerts' },
  { v: 'probe', label: 'CLIP probes' },
]
export const TIMES = [
  { v: '1', label: 'Last 1h' }, { v: '6', label: 'Last 6h' }, { v: '24', label: 'Last 24h' },
  { v: '72', label: 'Last 3d' }, { v: '168', label: 'Last 7d' }, { v: '0', label: 'All time' },
]
function rangeLabel(f: ArchiveFilters): string {
  const fmt = (ms?: string) => (ms ? new Date(Number(ms)).toLocaleDateString([], { month: 'short', day: 'numeric' }) : '…')
  return `${fmt(f.sinceMs)} → ${fmt(f.untilMs)}`
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
      <Dropdown variant="chip" icon={<IconVideo size={15} />} value={filters.channelId || ''} onChange={(v) => onChange({ channelId: v })}
        options={[{ value: '', label: 'All streams' }, ...channels.map((c) => ({ value: String(c.id), label: c.title }))]} />
      <Dropdown variant="chip" icon={<IconFilter size={15} />} value={filters.source || ''} onChange={(v) => onChange({ source: v, probeId: v === 'probe' ? filters.probeId : undefined })}
        options={SOURCES.map((s) => ({ value: s.v, label: s.label }))} />
      {filters.source === 'probe' && (
        <Dropdown
          variant="chip"
          icon={<IconFilter size={15} />}
          value={filters.probeId || ''}
          disabled={probesLoading}
          title={probesLoading ? 'Loading archived probes' : 'Filter archived CLIP probe hits'}
          onChange={(v) => onChange({ probeId: v })}
          options={[
            { value: '', label: probesLoading ? 'Loading probes…' : 'All CLIP probes' },
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
      <Dropdown variant="chip" value={filters.rows || '24'} onChange={(v) => onChange({ rows: v })}
        options={['12', '24', '36', '48'].map((r) => ({ value: r, label: `${r} rows` }))} />

      <button className="btn" type="button" onClick={onRefresh} disabled={loading || probesLoading} title="Reload channel and archive probe filters">
        <IconRefresh size={15} /> Refresh filters
      </button>
      <button className="btn primary" type="button" onClick={() => onLoad()} disabled={loading}>
        {loading ? <IconRefresh size={15} className="spin" /> : <IconDownload size={15} />}
        Load archive
      </button>
      </div>
    </div>
  )
}
