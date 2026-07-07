import { IconVideo, IconFilter, IconClock, IconArrowsSort, IconDownload, IconRefresh } from '@tabler/icons-react'
import type { Channel, ArchiveFilters } from '../../api/types'

const SOURCES = [
  { v: '', label: 'All evidence' },
  { v: 'vlm_summary', label: 'Video descriptions' },
  { v: 'vlm_alert', label: 'VLM alerts' },
  { v: 'probe', label: 'CLIP probes' },
]
const TIMES = [
  { v: '1', label: 'Last 1h' }, { v: '6', label: 'Last 6h' }, { v: '24', label: 'Last 24h' },
  { v: '72', label: 'Last 3d' }, { v: '168', label: 'Last 7d' }, { v: '0', label: 'All time' },
]
const ROWS = ['12', '24', '36', '48']

export function FilterBar({
  filters, channels, onChange, onLoad, loading, count,
}: {
  filters: ArchiveFilters
  channels: Channel[]
  onChange: (f: Partial<ArchiveFilters>) => void
  onLoad: () => void
  loading: boolean
  count: string
}) {
  return (
    <div className="filter-bar">
      <label className="qf">
        <IconVideo size={15} />
        <select value={filters.channelId || ''} onChange={(e) => onChange({ channelId: e.target.value })}>
          <option value="">All streams</option>
          {channels.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
        </select>
      </label>
      <label className="qf">
        <IconFilter size={15} />
        <select value={filters.source || ''} onChange={(e) => onChange({ source: e.target.value })}>
          {SOURCES.map((s) => <option key={s.v} value={s.v}>{s.label}</option>)}
        </select>
      </label>
      <label className="qf">
        <IconClock size={15} />
        <select value={filters.hours || '24'} onChange={(e) => onChange({ hours: e.target.value })}>
          {TIMES.map((t) => <option key={t.v} value={t.v}>{t.label}</option>)}
        </select>
      </label>
      <label className="qf">
        <IconArrowsSort size={15} />
        <select value={filters.sortBy || 'similarity'} onChange={(e) => onChange({ sortBy: e.target.value })}>
          <option value="similarity">Similarity</option>
          <option value="time">Newest</option>
        </select>
      </label>
      <label className="qf">
        <select value={filters.rows || '24'} onChange={(e) => onChange({ rows: e.target.value })}>
          {ROWS.map((r) => <option key={r} value={r}>{r} rows</option>)}
        </select>
      </label>
      <button className="btn primary" onClick={onLoad} disabled={loading}>
        {loading ? <IconRefresh size={15} className="spin" /> : <IconDownload size={15} />}
        Load archive
      </button>
      <span className="count-note">{count}</span>
    </div>
  )
}
