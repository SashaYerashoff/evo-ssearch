import { useEffect, useState, useCallback, useRef } from 'react'
import {
  IconFilter, IconAdjustmentsHorizontal, IconLetterT, IconPhoto, IconSearch, IconDownload, IconX, IconSparkles,
} from '@tabler/icons-react'
import type { Channel, Detection, ArchiveFilters } from '../../api/types'
import type { AgentDrive } from '../../App'
import { api } from '../../api/client'
import { listArchive, searchText, normalizeDetection, detectionsFromResult } from '../../api/detections'
import { FilterBar } from './FilterBar'
import { DetectionCard } from './DetectionCard'
import { InspectorModal } from './InspectorModal'

export type ArchiveTool = null | 'filters' | 'search' | 'text' | 'image'
type Tool = Exclude<ArchiveTool, null>

// tools whose returned frames get rendered into the archive grid
const VIEW_TOOLS = new Set(['search_archive', 'search_text', 'search_detections', 'search_folder', 'get_detections', 'get_video_summaries'])
// view tools that carry a free-text query we can type into the search box for show
const TYPING_TOOLS = new Set(['search_archive', 'search_text', 'search_detections', 'search_folder'])
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))
function prettyTool(name: string): string {
  const map: Record<string, string> = {
    search_archive: 'searching the archive', search_detections: 'searching the archive', search_folder: 'searching frames',
    get_detections: 'pulling detections', get_detection_summary: 'summarising detections',
    describe_frame: 'describing a frame', list_channels: 'listing channels', survey_channels: 'surveying channels',
    calibrate_probe_from_archive: 'calibrating a probe', create_probe: 'creating a probe', update_probe: 'updating a probe',
    list_probes: 'listing probes', generate_report: 'generating a report', create_bookmark: 'bookmarking a frame',
    get_video_summaries: 'reading video summaries', normalize_time_window: 'resolving the time window',
  }
  return map[name] || name.replace(/_/g, ' ')
}

// map an agent tool's time args back onto the FilterBar preset so the dropdown visibly changes
function agentHoursFromArgs(args: any): string | null {
  let hrs: number | null = null
  if (args?.hours != null) hrs = Number(args.hours)
  else if (args?.days != null) hrs = Number(args.days) * 24
  else if (args?.since_ms != null) {
    const since = Number(args.since_ms)
    if (since <= 0) return '0'
    const until = args?.until_ms != null ? Number(args.until_ms) : Date.now()
    hrs = (until - since) / 3_600_000
  }
  if (hrs == null || !isFinite(hrs)) return null
  if (hrs <= 0 || hrs > 168 * 1.5) return '0' // <=0 or > ~10 days → "All time"
  const presets = [1, 6, 24, 72, 168]
  return String(presets.reduce((b, p) => (Math.abs(p - hrs!) < Math.abs(b - hrs!) ? p : b), presets[0]))
}

const DEFAULT_FILTERS: ArchiveFilters = { source: '', hours: '24', sortBy: 'similarity', rows: '24' }

const TOOL_META: Record<Tool, { title: string; icon: JSX.Element; width: number }> = {
  filters: { title: 'Archive filters', icon: <IconFilter size={16} />, width: 384 },
  search: { title: 'Search controls', icon: <IconAdjustmentsHorizontal size={16} />, width: 340 },
  text: { title: 'Text query', icon: <IconLetterT size={16} />, width: 460 },
  image: { title: 'Image query', icon: <IconPhoto size={16} />, width: 360 },
}

const SOURCES = [
  { v: '', label: 'All evidence' }, { v: 'vlm_summary', label: 'Video descriptions' },
  { v: 'vlm_alert', label: 'VLM alerts' }, { v: 'probe', label: 'CLIP probes' },
]
const TIMES = [
  { v: '1', label: 'Last 1h' }, { v: '6', label: 'Last 6h' }, { v: '24', label: 'Last 24h' },
  { v: '72', label: 'Last 3d' }, { v: '168', label: 'Last 7d' }, { v: '0', label: 'All time' },
]

export function ArchiveScreen({
  channels, tool, drive, noAnim, onFilters, onToolHandled,
}: {
  channels: Channel[]
  tool: ArchiveTool
  drive?: AgentDrive | null
  noAnim?: boolean
  onFilters?: (f: ArchiveFilters) => void
  onToolHandled: () => void
}) {
  const [filters, setFilters] = useState<ArchiveFilters>(DEFAULT_FILTERS)
  const [items, setItems] = useState<Detection[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const [selected, setSelected] = useState<Detection | null>(null)
  const [minMatch, setMinMatch] = useState(0)
  const [modalTool, setModalTool] = useState<Tool | null>(null)
  const [textValue, setTextValue] = useState('')
  const [agentStep, setAgentStep] = useState<string | null>(null)
  const [agentTyping, setAgentTyping] = useState(false)
  const typeToken = useRef(0)

  const patch = (p: Partial<ArchiveFilters>) => setFilters((f) => ({ ...f, ...p }))

  const runLoad = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const { items, total } = await listArchive(filters, channels)
      setItems(items); setNote(`${items.length} of ${total} frames`)
    } catch (e: any) { setError(e?.message || 'Archive load failed'); setItems([]) }
    finally { setLoading(false) }
  }, [filters, channels])

  const runText = useCallback(async (q: string) => {
    setLoading(true); setError(null)
    try {
      const results = await searchText(q, filters, channels)
      setItems(results); setNote(`${results.length} matches · “${q}”`)
    } catch (e: any) { setError(e?.message || 'Text search failed') }
    finally { setLoading(false) }
  }, [filters, channels])

  const runImageSearch = useCallback(async (blob: Blob, label: string) => {
    setLoading(true); setError(null)
    try {
      const form = new FormData()
      form.append('image', blob, 'ref.jpg')
      form.append('limit', filters.rows || '24')
      form.append('sort_by', filters.sortBy || 'similarity')
      if (filters.channelId) form.append('channel_id', filters.channelId)
      if (filters.source) form.append('source', filters.source)
      const res = await api.postForm('/detections/search_image', form)
      const cmap = new Map(channels.map((c) => [c.id, c.title]))
      const results = (res.results || []).map((x: any) => normalizeDetection(x, cmap))
      setItems(results); setNote(`${results.length} similar · ${label}`)
    } catch (e: any) { setError(e?.message || 'Image search failed') }
    finally { setLoading(false) }
  }, [filters, channels])

  const runSimilar = useCallback((d: Detection) => {
    setSelected(null)
    if (!d.thumbnail) { setError('No image on this frame.'); return }
    const bin = atob(d.thumbnail)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    runImageSearch(new Blob([bytes], { type: 'image/jpeg' }), `“${d.probeName}”`)
  }, [runImageSearch])

  // agent "types" the query into the search box (visual only — no fetch; results come from the tool_result)
  const animateTyping = useCallback(async (q: string) => {
    if (noAnim) return
    const token = ++typeToken.current
    setModalTool('text'); setAgentTyping(true); setTextValue('')
    await sleep(240)
    for (let i = 1; i <= q.length; i++) {
      if (typeToken.current !== token) return
      setTextValue(q.slice(0, i)); await sleep(34)
    }
  }, [noAnim])

  useEffect(() => { runLoad() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (tool) { setModalTool(tool); onToolHandled() } }, [tool]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { onFilters?.(filters) }, [filters]) // eslint-disable-line react-hooks/exhaustive-deps

  // mirror each agent action onto the working console, and render view-tool results into the grid
  useEffect(() => {
    if (!drive) return
    const { name, args, done, error, result } = drive
    if (!done) {
      setAgentStep(prettyTool(name))
      if (VIEW_TOOLS.has(name)) {
        // agent drives the console controls: channel / source / time / sort / rows visibly change
        if (args?.channel_id != null) {
          const ch = channels.find((c) => String(c.id) === String(args.channel_id))
          if (ch) patch({ channelId: String(ch.id) })
        }
        if (args?.source) patch({ source: String(args.source) })
        const h = agentHoursFromArgs(args)
        if (h != null) patch({ hours: h, sinceMs: undefined, untilMs: undefined })
        if (args?.sort_by) patch({ sortBy: String(args.sort_by) === 'time' ? 'time' : 'similarity' })
        if (args?.limit != null && isFinite(Number(args.limit))) {
          const ps = [12, 24, 36, 48]; const n = Number(args.limit)
          patch({ rows: String(ps.reduce((b, p) => (Math.abs(p - n) < Math.abs(b - n) ? p : b), ps[0])) })
        }
        const q = String(args?.query || args?.event_query || args?.positive_query || args?.text || '').trim()
        if (TYPING_TOOLS.has(name) && q) animateTyping(q)
      }
      return
    }
    // done → route the tool's returned frames into the grid
    if (VIEW_TOOLS.has(name)) {
      typeToken.current++ // stop any in-flight typing
      setAgentTyping(false); setModalTool(null)
      if (!error) {
        const found = detectionsFromResult(result, channels)
        setItems(found)
        setNote(`Agent · ${found.length} frame${found.length === 1 ? '' : 's'} · ${prettyTool(name)}`)
        setError(found.length ? null : 'Agent returned no frames for this query.')
      }
    }
    setAgentStep(error ? `${prettyTool(name)} — failed` : prettyTool(name))
    const t = setTimeout(() => setAgentStep(null), error ? 2600 : 700)
    return () => clearTimeout(t)
  }, [drive?.seq]) // eslint-disable-line react-hooks/exhaustive-deps

  const displayed = minMatch > 0 ? items.filter((d) => (d.matchPct ?? 0) >= minMatch) : items

  function toDtLocal(ms?: string) {
    if (!ms) return ''
    const d = new Date(Number(ms)); const p = (n: number) => String(n).padStart(2, '0')
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`
  }
  const fromDt = (v: string) => (v ? String(new Date(v).getTime()) : undefined)

  function toolBody(t: Tool) {
    if (t === 'filters') return (
      <div className="wform" style={{ minWidth: 260 }}>
        <div className="wfield"><label>Stream</label>
          <select value={filters.channelId || ''} onChange={(e) => patch({ channelId: e.target.value })}>
            <option value="">All streams</option>
            {channels.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
          </select>
        </div>
        <div className="wgrid">
          <div className="wfield"><label>Source</label>
            <select value={filters.source || ''} onChange={(e) => patch({ source: e.target.value })}>
              {SOURCES.map((s) => <option key={s.v} value={s.v}>{s.label}</option>)}
            </select>
          </div>
          <div className="wfield"><label>Time range</label>
            <select value={filters.hours || '24'} onChange={(e) => patch({ hours: e.target.value, sinceMs: undefined, untilMs: undefined })}>
              {TIMES.map((tt) => <option key={tt.v} value={tt.v}>{tt.label}</option>)}
            </select>
          </div>
        </div>
        <div className="wfield"><label>From</label>
          <input type="datetime-local" value={toDtLocal(filters.sinceMs)} onChange={(e) => patch({ sinceMs: fromDt(e.target.value) })} />
        </div>
        <div className="wfield"><label>To</label>
          <input type="datetime-local" value={toDtLocal(filters.untilMs)} onChange={(e) => patch({ untilMs: fromDt(e.target.value) })} />
        </div>
        <button className="btn primary" style={{ justifyContent: 'center' }} onClick={() => { runLoad(); setModalTool(null) }}>
          <IconDownload size={15} /> Load archive
        </button>
      </div>
    )
    if (t === 'search') return (
      <div className="wform" style={{ minWidth: 250 }}>
        <div className="wgrid">
          <div className="wfield"><label>Sort by</label>
            <select value={filters.sortBy || 'similarity'} onChange={(e) => patch({ sortBy: e.target.value })}>
              <option value="similarity">Similarity</option><option value="time">Newest</option>
            </select>
          </div>
          <div className="wfield"><label>Results</label>
            <select value={filters.rows || '24'} onChange={(e) => patch({ rows: e.target.value })}>
              {['12', '24', '36', '48'].map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
        </div>
        <div className="wfield">
          <label>Min match: {minMatch}%</label>
          <input type="range" min={0} max={100} step={1} value={minMatch} onChange={(e) => setMinMatch(Number(e.target.value))} />
        </div>
        <div className="wnote">Hides frames below the match threshold ({displayed.length}/{items.length} shown).</div>
      </div>
    )
    if (t === 'text') return (
      <form className="wform" style={{ minWidth: 320 }} onSubmit={(e) => { e.preventDefault(); const v = textValue.trim(); if (v) { runText(v); setModalTool(null) } }}>
        <div className="wnote">
          {agentTyping ? <><IconSparkles size={13} /> Agent is typing a query…</> : 'Natural-language search over archived frames. Current filters apply.'}
        </div>
        <div className="wrow">
          <input
            name="q" placeholder="Describe archived scene…" autoFocus={!agentTyping}
            value={textValue} readOnly={agentTyping}
            onChange={(e) => setTextValue(e.target.value)}
            className={agentTyping ? 'agent-caret' : ''}
            style={{ flex: 1, background: 'var(--void-tile)', border: '1px solid var(--line-2)', borderRadius: 9, color: 'var(--text)', padding: '9px 11px', outline: 'none' }}
          />
          <button className="btn primary" disabled={agentTyping}><IconSearch size={15} /> Search</button>
        </div>
      </form>
    )
    // image
    return (
      <div className="wform" style={{ minWidth: 250 }}>
        <div className="wnote">Upload a reference image; runs visual similarity over archived frames.</div>
        <label className="btn" style={{ justifyContent: 'center' }}>
          <IconPhoto size={15} /> Choose image
          <input type="file" accept="image/*" style={{ display: 'none' }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) { runImageSearch(f, f.name); setModalTool(null) } }} />
        </label>
      </div>
    )
  }

  return (
    <div className="center-scroll">
      {(agentStep || agentTyping) && (
        <div className="agent-driving">
          <span className="ad-dot" /><IconSparkles size={15} />
          <span>Agent is <b>{agentStep || 'searching the archive'}</b> — watch the console</span>
        </div>
      )}
      <FilterBar filters={filters} channels={channels} onChange={patch} onLoad={runLoad} loading={loading} count={note} />

      {error && <div className="empty-state" style={{ color: 'var(--danger)', padding: 30 }}>{error}</div>}
      {loading && items.length === 0 && <div className="loading-state"><div className="spinner" /><div>Loading archive…</div></div>}
      {!loading && !error && displayed.length === 0 && <div className="empty-state">No archived frames for these filters.</div>}

      {displayed.length > 0 && (
        <div className="card-grid">
          {displayed.map((d) => <DetectionCard key={d.key} d={d} onClick={() => setSelected(d)} />)}
        </div>
      )}

      {selected && <InspectorModal d={selected} onClose={() => setSelected(null)} onFindSimilar={runSimilar} />}

      {modalTool && (
        <div className={`scrim ${agentTyping ? 'driven' : ''}`} onClick={() => { if (!agentTyping) setModalTool(null) }}>
          <div className={`modal ${agentTyping ? 'driven' : ''}`} style={{ maxWidth: TOOL_META[modalTool].width + 44 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                {TOOL_META[modalTool].icon}{TOOL_META[modalTool].title}
              </div>
              <button className="modal-close" onClick={() => setModalTool(null)}><IconX size={18} /></button>
            </div>
            <div className="modal-body">{toolBody(modalTool)}</div>
          </div>
        </div>
      )}
    </div>
  )
}
