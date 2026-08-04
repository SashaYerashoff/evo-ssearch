import { useEffect, useState, useCallback, useMemo, useRef, type ReactNode } from 'react'
import {
  IconAdjustmentsHorizontal,
  IconAlertTriangle,
  IconFilter,
  IconLetterT,
  IconLoader2,
  IconPhoto,
  IconSearch,
  IconSparkles,
} from '@tabler/icons-react'
import type { Channel, Detection, ArchiveFilters } from '../../api/types'
import type { AgentDrive } from '../../App'
import { api } from '../../api/client'
import {
  buildArchiveSearchPayload,
  detectionsFromResult,
  getArchiveProbeOptions,
  detImageSrc,
  listArchive,
  normalizeDetection,
  searchText,
  type ArchiveProbeOption,
  type ArchiveSearchCoverage,
} from '../../api/detections'
import { FilterBar, TIMES } from './FilterBar'
import { ToolTabs } from '../shell/ToolTabs'
import { DetectionCard } from './DetectionCard'
import { InspectorModal } from './InspectorModal'
import {
  archiveScoreRange,
  archiveScoreThreshold,
  formatArchiveScore,
  passesArchiveScoreThreshold,
} from './archiveScore'
import { archiveCoverageMessages } from './archiveCoverage'

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

export function ArchiveScreen({
  navigation, channels, drive, similarDrive, noAnim, canReportFeedback, canReportIncidents, canExport, onFilters, onRefreshChannels,
  onSimilarDriveHandled,
}: {
  navigation?: ReactNode
  channels: Channel[]
  drive?: AgentDrive | null
  similarDrive?: { detection: Detection; seq: number } | null
  noAnim?: boolean
  canReportFeedback?: boolean
  canReportIncidents?: boolean
  canExport?: boolean
  onFilters?: (f: ArchiveFilters) => void
  onRefreshChannels?: () => Promise<void> | void
  onSimilarDriveHandled?: () => void
}) {
  const [filters, setFilters] = useState<ArchiveFilters>(DEFAULT_FILTERS)
  const [items, setItems] = useState<Detection[]>([])
  const [loading, setLoading] = useState(false)
  const [textSearchPending, setTextSearchPending] = useState(false)
  const [searchCoverage, setSearchCoverage] = useState<ArchiveSearchCoverage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const [selected, setSelected] = useState<Detection | null>(null)
  const [scoreSliderPercent, setScoreSliderPercent] = useState(0)
  const [textValue, setTextValue] = useState('')
  const [agentStep, setAgentStep] = useState<string | null>(null)
  const [agentTyping, setAgentTyping] = useState(false)
  const [nextOffset, setNextOffset] = useState(0)
  const [total, setTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [resultMode, setResultMode] = useState<'list' | 'search'>('list')
  const [appliedFilters, setAppliedFilters] = useState<ArchiveFilters | null>(null)
  const [probeOptions, setProbeOptions] = useState<ArchiveProbeOption[]>([])
  const [probesLoading, setProbesLoading] = useState(false)
  const [filterRefresh, setFilterRefresh] = useState(0)
  // horizontal accordion: exactly one tool block is expanded, the rest collapse to summary chips
  const [openTool, setOpenTool] = useState<'filters' | 'text' | 'image'>('filters')
  const typeToken = useRef(0)
  const requestSeq = useRef(0)
  const probeRequestSeq = useRef(0)
  const loadingRef = useRef(false)
  const loadMoreRef = useRef<HTMLDivElement>(null)
  const resultsScrollRef = useRef<HTMLDivElement>(null)
  const nextOffsetRef = useRef(0)   // read by the observer without re-creating it each load
  // Distinguishes a live agent run from re-playing a finished drive on (re)mount:
  // only true once we've actually seen an in-progress action this mount.
  const sawAgentProgress = useRef(false)

  const patch = useCallback((p: Partial<ArchiveFilters>) => {
    requestSeq.current++
    loadingRef.current = false
    setLoading(false)
    setTextSearchPending(false)
    setSearchCoverage(null)
    setNextOffset(0)
    setFilters((f) => ({ ...f, ...p }))
  }, [])

  const runLoad = useCallback(async (requestedOffset = 0, append = false) => {
    if (loadingRef.current) return
    loadingRef.current = true
    const seq = ++requestSeq.current
    setTextSearchPending(false)
    if (!append) setSearchCoverage(null)
    setLoading(true); setError(null)
    try {
      const result = await listArchive(filters, channels, requestedOffset)
      if (requestSeq.current !== seq) return
      const last = result.offset + result.items.length
      setItems((current) => {
        if (!append) return result.items
        const existing = new Set(current.map((item) => item.key))
        return [...current, ...result.items.filter((item) => !existing.has(item.key))]
      })
      setNextOffset(last)
      setTotal(result.total)
      setHasMore(result.hasMore)
      setResultMode('list')
      if (!append) setScoreSliderPercent(0)
      setAppliedFilters({ ...filters })
      if (!append) setSelected(null)
      setNote(result.items.length ? `${last} loaded` : '0 loaded')
    } catch (e: any) {
      if (requestSeq.current !== seq) return
      setError(e?.message || (append ? 'Could not load more archive matches' : 'Archive load failed'))
      if (!append) {
        setItems([])
        setTotal(0)
        setNextOffset(0)
      }
      setHasMore(false)
    } finally {
      if (requestSeq.current === seq) {
        loadingRef.current = false
        setLoading(false)
      }
    }
  }, [filters, channels])

  const runText = useCallback(async (q: string) => {
    loadingRef.current = true
    const seq = ++requestSeq.current
    setTextSearchPending(true)
    setLoading(true); setError(null)
    try {
      const searchResult = await searchText(q, filters, channels)
      if (requestSeq.current !== seq) return
      const results = searchResult.items
      setSearchCoverage(searchResult.coverage)
      setItems(results); setNote(`${results.length} matches · “${q}”`)
      setScoreSliderPercent(0)
      setNextOffset(0); setTotal(results.length); setHasMore(false); setResultMode('search')
      setAppliedFilters({ ...filters }); setSelected(null)
    } catch (e: any) {
      if (requestSeq.current === seq) {
        setSearchCoverage(null)
        setError(e?.message || 'Text search failed')
      }
    } finally {
      if (requestSeq.current === seq) {
        loadingRef.current = false
        setTextSearchPending(false)
        setLoading(false)
      }
    }
  }, [filters, channels])

  const runImageSearch = useCallback(async (blob: Blob, label: string) => {
    loadingRef.current = true
    const seq = ++requestSeq.current
    setTextSearchPending(false)
    setLoading(true); setError(null)
    try {
      const form = new FormData()
      form.append('image', blob, 'ref.jpg')
      for (const [key, value] of Object.entries(buildArchiveSearchPayload(filters))) {
        if (value !== undefined && value !== '') form.append(key, String(value))
      }
      const res = await api.postForm('/detections/search_image', form)
      if (requestSeq.current !== seq) return
      const cmap = new Map(channels.map((c) => [c.id, c.title]))
      const results = (res.results || []).map((x: any) => normalizeDetection(x, cmap))
      setSearchCoverage((res.coverage && typeof res.coverage === 'object') ? res.coverage : null)
      setItems(results); setNote(`${results.length} similar · ${label}`)
      setScoreSliderPercent(0)
      setNextOffset(0); setTotal(results.length); setHasMore(false); setResultMode('search')
      setAppliedFilters({ ...filters }); setSelected(null)
    } catch (e: any) {
      if (requestSeq.current === seq) {
        setSearchCoverage(null)
        setError(e?.message || 'Image search failed')
      }
    } finally {
      if (requestSeq.current === seq) {
        loadingRef.current = false
        setLoading(false)
      }
    }
  }, [filters, channels])

  const runSimilar = useCallback(async (d: Detection) => {
    setSelected(null)
    try {
      let blob: Blob
      if (d.thumbnail) {
        const bin = atob(d.thumbnail)
        const bytes = new Uint8Array(bin.length)
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
        blob = new Blob([bytes], { type: 'image/jpeg' })
      } else {
        const source = detImageSrc(d)
        if (!source) throw new Error('No image on this frame.')
        const response = await fetch(source, { credentials: 'same-origin' })
        if (!response.ok) throw new Error(`Could not load the reference frame (${response.status}).`)
        blob = await response.blob()
      }
      await runImageSearch(blob, `“${d.probeName}”`)
    } catch (exception: any) {
      setError(exception?.message || 'Could not search from this frame.')
    }
  }, [runImageSearch])

  useEffect(() => {
    if (!similarDrive) return
    void runSimilar(similarDrive.detection)
    onSimilarDriveHandled?.()
  }, [similarDrive?.seq]) // eslint-disable-line react-hooks/exhaustive-deps

  // agent "types" the query into the search box (visual only — no fetch; results come from the tool_result)
  const animateTyping = useCallback(async (q: string) => {
    if (noAnim) return
    const token = ++typeToken.current
    setAgentTyping(true); setTextValue('')
    await sleep(240)
    for (let i = 1; i <= q.length; i++) {
      if (typeToken.current !== token) return
      setTextValue(q.slice(0, i)); await sleep(34)
    }
  }, [noAnim])

  useEffect(() => {
    if (!similarDrive) void runLoad()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { onFilters?.(filters) }, [filters]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const seq = ++probeRequestSeq.current
    if (filters.source !== 'probe') {
      setProbeOptions([])
      setProbesLoading(false)
      return
    }
    setProbesLoading(true)
    getArchiveProbeOptions(filters)
      .then((options) => {
        if (probeRequestSeq.current === seq) setProbeOptions(options)
      })
      .catch(() => {
        if (probeRequestSeq.current === seq) setProbeOptions([])
      })
      .finally(() => {
        if (probeRequestSeq.current === seq) setProbesLoading(false)
      })
  }, [
    filters.source,
    filters.channelId,
    filters.channelIds?.join(','),
    filters.hours,
    filters.sinceMs,
    filters.untilMs,
    filterRefresh,
  ])

  const refreshFilters = useCallback(async () => {
    await onRefreshChannels?.()
    setFilterRefresh((n) => n + 1)
  }, [onRefreshChannels])

  // mirror each agent action onto the working console, and render view-tool results into the grid
  useEffect(() => {
    if (!drive) return
    const { name, args, done, error, result } = drive
    if (VIEW_TOOLS.has(name)) {
      if (Array.isArray(args?.channel_ids)) {
        const selected = channels
          .filter((channel) => args.channel_ids.some((id: unknown) => String(id) === String(channel.id)))
          .map((channel) => String(channel.id))
        patch({
          channelIds: selected,
          channelId: selected.length === 1 ? selected[0] : undefined,
        })
      }
      if (args?.channel_id != null) {
        const ch = channels.find((c) => String(c.id) === String(args.channel_id))
        if (ch) patch({ channelIds: [String(ch.id)], channelId: String(ch.id) })
      }
      if (args?.source) patch({ source: String(args.source) })
      if (args?.probe_id) patch({ source: 'probe', probeId: String(args.probe_id) })
      const h = agentHoursFromArgs(args)
      if (h != null) patch({ hours: h, sinceMs: undefined, untilMs: undefined })
      if (args?.sort_by) patch({ sortBy: String(args.sort_by) === 'time' ? 'time' : 'similarity' })
      if (args?.limit != null && isFinite(Number(args.limit))) {
        const ps = [12, 24, 36, 48]; const n = Number(args.limit)
        patch({ rows: String(ps.reduce((b, p) => (Math.abs(p - n) < Math.abs(b - n) ? p : b), ps[0])) })
      }
      if (done && (args?.channel_id != null || Array.isArray(args?.channel_ids) || args?.source || h != null || args?.sort_by || args?.limit != null)) {
        setOpenTool('filters')
      }
    }
    if (!done) {
      sawAgentProgress.current = true
      setAgentStep(prettyTool(name))
      if (VIEW_TOOLS.has(name)) {
        // agent drives the console controls: channel / source / time / sort / rows visibly change
        const q = String(args?.query || args?.event_query || args?.positive_query || args?.text || '').trim()
        if (TYPING_TOOLS.has(name) && q) { setOpenTool('text'); animateTyping(q) }
        else if (args?.channel_id != null || Array.isArray(args?.channel_ids) || args?.source || args?.sort_by || args?.limit != null) setOpenTool('filters')
      }
      return
    }
    // done → route the tool's returned frames into the grid
    if (VIEW_TOOLS.has(name)) {
      typeToken.current++ // stop any in-flight typing
      requestSeq.current++
      loadingRef.current = false
      setLoading(false)
      setTextSearchPending(false)
      setSearchCoverage(null)
      setAgentTyping(false)
      if (!error) {
        const found = detectionsFromResult(result, channels)
        setItems(found)
        setScoreSliderPercent(0)
        setNextOffset(0); setTotal(found.length); setHasMore(false); setResultMode('search')
        setAppliedFilters({ ...filters })
        setNote(`Agent · ${found.length} frame${found.length === 1 ? '' : 's'} · ${prettyTool(name)}`)
        setError(found.length ? null : 'Agent returned no frames for this query.')
        if (args?.open_detection_id != null) {
          const requestedId = String(args.open_detection_id)
          const requested = found.find((item) => String(item.id) === requestedId)
          if (requested) setSelected(requested)
        }
      }
    }
    // A finished drive replayed on (re)mount — with no in-progress step seen — is stale:
    // load its frames but don't flash the "Agent is searching…" banner when nothing is running.
    if (!sawAgentProgress.current) { setAgentStep(null); return }
    sawAgentProgress.current = false
    setAgentStep(error ? `${prettyTool(name)} — failed` : prettyTool(name))
    const t = setTimeout(() => setAgentStep(null), error ? 2600 : 700)
    return () => clearTimeout(t)
  }, [drive?.seq]) // eslint-disable-line react-hooks/exhaustive-deps

  const scoreRange = useMemo(() => archiveScoreRange(items), [items])
  const scoreThreshold = archiveScoreThreshold(scoreRange, scoreSliderPercent)
  const displayed = items.filter((d) => passesArchiveScoreThreshold(d, scoreThreshold))
  const filtersDirty = !!appliedFilters && JSON.stringify(appliedFilters) !== JSON.stringify(filters)
  const archiveMatchCount = resultMode === 'list' ? total : items.length
  const coverageMessages = useMemo(
    () => archiveCoverageMessages(searchCoverage, channels),
    [searchCoverage, channels],
  )

  // keep the offset in a ref so the observer reads the latest value WITHOUT being torn down
  // and rebuilt on every load (which was re-firing instantly and loading the whole archive)
  useEffect(() => { nextOffsetRef.current = nextOffset }, [nextOffset])

  useEffect(() => {
    const sentinel = loadMoreRef.current
    if (!sentinel || !hasMore || resultMode !== 'list' || filtersDirty) return
    // one batch per genuine scroll-into-view — the observer is stable, so it never chains
    // into a runaway that loads (and re-renders) the entire archive at once
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting && !loadingRef.current) void runLoad(nextOffsetRef.current, true)
    }, { root: resultsScrollRef.current, rootMargin: '400px 0px' })
    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [filtersDirty, hasMore, resultMode, runLoad])

  // live summaries shown on collapsed chips
  const filtersSummary = [
    filters.channelIds?.length
      ? (filters.channelIds.length === 1
          ? (channels.find((c) => String(c.id) === filters.channelIds?.[0])?.title || `ch ${filters.channelIds[0]}`)
          : `${filters.channelIds.length} streams`)
      : filters.channelId
        ? (channels.find((c) => String(c.id) === filters.channelId)?.title || `ch ${filters.channelId}`)
        : 'All streams',
    filters.source === 'probe' && filters.probeId
      ? (probeOptions.find((p) => p.id === filters.probeId)?.name || filters.probeId)
      : null,
    (filters.sinceMs || filters.untilMs) ? 'custom range' : (TIMES.find((t) => t.v === (filters.hours || '24'))?.label || 'Last 24h'),
  ].filter(Boolean).join(' · ')
  const q = textValue.trim()
  const scoreLabel = !scoreRange.hasScores
    ? 'No scores'
    : !scoreRange.hasSpread
      ? `All @ ${formatArchiveScore(scoreRange.min)}`
      : scoreThreshold > 0
        ? `≥ ${formatArchiveScore(scoreThreshold)}`
        : 'All'
  const textSummary = textSearchPending
    ? 'Searching archive…'
    : (q ? `“${q.length > 26 ? q.slice(0, 26) + '…' : q}”` : '—') + (scoreThreshold > 0 ? ` · ${scoreLabel}` : '')

  const TOOL_META: Record<typeof openTool, { Icon: any; label: string; summary: string }> = {
    filters: { Icon: IconFilter, label: 'Filters', summary: filtersSummary },
    text: { Icon: IconLetterT, label: 'Text query', summary: textSummary },
    image: { Icon: IconPhoto, label: 'Image', summary: '—' },
  }

  // active tool's controls, shown to the right of the fixed tab strip
  const expanded = () => {
    if (openTool === 'filters') return (
      <div className="atp-open" key="filters">
        <FilterBar
          filters={filters}
          channels={channels}
          probes={probeOptions}
          probesLoading={probesLoading}
          onChange={patch}
          onLoad={() => runLoad(0)}
          onRefresh={refreshFilters}
          loading={loading}
        />
      </div>
    )
    if (openTool === 'text') return (
      <div className="atp-open atp-group atp-textgroup" key="text" aria-busy={textSearchPending}>
        <span className="atp-glabel"><IconLetterT size={13} /> Text query</span>
        <form className="atp-text" onSubmit={(e) => {
          e.preventDefault()
          const v = textValue.trim()
          if (v && !textSearchPending) void runText(v)
        }}>
          <input placeholder="describe an archived scene…" autoFocus={!agentTyping}
            value={textValue} readOnly={agentTyping} className={agentTyping ? 'agent-caret' : ''}
            onChange={(e) => setTextValue(e.target.value)} />
          <button className="btn primary atp-search-submit" disabled={agentTyping || textSearchPending || !textValue.trim()}>
            {textSearchPending ? <IconLoader2 className="spin" size={15} /> : <IconSearch size={15} />}
            {textSearchPending ? 'Searching…' : 'Search'}
          </button>
        </form>
        <span className="sr-only" role="status" aria-live="polite">
          {textSearchPending ? 'Semantic archive search in progress.' : ''}
        </span>
        {/* similarity threshold for the text-query results — filters out weak matches */}
        <div className="atp-min">
          <label title="The slider is normalized to the score range returned by this query">
            <IconAdjustmentsHorizontal size={12} /> Min match · {scoreLabel}
          </label>
          <input
            type="range" min={0} max={100} step={1}
            value={scoreRange.hasSpread ? scoreSliderPercent : 0}
            disabled={!scoreRange.hasSpread}
            onChange={(e) => setScoreSliderPercent(Number(e.target.value))}
          />
          {scoreRange.hasScores && (
            <span className="atp-score-range">
              {displayed.length}/{items.length} · range {formatArchiveScore(scoreRange.min)}–{formatArchiveScore(scoreRange.max)}
            </span>
          )}
        </div>
      </div>
    )
    return (
      <div className="atp-open atp-group" key="image">
        <span className="atp-glabel"><IconPhoto size={13} /> Image query</span>
        <label className="btn atp-image" title="Upload a reference image — visual similarity search">
          <IconPhoto size={15} /> Choose image
          <input type="file" accept="image/*" style={{ display: 'none' }}
            onChange={(e) => { const file = e.target.files?.[0]; if (file) runImageSearch(file, file.name); e.currentTarget.value = '' }} />
        </label>
      </div>
    )
  }

  return (
    <div className="center-scroll archive-screen">
      {(agentStep || agentTyping) && (
        <div className="agent-driving">
          <span className="ad-dot" /><IconSparkles size={15} />
          <span>Agent is <b>{agentStep || 'searching the archive'}</b> — watch the console</span>
        </div>
      )}
      <ToolTabs
        tabs={(['filters', 'text', 'image'] as const).map((t) => {
          const { Icon, label, summary } = TOOL_META[t]
          return { id: t, icon: <Icon size={13} />, label, summary }
        })}
        active={openTool}
        onSelect={(id) => setOpenTool(id as typeof openTool)}
        leading={navigation}
      >
        {expanded()}
      </ToolTabs>

      <div className="archive-results-head" role="status" aria-live="polite">
        <div className="archive-results-count">
          <strong>{archiveMatchCount.toLocaleString()}</strong>
          <span>{archiveMatchCount === 1 ? 'archive match' : 'archive matches'}</span>
        </div>
        <div className="archive-results-context">
          {textSearchPending
            ? `Searching archive for “${q}” · current results remain visible`
            : (note || `${items.length} loaded`)}
          {filtersDirty ? ' · Filters changed — load to apply' : ''}
        </div>
      </div>

      <div ref={resultsScrollRef} className="archive-results-scroll">
        {coverageMessages.length > 0 && (
          <div className="archive-coverage-notice" role="status" aria-live="polite">
            <IconAlertTriangle size={16} />
            <div>{coverageMessages.map((message) => <div key={message}>{message}</div>)}</div>
          </div>
        )}

        {error && <div className="empty-state" style={{ color: 'var(--danger)', padding: 30 }}>{error}</div>}
        {loading && items.length === 0 && (
          <div className="loading-state">
            <div className="spinner" />
            <div>{textSearchPending ? 'Searching semantic archive…' : 'Loading archive…'}</div>
          </div>
        )}
        {!loading && !error && displayed.length === 0 && <div className="empty-state">No archived frames for these filters.</div>}

        {displayed.length > 0 && (
          <div className="card-grid">
            {displayed.map((d) => <DetectionCard key={d.key} d={d} onClick={() => setSelected(d)} />)}
          </div>
        )}

        {resultMode === 'list' && !filtersDirty && (
          <div ref={loadMoreRef} className="archive-load-more">
            {loading && items.length > 0 && <><div className="spinner" /><span>Loading more matches…</span></>}
            {!loading && hasMore && <span>Scroll for more</span>}
            {!loading && !hasMore && items.length > 0 && !error && <span>All archive matches loaded</span>}
            {!loading && error && items.length > 0 && (
              <button type="button" className="btn" onClick={() => runLoad(nextOffset, true)}>Retry loading more</button>
            )}
          </div>
        )}
      </div>

      {selected && (
        <InspectorModal
          d={selected}
          channels={channels}
          canReportFeedback={!!canReportFeedback}
          canReportIncidents={!!canReportIncidents}
          canExport={!!canExport}
          onClose={() => setSelected(null)}
          onFindSimilar={runSimilar}
        />
      )}
    </div>
  )
}
