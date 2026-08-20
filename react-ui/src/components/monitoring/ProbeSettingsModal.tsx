import { useEffect, useRef, useState } from 'react'
import {
  IconX, IconPlus, IconTrash, IconPhoto, IconDeviceFloppy, IconVideoOff,
  IconChevronRight, IconCrop, IconPlayerPlay, IconPlayerStop, IconBroadcast, IconAlertTriangle,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import {
  authorizeProbeInput,
  probeMutationRequiresBookmarkPermission,
  probeRangeDurationMs,
  probesApi,
  type ChannelStatus,
  type PatchAttentionResult,
  type Probe,
  type ProbeInput,
  type ProbeLiveSignal,
  type ProbeThresholdDefaults,
  type RoiNorm,
  type SemanticPresenceClass,
} from '../../api/probes'
import { recentFrameUrl } from '../../api/video'
import { Dropdown } from '../shell/Dropdown'
import { ProbeSparkline } from './ProbeCard'
import { SemanticPresenceCard } from './SemanticPresenceCard'

const SEVERITIES = ['info', 'low', 'normal', 'high', 'critical']

interface Draft {
  id?: string
  name: string
  channel_id?: number
  enabled: boolean
  pairs: { pos: string; neg: string }[]
  pos_floor: number
  margin: number
  bookmark: boolean
  severity: string
  cooldown: number
  dedupe: number
  imgData: string | null
  imgName: string
  imgFloor: number
  imgEnabled: boolean
  roiOn: boolean
  roi: RoiNorm | null
}

function normRoi(raw: any): RoiNorm | null {
  if (!raw) return null
  const r = Array.isArray(raw) ? { x: raw[0], y: raw[1], w: raw[2], h: raw[3] } : raw
  const x = Number(r.x), y = Number(r.y), w = Number(r.w), h = Number(r.h)
  if (![x, y, w, h].every(isFinite) || w <= 0.01 || h <= 0.01) return null
  return { x: Math.max(0, x), y: Math.max(0, y), w: Math.min(1, w), h: Math.min(1, h) }
}

function fromProbe(p: Probe | null, channels: Channel[], defaults: ProbeThresholdDefaults): Draft {
  // positives[]/negatives[] are authoritative — zip them into editable pairs
  const pos = p?.positives || []
  const neg = p?.negatives || []
  const len = Math.max(pos.length, neg.length, 1)
  const pairs = Array.from({ length: len }, (_, i) => ({ pos: pos[i] || '', neg: neg[i] || '' }))
  return {
    id: p?.id,
    name: p?.name || '',
    channel_id: p?.channel_id ?? channels[0]?.id,
    enabled: p?.enabled ?? true,
    pairs,
    pos_floor: p?.pos_floor ?? defaults.pos_floor,
    margin: p?.margin ?? defaults.margin,
    bookmark: p?.bookmark ?? false,
    severity: p?.severity || 'info',
    cooldown: p?.bookmark_cooldown_sec ?? 8,
    dedupe: p?.bookmark_dedupe_window_sec ?? 20,
    imgData: p?.image_probe?.data ?? null,
    imgName: p?.image_probe?.name || '',
    imgFloor: p?.image_probe?.pos_floor ?? 0.7,
    imgEnabled: p?.image_probe?.enabled ?? false,
    roiOn: !!p?.roi_enabled,
    roi: normRoi(p?.roi_norm),
  }
}

const pct = (v: number) => `${Math.round(v * 100)}%`

export function ProbeSettingsModal({ probe, channels, busy, canControlCapture, canCreateBookmarks, defaults, onClose, onSave, onCasted }: {
  probe: Probe | null
  channels: Channel[]
  busy: boolean
  canControlCapture: boolean
  canCreateBookmarks: boolean
  defaults: ProbeThresholdDefaults
  onClose: () => void
  onSave: (input: ProbeInput) => Promise<Probe | null>
  onCasted?: () => void
}) {
  const [d, setD] = useState<Draft>(() => fromProbe(probe, channels, defaults))
  const bookmarkMutationBlocked = probeMutationRequiresBookmarkPermission(probe, canCreateBookmarks)
  const set = (p: Partial<Draft>) => setD((x) => ({ ...x, ...p }))
  const setPair = (i: number, p: Partial<{ pos: string; neg: string }>) =>
    setD((x) => ({ ...x, pairs: x.pairs.map((pr, j) => (j === i ? { ...pr, ...p } : pr)) }))

  // pre-open a collapsed section when it already holds non-default settings (editing)
  const [openTuning] = useState(() => (
    (probe?.pos_floor ?? defaults.pos_floor) !== defaults.pos_floor
    || (probe?.margin ?? defaults.margin) !== defaults.margin
  ))
  const [openImage] = useState(() => !!probe?.image_probe?.data)
  // the two bottom dropdowns are controlled: their toggle sits OUTSIDE the scroll area and
  // never moves — the panel opens into the reserved space above it, so nothing jumps.
  const [techOpen, setTechOpen] = useState(false)
  const [advOpen, setAdvOpen] = useState(() => openTuning || openImage)

  // live preview + capture status of the selected channel
  const [previewSrc, setPreviewSrc] = useState('')
  const [pvErr, setPvErr] = useState(true)
  const [scoredFrameSrc, setScoredFrameSrc] = useState('')
  const [st, setSt] = useState<ChannelStatus>({ channel_id: d.channel_id ?? 0 })
  const [displayedSignal, setDisplayedSignal] = useState<ProbeLiveSignal | null>(null)
  const [signalFrameError, setSignalFrameError] = useState(false)
  const [applyMessage, setApplyMessage] = useState<string | null>(null)
  const [patchAttention, setPatchAttention] = useState<PatchAttentionResult | null>(null)
  const [patchBusyKey, setPatchBusyKey] = useState<string | null>(null)
  const [patchError, setPatchError] = useState<string | null>(null)
  const [captureBusy, setCaptureBusy] = useState(false)
  const previewBlobUrlRef = useRef<string | null>(null)
  const scoredFrameBlobUrlRef = useRef<string | null>(null)
  useEffect(() => {
    let disposed = false
    let requestNumber = 0
    let lastGoodAtMs = 0
    let previewTimer: number | null = null
    let signalTimer: number | null = null
    let previewController: AbortController | null = null
    let signalFrameController: AbortController | null = null
    let displayedTimestampMs = 0
    const revokeTimers = new Map<number, string>()
    const previousBlobUrl = previewBlobUrlRef.current
    previewBlobUrlRef.current = null
    if (previousBlobUrl) URL.revokeObjectURL(previousBlobUrl)
    const previousScoredBlobUrl = scoredFrameBlobUrlRef.current
    scoredFrameBlobUrlRef.current = null
    if (previousScoredBlobUrl) URL.revokeObjectURL(previousScoredBlobUrl)
    setPreviewSrc('')
    setPvErr(true)
    setScoredFrameSrc('')
    setSt({ channel_id: d.channel_id ?? 0 })
    setDisplayedSignal(null)
    setSignalFrameError(false)
    setPatchAttention(null)
    setPatchBusyKey(null)
    setPatchError(null)

    const schedulePreview = (delayMs: number) => {
      if (previewTimer != null) window.clearTimeout(previewTimer)
      previewTimer = window.setTimeout(loadPreview, delayMs)
    }
    const previewHasExpired = () => (
      lastGoodAtMs === 0 || Date.now() - lastGoodAtMs >= 15_000
    )
    const retireBlobUrl = (url: string) => {
      const timer = window.setTimeout(() => {
        revokeTimers.delete(timer)
        URL.revokeObjectURL(url)
      }, 5_000)
      revokeTimers.set(timer, url)
    }
    async function loadPreview() {
      // The operator preview must remain live independently of semantic
      // throughput. P/N/M is paired with a separate exact scored frame below.
      if (disposed || d.channel_id == null) return
      const controller = new AbortController()
      previewController = controller
      const watchdog = window.setTimeout(() => controller.abort(), 12_000)
      try {
        const response = await fetch(
          recentFrameUrl(d.channel_id, ++requestNumber),
          { credentials: 'include', cache: 'no-store', signal: controller.signal },
        )
        const contentType = String(response.headers.get('content-type') || '').toLowerCase()
        if (!response.ok || !contentType.startsWith('image/')) {
          throw new Error(`Preview frame unavailable (${response.status})`)
        }
        const blob = await response.blob()
        if (disposed) return
        const nextBlobUrl = URL.createObjectURL(blob)
        const oldBlobUrl = previewBlobUrlRef.current
        previewBlobUrlRef.current = nextBlobUrl
        lastGoodAtMs = Date.now()
        setPreviewSrc(nextBlobUrl)
        setPvErr(false)
        if (oldBlobUrl) retireBlobUrl(oldBlobUrl)
        schedulePreview(1_000)
      } catch {
        if (disposed) return
        // A transient slow/error response must not blank a frame that was
        // already loaded successfully. Escalate only after a bounded period
        // without any good replacement, while retrying without overlap.
        if (previewHasExpired()) setPvErr(true)
        schedulePreview(2_000)
      } finally {
        window.clearTimeout(watchdog)
        if (previewController === controller) previewController = null
      }
    }

    async function loadSignalFrame(signal: ProbeLiveSignal): Promise<boolean> {
      const timestampMs = Number(signal.timestamp_ms)
      if (!d.id || !Number.isFinite(timestampMs) || timestampMs <= 0) return false
      if (timestampMs === displayedTimestampMs && scoredFrameBlobUrlRef.current) {
        setDisplayedSignal(signal)
        setSignalFrameError(false)
        return true
      }
      const controller = new AbortController()
      signalFrameController = controller
      const watchdog = window.setTimeout(() => controller.abort(), 12_000)
      try {
        const frameUrl = signal.frame_url
          || `/probes/signal_frame/${d.channel_id}/${timestampMs}`
        const response = await fetch(frameUrl, {
          credentials: 'include',
          cache: 'no-store',
          signal: controller.signal,
        })
        const contentType = String(response.headers.get('content-type') || '').toLowerCase()
        if (!response.ok || !contentType.startsWith('image/')) return false
        const blob = await response.blob()
        if (disposed) return false
        const nextBlobUrl = URL.createObjectURL(blob)
        const oldBlobUrl = scoredFrameBlobUrlRef.current
        scoredFrameBlobUrlRef.current = nextBlobUrl
        displayedTimestampMs = timestampMs
        // React batches these updates: the operator never sees new P/N/M on
        // the previous image or a new image carrying the previous score.
        setScoredFrameSrc(nextBlobUrl)
        setDisplayedSignal(signal)
        setSignalFrameError(false)
        if (oldBlobUrl) retireBlobUrl(oldBlobUrl)
        return true
      } finally {
        window.clearTimeout(watchdog)
        if (signalFrameController === controller) signalFrameController = null
      }
    }

    const scheduleSignal = (delayMs: number) => {
      if (signalTimer != null) window.clearTimeout(signalTimer)
      signalTimer = window.setTimeout(pollSignal, delayMs)
    }
    async function pollSignal() {
      if (disposed || d.channel_id == null) return
      try {
        const nextStatus = await probesApi.status(d.channel_id, d.id)
        if (disposed) return
        setSt(nextStatus)
        if (d.id && nextStatus.live_signal) {
          const paired = await loadSignalFrame(nextStatus.live_signal)
          if (!disposed && !paired) {
            setSignalFrameError(true)
          }
        }
      } catch {
        // Keep the last complete image+score pair during a transient control
        // plane stall. The age badge makes that retained evidence explicit.
        if (!disposed) setSignalFrameError(true)
      } finally {
        if (!disposed) scheduleSignal(800)
      }
    }

    void pollSignal()
    void loadPreview()
    return () => {
      disposed = true
      if (previewTimer != null) window.clearTimeout(previewTimer)
      if (signalTimer != null) window.clearTimeout(signalTimer)
      previewController?.abort()
      signalFrameController?.abort()
      for (const [timer, url] of revokeTimers) {
        window.clearTimeout(timer)
        URL.revokeObjectURL(url)
      }
      revokeTimers.clear()
      const blobUrl = previewBlobUrlRef.current
      previewBlobUrlRef.current = null
      if (blobUrl) URL.revokeObjectURL(blobUrl)
      const scoredBlobUrl = scoredFrameBlobUrlRef.current
      scoredFrameBlobUrlRef.current = null
      if (scoredBlobUrl) URL.revokeObjectURL(scoredBlobUrl)
    }
  }, [d.channel_id, d.id])

  // The preview reads only EVA's captured-frame ring. An enabled saved probe,
  // and especially a direct V4L2 source, must therefore own a capture session
  // before the first preview/status poll. Previously the modal displayed a
  // misleading "Stop stream" button while the backend was idle and waited
  // forever for a frame that nobody was producing.
  useEffect(() => {
    const channelId = d.channel_id
    const selectedChannel = channels.find((channel) => channel.id === channelId)
    const directLocalSource = selectedChannel?.source === 'local_v4l2'
    if (
      !canControlCapture
      || channelId == null
      || !d.enabled
      || (!d.id && !directLocalSource)
    ) return
    let disposed = false
    setCaptureBusy(true)
    void probesApi.startCapture(channelId, probe?.fps)
      .then((response) => {
        if (disposed) return
        const state = response?.state || {}
        setSt((current) => ({
          ...current,
          channel_id: channelId,
          runtime_state: state.running === false ? 'idle' : 'running',
          capture_error: null,
        }))
      })
      .catch((error: any) => {
        if (disposed) return
        setSt((current) => ({
          ...current,
          channel_id: channelId,
          runtime_state: 'idle',
          capture_error: error?.message || 'Failed to start capture',
        }))
      })
      .finally(() => { if (!disposed) setCaptureBusy(false) })
    return () => { disposed = true }
  }, [canControlCapture, channels, d.channel_id, d.enabled, d.id, probe?.fps])

  // ROI drawing on the preview
  const pvRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<{ x: number; y: number } | null>(null)
  const relPoint = (e: React.MouseEvent) => {
    const r = pvRef.current!.getBoundingClientRect()
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) }
  }
  const roiDown = (e: React.MouseEvent) => { if (!d.roiOn || pvErr) return; e.preventDefault(); dragRef.current = relPoint(e) }
  const roiMove = (e: React.MouseEvent) => {
    if (!d.roiOn || !dragRef.current) return
    const p = relPoint(e), s = dragRef.current
    set({ roi: { x: Math.min(s.x, p.x), y: Math.min(s.y, p.y), w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y) } })
  }
  const roiUp = () => {
    if (!dragRef.current) return
    dragRef.current = null
    setD((x) => (x.roi && (x.roi.w < 0.02 || x.roi.h < 0.02) ? { ...x, roi: null } : x))
  }

  // cast panel
  const [castOpen, setCastOpen] = useState(false)
  const [castSel, setCastSel] = useState<Set<number>>(new Set())
  const [castConflict, setCastConflict] = useState<'skip' | 'create' | 'update'>('skip')
  const [castEnable, setCastEnable] = useState(true)
  const [castCopyRoi, setCastCopyRoi] = useState(false)
  const [castBusy, setCastBusy] = useState(false)
  const [castMsg, setCastMsg] = useState<{ text: string; err?: boolean } | null>(null)

  function onFile(f: File) {
    const r = new FileReader()
    r.onload = () => { const s = String(r.result || ''); set({ imgData: s.includes(',') ? s.split(',')[1] : s, imgName: f.name, imgEnabled: true }) }
    r.readAsDataURL(f)
  }

  function buildInput(): ProbeInput {
    const positives = d.pairs.map((p) => p.pos.trim()).filter(Boolean)
    const negatives = d.pairs.map((p) => p.neg.trim()).filter(Boolean)
    return authorizeProbeInput({
      id: d.id,
      name: d.name.trim() || 'Untitled probe',
      channel_id: d.channel_id,
      enabled: d.enabled,
      pairs: d.pairs.filter((p) => p.pos.trim() || p.neg.trim()).map((p) => ({ positive: p.pos.trim(), negative: p.neg.trim() })),
      positives, negatives,
      pos_floor: d.pos_floor, margin: d.margin, top_k: 6,
      window_sec: probe?.window_sec ?? 300,
      severity: d.severity,
      bookmark: d.bookmark,
      bookmark_cooldown_sec: d.cooldown,
      bookmark_dedupe_window_sec: d.dedupe,
      image_probe: d.imgData ? { data: d.imgData, name: d.imgName, pos_floor: d.imgFloor, enabled: d.imgEnabled } : null,
      roi_enabled: d.roiOn && !!d.roi,
      roi_norm: d.roiOn ? d.roi : null,
    }, canCreateBookmarks)
  }

  async function startStream(channelId = d.channel_id) {
    if (channelId == null || captureBusy) return
    setCaptureBusy(true)
    try {
      const response = await probesApi.startCapture(channelId, probe?.fps)
      const state = response?.state || {}
      setSt((current) => ({
        ...current,
        channel_id: channelId,
        runtime_state: state.running === false ? 'idle' : 'running',
        capture_error: null,
      }))
    } catch (error: any) {
      setSt((current) => ({
        ...current,
        channel_id: channelId,
        runtime_state: 'idle',
        capture_error: error?.message || 'Failed to start capture',
      }))
    } finally {
      setCaptureBusy(false)
    }
  }

  async function stopStream() {
    if (d.channel_id == null || captureBusy) return
    setCaptureBusy(true)
    try {
      await probesApi.stopCapture(d.channel_id)
      setSt({ channel_id: d.channel_id, runtime_state: 'paused' })
    } catch (error: any) {
      setSt((current) => ({
        ...current,
        capture_error: error?.message || 'Failed to stop capture',
      }))
    } finally {
      setCaptureBusy(false)
    }
  }

  async function applyCast() {
    if (bookmarkMutationBlocked) {
      setCastMsg({ text: 'bookmarks:create is required to cast this bookmarked probe.', err: true })
      return
    }
    const ids = [...castSel]
    if (!ids.length) { setCastMsg({ text: 'Select at least one channel.', err: true }); return }
    const base = buildInput()
    delete base.id
    setCastBusy(true); setCastMsg({ text: `Casting to ${ids.length} channel${ids.length === 1 ? '' : 's'}…` })
    try {
      const res = await probesApi.cast({
        ...base,
        enabled: castEnable,
        channel_ids: ids,
        conflict: castConflict,
        copy_roi: castCopyRoi,
        ...(castCopyRoi ? {} : { roi_enabled: false, roi_norm: null }),
      })
      const c = res.counts
      if (!c) throw new Error(res.error || 'Cast failed')
      setCastMsg({
        text: `Created ${c.created} · updated ${c.updated} · skipped ${c.skipped}${c.failed ? ` · failed ${c.failed}` : ''}`,
        err: c.failed > 0,
      })
      onCasted?.()
    } catch (e: any) { setCastMsg({ text: e?.message || 'Cast failed', err: true }) }
    finally { setCastBusy(false) }
  }

  const roiLabel = d.roiOn
    ? (d.roi ? `ROI ${pct(d.roi.w)} × ${pct(d.roi.h)} @ ${pct(d.roi.x)}, ${pct(d.roi.y)}` : 'ROI enabled — draw on the preview')
    : 'Full frame matching'
  const rangeDurationMs = probeRangeDurationMs(st)
  const rangeLabel = rangeDurationMs != null ? `${Math.round(rangeDurationMs / 1000)}s` : 'N/A'
  const lastLabel = st.last_timestamp_ms ? new Date(st.last_timestamp_ms).toLocaleTimeString() : 'n/a'
  // Only expose scores whose exact frame has finished loading. `st.live_signal`
  // may already point at the next sample while its JPEG is still in flight.
  const signal = displayedSignal
  const signalTime = signal?.timestamp_ms
    ? new Date(Number(signal.timestamp_ms)).toLocaleTimeString()
    : null
  const signalAgeSec = signal?.age_ms == null
    ? null
    : Math.max(0, Number(signal.age_ms) / 1000)

  async function inspectPresenceClass(item: SemanticPresenceClass) {
    const channelId = Number(d.channel_id)
    const timestampMs = Number(signal?.timestamp_ms)
    const classKey = String(item.key || item.label || '').trim()
    if (!Number.isFinite(channelId) || channelId <= 0 || !Number.isFinite(timestampMs) || timestampMs <= 0) {
      setPatchError('Wait for an exact scored frame before inspecting patches.')
      return
    }
    if (!classKey || patchBusyKey) return
    setPatchBusyKey(classKey)
    setPatchError(null)
    try {
      const result = await probesApi.patchAttention(channelId, timestampMs, classKey)
      setPatchAttention(result)
    } catch (error: any) {
      setPatchError(error?.message || 'Patch inspection failed.')
    } finally {
      setPatchBusyKey(null)
    }
  }

  async function applyProbe() {
    setApplyMessage(null)
    const saved = await onSave(buildInput())
    if (!saved) return
    setD((current) => ({ ...current, id: saved.id }))
    if (canControlCapture && saved.enabled !== false && saved.channel_id != null) {
      await startStream(saved.channel_id)
    }
    setApplyMessage('Applied. Live P/N/M now follows the current stream.')
  }

  const captureRunning = st.runtime_state === 'running'

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal probe-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">{d.id ? 'Semantic probe settings' : 'New semantic probe'}</div>
          <button className="modal-close" onClick={onClose}><IconX size={18} /></button>
        </div>

        <div className="modal-body probe-flow">
          {bookmarkMutationBlocked && (
            <div className="set-denied">
              <IconAlertTriangle size={15} />
              This probe creates bookmarks. It is read-only without bookmarks:create.
            </div>
          )}

          {/* Concept A — hero preview on the left, the essence (name/channel/what-to-detect) on the right */}
          <div className="probe-cols">
          <div className="probe-col-left">
          <div className="probe-col-scroll">
          <div className={`probe-preview ${pvErr ? 'err' : ''} ${d.roiOn ? 'roi-mode' : ''}`} ref={pvRef}
            onMouseDown={roiDown} onMouseMove={roiMove} onMouseUp={roiUp} onMouseLeave={roiUp}>
            {previewSrc && <img src={previewSrc} alt="stream preview" draggable={false} onError={() => setPvErr(true)} />}
            {pvErr && <div className="vid-overlay"><IconVideoOff size={18} /> PREVIEW UNAVAILABLE</div>}
            {!pvErr && <div className="probe-frame-stamp live">Live operator preview</div>}
            {d.roiOn && d.roi && (
              <div className="probe-roi-rect" style={{ left: pct(d.roi.x), top: pct(d.roi.y), width: pct(d.roi.w), height: pct(d.roi.h) }} />
            )}
          </div>

          {/* Keep the primary flow quiet: actions stay visible, diagnostics are available on demand. */}
          <div className="probe-roi-row">
            <button className={`mon-btn sm ${d.roiOn ? 'accent' : ''}`} onClick={() => set({ roiOn: !d.roiOn })}>
              <IconCrop size={14} /> ROI {d.roiOn ? 'ON' : 'OFF'}
            </button>
            <button className="mon-btn sm" disabled={!d.roi} onClick={() => set({ roi: null })}>Clear ROI</button>
            {canControlCapture && (
              <button
                className={`mon-btn sm ${captureRunning ? '' : 'accent'}`}
                disabled={captureBusy}
                onClick={() => { void (captureRunning ? stopStream() : startStream()) }}
              >
                {captureRunning
                  ? <><IconPlayerStop size={14} /> Stop stream</>
                  : <><IconPlayerPlay size={14} /> {captureBusy ? 'Starting…' : 'Start stream'}</>}
              </button>
            )}
          </div>

          {techOpen && (
            <div className="probe-panel">
              <div className="probe-tech-grid">
                <div><span>Stream</span><b className={pvErr ? 'is-bad' : 'is-good'}>{pvErr ? 'Failed' : 'Ready'}</b></div>
                <div><span>Capture</span><b>{(st.frames ?? 0) > 0 ? 'Active' : 'Warming'}</b></div>
                <div><span>Frames</span><b>{st.frames ?? 0}</b></div>
                <div><span>Range</span><b>{rangeLabel}</b></div>
                <div><span>Last snapshot</span><b>{lastLabel}</b></div>
                <div><span>Matching area</span><b>{roiLabel}</b></div>
              </div>
              <div className="probe-tech-note"><strong>Prompt pairs</strong> Positive describes what to spot; negative describes a similar scene to suppress.</div>
            </div>
          )}
          </div>{/* /probe-col-scroll */}
          <button className={`probe-tab ${techOpen ? 'on' : ''}`} onClick={() => setTechOpen((v) => !v)}>
            <IconChevronRight size={15} className="acc-chev" /> Technical details
          </button>
          </div>{/* /probe-col-left */}

          {/* the essence: identity + what to detect */}
          <div className="probe-col-right">
          <div className="probe-col-scroll">
          <div className="wfield"><label>Probe name</label>
            <input value={d.name} onChange={(e) => set({ name: e.target.value })} placeholder="Descriptive name" autoFocus />
          </div>
          <div className="wfield"><label>Channel</label>
            <Dropdown value={String(d.channel_id ?? '')} onChange={(v) => set({ channel_id: Number(v) })}
              options={channels.map((c) => ({ value: String(c.id), label: c.title }))} />
          </div>
          <div className="probe-detect-card">
          <div className="mon-heading">What to detect</div>
          <div className="mon-pairs">
            <div className="probe-pair-head"><span>Positive scene</span><span>Negative look-alike</span><span /></div>
            {d.pairs.map((pr, i) => (
              <div key={i} className="mon-pair">
                <input placeholder="e.g. person sitting on a chair" value={pr.pos} onChange={(e) => setPair(i, { pos: e.target.value })} />
                <input placeholder="e.g. empty room" value={pr.neg} onChange={(e) => setPair(i, { neg: e.target.value })} />
                <button className="mon-icobtn danger" title="Remove pair" disabled={d.pairs.length === 1}
                  onClick={() => set({ pairs: d.pairs.filter((_, j) => j !== i) })}><IconTrash size={14} /></button>
              </div>
            ))}
            <button className="mon-btn" onClick={() => set({ pairs: [...d.pairs, { pos: '', neg: '' }] })}><IconPlus size={14} /> Add pair</button>
          </div>
          </div>{/* /probe-detect-card */}

          <div className={`probe-live-signal ${signal?.stale ? 'stale' : (st.semantic_state || 'warming_up')}`}>
            <div className="probe-live-head">
              <div>
                <span>Live semantic signal</span>
                <b>{st.semantic_state === 'degraded'
                  ? 'Scorer unavailable'
                  : signal?.stale
                    ? `stale · ${signalTime || 'unknown sample time'}`
                  : signal
                    ? `${String(signal.threshold_state || 'sample').replace(/_/g, ' ')} · ${signalTime || 'now'}`
                    : d.id
                      ? 'Waiting for the next indexed frame'
                      : 'Apply the probe to start scoring'}</b>
              </div>
              <i>{st.embedding_backend || defaults.embedding_backend || 'embedding'}</i>
            </div>
            {st.semantic_error ? (
              <div className="probe-live-error"><IconAlertTriangle size={15} /> {st.semantic_error}</div>
            ) : (
              <div className="probe-signal-pair">
                <div className={`probe-scored-frame ${signal?.stale ? 'stale' : ''}`}>
                  {scoredFrameSrc ? (
                    <img src={scoredFrameSrc} alt="exact frame scored by semantic probe" />
                  ) : (
                    <div className="probe-scored-placeholder"><IconVideoOff size={16} /> Waiting for scored frame</div>
                  )}
                  <div className="probe-scored-stamp">
                    Scored frame · {signalTime || 'waiting'}
                    {signalAgeSec != null ? ` · ${signalAgeSec.toFixed(signalAgeSec < 10 ? 1 : 0)}s old` : ''}
                  </div>
                </div>
                <div className="probe-live-values">
                  <div><span>P</span><b>{signal?.pos_score == null ? '—' : Number(signal.pos_score).toFixed(3)}</b><em>floor {d.pos_floor.toFixed(3)}</em></div>
                  <div><span>N</span><b>{signal?.neg_score == null ? '—' : Number(signal.neg_score).toFixed(3)}</b><em>negative</em></div>
                  <div><span>M</span><b>{signal?.margin == null ? '—' : Number(signal.margin).toFixed(3)}</b><em>floor {d.margin.toFixed(3)}</em></div>
                </div>
              </div>
            )}
            {probe && !st.semantic_error && (
              <div className="probe-live-pulse">
                <ProbeSparkline probe={probe} history={st.signal_history} />
              </div>
            )}
            {!st.semantic_error && signalFrameError && (
              <div className="probe-live-warning"><IconAlertTriangle size={15} /> Waiting for the next complete scored frame. The last matched image and values are retained.</div>
            )}
            {!st.semantic_error && st.embedding_calibration_state && st.embedding_calibration_state !== 'calibrated' && (
              <div className="probe-live-error"><IconAlertTriangle size={15} /> Legacy thresholds are in shadow mode. Apply after reviewing the live SigLIP2 scores to reactivate alerts/bookmarks.</div>
            )}
            <div className="probe-live-note">P/N/M belongs only to the scored frame above. The large preview at left stays live for operator framing and ROI.</div>
          </div>

          <SemanticPresenceCard
            presence={st.semantic_presence}
            compact
            maxClasses={10}
            contextTexts={[d.name, ...d.pairs.map((pair) => pair.pos)]}
            onInspect={inspectPresenceClass}
            busyKey={patchBusyKey}
            activeKey={patchAttention?.class_key}
          />

          {(patchAttention || patchError) && (
            <section className="probe-patch-card" aria-label="Experimental patch affinity">
              <div className="probe-patch-head">
                <div>
                  <span>Experimental patch affinity</span>
                  <b>{patchAttention
                    ? `${patchAttention.label} · exact frame ${new Date(patchAttention.timestamp_ms).toLocaleTimeString()}`
                    : 'Inspection unavailable'}</b>
                </div>
                {patchAttention && <i>{patchAttention.grid.rows} × {patchAttention.grid.cols} · ephemeral</i>}
              </div>
              {patchError && <div className="probe-live-error"><IconAlertTriangle size={15} /> {patchError}</div>}
              {patchAttention && (
                <>
                  <div
                    className="probe-patch-frame"
                    style={{ aspectRatio: `${patchAttention.image?.width || 16} / ${patchAttention.image?.height || 9}` }}
                  >
                    <img src={patchAttention.frame_url} alt={`exact frame patch affinity for ${patchAttention.label}`} />
                    <div
                      className="probe-patch-grid"
                      style={{ gridTemplateColumns: `repeat(${patchAttention.grid.cols}, minmax(0, 1fr))` }}
                      aria-hidden="true"
                    >
                      {patchAttention.heatmap.map((value, index) => (
                        <i
                          key={index}
                          style={{ opacity: Math.max(0, Math.min(0.78, Number(value) * 0.78)) }}
                        />
                      ))}
                    </div>
                    {patchAttention.suggested_roi && (
                      <div
                        className="probe-patch-roi"
                        style={{
                          left: pct(patchAttention.suggested_roi.x),
                          top: pct(patchAttention.suggested_roi.y),
                          width: pct(patchAttention.suggested_roi.w),
                          height: pct(patchAttention.suggested_roi.h),
                        }}
                      />
                    )}
                  </div>
                  <div className="probe-patch-actions">
                    <span>
                      Relative contrast {Number(patchAttention.raw_range?.contrast || 0).toFixed(4)}.
                      This is a localization hint, not an object box.
                    </span>
                    <button
                      type="button"
                      className="mon-btn sm"
                      disabled={!patchAttention.suggested_roi}
                      onClick={() => {
                        if (patchAttention.suggested_roi) {
                          set({ roiOn: true, roi: patchAttention.suggested_roi })
                        }
                      }}
                    >
                      Use suggested ROI
                    </button>
                  </div>
                </>
              )}
            </section>
          )}

          {advOpen && (
            <div className="probe-panel">
              <div className="probe-sub">
                <div className="probe-block-head">Detection tuning</div>
                <div className="mon-help">
                  Active {defaults.embedding_backend || 'embedding'} defaults: P {defaults.pos_floor} · M {defaults.margin}
                  {defaults.embedding_model ? ` · ${defaults.embedding_model}` : ''}
                </div>
                <div className="wgrid">
                  <div className="wfield"><label>Positive floor</label>
                    <input type="number" step="0.01" value={d.pos_floor} onChange={(e) => set({ pos_floor: Number(e.target.value) })} />
                    <div className="mon-help">Minimum similarity for a hit (0–1). Higher = stricter.</div>
                  </div>
                  <div className="wfield"><label>Margin</label>
                    <input type="number" step="0.01" min="0" value={d.margin} onChange={(e) => set({ margin: Number(e.target.value) })} />
                    <div className="mon-help">How far positive must beat negative.</div>
                  </div>
                </div>
              </div>

              {canCreateBookmarks && (
                <div className="probe-sub">
                  <div className="probe-block-head">Bookmarks</div>
                  <label className="mon-check tile"><input type="checkbox" checked={d.bookmark} onChange={(e) => set({ bookmark: e.target.checked })} /> Make bookmarks in Luxriot on hits</label>
                  {d.bookmark && (
                    <>
                      <div className="wfield"><label>Severity</label>
                        <Dropdown value={d.severity} onChange={(v) => set({ severity: v })} options={SEVERITIES.map((s) => ({ value: s, label: s }))} />
                      </div>
                      <div className="wgrid">
                        <div className="wfield"><label>Cooldown (s)</label>
                          <input type="number" step="0.5" min="0" value={d.cooldown} onChange={(e) => set({ cooldown: Number(e.target.value) })} />
                        </div>
                        <div className="wfield"><label>Dedup window (s)</label>
                          <input type="number" step="0.5" min="0.5" value={d.dedupe} onChange={(e) => set({ dedupe: Number(e.target.value) })} />
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}

              <div className="probe-sub">
                <div className="probe-block-head">Image probe</div>
                <div className="mon-help">Optional reference image — frames similar to it also count as hits.</div>
                {!d.imgData ? (
                  <label className="mon-btn"><IconPhoto size={14} /> Choose image
                    <input type="file" accept="image/*" style={{ display: 'none' }}
                      onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.currentTarget.value = '' }} />
                  </label>
                ) : (
                  <div className="probe-img-box">
                    <img className="probe-img" src={`data:image/jpeg;base64,${d.imgData}`} alt="probe reference" />
                    <div className="probe-img-side">
                      <div className="probe-img-name" title={d.imgName}>{d.imgName || 'reference.jpg'}</div>
                      <label className="mon-check"><input type="checkbox" checked={d.imgEnabled} onChange={(e) => set({ imgEnabled: e.target.checked })} /> Enabled</label>
                      <div className="wfield"><label>Min match</label>
                        <input type="number" step="0.01" min="0" max="1" value={d.imgFloor} onChange={(e) => set({ imgFloor: Number(e.target.value) })} />
                      </div>
                      <div className="probe-img-actions">
                        <label className="mon-btn sm"><IconPhoto size={13} /> Replace
                          <input type="file" accept="image/*" style={{ display: 'none' }}
                            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.currentTarget.value = '' }} />
                        </label>
                        <button className="mon-btn sm danger" onClick={() => set({ imgData: null, imgName: '', imgEnabled: false })}><IconTrash size={13} /> Clear</button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
          </div>{/* /probe-col-scroll */}
          <button className={`probe-tab ${advOpen ? 'on' : ''}`} onClick={() => setAdvOpen((v) => !v)}>
            <IconChevronRight size={15} className="acc-chev" /> Advanced settings
            <span className="acc-sub">floor {d.pos_floor}{canCreateBookmarks && d.bookmark ? ' · bookmarks' : ''}{d.imgData ? ' · image' : ''}</span>
          </button>
          </div>{/* /probe-col-right */}
          </div>{/* /probe-cols */}
        </div>

        <div className="probe-footer">
          <label className="mon-check tile"><input type="checkbox" checked={d.enabled} onChange={(e) => set({ enabled: e.target.checked })} /> Probe enabled</label>
          <div className="probe-footer-actions">
            <button className="mon-btn" onClick={onClose}>Close</button>
            <button className="mon-btn" disabled={bookmarkMutationBlocked} title={bookmarkMutationBlocked ? 'Requires bookmarks:create' : undefined} onClick={() => {
              setCastSel(new Set(d.channel_id != null ? [d.channel_id] : []))
              setCastCopyRoi(false); setCastMsg(null); setCastOpen(true)
            }}><IconBroadcast size={15} /> Cast</button>
            <button
              className="mon-btn accent"
              disabled={busy || bookmarkMutationBlocked}
              title={bookmarkMutationBlocked ? 'Requires bookmarks:create' : undefined}
              onClick={applyProbe}
            >
              <IconDeviceFloppy size={15} /> {busy ? 'Applying…' : 'Apply probe'}
            </button>
          </div>
        </div>
        {applyMessage && <div className="probe-apply-status">{applyMessage}</div>}

        {/* cast panel — copy this probe onto many channels */}
        {castOpen && (
          <div className="scrim" onClick={() => setCastOpen(false)}>
            <div className="modal probe-cast" onClick={(e) => e.stopPropagation()}>
              <div className="modal-head">
                <div className="modal-title">Cast probe to channels</div>
                <button className="modal-close" onClick={() => setCastOpen(false)}><IconX size={16} /></button>
              </div>
              <div className="modal-body probe-cast-body">
                <div className="probe-cast-tools">
                  <button className="mon-btn sm" onClick={() => setCastSel(new Set(channels.map((c) => c.id)))}>All</button>
                  <button className="mon-btn sm" onClick={() => setCastSel(new Set())}>None</button>
                  <button className="mon-btn sm" disabled={d.channel_id == null}
                    onClick={() => setCastSel(new Set(d.channel_id != null ? [d.channel_id] : []))}>Current</button>
                  <span className="mon-help" style={{ marginLeft: 'auto', marginTop: 0 }}>{castSel.size} selected</span>
                </div>
                <div className="probe-cast-list">
                  {channels.map((c) => (
                    <label key={c.id} className="probe-cast-ch">
                      <input type="checkbox" checked={castSel.has(c.id)}
                        onChange={(e) => setCastSel((s) => { const n = new Set(s); if (e.target.checked) n.add(c.id); else n.delete(c.id); return n })} />
                      <span>{c.title}</span>
                    </label>
                  ))}
                </div>
                <div className="wgrid">
                  <div className="wfield"><label>If probe already exists</label>
                    <Dropdown value={castConflict} onChange={(v) => setCastConflict(v as any)}
                      options={[
                        { value: 'skip', label: 'Skip channel' },
                        { value: 'create', label: 'Create a copy' },
                        ...(canCreateBookmarks ? [{ value: 'update', label: 'Update existing' }] : []),
                      ]} />
                  </div>
                  <div className="probe-cast-flags">
                    <label className="mon-check"><input type="checkbox" checked={castEnable} onChange={(e) => setCastEnable(e.target.checked)} /> Enable probes</label>
                    <label className="mon-check"><input type="checkbox" checked={castCopyRoi} disabled={!(d.roiOn && d.roi)}
                      onChange={(e) => setCastCopyRoi(e.target.checked)} /> Copy ROI</label>
                  </div>
                </div>
                {castMsg && <div className={`set-status ${castMsg.err ? 'err' : 'ok'}`}>{castMsg.text}</div>}
              </div>
              <div className="probe-footer">
                <span />
                <div className="probe-footer-actions">
                  <button className="mon-btn" onClick={() => setCastOpen(false)}>Close</button>
                  <button className="mon-btn accent" disabled={castBusy || castSel.size === 0} onClick={applyCast}>
                    <IconBroadcast size={15} /> {castBusy ? 'Casting…' : `Cast to ${castSel.size} channel${castSel.size === 1 ? '' : 's'}`}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
