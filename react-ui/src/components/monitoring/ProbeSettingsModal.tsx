import { useEffect, useRef, useState } from 'react'
import {
  IconX, IconPlus, IconTrash, IconPhoto, IconDeviceFloppy, IconVideoOff,
  IconChevronRight, IconCrop, IconPlayerStop, IconBroadcast, IconAlertTriangle,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import {
  authorizeProbeInput,
  probeMutationRequiresBookmarkPermission,
  probeRangeDurationMs,
  probesApi,
  type Probe,
  type ProbeInput,
  type RoiNorm,
} from '../../api/probes'
import { recentFrameUrl } from '../../api/video'
import { Dropdown } from '../shell/Dropdown'

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

function fromProbe(p: Probe | null, channels: Channel[]): Draft {
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
    pos_floor: p?.pos_floor ?? 0.2,
    margin: p?.margin ?? 0.05,
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

export function ProbeSettingsModal({ probe, channels, busy, canControlCapture, canCreateBookmarks, onClose, onSave, onCasted }: {
  probe: Probe | null
  channels: Channel[]
  busy: boolean
  canControlCapture: boolean
  canCreateBookmarks: boolean
  onClose: () => void
  onSave: (input: ProbeInput) => void
  onCasted?: () => void
}) {
  const [d, setD] = useState<Draft>(() => fromProbe(probe, channels))
  const bookmarkMutationBlocked = probeMutationRequiresBookmarkPermission(probe, canCreateBookmarks)
  const set = (p: Partial<Draft>) => setD((x) => ({ ...x, ...p }))
  const setPair = (i: number, p: Partial<{ pos: string; neg: string }>) =>
    setD((x) => ({ ...x, pairs: x.pairs.map((pr, j) => (j === i ? { ...pr, ...p } : pr)) }))

  // pre-open a collapsed section when it already holds non-default settings (editing)
  const [openTuning] = useState(() => (probe?.pos_floor ?? 0.2) !== 0.2 || (probe?.margin ?? 0.05) !== 0.05)
  const [openBookmarks] = useState(() => !!probe?.bookmark)
  const [openImage] = useState(() => !!probe?.image_probe?.data)

  // live preview + capture status of the selected channel
  const [bust, setBust] = useState(1)
  const [pvErr, setPvErr] = useState(true)
  const [st, setSt] = useState<{
    frames?: number
    time_range_ms?: number | [number, number] | null
    first_timestamp_ms?: number | null
    last_timestamp_ms?: number | null
  }>({})
  useEffect(() => {
    setBust((b) => b + 1); setPvErr(true); setSt({})
    const poll = () => { if (d.channel_id != null) probesApi.status(d.channel_id).then(setSt).catch(() => {}) }
    poll()
    const t = window.setInterval(() => { setBust((b) => b + 1); poll() }, 4000)
    return () => window.clearInterval(t)
  }, [d.channel_id])
  const previewSrc = d.channel_id != null ? recentFrameUrl(d.channel_id, bust) : ''

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

  async function stopStream() {
    if (d.channel_id == null) return
    try { await probesApi.stopCapture(d.channel_id); setSt({}) } catch { /* ignore */ }
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

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal probe-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">{d.id ? 'CLIP probe settings' : 'New CLIP probe'}</div>
          <button className="modal-close" onClick={onClose}><IconX size={18} /></button>
        </div>

        <div className="modal-body probe-flow">
          {bookmarkMutationBlocked && (
            <div className="set-denied">
              <IconAlertTriangle size={15} />
              This probe creates bookmarks. It is read-only without bookmarks:create.
            </div>
          )}

          {/* 1 · where the probe watches */}
          <div className="wgrid">
            <div className="wfield"><label>Probe name</label>
              <input value={d.name} onChange={(e) => set({ name: e.target.value })} placeholder="Descriptive name" autoFocus />
            </div>
            <div className="wfield"><label>Channel</label>
              <Dropdown value={String(d.channel_id ?? '')} onChange={(v) => set({ channel_id: Number(v) })}
                options={channels.map((c) => ({ value: String(c.id), label: c.title }))} />
            </div>
          </div>

          <div className={`probe-preview ${pvErr ? 'err' : ''} ${d.roiOn ? 'roi-mode' : ''}`} ref={pvRef}
            onMouseDown={roiDown} onMouseMove={roiMove} onMouseUp={roiUp} onMouseLeave={roiUp}>
            {previewSrc && <img className={pvErr ? 'preview-pending' : undefined} src={previewSrc} alt="stream preview" draggable={false}
              onLoad={() => setPvErr(false)} onError={() => setPvErr(true)} />}
            {pvErr && <div className="vid-overlay"><IconVideoOff size={18} /> PREVIEW UNAVAILABLE</div>}
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
            {canControlCapture && <button className="mon-btn sm" onClick={stopStream}><IconPlayerStop size={14} /> Stop stream</button>}
          </div>

          <details className="probe-acc probe-tech">
            <summary>
              <IconChevronRight size={15} className="acc-chev" /> Technical details
            </summary>
            <div className="acc-body">
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
          </details>

          {/* 2 · what to detect — the essence of the probe */}
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

          {/* 3 · optional settings, collapsed with live summaries */}
          <details className="probe-acc" open={openTuning || undefined}>
            <summary>
              <IconChevronRight size={15} className="acc-chev" /> Detection tuning
              <span className="acc-sub">floor {d.pos_floor} · margin {d.margin}</span>
            </summary>
            <div className="acc-body">
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
          </details>

          {canCreateBookmarks && <details className="probe-acc" open={openBookmarks || undefined}>
            <summary>
              <IconChevronRight size={15} className="acc-chev" /> Bookmarks
              <span className="acc-sub">{d.bookmark ? `on · ${d.severity} · cooldown ${d.cooldown}s` : 'off'}</span>
            </summary>
            <div className="acc-body">
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
          </details>}

          <details className="probe-acc" open={openImage || undefined}>
            <summary>
              <IconChevronRight size={15} className="acc-chev" /> Image probe
              <span className="acc-sub">{d.imgData ? (d.imgEnabled ? `${d.imgName || 'image'} · min ${d.imgFloor}` : 'set · disabled') : 'none'}</span>
            </summary>
            <div className="acc-body">
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
          </details>
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
              onClick={() => onSave(buildInput())}
            >
              <IconDeviceFloppy size={15} /> {busy ? 'Saving…' : 'Save probe'}
            </button>
          </div>
        </div>

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
