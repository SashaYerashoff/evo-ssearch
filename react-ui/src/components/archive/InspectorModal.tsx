import { useEffect, useMemo, useState } from 'react'
import {
  IconAlertTriangle,
  IconArrowsMaximize,
  IconChevronLeft,
  IconChevronRight,
  IconCopy,
  IconDownload,
  IconFileDescription,
  IconFlag,
  IconMessage,
  IconPhoto,
  IconPlayerPlay,
  IconPlayerStop,
  IconX,
} from '@tabler/icons-react'
import type { Channel, Detection } from '../../api/types'
import {
  archivePlaybackUrl,
  batchFrameNumber,
  describeFrame,
  detImageSrc,
  falsePositiveExportUrl,
  fullDetectionImageSrc,
  getAlertFeedback,
  loadDetectionBatchFrames,
  saveAlertFeedback,
  type AlertFeedbackReason,
} from '../../api/detections'
import type { IncidentDraftInput } from '../../api/incidents'
import { IncidentModal } from '../incidents/IncidentModal'

const FALLBACK_REASONS: AlertFeedbackReason[] = [
  { code: 'no_relevant_event', label: 'No relevant event' },
  { code: 'benign_activity', label: 'Benign activity' },
  { code: 'wrong_object_or_actor', label: 'Wrong object or actor' },
  { code: 'duplicate_or_stale', label: 'Duplicate or stale alert' },
  { code: 'poor_visual_quality', label: 'Poor visual quality' },
]

function fmtFull(ms: number | null): string {
  if (!ms) return '—'
  return new Date(ms).toLocaleString([], {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function summaryText(detection: Detection): string {
  const payload = detection.raw?.payload || {}
  const value = payload.summary
    ?? detection.raw?.summary
    ?? payload.description
    ?? detection.raw?.description
    ?? ''
  return typeof value === 'string' ? value.trim() : ''
}

function frameLabel(detection: Detection, index: number): string {
  const frame = batchFrameNumber(detection)
  return frame === Number.MAX_SAFE_INTEGER ? `Frame ${index + 1}` : `Frame ${frame}`
}

function sameDetection(left: Detection, right: Detection): boolean {
  if (left.id != null && right.id != null && String(left.id) === String(right.id)) return true
  const leftFrame = batchFrameNumber(left)
  const rightFrame = batchFrameNumber(right)
  return leftFrame !== Number.MAX_SAFE_INTEGER
    && leftFrame === rightFrame
    && left.channelId === right.channelId
}

function frameMarkers(frame: Detection, base: Detection): string[] {
  const payload = frame.raw?.payload || {}
  const markers: string[] = []
  if (payload.is_cover) markers.push('COVER')
  if (frame.source === 'vlm_alert') markers.push('ALERT')
  if (sameDetection(frame, base)) markers.push('MATCH')
  return markers
}

export function InspectorModal({
  d,
  channels,
  canReportFeedback,
  canReportIncidents,
  canExport,
  onClose,
  onFindSimilar,
}: {
  d: Detection
  channels: Channel[]
  canReportFeedback: boolean
  canReportIncidents: boolean
  canExport: boolean
  onClose: () => void
  onFindSimilar: (d: Detection) => void
}) {
  const [frames, setFrames] = useState<Detection[]>([d])
  const [activeIndex, setActiveIndex] = useState(0)
  const [framesLoading, setFramesLoading] = useState(false)
  const [framesError, setFramesError] = useState('')
  const [desc, setDesc] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [zoom, setZoom] = useState(false)
  const [feedbackOpen, setFeedbackOpen] = useState(false)
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const [feedbackSaving, setFeedbackSaving] = useState(false)
  const [feedbackStatus, setFeedbackStatus] = useState('')
  const [feedbackReasons, setFeedbackReasons] = useState(FALLBACK_REASONS)
  const [feedbackReason, setFeedbackReason] = useState('')
  const [feedbackNote, setFeedbackNote] = useState('')
  const [hasFeedback, setHasFeedback] = useState(false)
  const [incidentDraft, setIncidentDraft] = useState<IncidentDraftInput | null>(null)
  const [playback, setPlayback] = useState<{
    state: 'idle' | 'loading' | 'decoding' | 'playing' | 'error'
    url?: string
    detail?: string
  }>({ state: 'idle' })
  const [playbackElapsed, setPlaybackElapsed] = useState(0)

  const active = frames[activeIndex] || d
  const previewSrc = detImageSrc(active)
  const fullSrc = fullDetectionImageSrc(active)
  const [src, setSrc] = useState(fullSrc)
  const summary = summaryText(d)
  const alertDetectionId = d.source === 'vlm_alert'
    && Number.isInteger(Number(d.id))
    && Number(d.id) > 0
    ? Number(d.id)
    : null
  const incidentAnchorId = Number.isInteger(Number(active.id)) && Number(active.id) > 0
    ? Number(active.id)
    : null
  const incidentChannelId = Number.isInteger(Number(active.channelId)) && Number(active.channelId) > 0
    ? Number(active.channelId)
    : null
  const scores = [
    active.posScore != null && `P ${active.posScore.toFixed(2)}`,
    active.negScore != null && `N ${active.negScore.toFixed(2)}`,
    active.margin != null && `M ${active.margin.toFixed(2)}`,
  ].filter(Boolean).join(' · ')

  useEffect(() => {
    setSrc(fullSrc)
    setDesc(null)
    setPlayback({ state: 'idle' })
  }, [active.key, fullSrc])

  useEffect(() => {
    if (!['loading', 'decoding'].includes(playback.state)) {
      setPlaybackElapsed(0)
      return
    }
    const startedAt = Date.now()
    setPlaybackElapsed(0)
    const timer = window.setInterval(() => {
      setPlaybackElapsed(Math.max(0, Math.floor((Date.now() - startedAt) / 1_000)))
    }, 1_000)
    return () => window.clearInterval(timer)
  }, [playback.state])

  useEffect(() => {
    if (playback.state !== 'loading') return
    const timer = window.setTimeout(() => {
      setPlayback({
        state: 'error',
        detail: 'The recorder did not begin the archive response within 120 seconds.',
      })
    }, 120_000)
    return () => window.clearTimeout(timer)
  }, [playback.state])

  useEffect(() => {
    if (playback.state !== 'decoding') return
    const timer = window.setTimeout(() => {
      setPlayback({
        state: 'error',
        detail: 'The recorder segment arrived, but the browser did not make it playable within 12 seconds.',
      })
    }, 12_000)
    return () => window.clearTimeout(timer)
  }, [playback.state])

  useEffect(() => {
    let alive = true
    setFrames([d])
    setActiveIndex(0)
    setFramesError('')
    if (!['vlm_summary', 'vlm_alert'].includes(d.source)) return () => { alive = false }
    setFramesLoading(true)
    loadDetectionBatchFrames(d, channels)
      .then((loaded) => {
        if (!alive) return
        setFrames(loaded.length ? loaded : [d])
        const selectedIndex = loaded.findIndex((frame) => sameDetection(frame, d))
        setActiveIndex(selectedIndex >= 0 ? selectedIndex : 0)
      })
      .catch((exception: any) => {
        if (!alive) return
        setFrames([d])
        setActiveIndex(0)
        setFramesError(exception?.message || 'Neighboring batch frames are unavailable.')
      })
      .finally(() => {
        if (alive) setFramesLoading(false)
      })
    return () => { alive = false }
  }, [d, channels])

  useEffect(() => {
    let alive = true
    setFeedbackOpen(false)
    setFeedbackStatus('')
    setFeedbackReason('')
    setFeedbackNote('')
    setHasFeedback(false)
    setFeedbackReasons(FALLBACK_REASONS)
    if (!alertDetectionId) return () => { alive = false }
    setFeedbackLoading(true)
    getAlertFeedback(alertDetectionId)
      .then((response) => {
        if (!alive) return
        if (response.reason_options?.length) setFeedbackReasons(response.reason_options)
        if (response.feedback) {
          setFeedbackReason(String(response.feedback.reason_code || ''))
          setFeedbackNote(String(response.feedback.note || ''))
          setHasFeedback(true)
          setFeedbackStatus('Saved operator annotation')
        }
      })
      .catch((exception: any) => {
        if (alive) setFeedbackStatus(exception?.message || 'Feedback is unavailable.')
      })
      .finally(() => {
        if (alive) setFeedbackLoading(false)
      })
    return () => { alive = false }
  }, [alertDetectionId])

  const activeMarkers = useMemo(() => frameMarkers(active, d), [active, d])

  async function describe() {
    setBusy(true)
    setDesc(null)
    try { setDesc(await describeFrame(active)) }
    catch (exception: any) { setDesc(`Error: ${exception?.message || 'describe failed'}`) }
    finally { setBusy(false) }
  }

  async function saveFeedback() {
    if (!alertDetectionId) return
    if (!feedbackReason) {
      setFeedbackStatus('Choose one reason before saving.')
      return
    }
    setFeedbackSaving(true)
    setFeedbackStatus('Saving…')
    try {
      const response = await saveAlertFeedback(alertDetectionId, feedbackReason, feedbackNote.trim())
      if (response.reason_options?.length) setFeedbackReasons(response.reason_options)
      setHasFeedback(true)
      setFeedbackStatus('Saved. Agent reports and L3 can use this annotation.')
    } catch (exception: any) {
      setFeedbackStatus(exception?.message || 'Feedback save failed.')
    } finally {
      setFeedbackSaving(false)
    }
  }

  function stopPlayback(detail?: string) {
    setPlayback(detail ? { state: 'error', detail } : { state: 'idle' })
  }

  function playArchive() {
    if (playback.state === 'loading') return
    const url = archivePlaybackUrl(active)
    if (!url) {
      setPlayback({ state: 'error', detail: 'This evidence has no recorder timestamp.' })
      return
    }
    setPlayback({
      state: 'loading',
      url,
      detail: 'Preparing recorder archive around the evidence frame…',
    })
  }

  const canPlayArchive = ['vlm_alert', 'vlm_summary'].includes(active.source)
    && active.channelId != null
    && active.tsMs != null

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal inspect-modal archive-review-modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">Archive research review</div>
            <div className="archive-review-subtitle">
              {d.probeName} · {d.sourceLabel}
              {activeMarkers.length ? ` · ${activeMarkers.join(' · ')}` : ''}
            </div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close review"><IconX size={18} /></button>
        </div>

        <div className="modal-body archive-review-body">
          <div className="archive-review-evidence">
            <div className="inspect-frame archive-review-frame">
              {['loading', 'decoding', 'playing'].includes(playback.state) && playback.url ? (
                <video
                  src={playback.url}
                  autoPlay
                  muted
                  loop
                  controls
                  playsInline
                  preload="auto"
                  onLoadStart={() => setPlayback((current) => (
                    current.url === playback.url
                      ? { ...current, state: 'loading', detail: 'Preparing recorder archive around the evidence frame…' }
                      : current
                  ))}
                  onLoadedMetadata={() => setPlayback((current) => (
                    current.url === playback.url
                      ? { ...current, state: 'decoding', detail: 'Archive metadata loaded; buffering video…' }
                      : current
                  ))}
                  onCanPlay={() => setPlayback((current) => (
                    current.url === playback.url
                      ? { ...current, state: 'playing', detail: 'Archive playback ready' }
                      : current
                  ))}
                  onPlaying={() => setPlayback((current) => (
                    current.url === playback.url
                      ? { ...current, state: 'playing', detail: 'Archive playback' }
                      : current
                  ))}
                  onError={() => stopPlayback('The browser could not load or decode the recorder archive segment.')}
                />
              ) : src ? (
                <>
                  <img
                    src={src}
                    alt={frameLabel(active, activeIndex)}
                    onClick={() => setZoom(true)}
                    onError={() => {
                      if (previewSrc && src !== previewSrc) setSrc(previewSrc)
                      else setSrc('')
                    }}
                  />
                  <button className="inspect-expand" title="View full frame" onClick={() => setZoom(true)}>
                    <IconArrowsMaximize size={15} />
                  </button>
                </>
              ) : (
                <div className="inspect-noimg"><IconPhoto size={30} /> No frame</div>
              )}
              {playback.state !== 'idle' && playback.detail && (
                <div className={`archive-playback-status ${playback.state}`}>
                  {playback.detail}
                  {['loading', 'decoding'].includes(playback.state) ? ` · ${playbackElapsed}s` : ''}
                </div>
              )}
            </div>

            <div className="archive-review-frame-nav">
              <button
                className="btn icon"
                disabled={activeIndex <= 0}
                onClick={() => setActiveIndex((index) => Math.max(0, index - 1))}
                aria-label="Previous batch frame"
              >
                <IconChevronLeft size={16} />
              </button>
              <span>{frameLabel(active, activeIndex)} of {frames.length} · {fmtFull(active.tsMs)}</span>
              <button
                className="btn icon"
                disabled={activeIndex >= frames.length - 1}
                onClick={() => setActiveIndex((index) => Math.min(frames.length - 1, index + 1))}
                aria-label="Next batch frame"
              >
                <IconChevronRight size={16} />
              </button>
            </div>

            <div className="archive-review-filmstrip" aria-label="Batch frames">
              {framesLoading && <div className="archive-review-filmstrip-status">Loading batch frames…</div>}
              {!framesLoading && frames.map((frame, index) => {
                const image = detImageSrc(frame)
                const markers = frameMarkers(frame, d)
                return (
                  <button
                    key={frame.key}
                    className={`archive-review-strip-frame ${index === activeIndex ? 'active' : ''}`}
                    onClick={() => setActiveIndex(index)}
                    title={`${frameLabel(frame, index)}${markers.length ? ` · ${markers.join(' · ')}` : ''}`}
                  >
                    {image ? <img src={image} alt={frameLabel(frame, index)} loading="lazy" /> : <IconPhoto size={20} />}
                    {markers.length > 0 && <i>{markers.join(' · ')}</i>}
                    <span>{frameLabel(frame, index)}</span>
                  </button>
                )
              })}
              {!framesLoading && frames.length <= 1 && !framesError && (
                <div className="archive-review-filmstrip-status">No neighboring batch frames returned.</div>
              )}
            </div>
            {framesError && (
              <div className="archive-review-warning"><IconAlertTriangle size={14} /> {framesError}</div>
            )}
          </div>

          <aside className="inspect-side archive-review-side">
            <div className="kv">
              <span className="k">Channel</span><span className="v">{active.channelTitle || `ch ${active.channelId ?? '—'}`}</span>
              <span className="k">Source</span><span className="v">{active.sourceLabel}</span>
              <span className="k">Time</span><span className="v">{fmtFull(active.tsMs)}</span>
              <span className="k">Severity</span><span className="v archive-review-severity">{active.severity}</span>
              {(active.matchPct != null || scores) && (
                <>
                  <span className="k">Match</span>
                  <span className="v">{active.matchPct != null ? `${active.matchPct}%` : '—'}{scores ? ` · ${scores}` : ''}</span>
                </>
              )}
            </div>

            {summary && (
              <section className="archive-review-summary">
                <h3>Video description for the L0 batch</h3>
                <div>{summary}</div>
              </section>
            )}
            {(busy || desc) && <div className="desc-box">{busy ? 'Generating description…' : desc}</div>}

            <div className="modal-actions">
              {canPlayArchive && (
                ['loading', 'decoding', 'playing'].includes(playback.state)
                  ? (
                    <button className="btn" onClick={() => stopPlayback()}>
                      <IconPlayerStop size={15} /> {playback.state === 'playing' ? 'Stop playback' : 'Cancel archive'}
                    </button>
                  )
                  : (
                    <button className="btn primary" onClick={playArchive}>
                      <IconPlayerPlay size={15} /> {playback.state === 'error' ? 'Retry archive video' : 'Play archive video'}
                    </button>
                  )
              )}
              <button className="btn" onClick={describe} disabled={busy || !detImageSrc(active)}>
                <IconMessage size={15} /> Describe frame
              </button>
              <button className="btn" onClick={() => onFindSimilar(active)} disabled={!detImageSrc(active)}>
                <IconCopy size={15} /> Find similar
              </button>
              {canReportIncidents && incidentAnchorId && incidentChannelId && (
                <button
                  className="btn incident-report-open"
                  onClick={() => setIncidentDraft({
                    channel_id: incidentChannelId,
                    anchor_detection_id: incidentAnchorId,
                  })}
                >
                  <IconFileDescription size={15} /> Report incident
                </button>
              )}
              {alertDetectionId && canReportFeedback && (
                <button className="btn archive-feedback-open" onClick={() => setFeedbackOpen((open) => !open)}>
                  <IconFlag size={15} /> {hasFeedback ? 'Edit false-positive report' : 'Report false positive'}
                </button>
              )}
            </div>

            {alertDetectionId && feedbackOpen && (
              <section className="archive-feedback-panel">
                <h3>Operator feedback</h3>
                <p>What made this VLM alert a false positive?</p>
                <div className="archive-feedback-reasons">
                  {feedbackReasons.map((reason) => (
                    <label key={reason.code}>
                      <input
                        type="radio"
                        name={`archive-feedback-${alertDetectionId}`}
                        value={reason.code}
                        checked={feedbackReason === reason.code}
                        onChange={() => setFeedbackReason(reason.code)}
                      />
                      <span>{reason.label}</span>
                    </label>
                  ))}
                </div>
                <label className="archive-feedback-note">
                  Note for agent and L3
                  <textarea
                    value={feedbackNote}
                    onChange={(event) => setFeedbackNote(event.target.value)}
                    placeholder="Optional context…"
                    rows={3}
                  />
                </label>
                <div className="archive-feedback-actions">
                  <button className="btn primary" disabled={feedbackSaving || feedbackLoading} onClick={saveFeedback}>
                    {feedbackSaving ? 'Saving…' : hasFeedback ? 'Update feedback' : 'Save feedback'}
                  </button>
                  {canExport && (
                    <>
                      <a className="btn" href={falsePositiveExportUrl('md', d.channelId)} download>
                        <IconDownload size={14} /> MD
                      </a>
                      <a className="btn" href={falsePositiveExportUrl('xml', d.channelId)} download>
                        <IconDownload size={14} /> XML
                      </a>
                    </>
                  )}
                </div>
                {feedbackStatus && <div className="archive-feedback-status">{feedbackStatus}</div>}
              </section>
            )}
          </aside>
        </div>
      </div>

      {zoom && src && (
        <div className="inspect-zoom" onClick={(event) => { event.stopPropagation(); setZoom(false) }}>
          <img src={src} alt={frameLabel(active, activeIndex)} onClick={(event) => event.stopPropagation()} />
          <button
            className="modal-close inspect-zoom-close"
            onClick={(event) => { event.stopPropagation(); setZoom(false) }}
            aria-label="Close full frame"
          >
            <IconX size={22} />
          </button>
        </div>
      )}
      {incidentDraft && (
        <IncidentModal
          draftInput={incidentDraft}
          canExport={canExport}
          onClose={() => setIncidentDraft(null)}
        />
      )}
    </div>
  )
}
