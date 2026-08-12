import {
  IconExternalLink,
  IconPhoto,
  IconPlayerPlay,
  IconPlayerStop,
  IconSettings,
  IconTrash,
} from '@tabler/icons-react'
import type { ChannelStatus, Probe } from '../../api/probes'
import { hitImageSrc } from '../../api/probes'
import {
  gateText,
  lastHit,
  ProbeOriginBadge,
  ProbeSparkline,
  PROBE_ORIGIN_LABELS,
  type ProbeStatus,
} from './ProbeCard'
import { probeOrigin, probeTemporaryTtl } from './probeBoard'
import { SemanticPresenceCard } from './SemanticPresenceCard'

const n3 = (v?: number | null) => (v == null ? '—' : Number(v).toFixed(3))
function fmtDateTime(ms?: number | null): string {
  if (!ms) return 'never'
  return new Date(Number(ms)).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' })
}

export function ProbeInspector({ probe, status, runtime, busy, settingsBlockedReason, onSettings, onRun, onDelete, onOpenParentAlert }: {
  probe: Probe
  status: ProbeStatus
  runtime?: ChannelStatus | null
  busy: boolean
  settingsBlockedReason?: string
  onSettings?: () => void
  onRun?: () => void
  onDelete?: () => void
  onOpenParentAlert?: (probe: Probe) => void
}) {
  const hit = lastHit(probe)
  const signal = runtime?.live_signal || hit
  const src = hitImageSrc(hit)
  const origin = probeOrigin(probe)
  const originView = PROBE_ORIGIN_LABELS[origin]
  const ttl = probeTemporaryTtl(probe)
  return (
    <div className="mon-panel probe-inspector-panel">
      <div className="mon-panel-title">Selected semantic probe</div>

      {/* who: one clean identity row */}
      <div className="pi-head">
        <div className="pi-thumb">{src ? <img src={src} alt={probe.name || 'probe'} /> : <IconPhoto size={18} />}</div>
        <div className="pi-id">
          <div className="pi-name">{probe.name || 'Untitled probe'}</div>
          <div className="pi-line">Ch {probe.channel_id ?? '—'} · Last event {fmtDateTime(hit?.timestamp_ms ?? hit?.recorded_at_ms)}</div>
        </div>
        <div className="pi-head-tags">
          <span className={`pc-badge ${status}`}>{status.toUpperCase()}</span>
          <ProbeOriginBadge probe={probe} />
          {ttl && <span className={`probe-ttl ${ttl.expired ? 'expired' : ''}`} title={ttl.title}>{ttl.text}</span>}
        </div>
      </div>

      {/* what matters: the last signal, front and centre */}
      <div className="pi-sec">Last signal</div>
      <ProbeSparkline probe={probe} history={runtime?.signal_history} />
      {runtime?.semantic_error && (
        <div className="probe-live-error pi-live-error">
          Embedding/scoring unavailable: {runtime.semantic_error}
        </div>
      )}
      {!runtime?.semantic_error && probe.embedding_calibration_state && probe.embedding_calibration_state !== 'calibrated' && (
        <div className="probe-live-error pi-live-error">
          This probe was created in another embedding space. Review its live P/N/M and Apply it before allowing alerts or bookmarks.
        </div>
      )}
      <div className="pi-scores">
        <div><span>Positive</span><b className="pos">{n3(signal?.pos_score)}</b></div>
        <div><span>Negative</span><b>{n3(signal?.neg_score)}</b></div>
        <div><span>Margin</span><b className="mar">{n3(signal?.margin)}</b></div>
      </div>
      <div className="pi-signal-meta">
        {signal
          ? `${runtime?.live_signal ? 'Live pre-threshold sample' : 'Last threshold hit'} · ${String(runtime?.live_signal?.threshold_state || 'hit').replace(/_/g, ' ')}`
          : runtime?.semantic_state === 'warming_up'
            ? 'Waiting for indexed frames.'
            : 'No score has been computed yet.'}
      </div>
      <SemanticPresenceCard
        presence={runtime?.semantic_presence}
        compact
        contextTexts={[probe.name || '', ...(probe.positives || [])]}
      />

      {/* how it's set up: one quiet scannable list */}
      <div className="pi-sec">Configuration</div>
      <div className="pi-config">
        <div className="pi-row"><span>Created by</span><b>{originView.label}</b></div>
        {origin === 'agent' && probe.origin_meta?.plan_id && (
          <div className="pi-row"><span>Approval plan</span><b>{probe.origin_meta.plan_id}</b></div>
        )}
        {origin === 'auto' && (
          <>
            <div className="pi-row">
              <span>Parent alert</span>
              <b title={probe.parent_alert_description}>{probe.parent_alert_title || probe.parent_alert_id || 'unknown'}</b>
            </div>
            {probe.parent_alert_timestamp_ms && (
              <div className="pi-row"><span>Alert time</span><b>{fmtDateTime(probe.parent_alert_timestamp_ms)}</b></div>
            )}
          </>
        )}
        <div className="pi-row"><span>Text pairs</span><b>{(probe.positives?.length ?? 0)} positive · {(probe.negatives?.length ?? 0)} negative</b></div>
        <div className="pi-row"><span>Image probe</span><b>{probe.image_probe?.enabled ? 'on' : 'off'}</b></div>
        <div className="pi-row"><span>Floor · margin</span><b>{n3(probe.pos_floor)} · {n3(probe.margin)}</b></div>
        <div className="pi-row"><span>Query window</span><b>{probe.window_sec ?? '—'}s</b></div>
        <div className="pi-row"><span>Bookmark gate</span><b>{gateText(probe)}</b></div>
      </div>

      <div className="pi-actions">
        {origin === 'auto' && probe.parent_alert_id && probe.parent_alert_timestamp_ms && onOpenParentAlert && (
          <button className="mon-btn" onClick={() => onOpenParentAlert(probe)}>
            <IconExternalLink size={15} /> Open parent alert in archive
          </button>
        )}
        {(onSettings || settingsBlockedReason) && (
          <button className="mon-btn" disabled={!!settingsBlockedReason} title={settingsBlockedReason} onClick={onSettings}>
            <IconSettings size={15} /> Semantic probe settings
          </button>
        )}
        {onRun && (status === 'running'
          ? <button className="mon-btn accent stop" disabled={busy} onClick={onRun}><IconPlayerStop size={15} /> Stop probe</button>
          : <button className="mon-btn accent" disabled={busy} onClick={onRun}><IconPlayerPlay size={15} /> Run probe</button>)}
        {onDelete && <button className="mon-btn danger" disabled={busy} onClick={onDelete}><IconTrash size={15} /> Delete probe</button>}
      </div>
    </div>
  )
}
