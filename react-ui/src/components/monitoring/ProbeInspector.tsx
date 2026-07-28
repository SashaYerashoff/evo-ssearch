import { IconSettings, IconPlayerPlay, IconPlayerStop, IconTrash, IconPhoto } from '@tabler/icons-react'
import type { Probe } from '../../api/probes'
import { hitImageSrc } from '../../api/probes'
import { gateText, lastHit, type ProbeStatus } from './ProbeCard'

const n3 = (v?: number | null) => (v == null ? '—' : Number(v).toFixed(3))
function fmtDateTime(ms?: number | null): string {
  if (!ms) return 'never'
  return new Date(Number(ms)).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' })
}

export function ProbeInspector({ probe, status, busy, settingsBlockedReason, onSettings, onRun, onDelete }: {
  probe: Probe
  status: ProbeStatus
  busy: boolean
  settingsBlockedReason?: string
  onSettings?: () => void
  onRun?: () => void
  onDelete?: () => void
}) {
  const hit = lastHit(probe)
  const src = hitImageSrc(hit)
  return (
    <div className="mon-panel">
      <div className="mon-panel-title">Selected CLIP probe</div>

      {/* who: one clean identity row */}
      <div className="pi-head">
        <div className="pi-thumb">{src ? <img src={src} alt={probe.name || 'probe'} /> : <IconPhoto size={18} />}</div>
        <div className="pi-id">
          <div className="pi-name">{probe.name || 'Untitled probe'}</div>
          <div className="pi-line">Ch {probe.channel_id ?? '—'} · Last event {fmtDateTime(hit?.timestamp_ms ?? hit?.recorded_at_ms)}</div>
        </div>
        <span className={`pc-badge ${status}`}>{status.toUpperCase()}</span>
      </div>

      {/* what matters: the last signal, front and centre */}
      <div className="pi-sec">Last signal</div>
      <div className="pi-scores">
        <div><span>Positive</span><b className="pos">{n3(hit?.pos_score)}</b></div>
        <div><span>Negative</span><b>{n3(hit?.neg_score)}</b></div>
        <div><span>Margin</span><b className="mar">{n3(hit?.margin)}</b></div>
      </div>

      {/* how it's set up: one quiet scannable list */}
      <div className="pi-sec">Configuration</div>
      <div className="pi-config">
        <div className="pi-row"><span>Text pairs</span><b>{(probe.positives?.length ?? 0)} positive · {(probe.negatives?.length ?? 0)} negative</b></div>
        <div className="pi-row"><span>Image probe</span><b>{probe.image_probe?.enabled ? 'on' : 'off'}</b></div>
        <div className="pi-row"><span>Floor · margin</span><b>{n3(probe.pos_floor)} · {n3(probe.margin)}</b></div>
        <div className="pi-row"><span>Query window</span><b>{probe.window_sec ?? '—'}s</b></div>
        <div className="pi-row"><span>Bookmark gate</span><b>{gateText(probe)}</b></div>
      </div>

      <div className="pi-actions">
        {(onSettings || settingsBlockedReason) && (
          <button className="mon-btn" disabled={!!settingsBlockedReason} title={settingsBlockedReason} onClick={onSettings}>
            <IconSettings size={15} /> CLIP probe settings
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
