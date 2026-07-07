import { IconSettings, IconPlayerPlay, IconTrash } from '@tabler/icons-react'
import type { Probe } from '../../api/probes'
import { hitImageSrc } from '../../api/probes'
import { gateText, lastHit, type ProbeStatus } from './ProbeCard'

const n3 = (v?: number | null) => (v == null ? '—' : Number(v).toFixed(3))
function fmtDateTime(ms?: number | null): string {
  if (!ms) return '—'
  return new Date(Number(ms)).toLocaleString([], { month: 'numeric', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' })
}

export function ProbeInspector({ probe, status, busy, onSettings, onRun, onDelete }: {
  probe: Probe
  status: ProbeStatus
  busy: boolean
  onSettings: () => void
  onRun: () => void
  onDelete: () => void
}) {
  const hit = lastHit(probe)
  const src = hitImageSrc(hit)
  return (
    <div className="mon-panel">
      <div className="mon-panel-title">Selected CLIP probe</div>
      <div className="mon-panel-sub">Current state, last signal, and direct engineer actions.</div>
      <div className="pi-status">{status.toUpperCase()} · Ch {probe.channel_id ?? '—'}</div>

      <div className="pi-head">
        <div className="pi-thumb">{src ? <img src={src} alt={probe.name || 'probe'} /> : null}</div>
        <div>
          <span className={`pc-badge ${status}`}>{status.toUpperCase()}</span>
          <div className="pi-name">{probe.name || 'Untitled probe'}</div>
          <div className="pi-line">Channel {probe.channel_id ?? '—'}</div>
          <div className="pi-line">Last event: {fmtDateTime(hit?.timestamp_ms ?? hit?.recorded_at_ms)}</div>
        </div>
      </div>

      <div className="pi-field"><div className="pi-k">Scores</div><div className="pi-v mono">P: {n3(hit?.pos_score)} · N: {n3(hit?.neg_score)} · M: {n3(hit?.margin)}</div></div>
      <div className="pi-field"><div className="pi-k">Bookmark gate</div><div className="pi-v">{gateText(probe)}</div></div>
      <div className="pi-field"><div className="pi-k">Text pairs</div><div className="pi-v">{(probe.positives?.length ?? 0)} positive · {(probe.negatives?.length ?? 0)} negative</div></div>
      <div className="pi-field"><div className="pi-k">Image probe</div><div className="pi-v">{probe.image_probe?.enabled ? 'on' : 'off'}</div></div>
      <div className="pi-field"><div className="pi-k">Thresholds</div><div className="pi-v mono">floor {n3(probe.pos_floor)} · margin {n3(probe.margin)} · cooldown {probe.window_sec ?? '—'}s</div></div>

      <div className="pi-actions">
        <button className="mon-btn" onClick={onSettings}><IconSettings size={15} /> CLIP probe settings</button>
        <button className="mon-btn accent" disabled={busy} onClick={onRun}><IconPlayerPlay size={15} /> Run probe</button>
        <button className="mon-btn danger" disabled={busy} onClick={onDelete}><IconTrash size={15} /> Delete probe</button>
      </div>
    </div>
  )
}
