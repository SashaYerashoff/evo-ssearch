import { useState } from 'react'
import { IconX, IconMessage, IconCopy, IconPhoto } from '@tabler/icons-react'
import type { Detection } from '../../api/types'
import { thumbSrc, describeFrame } from '../../api/detections'

function fmtFull(ms: number | null): string {
  if (!ms) return '—'
  return new Date(ms).toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

export function InspectorModal({
  d, onClose, onFindSimilar,
}: {
  d: Detection
  onClose: () => void
  onFindSimilar: (d: Detection) => void
}) {
  const src = thumbSrc(d)
  const [desc, setDesc] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function describe() {
    setBusy(true); setDesc(null)
    try { setDesc(await describeFrame(d)) }
    catch (e: any) { setDesc('Error: ' + (e?.message || 'describe failed')) }
    finally { setBusy(false) }
  }

  const scores = [
    d.posScore != null && `P ${d.posScore.toFixed(2)}`,
    d.negScore != null && `N ${d.negScore.toFixed(2)}`,
    d.margin != null && `M ${d.margin.toFixed(2)}`,
  ].filter(Boolean).join(' · ')

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">Inspector · {d.probeName}</div>
          <button className="modal-close" onClick={onClose}><IconX size={18} /></button>
        </div>
        <div className="modal-body">
          <div className="modal-frame">
            {src ? <img src={src} alt={d.probeName} /> : <IconPhoto size={30} />}
          </div>
          <div className="kv">
            <span className="k">Channel</span><span className="v">{d.channelTitle || `ch ${d.channelId ?? '—'}`}</span>
            <span className="k">Source</span><span className="v">{d.sourceLabel}</span>
            <span className="k">Time</span><span className="v">{fmtFull(d.tsMs)}</span>
            <span className="k">Severity</span><span className="v" style={{ color: 'var(--danger)' }}>{d.severity}</span>
            {(d.matchPct != null || scores) && (
              <>
                <span className="k">Match</span>
                <span className="v">{d.matchPct != null ? `${d.matchPct}%` : '—'}{scores ? ` · ${scores}` : ''}</span>
              </>
            )}
          </div>
          {(busy || desc) && (
            <div className="desc-box">{busy ? 'Generating description…' : desc}</div>
          )}
          <div className="modal-actions">
            <button className="btn" onClick={describe} disabled={busy}><IconMessage size={15} /> Describe frame</button>
            <button className="btn" onClick={() => onFindSimilar(d)}><IconCopy size={15} /> Find similar</button>
          </div>
        </div>
      </div>
    </div>
  )
}
