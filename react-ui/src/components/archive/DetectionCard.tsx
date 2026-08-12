import { useEffect, useState } from 'react'
import { IconVideo } from '@tabler/icons-react'
import type { Detection } from '../../api/types'
import { detImageSrc } from '../../api/detections'

function fmtTime(ms: number | null): string {
  if (!ms) return '—'
  const d = new Date(ms)
  return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

export function DetectionCard({ d, onClick }: { d: Detection; onClick: () => void }) {
  const src = detImageSrc(d)
  const [imageFailed, setImageFailed] = useState(false)
  const sev = d.severity.toLowerCase()
  useEffect(() => setImageFailed(false), [src])
  return (
    <button className="card archive-card" onClick={onClick}>
      <div className="card-thumb">
        <span className="card-tag">{d.sourceLabel}</span>
        {src && !imageFailed
          ? <img src={src} alt={d.probeName} loading="lazy" onError={() => setImageFailed(true)} />
          : <span className="card-thumb-missing"><IconVideo size={22} /> Preview unavailable</span>}
      </div>
      <div className="card-body">
        <div className="card-title-row">
          <span className="card-title">{d.probeName}</span>
          {['critical', 'high'].includes(sev) && <span className={`sev ${sev}`}>{sev}</span>}
        </div>
        <div className="card-meta">
          {d.channelTitle ? d.channelTitle.slice(0, 22) : `ch ${d.channelId ?? '—'}`}
          {' · '}{fmtTime(d.tsMs)}
          {d.matchPct != null && <> · {d.matchPct}%</>}
        </div>
      </div>
    </button>
  )
}
