import { IconEye, IconVideo, IconRadar2, IconPlug, IconPlugConnectedX, IconActivity, IconPlayerPause } from '@tabler/icons-react'
import type { AuthUser } from '../../api/types'
import type { StatusData } from '../../App'
import { BenchmarkButton } from '../monitoring/BenchmarkButton'

export function TopBar({
  user, status, noAnim, onToggleNoAnim,
}: {
  user: AuthUser
  status: StatusData
  noAnim: boolean
  onToggleNoAnim: () => void
}) {
  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-mark"><IconEye size={19} stroke={2} /></div>
        <div>
          <div className="brand-name">Luxriot · EVA AI</div>
          <div className="brand-sub">β 0.8.3 · {user.displayName || user.username}</div>
        </div>
      </div>

      <div className="top-right">
      <div className="status-strip">
        {status.luxriot ? (
          <span className="status-chip" style={{ color: 'var(--success)' }}>
            <span className="dot pulse" style={{ background: 'var(--success)', color: 'var(--success)' }} />
            <IconPlug size={14} /> Luxriot connected
          </span>
        ) : (
          <span className="status-chip" style={{ color: 'var(--danger)' }}>
            <span className="dot" style={{ background: 'var(--danger)' }} />
            <IconPlugConnectedX size={14} /> Luxriot offline
          </span>
        )}
        <span className="status-chip"><IconVideo size={14} /> {status.channels} channels</span>
        <span className="status-chip">
          <span className="dot" style={{ background: status.agent === 'working' ? 'var(--accent)' : 'var(--text-mut)' }} />
          Agent {status.agent}
        </span>
        <span className="status-chip"><IconRadar2 size={14} /> {status.probes} probe{status.probes === 1 ? '' : 's'} active</span>
      </div>
        <BenchmarkButton />
        <button
          className={`motion-toggle ${noAnim ? 'off' : ''}`}
          onClick={onToggleNoAnim}
          title="Toggle all interface animations"
        >
          {noAnim ? <IconPlayerPause size={14} /> : <IconActivity size={14} />}
          {noAnim ? 'Animations off' : 'Animations on'}
        </button>
      </div>
    </div>
  )
}
