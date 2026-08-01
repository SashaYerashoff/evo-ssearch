import {
  IconActivity,
  IconColorSwatch,
  IconPlayerPause,
  IconPlug,
  IconPlugConnectedX,
  IconRadar2,
  IconVideo,
} from '@tabler/icons-react'
import type { AuthUser } from '../../api/types'
import type { StatusData } from '../../App'
import { BenchmarkButton } from '../monitoring/BenchmarkButton'

export function TopBar({
  user, status, section, noAnim, canBenchmark, appVersion, onToggleNoAnim, onAppearance, onBrand,
}: {
  user: AuthUser
  status: StatusData
  section: string
  noAnim: boolean
  canBenchmark: boolean
  appVersion: string
  onToggleNoAnim: () => void
  onAppearance: () => void
  onBrand: () => void
}) {
  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-btn" role="button" tabIndex={0} title="EVA AI home"
          onClick={onBrand}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onBrand() }}>
          <div className="brand-top">
            <img className="brand-logo" src="/branding/logo" alt="Luxriot logo" />
            <span className="brand-main">EVA AI</span>
            <span className="brand-ver">{appVersion ? `v${appVersion}` : 'version unavailable'}</span>
          </div>
          <div className="brand-tagline">Smart Image Search and Understanding</div>
        </div>
        <div className="top-section">{section}</div>
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
        <span className="status-chip"><IconRadar2 size={14} /> {status.probes} configured probe{status.probes === 1 ? '' : 's'}</span>
        <span className="status-chip" title={user.username}>{user.displayName || user.username}</span>
      </div>
        {canBenchmark && <BenchmarkButton />}
        <button
          className="motion-toggle appearance-toggle"
          onClick={onAppearance}
          title="Theme and appearance"
          aria-label="Open theme and appearance settings"
        >
          <IconColorSwatch size={14} />
          Appearance
        </button>
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
