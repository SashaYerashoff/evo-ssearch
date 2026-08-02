import type { AuthUser } from '../../api/types'
import type { StatusData } from '../../App'

export function StatusConsole({ user, status }: { user: AuthUser; status: StatusData }) {
  return (
    <div className="status-console" role="status" aria-label="EVA runtime status">
      <span className={status.luxriot ? 'ok' : 'err'}>
        <i className="status-console-dot" /> Luxriot {status.luxriot ? 'connected' : 'offline'}
      </span>
      <i className="status-console-sep" />
      <span>{status.channels} channel{status.channels === 1 ? '' : 's'}</span>
      <i className="status-console-sep" />
      <span className={status.agent === 'working' ? 'active' : ''}>Agent {status.agent}</span>
      <i className="status-console-sep status-console-secondary" />
      <span className="status-console-secondary">{status.probes} configured probe{status.probes === 1 ? '' : 's'}</span>
      <span className="status-console-user" title={user.username}>{user.displayName || user.username}</span>
    </div>
  )
}
