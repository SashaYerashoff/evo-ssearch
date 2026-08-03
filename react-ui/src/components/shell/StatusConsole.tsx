import type { AuthUser } from '../../api/types'
import type { StatusData } from '../../App'
import { useI18n } from '../../i18n/I18nProvider'

export function StatusConsole({ user, status }: { user: AuthUser; status: StatusData }) {
  const { t } = useI18n()
  const luxriot = {
    checking: { className: '', label: t('status.checking') },
    connected: { className: 'ok', label: t('status.connected') },
    stale: { className: 'warn', label: t('status.stale') },
    offline: { className: 'err', label: t('status.offline') },
  }[status.luxriot]
  return (
    <div className="status-console" role="status" aria-label="EVA runtime status">
      <span className={luxriot.className} title={status.luxriotDetail}>
        <i className="status-console-dot" /> {t('status.luxriot')} {luxriot.label}
      </span>
      <i className="status-console-sep" />
      <span>{status.channels} {t(status.channels === 1 ? 'status.channel' : 'status.channels')}</span>
      <i className="status-console-sep" />
      <span className={status.agent === 'working' ? 'active' : ''}>{t('status.agent')} {t(status.agent === 'working' ? 'status.working' : 'status.idle')}</span>
      <i className="status-console-sep status-console-secondary" />
      <span className="status-console-secondary">{status.probes} {t(status.probes === 1 ? 'status.probe' : 'status.probes')}</span>
      <span className="status-console-user" title={user.username}>{user.displayName || user.username}</span>
    </div>
  )
}
