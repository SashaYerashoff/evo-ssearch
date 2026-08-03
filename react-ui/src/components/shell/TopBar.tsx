import { IconColorSwatch } from '@tabler/icons-react'
import { useI18n } from '../../i18n/I18nProvider'

export function TopBar({
  section, appVersion, onAppearance, onBrand,
}: {
  section: string
  appVersion: string
  onAppearance: () => void
  onBrand: () => void
}) {
  const { t } = useI18n()
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
        <button
          className="motion-toggle appearance-toggle"
          onClick={onAppearance}
          title={t('appearance.open')}
          aria-label={t('appearance.open')}
        >
          <IconColorSwatch size={14} />
          {t('appearance.title')}
        </button>
      </div>
    </div>
  )
}
