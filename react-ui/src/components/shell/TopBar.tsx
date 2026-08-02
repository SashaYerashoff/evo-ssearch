import { IconColorSwatch } from '@tabler/icons-react'

export function TopBar({
  section, appVersion, onAppearance, onBrand,
}: {
  section: string
  appVersion: string
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
        <button
          className="motion-toggle appearance-toggle"
          onClick={onAppearance}
          title="Theme and appearance"
          aria-label="Open theme and appearance settings"
        >
          <IconColorSwatch size={14} />
          Appearance
        </button>
      </div>
    </div>
  )
}
