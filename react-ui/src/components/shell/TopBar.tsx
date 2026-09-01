export function TopBar({
  section, appVersion, onBrand,
}: {
  section: string
  appVersion: string
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
        </div>
        <div className="top-section">{section}</div>
      </div>

    </div>
  )
}
