export function TopBar({
  section, subsection, appVersion, onBrand,
}: {
  section: string
  /** Where the operator is inside the section, shown as the next crumb. */
  subsection?: string | null
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
        <div className="top-section">
          <span className={subsection ? 'top-crumb' : undefined}>{section}</span>
          {subsection && (
            <>
              <span className="top-crumb-sep" aria-hidden="true">/</span>
              <span className="top-crumb current">{subsection}</span>
            </>
          )}
        </div>
      </div>

    </div>
  )
}
