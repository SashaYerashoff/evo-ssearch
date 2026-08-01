import {
  IconFileDescription,
  IconLogout,
  IconPhotoSearch,
  IconSettings,
  IconTargetArrow,
} from '@tabler/icons-react'

export type SectionId = 'home' | 'archive' | 'video' | 'monitoring' | 'agent'

export const SECTION_LABELS: Record<SectionId, string> = {
  home: 'Home',
  archive: 'Archive',
  video: 'Stream summaries',
  monitoring: 'Probes',
  agent: 'Agent',
}

const NAV: { id: SectionId; label: string; Icon: any }[] = [
  { id: 'archive', label: 'Archive', Icon: IconPhotoSearch },
  { id: 'video', label: 'Summaries', Icon: IconFileDescription },
  { id: 'monitoring', label: 'Probes', Icon: IconTargetArrow },
]

export function LeftRail({
  active, visibleSections, showSettings, onNavigate, onSettings, onLogout,
}: {
  active: SectionId
  visibleSections: SectionId[]
  showSettings: boolean
  onNavigate: (s: SectionId) => void
  onSettings: () => void
  onLogout: () => void
}) {
  return (
    <div className="rail-wrap">
      <div className="rail">
        {NAV.filter(({ id }) => visibleSections.includes(id)).map(({ id, label, Icon }) => (
          <button key={id} className={`rail-item ${active === id ? 'on' : ''}`} onClick={() => onNavigate(id)} title={label}>
            <span className="ricon"><Icon size={23} stroke={1.8} /></span>
            <span className="rail-label">{label}</span>
          </button>
        ))}
        {showSettings && (
          <>
            <div className="rail-sep" />
            <button className="rail-item" onClick={onSettings} title="Settings">
              <span className="ricon"><IconSettings size={22} stroke={1.8} /></span>
              <span className="rail-label">Settings</span>
            </button>
          </>
        )}
        <button className="rail-item danger" onClick={onLogout} title="Log out">
          <span className="ricon"><IconLogout size={22} stroke={1.8} /></span>
          <span className="rail-label">Log out</span>
        </button>
      </div>
    </div>
  )
}
