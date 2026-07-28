import { IconEye, IconArchive, IconDeviceTv, IconRadar2, IconSettings, IconLogout } from '@tabler/icons-react'

export type SectionId = 'home' | 'archive' | 'video' | 'monitoring' | 'agent'

export const SECTION_LABELS: Record<SectionId, string> = {
  home: 'Home', archive: 'Archive', video: 'Video', monitoring: 'Monitoring', agent: 'Agent',
}

const NAV: { id: SectionId; label: string; Icon: any }[] = [
  { id: 'home', label: 'Home', Icon: IconEye },
  { id: 'archive', label: 'Archive', Icon: IconArchive },
  { id: 'video', label: 'Video', Icon: IconDeviceTv },
  { id: 'monitoring', label: 'Monitoring', Icon: IconRadar2 },
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
            <span className="ricon"><Icon size={26} stroke={1.8} /></span>
          </button>
        ))}
        {showSettings && (
          <>
            <div className="rail-sep" />
            <button className="rail-item" onClick={onSettings} title="Settings">
              <span className="ricon"><IconSettings size={26} stroke={1.8} /></span>
            </button>
          </>
        )}
        <button className="rail-item danger" onClick={onLogout} title="Log out">
          <span className="ricon"><IconLogout size={26} stroke={1.8} /></span>
        </button>
      </div>
    </div>
  )
}
