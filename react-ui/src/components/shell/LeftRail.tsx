import {
  IconFileDescription,
  IconLogout,
  IconMenu2,
  IconPhotoSearch,
  IconSettings,
  IconTargetArrow,
} from '@tabler/icons-react'
import { useEffect } from 'react'

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

export function MainMenuButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  if (open) return null
  return (
    <button
      className="menu-header-button"
      aria-expanded={false}
      aria-controls="eva-main-menu"
      title="Open main menu"
      onClick={onToggle}
    >
      <IconMenu2 size={20} stroke={2} />
      <span>MENU</span>
    </button>
  )
}

export function LeftRail({
  active, visibleSections, showSettings, open, showTrigger, onOpenChange, onNavigate, onSettings, onLogout,
}: {
  active: SectionId
  visibleSections: SectionId[]
  showSettings: boolean
  open: boolean
  showTrigger: boolean
  onOpenChange: (open: boolean) => void
  onNavigate: (s: SectionId) => void
  onSettings: () => void
  onLogout: () => void
}) {
  useEffect(() => {
    if (!open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onOpenChange(false)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open, onOpenChange])

  const navigate = (id: SectionId) => {
    onNavigate(id)
    onOpenChange(false)
  }

  return (
    <div className={`menu-shell ${open ? 'open' : ''}`}>
      {open && <button className="menu-dismiss" onClick={() => onOpenChange(false)} aria-label="Close menu" />}
      {showTrigger && !open && (
        <button
          className="menu-ear"
          aria-expanded={false}
          aria-controls="eva-main-menu"
          title="Open menu"
          onClick={() => onOpenChange(true)}
        >
          <IconMenu2 size={15} stroke={2} />
          <span className="txt">MENU</span>
        </button>
      )}
      <nav className="menu-drawer" id="eva-main-menu" aria-hidden={!open}>
        <div className="menu-title">Navigation</div>
        {NAV.filter(({ id }) => visibleSections.includes(id)).map(({ id, label, Icon }) => (
          <button key={id} className={`menu-item ${active === id ? 'on' : ''}`} onClick={() => navigate(id)}>
            <span className="ricon"><Icon size={23} stroke={1.8} /></span>
            <span className="menu-label">{label}</span>
          </button>
        ))}
        {showSettings && (
          <>
            <div className="menu-sep" />
            <button className="menu-item" onClick={() => { onSettings(); onOpenChange(false) }}>
              <span className="ricon"><IconSettings size={22} stroke={1.8} /></span>
              <span className="menu-label">Settings</span>
            </button>
          </>
        )}
        <button className="menu-item danger" onClick={() => { onOpenChange(false); onLogout() }}>
          <span className="ricon"><IconLogout size={22} stroke={1.8} /></span>
          <span className="menu-label">Log out</span>
        </button>
      </nav>
    </div>
  )
}
