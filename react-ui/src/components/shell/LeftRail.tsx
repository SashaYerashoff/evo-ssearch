import {
  IconFileDescription,
  IconLogout,
  IconMenu2,
  IconPhotoSearch,
  IconSettings,
  IconTargetArrow,
} from '@tabler/icons-react'
import { useEffect, useState } from 'react'

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
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open])

  const navigate = (id: SectionId) => {
    onNavigate(id)
    setOpen(false)
  }

  return (
    <div className={`menu-shell ${open ? 'open' : ''}`}>
      {open && <button className="menu-dismiss" onClick={() => setOpen(false)} aria-label="Close menu" />}
      <button
        className="menu-ear"
        aria-expanded={open}
        aria-controls="eva-main-menu"
        title={open ? 'Close menu' : 'Open menu'}
        onClick={() => setOpen((value) => !value)}
      >
        <IconMenu2 size={15} stroke={2} />
        <span className="txt">MENU</span>
      </button>
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
            <button className="menu-item" onClick={() => { onSettings(); setOpen(false) }}>
              <span className="ricon"><IconSettings size={22} stroke={1.8} /></span>
              <span className="menu-label">Settings</span>
            </button>
          </>
        )}
        <button className="menu-item danger" onClick={onLogout}>
          <span className="ricon"><IconLogout size={22} stroke={1.8} /></span>
          <span className="menu-label">Log out</span>
        </button>
      </nav>
    </div>
  )
}
