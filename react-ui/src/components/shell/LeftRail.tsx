import {
  IconFileDescription,
  IconLogout,
  IconMenu2,
  IconPhotoSearch,
  IconSettings,
  IconTargetArrow,
} from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useI18n, type TranslationKey } from '../../i18n/I18nProvider'

export type SectionId = 'home' | 'archive' | 'video' | 'monitoring' | 'agent'

export const SECTION_LABEL_KEYS: Record<SectionId, TranslationKey> = {
  home: 'nav.home',
  archive: 'nav.archive',
  video: 'nav.summaries',
  monitoring: 'nav.probes',
  agent: 'nav.agent',
}

const NAV: { id: SectionId; labelKey: TranslationKey; Icon: any }[] = [
  { id: 'archive', labelKey: 'nav.archive', Icon: IconPhotoSearch },
  { id: 'video', labelKey: 'nav.summariesShort', Icon: IconFileDescription },
  { id: 'monitoring', labelKey: 'nav.probes', Icon: IconTargetArrow },
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
  const { t } = useI18n()
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
      {open && <button className="menu-dismiss" onClick={() => setOpen(false)} aria-label={t('nav.closeMenu')} />}
      <button
        className="menu-ear"
        aria-expanded={open}
        aria-controls="eva-main-menu"
        title={open ? t('nav.closeMenu') : t('nav.openMenu')}
        onClick={() => setOpen((value) => !value)}
      >
        <IconMenu2 size={15} stroke={2} />
        <span className="txt">{t('nav.menu')}</span>
      </button>
      <nav className="menu-drawer" id="eva-main-menu" aria-hidden={!open}>
        <div className="menu-title">{t('nav.navigation')}</div>
        {NAV.filter(({ id }) => visibleSections.includes(id)).map(({ id, labelKey, Icon }) => (
          <button key={id} className={`menu-item ${active === id ? 'on' : ''}`} onClick={() => navigate(id)}>
            <span className="ricon"><Icon size={23} stroke={1.8} /></span>
            <span className="menu-label">{t(labelKey)}</span>
          </button>
        ))}
        {showSettings && (
          <>
            <div className="menu-sep" />
            <button className="menu-item" onClick={() => { onSettings(); setOpen(false) }}>
              <span className="ricon"><IconSettings size={22} stroke={1.8} /></span>
              <span className="menu-label">{t('nav.settings')}</span>
            </button>
          </>
        )}
        <button className="menu-item danger" onClick={onLogout}>
          <span className="ricon"><IconLogout size={22} stroke={1.8} /></span>
          <span className="menu-label">{t('nav.logout')}</span>
        </button>
      </nav>
    </div>
  )
}
