import {
  IconFileDescription,
  IconLogout,
  IconPhotoSearch,
  IconSettings,
  IconTargetArrow,
} from '@tabler/icons-react'
import { useEffect } from 'react'
import { useI18n, type TranslationKey } from '../../i18n/I18nProvider'

export type SectionId = 'home' | 'archive' | 'video' | 'monitoring' | 'agent'

export const SECTION_LABEL_KEYS: Record<SectionId, TranslationKey> = {
  home: 'nav.home',
  archive: 'nav.archive',
  video: 'nav.summaries',
  monitoring: 'nav.probes',
  agent: 'nav.agent',
}

const NAV = [
  { id: 'archive', labelKey: 'nav.archive', Icon: IconPhotoSearch },
  { id: 'video', labelKey: 'nav.summariesShort', Icon: IconFileDescription },
  { id: 'monitoring', labelKey: 'nav.probes', Icon: IconTargetArrow },
] satisfies { id: SectionId; labelKey: TranslationKey; Icon: typeof IconPhotoSearch }[]

type MenuRailTriggerProps = {
  open: boolean
  onToggle: () => void
}

export function MenuRailTrigger({ open, onToggle }: MenuRailTriggerProps) {
  const { t } = useI18n()
  return (
    <button
      className={`menu-rail-trigger ${open ? 'open' : ''}`}
      aria-label={open ? t('nav.closeMenu') : t('nav.openMenu')}
      aria-expanded={open}
      aria-controls="eva-main-menu"
      title={open ? t('nav.closeMenu') : t('nav.openMenu')}
      onClick={onToggle}
    >
      <span className="menu-rail-label" aria-hidden="true">MENU</span>
    </button>
  )
}

export function LeftRail({
  active,
  visibleSections,
  showSettings,
  showTrigger,
  open,
  onOpenChange,
  onNavigate,
  onSettings,
  onLogout,
}: {
  active: SectionId
  visibleSections: SectionId[]
  showSettings: boolean
  showTrigger: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  onNavigate: (section: SectionId) => void
  onSettings: () => void
  onLogout: () => void
}) {
  const { t } = useI18n()
  useEffect(() => {
    if (!open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onOpenChange(false)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open, onOpenChange])

  const items = NAV.filter(({ id }) => visibleSections.includes(id))
  const navigate = (section: SectionId) => {
    onNavigate(section)
    onOpenChange(false)
  }

  return (
    <aside className={`menu-shell ${open ? 'open' : ''}`}>
      {open && <button className="menu-dismiss" onClick={() => onOpenChange(false)} aria-label={t('nav.closeMenu')} />}
      {showTrigger && (
        <MenuRailTrigger
          open={open}
          onToggle={() => onOpenChange(!open)}
        />
      )}

      <nav className="menu-drawer" id="eva-main-menu" aria-hidden={!open}>
        <div className="menu-title">{t('nav.navigation')}</div>
        {items.map(({ id, labelKey, Icon }) => (
          <button key={id} className={`menu-item ${active === id ? 'on' : ''}`} onClick={() => navigate(id)}>
            <span className="ricon"><Icon size={22} stroke={1.8} /></span>
            <span className="menu-label">{t(labelKey)}</span>
          </button>
        ))}
        {showSettings && (
          <>
            <div className="menu-sep" />
            <button className="menu-item" onClick={() => { onSettings(); onOpenChange(false) }}>
              <span className="ricon"><IconSettings size={21} stroke={1.8} /></span>
              <span className="menu-label">{t('nav.settings')}</span>
            </button>
          </>
        )}
        <button className="menu-item danger" onClick={() => { onOpenChange(false); onLogout() }}>
          <span className="ricon"><IconLogout size={21} stroke={1.8} /></span>
          <span className="menu-label">{t('nav.logout')}</span>
        </button>
      </nav>
    </aside>
  )
}
