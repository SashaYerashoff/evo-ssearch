import {
  IconFileDescription,
  IconLogout,
  IconMenu2,
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

const NAV: { id: SectionId; labelKey: TranslationKey; Icon: any }[] = [
  { id: 'archive', labelKey: 'nav.archive', Icon: IconPhotoSearch },
  { id: 'video', labelKey: 'nav.summariesShort', Icon: IconFileDescription },
  { id: 'monitoring', labelKey: 'nav.probes', Icon: IconTargetArrow },
]

export function MainMenuButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const { t } = useI18n()
  if (open) return null
  return (
    <button
      className="menu-header-button"
      aria-expanded={false}
      aria-controls="eva-main-menu"
      title={t('nav.openMenu')}
      onClick={onToggle}
    >
      <IconMenu2 size={20} stroke={2} />
      <span>{t('nav.menu')}</span>
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
  const { t } = useI18n()
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
      {open && <button className="menu-dismiss" onClick={() => onOpenChange(false)} aria-label={t('nav.closeMenu')} />}
      {showTrigger && !open && (
        <button
          className="menu-ear"
          aria-expanded={false}
          aria-controls="eva-main-menu"
          title={t('nav.openMenu')}
          onClick={() => onOpenChange(true)}
        >
          <IconMenu2 size={15} stroke={2} />
          <span className="txt">{t('nav.menu')}</span>
        </button>
      )}
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
            <button className="menu-item" onClick={() => { onSettings(); onOpenChange(false) }}>
              <span className="ricon"><IconSettings size={22} stroke={1.8} /></span>
              <span className="menu-label">{t('nav.settings')}</span>
            </button>
          </>
        )}
        <button className="menu-item danger" onClick={() => { onOpenChange(false); onLogout() }}>
          <span className="ricon"><IconLogout size={22} stroke={1.8} /></span>
          <span className="menu-label">{t('nav.logout')}</span>
        </button>
      </nav>
    </div>
  )
}
