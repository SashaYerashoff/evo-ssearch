import { useEffect, useRef, type ReactNode } from 'react'

// Folder-style tool tabs around a shared control panel. Most screens keep tabs
// above the panel; workspace screens may place them below when controls are global.
export interface ToolTab { id: string; icon: ReactNode; label: string; badge?: string }

const HORIZONTAL_RAIL_SELECTOR = '.toolbar-scroll-rail, .atp-tabrow, .atp-textgroup'
const VERTICAL_POPOVER_SELECTOR = '.dd-pop, .qf-pop, .daterange-pop, .archive-channel-picker-pop'

function wheelPixels(event: WheelEvent, rail: HTMLElement): number {
  if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) return event.deltaY * 24
  if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) return event.deltaY * rail.clientWidth
  return event.deltaY
}

export function ToolTabs({ tabs, active, onSelect, leading, reserveLeading = false, hideTabs = false, tabsPosition = 'top', children }: {
  tabs: ToolTab[]
  active: string
  onSelect: (id: string) => void
  leading?: ReactNode
  reserveLeading?: boolean
  hideTabs?: boolean
  tabsPosition?: 'top' | 'bottom'
  children: ReactNode
}) {
  const hasLeadingColumn = Boolean(leading) || reserveLeading
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const root = rootRef.current
    if (!root) return

    const onWheel = (event: WheelEvent) => {
      const target = event.target as HTMLElement | null
      if (!target || target.closest(VERTICAL_POPOVER_SELECTOR)) return

      const rail = target.closest<HTMLElement>(HORIZONTAL_RAIL_SELECTOR)
      if (!rail || rail.scrollWidth <= rail.clientWidth + 1) return
      if (Math.abs(event.deltaX) >= Math.abs(event.deltaY) || event.deltaY === 0) return

      const delta = wheelPixels(event, rail)
      const maxLeft = Math.max(0, rail.scrollWidth - rail.clientWidth)
      const nextLeft = Math.max(0, Math.min(maxLeft, rail.scrollLeft + delta))
      if (Math.abs(nextLeft - rail.scrollLeft) < 0.5) return

      rail.scrollLeft = nextLeft
      event.preventDefault()
      event.stopPropagation()
    }

    root.addEventListener('wheel', onWheel, { passive: false })
    return () => root.removeEventListener('wheel', onWheel)
  }, [])

  const tabRow = hideTabs ? null : (
    <div className="atp-tabrow">
      {tabs.map((t) => (
        <button
          key={t.id}
          className={`atp-tab ${active === t.id ? 'on' : ''}`}
          onClick={() => onSelect(t.id)}
          title={t.label}
          aria-pressed={active === t.id}
        >
          <b>
            {t.icon} {t.label}
            {t.badge && <em className="atp-tab-badge">{t.badge}</em>}
          </b>
        </button>
      ))}
    </div>
  )

  return (
    <div ref={rootRef} className={`tool-tabs ${hasLeadingColumn ? 'with-leading' : ''} ${hideTabs ? 'without-tabs' : ''} ${tabsPosition === 'bottom' ? 'tabs-bottom' : ''}`}>
      <div className="atp-tabpanel">
        {tabsPosition === 'top' && tabRow}
        <div className="atp-content-row">
          {hasLeadingColumn && <div className="tool-tabs-leading">{leading}</div>}
          <div className="atp-tabpanel-content">{children}</div>
        </div>
        {tabsPosition === 'bottom' && tabRow}
      </div>
    </div>
  )
}
