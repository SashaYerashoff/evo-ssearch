import type { ReactNode } from 'react'

// Folder-style tool tabs: a fixed tab row on top, the active tab's controls
// open in a panel right below. Tabs never move — the active one lights up.
export interface ToolTab { id: string; icon: ReactNode; label: string; summary: string; badge?: string }

export function ToolTabs({ tabs, active, onSelect, leading, children }: {
  tabs: ToolTab[]
  active: string
  onSelect: (id: string) => void
  leading?: ReactNode
  children: ReactNode
}) {
  return (
    <div className={`tool-tabs ${leading ? 'with-leading' : ''}`}>
      {leading && <div className="tool-tabs-leading">{leading}</div>}
      <div className="atp-tabpanel">
        <div className="atp-tabrow">
          {tabs.map((t) => (
            <button
              key={t.id}
              className={`atp-tab ${active === t.id ? 'on' : ''}`}
              onClick={() => onSelect(t.id)}
              title={`${t.label}${t.summary && t.summary !== '—' ? ` · ${t.summary}` : ''}`}
              aria-pressed={active === t.id}
            >
              <b>
                {t.icon} {t.label}
                {t.badge && <em className="atp-tab-badge">{t.badge}</em>}
              </b>
              <i className="atp-tab-sep" aria-hidden="true" />
              <span className="atp-tab-summary">{t.summary}</span>
            </button>
          ))}
        </div>
        <div className="atp-tabpanel-content">{children}</div>
      </div>
    </div>
  )
}
