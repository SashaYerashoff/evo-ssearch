import { useRef, useState } from 'react'
import {
  IconArchive, IconDeviceTv, IconRadar2, IconSettings, IconLogout,
  IconFilter, IconAdjustmentsHorizontal, IconLetterT, IconPhoto,
  IconPin, IconPinnedFilled, IconRefresh, IconPlus,
} from '@tabler/icons-react'
import type { ArchiveTool } from '../archive/ArchiveScreen'
import type { MonitorAction } from '../monitoring/MonitoringScreen'

export type SectionId = 'archive' | 'video' | 'monitoring' | 'agent'

const NAV: { id: SectionId; label: string; Icon: any }[] = [
  { id: 'archive', label: 'Archive', Icon: IconArchive },
  { id: 'video', label: 'Video', Icon: IconDeviceTv },
  { id: 'monitoring', label: 'Monitoring', Icon: IconRadar2 },
]

const ARCHIVE_CHILDREN: { tool: ArchiveTool; label: string; Icon: any }[] = [
  { tool: 'filters', label: 'Archive filters', Icon: IconFilter },
  { tool: 'search', label: 'Search controls', Icon: IconAdjustmentsHorizontal },
  { tool: 'text', label: 'Text query', Icon: IconLetterT },
  { tool: 'image', label: 'Image query', Icon: IconPhoto },
]

const MONITOR_CHILDREN: { action: MonitorAction; label: string; Icon: any }[] = [
  { action: 'refresh', label: 'Refresh list', Icon: IconRefresh },
  { action: 'new', label: 'New CLIP probe', Icon: IconPlus },
]

export function LeftRail({
  active, pinned, onNavigate, onArchiveTool, onMonitorAction, onTogglePin, onLogout,
}: {
  active: SectionId
  pinned: boolean
  onNavigate: (s: SectionId) => void
  onArchiveTool: (t: ArchiveTool) => void
  onMonitorAction: (a: MonitorAction) => void
  onTogglePin: () => void
  onLogout: () => void
}) {
  const [open, setOpen] = useState(false)
  const expanded = open || pinned
  const timer = useRef<number | undefined>(undefined)
  const openNow = () => { if (timer.current) window.clearTimeout(timer.current); setOpen(true) }
  const scheduleClose = () => {
    if (timer.current) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setOpen(false), 140)
  }

  return (
    <div className={`rail-wrap ${pinned ? 'pinned' : ''}`}>
      {/* one element — expands in place on hover (overlay), or pinned open (pushes layout).
          hover handlers live on .rail itself so only the real menu area triggers expand. */}
      <div className={`rail ${expanded ? 'open' : ''} ${pinned ? 'pinned' : ''}`} onMouseEnter={openNow} onMouseLeave={scheduleClose}>
        {/* pin / fix at the top */}
        <button
          className={`rail-item rail-pin ${pinned ? 'on' : ''}`}
          onClick={onTogglePin}
          title={pinned ? 'Unpin menu' : 'Pin menu open'}
        >
          <span className="ricon">{pinned ? <IconPinnedFilled size={22} stroke={1.8} /> : <IconPin size={22} stroke={1.8} />}</span>
          <span className="rail-label">{pinned ? 'Unpin' : 'Pin menu'}</span>
        </button>
        <div className="rail-sep" />

        {NAV.map(({ id, label, Icon }) => (
          <div key={id}>
            <button className={`rail-item ${active === id ? 'on' : ''}`} onClick={() => onNavigate(id)} title={label}>
              <span className="ricon"><Icon size={26} stroke={1.8} /></span>
              <span className="rail-label">{label}</span>
            </button>
            {id === 'archive' && active === 'archive' && expanded && (
              <div className="rail-children">
                {ARCHIVE_CHILDREN.map(({ tool, label, Icon }) => (
                  <button key={label} className="rail-child" onClick={() => onArchiveTool(tool)}>
                    <span className="ricon"><Icon size={19} stroke={1.8} /></span>
                    <span className="rail-label">{label}</span>
                  </button>
                ))}
              </div>
            )}
            {id === 'monitoring' && active === 'monitoring' && expanded && (
              <div className="rail-children">
                {MONITOR_CHILDREN.map(({ action, label, Icon }) => (
                  <button key={label} className="rail-child" onClick={() => onMonitorAction(action)}>
                    <span className="ricon"><Icon size={19} stroke={1.8} /></span>
                    <span className="rail-label">{label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        <div className="rail-sep" />
        <button className="rail-item" onClick={() => onNavigate('archive')} title="Settings">
          <span className="ricon"><IconSettings size={26} stroke={1.8} /></span>
          <span className="rail-label">Settings</span>
        </button>
        <button className="rail-item danger" onClick={onLogout} title="Log out">
          <span className="ricon"><IconLogout size={26} stroke={1.8} /></span>
          <span className="rail-label">Log out</span>
        </button>
      </div>
    </div>
  )
}
