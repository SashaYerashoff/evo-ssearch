import { useEffect, useRef, useState, type ReactNode } from 'react'
import { IconChevronDown, IconDots } from '@tabler/icons-react'

export interface ToolbarAction {
  id: string
  label: string
  icon: ReactNode
  onSelect: () => void
  disabled?: boolean
  danger?: boolean
  active?: boolean
}

export function ToolbarActionMenu({ actions, label = 'Actions' }: {
  actions: ToolbarAction[]
  label?: string
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const closeOutside = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', closeOutside)
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('mousedown', closeOutside)
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [open])

  if (!actions.length) return null

  return (
    <div className="toolbar-action-menu" ref={ref}>
      <button
        type="button"
        className={`toolbar-actions-trigger ${open ? 'on' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <IconDots size={16} />
        <span>{label}</span>
        <IconChevronDown size={13} />
      </button>
      {open && (
        <div className="toolbar-actions-pop" role="menu">
          {actions.map((action) => (
            <button
              type="button"
              key={action.id}
              role="menuitem"
              className={`${action.danger ? 'danger' : ''} ${action.active ? 'active' : ''}`}
              disabled={action.disabled}
              onClick={() => {
                action.onSelect()
                setOpen(false)
              }}
            >
              {action.icon}
              <span>{action.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
