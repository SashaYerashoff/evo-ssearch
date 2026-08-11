import { useState, useRef, useEffect, type ReactNode } from 'react'
import { IconChevronDown } from '@tabler/icons-react'

export interface DropOption { value: string; label: string }

// Reusable custom dropdown (replaces native <select>). Two looks: 'chip' (filter bar) and 'field' (forms).
export function Dropdown({ value, options, onChange, icon, variant = 'field', title, disabled }: {
  value: string
  options: DropOption[]
  onChange: (v: string) => void
  icon?: ReactNode
  variant?: 'field' | 'chip'
  title?: string
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const sel = options.find((o) => o.value === value)
  return (
    <div className={`dd dd-${variant}`} ref={ref}>
      <button type="button" className="dd-btn" disabled={disabled} title={title} aria-label={title} onClick={() => !disabled && setOpen((v) => !v)}>
        {icon}<span className="dd-val">{sel?.label ?? value}</span><IconChevronDown size={13} className="dd-chev" />
      </button>
      {open && (
        <div className="dd-pop">
          {options.map((o) => (
            <button type="button" key={o.value} className={`dd-opt ${o.value === value ? 'on' : ''}`}
              onClick={() => { onChange(o.value); setOpen(false) }}>{o.label}</button>
          ))}
        </div>
      )}
    </div>
  )
}
