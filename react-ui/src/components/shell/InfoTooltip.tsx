import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { IconQuestionMark } from '@tabler/icons-react'

export function InfoTooltip({ label, children }: { label: string; children: ReactNode }) {
  const [hovered, setHovered] = useState(false)
  const [focused, setFocused] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [style, setStyle] = useState<CSSProperties>({ position: 'fixed', visibility: 'hidden' })
  const anchorRef = useRef<HTMLButtonElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const tooltipId = useId()
  const open = hovered || focused || pinned

  useLayoutEffect(() => {
    if (!open) return

    const place = () => {
      const anchor = anchorRef.current?.getBoundingClientRect()
      const tooltip = tooltipRef.current?.getBoundingClientRect()
      if (!anchor || !tooltip) return

      const gutter = 10
      const viewportPadding = 12
      const fitsAbove = anchor.top >= tooltip.height + gutter + viewportPadding
      const top = fitsAbove
        ? anchor.top - tooltip.height - gutter
        : anchor.bottom + gutter
      const preferredLeft = anchor.left + anchor.width / 2 - tooltip.width / 2
      const left = Math.min(
        window.innerWidth - tooltip.width - viewportPadding,
        Math.max(viewportPadding, preferredLeft),
      )

      setStyle({ position: 'fixed', top, left, visibility: 'visible' })
    }

    place()
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => {
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setPinned(false)
        setFocused(false)
        anchorRef.current?.blur()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [open])

  useEffect(() => {
    if (!pinned) return
    const onPointerDown = (event: PointerEvent) => {
      if (!anchorRef.current?.contains(event.target as Node)) setPinned(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [pinned])

  return (
    <span className="info-tooltip-anchor">
      <button
        ref={anchorRef}
        type="button"
        className={`info-tooltip-trigger ${open ? 'is-open' : ''}`}
        aria-label={`Help: ${label}`}
        aria-describedby={open ? tooltipId : undefined}
        aria-expanded={open}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onClick={(event) => {
          if (pinned) {
            setPinned(false)
            setFocused(false)
            event.currentTarget.blur()
          } else {
            setPinned(true)
          }
        }}
      >
        <IconQuestionMark size={11} stroke={2.2} />
      </button>
      {open && createPortal(
        <div ref={tooltipRef} id={tooltipId} role="tooltip" className="info-tooltip" style={style}>
          <div className="info-tooltip-title">{label}</div>
          <div className="info-tooltip-copy">{children}</div>
        </div>,
        document.body,
      )}
    </span>
  )
}
