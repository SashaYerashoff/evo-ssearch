import { useLayoutEffect, useState, type CSSProperties, type ReactNode, type RefObject } from 'react'
import { createPortal } from 'react-dom'

export function FloatingPopover({
  anchorRef,
  children,
  className,
  popoverRef,
  offset = 6,
  matchAnchorWidth = false,
}: {
  anchorRef: RefObject<HTMLElement>
  children: ReactNode
  className: string
  popoverRef?: RefObject<HTMLDivElement>
  offset?: number
  matchAnchorWidth?: boolean
}) {
  const [style, setStyle] = useState<CSSProperties>({ position: 'fixed', visibility: 'hidden' })

  useLayoutEffect(() => {
    const place = () => {
      const rect = anchorRef.current?.getBoundingClientRect()
      if (!rect) return
      setStyle({
        position: 'fixed',
        top: rect.bottom + offset,
        left: rect.left,
        minWidth: matchAnchorWidth ? rect.width : undefined,
      })
    }
    place()
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => {
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [anchorRef, matchAnchorWidth, offset])

  return createPortal(
    <div ref={popoverRef} className={className} style={style}>{children}</div>,
    document.body,
  )
}
