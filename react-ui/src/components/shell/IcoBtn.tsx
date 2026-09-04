import type { ReactNode } from 'react'

/**
 * Toolbar action rendered as an icon only — the label lives in the tooltip.
 * Shared by every console strip (Stream summaries, Probes) so the unpacked
 * action rails stay one size and one shape across sections.
 */
export function IcoBtn({ title, onClick, disabled, danger, active, children }: {
  title: string
  onClick: () => void
  disabled?: boolean
  danger?: boolean
  /** Marks a view-mode button as the current one (card / list). */
  active?: boolean
  children: ReactNode
}) {
  return (
    <button
      type="button"
      className={`icobtn${danger ? ' danger' : ''}${active ? ' on' : ''}`}
      title={title}
      aria-label={title}
      aria-pressed={active === undefined ? undefined : active}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}
