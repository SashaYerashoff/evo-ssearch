import { useState, useRef, useEffect, type RefObject } from 'react'
import { IconX, IconChevronLeft, IconChevronRight, IconTrash, IconCheck } from '@tabler/icons-react'
import { FloatingPopover } from '../shell/FloatingPopover'

const WD = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
const pad = (n: number) => String(n).padStart(2, '0')
const toInput = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
const fromMs = (ms?: string) => (ms ? new Date(Number(ms)) : null)

function monthGrid(view: Date): Date[] {
  const first = new Date(view.getFullYear(), view.getMonth(), 1)
  let start = first.getDay() - 1
  if (start < 0) start = 6 // Monday-first
  const cells: Date[] = []
  for (let i = 0; i < 42; i++) { const d = new Date(first); d.setDate(1 - start + i); cells.push(d) }
  return cells
}

function Calendar({ view, onView, selected, onPick }: { view: Date; onView: (d: Date) => void; selected: Date | null; onPick: (d: Date) => void }) {
  const cells = monthGrid(view)
  const sameDay = (a: Date, b: Date | null) => !!b && a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()
  return (
    <div className="cal">
      <div className="cal-head">
        <button type="button" className="cal-nav" onClick={() => onView(new Date(view.getFullYear(), view.getMonth() - 1, 1))}><IconChevronLeft size={16} /></button>
        <span>{view.toLocaleDateString([], { month: 'long', year: 'numeric' })}</span>
        <button type="button" className="cal-nav" onClick={() => onView(new Date(view.getFullYear(), view.getMonth() + 1, 1))}><IconChevronRight size={16} /></button>
      </div>
      <div className="cal-grid cal-wd">{WD.map((w) => <span key={w}>{w}</span>)}</div>
      <div className="cal-grid">
        {cells.map((d, i) => (
          <button key={i} type="button"
            className={`cal-day ${d.getMonth() !== view.getMonth() ? 'out' : ''} ${sameDay(d, selected) ? 'sel' : ''}`}
            onClick={() => onPick(d)}>{d.getDate()}</button>
        ))}
      </div>
    </div>
  )
}

export function DateRangeModal({ anchorRef, sinceMs, untilMs, onApply, onClear, onClose }: {
  anchorRef: RefObject<HTMLElement>
  sinceMs?: string
  untilMs?: string
  onApply: (since?: string, until?: string) => void
  onClear: () => void
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const now = new Date()
  const initFrom = fromMs(sinceMs) || new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
  const initTo = fromMs(untilMs) || now
  const [from, setFrom] = useState(toInput(initFrom))
  const [to, setTo] = useState(toInput(initTo))
  const [fromView, setFromView] = useState(new Date(initFrom.getFullYear(), initFrom.getMonth(), 1))
  const [toView, setToView] = useState(new Date(initTo.getFullYear(), initTo.getMonth(), 1))
  const fromValue = from ? new Date(from).getTime() : null
  const toValue = to ? new Date(to).getTime() : null
  const invalidRange = fromValue != null && toValue != null && fromValue > toValue

  // close on outside click — but ignore the calendar toggle button
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node
      if (!ref.current?.contains(target) && !anchorRef.current?.contains(target)) onClose()
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [anchorRef, onClose])

  const pickDate = (val: string, setVal: (s: string) => void) => (d: Date) => {
    const cur = val ? new Date(val) : new Date()
    setVal(toInput(new Date(d.getFullYear(), d.getMonth(), d.getDate(), cur.getHours() || 0, cur.getMinutes() || 0)))
  }
  const apply = () => {
    if (invalidRange) return
    onApply(fromValue != null ? String(fromValue) : undefined, toValue != null ? String(toValue) : undefined)
  }

  return (
    <FloatingPopover anchorRef={anchorRef} popoverRef={ref} className="daterange-pop floating-popover" offset={8}>
      <div className="dr-head">
        <span className="dr-title">Pick date range</span>
        <button className="modal-close" onClick={onClose}><IconX size={16} /></button>
      </div>
      <div className="dr-cols">
        <div className="dr-col">
          <div className="dr-label">From</div>
          <input type="datetime-local" value={from} onChange={(e) => { setFrom(e.target.value); if (e.target.value) setFromView(new Date(e.target.value)) }} />
          <Calendar view={fromView} onView={setFromView} selected={from ? new Date(from) : null} onPick={pickDate(from, setFrom)} />
        </div>
        <div className="dr-col">
          <div className="dr-label">To</div>
          <input type="datetime-local" value={to} onChange={(e) => { setTo(e.target.value); if (e.target.value) setToView(new Date(e.target.value)) }} />
          <Calendar view={toView} onView={setToView} selected={to ? new Date(to) : null} onPick={pickDate(to, setTo)} />
        </div>
      </div>
      {invalidRange && <div className="dr-error" role="alert">The end of the period must be after its start.</div>}
      <div className="dr-actions">
        <button className="mon-btn" onClick={() => { onClear(); onClose() }}><IconTrash size={15} /> Clear range</button>
        <button className="mon-btn accent" disabled={invalidRange} onClick={() => { apply(); onClose() }}><IconCheck size={15} /> Apply range</button>
      </div>
    </FloatingPopover>
  )
}
