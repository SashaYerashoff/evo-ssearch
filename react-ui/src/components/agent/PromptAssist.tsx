import { useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { IconRefresh, IconPlus, IconX } from '@tabler/icons-react'
import type { Channel } from '../../api/types'

// Guided prompt builder — the STAGE is derived from the input text, so the bubble row
// rebuilds both ways: forward as you type/pick, and backward as you erase. Erase past the
// end of a section and the bubbles fall back to the previous step.
//   [starters] → «channel» bubbles (filter as you type) → [look for] → example queries
const START = 'Search the video-description archive for'
const CONNECTOR = 'look for'
const QUERY_EXAMPLES = [
  'a person loitering near the entrance',
  'a vehicle stopped where it should not be',
  'someone carrying a large bag or package',
  'a person walking on the road or crossing',
  'any unusual activity in the last hour',
]

const STARTERS: { id: string; label: string; flow?: boolean; prompt?: string }[] = [
  { id: 'vlm', label: 'VLM alerts', prompt: 'Show the latest VLM alerts across all channels.' },
  { id: 'status', label: 'Stream status', prompt: 'Report the live stream and capture status for every channel.' },
  { id: 'report', label: 'Video report', prompt: 'Summarize video activity across all channels over the last 24 hours.' },
  { id: 'archive', label: 'Archive evidence', flow: true },
  { id: 'describe', label: 'Describe frame', prompt: 'Describe what the cameras are seeing right now.' },
]

const withChannel = (title: string) => `${START} “${title}” `
const withConnector = (title: string) => `${START} “${title}” ${CONNECTOR} `

const CUSTOM_KEY = 'eva.prompt-assist.custom-queries'
interface Parsed { stage: 0 | 1 | 2 | 3; channel: string; chanTail: string; query: string }
// stage 0 starters · 1 channel · 2 connector · 3 query
function parse(value: string): Parsed {
  if (!value.startsWith(START)) return { stage: 0, channel: '', chanTail: '', query: '' }
  const rest = value.slice(START.length)
  const closed = rest.match(/^\s*“([^”]*)”\s*([\s\S]*)$/)   // a fully quoted channel?
  if (!closed) {
    // still choosing the channel — tail is whatever was typed (minus a stray opening quote)
    return { stage: 1, channel: '', chanTail: rest.replace(/^[\s“"]+/, '').trim().toLowerCase(), query: '' }
  }
  const channel = closed[1]
  const after = closed[2]
  if (/^look for\b/i.test(after)) return { stage: 3, channel, chanTail: '', query: after.replace(/^look for\b\s*/i, '') }
  return { stage: 2, channel, chanTail: '', query: '' }   // channel done, connector not yet
}

export function PromptAssist({ channels, value, onChange, inputRef }: {
  channels: Channel[]
  value: string
  onChange: (text: string) => void
  inputRef: RefObject<HTMLTextAreaElement | null>
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const prevLen = useRef(value.length)

  // user-saved query bubbles (persisted) — reusable across sessions
  const [customQ, setCustomQ] = useState<string[]>(() => {
    try { const a = JSON.parse(localStorage.getItem(CUSTOM_KEY) || '[]'); return Array.isArray(a) ? a.filter((x) => typeof x === 'string').slice(0, 12) : [] } catch { return [] }
  })
  const persistCustom = (list: string[]) => { setCustomQ(list); try { localStorage.setItem(CUSTOM_KEY, JSON.stringify(list)) } catch { /* private mode */ } }

  const { stage, channel, chanTail, query } = parse(value)
  const matches = useMemo(
    () => channels.filter((c) => !chanTail || c.title.toLowerCase().includes(chanTail)),
    [channels, chanTail],
  )
  const freeform = !!chanTail && !channels.some((c) => c.title.toLowerCase() === chanTail)
  const rawTail = () => value.slice(START.length).replace(/^[\s“"]+/, '').replace(/["”\s]+$/, '')

  // stage-3 query suggestions: saved customs + defaults, filtered by what's typed after "look for"
  const queryPool = useMemo(() => [...customQ, ...QUERY_EXAMPLES.filter((e) => !customQ.includes(e))], [customQ])
  const q3 = query.trim().toLowerCase()
  const queryMatches = q3 ? queryPool.filter((e) => e.toLowerCase().includes(q3)) : queryPool
  const canSave = !!q3 && !queryPool.some((e) => e.toLowerCase() === q3)

  // focus always returns to the input (caret at end) — sync for reliability, rAF fixes the caret
  const focusInput = () => {
    const el = inputRef.current
    if (!el) return
    el.focus()
    const n = el.value.length
    el.setSelectionRange(n, n)
  }
  const commit = (text: string) => { onChange(text); focusInput(); requestAnimationFrame(focusInput) }

  const pickStarter = (s: typeof STARTERS[number]) => commit(s.flow ? `${START} ` : (s.prompt || ''))
  const pickChannel = (title: string) => title && commit(withChannel(title))
  const pickConnector = () => commit(withConnector(channel))
  const pickExample = (ex: string) => commit(`${withConnector(channel)}${ex}`)
  const reset = () => commit('')
  const saveCustom = () => { const q = query.trim(); if (!q) return; persistCustom([q, ...customQ.filter((x) => x.toLowerCase() !== q.toLowerCase())].slice(0, 12)); focusInput() }
  const removeCustom = (ex: string) => { persistCustom(customQ.filter((x) => x !== ex)); focusInput() }

  // typed an exact channel name (while typing forward) → auto-advance
  useEffect(() => {
    const grew = value.length > prevLen.current
    prevLen.current = value.length
    if (stage !== 1 || !chanTail || !grew) return
    const exact = channels.find((c) => c.title.toLowerCase() === chanTail)
    if (exact) commit(withChannel(exact.title))
  }, [value]) // eslint-disable-line react-hooks/exhaustive-deps

  // Tab drops the leftmost bubble of the current stage into the input
  const pickFirst = (): boolean => {
    if (stage === 0) { pickStarter(STARTERS[0]); return true }
    if (stage === 1) { if (matches[0]) { pickChannel(matches[0].title); return true } if (freeform) { pickChannel(rawTail()); return true } return false }
    if (stage === 2) { pickConnector(); return true }
    if (queryMatches[0]) pickExample(queryMatches[0])
    else if (canSave) saveCustom()
    return true
  }
  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Tab' && !e.shiftKey) { if (pickFirst()) e.preventDefault() } }
    el.addEventListener('keydown', onKey)
    return () => el.removeEventListener('keydown', onKey)
  }, [stage, value, channels, customQ]) // eslint-disable-line react-hooks/exhaustive-deps

  // mouse wheel scrolls the bubble row horizontally
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.deltaY || el.scrollWidth <= el.clientWidth) return
      el.scrollLeft += e.deltaY
      e.preventDefault()
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  return (
    <div className="agent-assist">
      <div className="aa-head">
        <span className="aa-label">Prompt assist</span>
        {stage > 0 && <button className="aa-reset" onClick={reset} title="Start over"><IconRefresh size={12} /> reset</button>}
      </div>

      {/* one fixed-height scrolling line for every stage — never wraps, never jumps */}
      <div className="aa-row" ref={scrollRef}>
        {stage === 0 && STARTERS.map((s) => (
          <button key={s.id} className={`aa-chip aa-starter ${s.flow ? 'aa-start' : ''}`} onClick={() => pickStarter(s)}>
            {s.label}
          </button>
        ))}

        {stage === 1 && (<>
          {matches.map((c) => (
            <button key={c.id} className="aa-chip aa-trunc" title={c.title} onClick={() => pickChannel(c.title)}>{c.title}</button>
          ))}
          {freeform && <button className="aa-chip aa-start aa-trunc" title={chanTail} onClick={() => pickChannel(rawTail())}>Use “{chanTail}”</button>}
        </>)}

        {stage === 2 && <button className="aa-chip aa-start" onClick={pickConnector}>{CONNECTOR}…</button>}

        {stage === 3 && (<>
          {queryMatches.map((ex) => (
            customQ.includes(ex) ? (
              <span key={ex} className="aa-chip aa-cc aa-trunc-wide" title={ex}>
                <button className="aa-cc-pick" onClick={() => pickExample(ex)}>{ex}</button>
                <button className="aa-cc-x" title="Remove saved bubble" onClick={() => removeCustom(ex)}><IconX size={11} /></button>
              </span>
            ) : (
              <button key={ex} className="aa-chip aa-trunc aa-trunc-wide" title={ex} onClick={() => pickExample(ex)}>{ex}</button>
            )
          ))}
          {canSave && (
            <button className="aa-chip aa-start aa-add" title={`Save “${query.trim()}” as a reusable bubble`} onClick={saveCustom}>
              <IconPlus size={13} /> save
            </button>
          )}
        </>)}
      </div>
    </div>
  )
}
