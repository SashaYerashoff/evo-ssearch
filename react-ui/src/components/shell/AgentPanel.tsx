import { useRef, useState, useEffect, useCallback } from 'react'
import {
  IconSparkles, IconX, IconArrowUp,
  IconPlus, IconTrash, IconPhoto, IconAlertTriangle, IconPencil, IconChevronRight,
  IconChevronDown, IconWand, IconCpu, IconVideo, IconDeviceGamepad2, IconArrowAutofitWidth, IconHistory,
  IconMaximize, IconMinimize, IconPlayerStop,
} from '@tabler/icons-react'
import type { Channel, ArchiveFilters } from '../../api/types'
import { agentSubmissionText, streamAgent, agentApi, type AgentEvent, type AgentSession, type AgentSkill } from '../../api/agent'
import { renderMarkdown } from '../agent/markdown'
import { ActionCard, type ToolAction } from '../agent/ActionCard'

export interface AgentAction { name: string; args: any; done: boolean; error?: string; result?: any }

const LS_SESSION = 'evs_agent_session_id'
const LS_WIDTH = 'evs_agent_half_width'
const MIN_W = 420          // agent panel never narrower than this
// keep enough console behind the panel for the toolbar to stay one line (~620px)
// and a full 5-wide row of result cards (rail + padding + 5 × 172 + gaps)
const MIN_CONSOLE = 1010
const maxWidth = () => Math.max(MIN_W, window.innerWidth - MIN_CONSOLE)
// effective half-width: use the saved value or a default a touch under half, always clamped
const DEFAULT_FRAC = 0.42
const effHalfWidth = (saved: number | null) =>
  Math.round(Math.max(MIN_W, Math.min(saved ?? window.innerWidth * DEFAULT_FRAC, maxWidth())))
const SUGGESTIONS = [
  'Search the archive for a man sitting on a chair',
  'Find people near the entrance in the last hour',
  'What needs attention right now?',
  'List active channels',
]

interface Note { id: number; message: string }
interface Msg {
  role: 'user' | 'assistant'
  text: string
  ts?: number
  imageB64?: string
  actions?: ToolAction[]
  notes?: Note[]
  status?: string
  streaming?: boolean
  error?: string
}

const labelTool = (n: string) => (n || '').replace(/_/g, ' ')

// format a session timestamp (backend may send seconds, ms, or ISO)
function fmtTs(v?: string | number): string {
  if (v == null) return ''
  const n = typeof v === 'number' ? v : Number(v)
  const d = isFinite(n) ? new Date(n < 1e12 ? n * 1000 : n) : new Date(String(v))
  return isNaN(d.getTime()) ? '' : d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
// readable session label: the real first message, unless it's the injected operator preamble
function sessionLabel(s: AgentSession): string {
  const t = (s.title || '').trim().replace(/^Operator request:\s*/i, '')
  if (t && !/^OPERATOR MODE/i.test(t)) return t
  return fmtTs(s.updated_at) || 'Untitled session'
}

// Build an "operator console context" preamble from the current archive filters so the
// server-side agent respects the operator's selected channel / time window / source.
function buildAgentContext(f: ArchiveFilters | null | undefined, channels: Channel[], operator: boolean): string {
  const lines: string[] = []
  if (operator) {
    lines.push(
      'OPERATOR MODE — you are operating the operator\'s live console UI (currently the Archive screen). Rules:',
      '- You MUST answer by calling tools that drive the console; NEVER answer archive/detection/channel questions from memory or from earlier messages.',
      '- Your tool calls are mirrored onto the console in real time (filters change, frames load) — so always act via a tool first, then summarise briefly.',
    )
  }
  if (f) {
    const now = Date.now()
    const hours = f.hours ?? '24'
    const until = f.untilMs ? Number(f.untilMs) : now
    const since = f.sinceMs ? Number(f.sinceMs) : (hours === '0' ? 0 : until - Number(hours) * 3600_000)
    const chTitle = f.channelId ? (channels.find((c) => String(c.id) === String(f.channelId))?.title || `channel ${f.channelId}`) : null
    const timeLabel = (f.sinceMs || f.untilMs) ? 'custom range' : hours === '0' ? 'all time' : `last ${hours}h`
    lines.push(
      'Console filter state (use these values for archive tool calls):',
      `- channel: ${chTitle ? `${chTitle} (channel_id=${f.channelId})` : 'all channels'}`,
      `- time window: ${timeLabel} → since_ms=${since}, until_ms=${until}`,
      `- source: ${f.source || 'all'}`,
      `- probe: ${f.source === 'probe' && f.probeId ? `${f.probeId} (probe_id)` : 'all'}`,
      'Tool guidance:',
      '- To show / list / browse archived frames or "the archive", call get_detections with the window above — it returns the stored detection frames. Do NOT use search_archive for this.',
      '- Use search_archive ONLY when the operator gives a specific visual query to match (an object, action, or scene).',
      '- Always pass since_ms/until_ms and channel_id (when set) explicitly; never fall back to a default 24h window.',
      '- If the operator names a different time range in their request below (e.g. "all time", "last week"), that overrides the window above.',
    )
  }
  return lines.join('\n')
}

export function AgentPanel({
  open, full, onClose, onToggleFull, channels, archiveFilters, onAction, onBusyChange,
  canManageModels, canManageSkills,
}: {
  open: boolean
  full: boolean
  onClose: () => void
  onToggleFull: () => void
  channels: Channel[]
  archiveFilters?: ArchiveFilters | null
  onAction: (a: AgentAction) => void
  onBusyChange?: (busy: boolean) => void
  canManageModels: boolean
  canManageSkills: boolean
}) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [imageB64, setImageB64] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [sessions, setSessions] = useState<AgentSession[]>([])
  const [curSession, setCurSession] = useState<string | null>(null)
  const [model, setModel] = useState('')
  const [modelDefault, setModelDefault] = useState('')
  const [skills, setSkills] = useState<AgentSkill[]>([])
  const [streams, setStreams] = useState<any[]>([])
  const [lightbox, setLightbox] = useState<{ url: string; title: string } | null>(null)
  const [skillModal, setSkillModal] = useState<{ mode: 'create' | 'edit'; name: string; slug: string; content: string } | null>(null)
  const [openMenu, setOpenMenu] = useState<'history' | 'skills' | 'model' | 'streams' | null>(null)
  const [operatorMode, setOperatorMode] = useState(true)
  const [halfWidth, setHalfWidth] = useState<number | null>(() => {
    const v = Number(localStorage.getItem(LS_WIDTH))
    return isFinite(v) && v >= MIN_W ? v : null
  })
  const [resizing, setResizing] = useState(false)

  const sessionRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const dockRef = useRef<HTMLDivElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const idRef = useRef(1)
  const loadedRef = useRef(false)

  const [, force] = useState(0)

  useEffect(() => { bodyRef.current?.scrollTo({ top: 1e9, behavior: 'smooth' }) }, [msgs, busy])
  useEffect(() => { onBusyChange?.(busy) }, [busy, onBusyChange])
  useEffect(() => () => abortRef.current?.abort(), [])

  // keep the panel width valid when the browser window is resized
  useEffect(() => {
    const onResize = () => { setHalfWidth((w) => (w == null ? w : Math.min(w, maxWidth()))); force((n) => n + 1) }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // close strip dropdowns on outside click
  useEffect(() => {
    if (!openMenu) return
    const onDown = (e: MouseEvent) => { if (!(e.target as HTMLElement)?.closest('.ag-strip')) setOpenMenu(null) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [openMenu])

  // first-open bootstrap
  useEffect(() => {
    if (!open || loadedRef.current) return
    loadedRef.current = true
    refreshSessions()
    agentApi.getConfig().then((c) => { setModel(c.model || c.default_model || ''); setModelDefault(c.default_model || '') }).catch(() => {})
    agentApi.skills().then((r) => setSkills(r.skills || [])).catch(() => {})
    refreshStreams()
    const last = localStorage.getItem(LS_SESSION)
    if (last) openSession(last)
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  function refreshSessions() { agentApi.sessions().then((r) => setSessions(r.sessions || [])).catch(() => {}) }
  function refreshStreams() { agentApi.streams().then((r) => setStreams(r.video_streams || [])).catch(() => {}) }

  const patchLast = useCallback((fn: (m: Msg) => Msg) =>
    setMsgs((all) => { const c = [...all]; const i = c.length - 1; if (i >= 0 && c[i].role === 'assistant') c[i] = fn(c[i]); return c }), [])

  const patchAction = useCallback((actionId: number, fn: (a: ToolAction) => ToolAction) =>
    setMsgs((all) => all.map((m) => m.actions?.some((a) => a.id === actionId)
      ? { ...m, actions: m.actions!.map((a) => (a.id === actionId ? fn(a) : a)) } : m)), [])

  async function openSession(id: string) {
    if (busy) return
    try {
      const data = await agentApi.session(id)
      setCurSession(id); sessionRef.current = id; localStorage.setItem(LS_SESSION, id)
      setMsgs((data.messages || []).map((m) => ({ role: m.role, text: m.content, ts: undefined })))
    } catch { /* ignore */ }
  }
  async function deleteSession(id: string) {
    try {
      await agentApi.deleteSession(id)
      if (curSession === id) { setCurSession(null); sessionRef.current = null; localStorage.removeItem(LS_SESSION); setMsgs([]) }
      refreshSessions()
    } catch { /* ignore */ }
  }
  function newSession() {
    if (busy) return
    setCurSession(null); sessionRef.current = null; localStorage.removeItem(LS_SESSION); setMsgs([]); setErr(null)
  }

  async function send(text: string) {
    const q = agentSubmissionText(text, imageB64)
    if (!q || busy) return
    setErr(null); setInput('')
    const img = imageB64; setImageB64(null)
    setMsgs((m) => [...m,
      { role: 'user', text: q, imageB64: img || undefined },
      { role: 'assistant', text: '', actions: [], notes: [], status: 'Thinking…', streaming: true },
    ])
    setBusy(true)
    const ctrl = new AbortController(); abortRef.current = ctrl

    // prepend live console context so the agent respects the operator's filters / drives the UI
    const ctx = buildAgentContext(archiveFilters, channels, operatorMode)
    const wire = ctx ? `${ctx}\n\nOperator request: ${q}` : q

    try {
      await streamAgent(wire, { sessionId: sessionRef.current, imageB64: img, operatorMode, signal: ctrl.signal }, (e: AgentEvent) => {
        if ((e.type === 'session' || e.type === 'done') && e.session_id) {
          sessionRef.current = e.session_id; setCurSession(e.session_id); localStorage.setItem(LS_SESSION, e.session_id)
        }
        if ((e.type === 'text' || e.type === 'token') && e.content) patchLast((m) => ({ ...m, text: m.text + e.content, status: '' }))
        else if (e.type === 'tool_call') {
          patchLast((m) => ({ ...m, status: `Running ${labelTool(e.name || '')}…` }))
          onAction({ name: e.name || '', args: e.args || {}, done: false })
        } else if (e.type === 'tool_result') {
          const planId = e.result?.approval?.plan_id || (e.result?.status === 'preview' ? e.result?.plan_id : null) || null
          const act: ToolAction = { id: ++idRef.current, name: e.name || '', result: e.result, error: e.error, planId }
          patchLast((m) => ({ ...m, actions: [...(m.actions || []), act], status: '' }))
          onAction({ name: e.name || '', args: {}, done: true, error: e.error, result: e.result })
        } else if (e.type === 'tool_progress') {
          patchLast((m) => ({ ...m, notes: [...(m.notes || []), { id: ++idRef.current, message: e.message || 'Working…' }], status: e.message || 'Working…' }))
        } else if (e.type === 'heartbeat') patchLast((m) => ({ ...m, status: m.text ? '' : 'Still working…' }))
        else if (e.type === 'error') { setErr(e.message || 'Agent error'); patchLast((m) => ({ ...m, error: e.message })) }
      })
    } catch (ex: any) {
      if (ex?.name !== 'AbortError') setErr(ex?.message || 'Agent request failed')
    } finally {
      setBusy(false); abortRef.current = null
      patchLast((m) => ({ ...m, streaming: false, status: '' }))
      refreshSessions()
    }
  }

  async function applyPlan(a: ToolAction) {
    if (!a.planId) return
    patchAction(a.id, (x) => ({ ...x, applying: true }))
    try {
      const res = await agentApi.executePlan(a.planId, sessionRef.current)
      if (!res.success) throw new Error(res.error || 'Apply failed')
      patchAction(a.id, (x) => ({ ...x, applying: false, applied: true }))
      // append the receipt card
      if (res.result) patchLast((m) => ({ ...m, actions: [...(m.actions || []), { id: ++idRef.current, name: a.name, result: res.result, applied: true }] }))
      refreshStreams()
    } catch (ex: any) {
      patchAction(a.id, (x) => ({ ...x, applying: false, error: ex?.message || 'Apply failed' }))
    }
  }

  function onFile(f: File) {
    const r = new FileReader()
    r.onload = () => { const s = String(r.result || ''); const b64 = s.includes(',') ? s.split(',')[1] : s; setImageB64(b64) }
    r.readAsDataURL(f)
  }

  async function saveSkill() {
    if (!skillModal) return
    const b = { name: skillModal.name.trim(), slug: skillModal.slug.trim(), content: skillModal.content }
    if (!b.name || !b.slug) return
    try {
      const res = skillModal.mode === 'create' ? await agentApi.createSkill(b) : await agentApi.updateSkill(b.slug, b)
      if (res.error) throw new Error(res.error)
      setSkillModal(null); agentApi.skills().then((r) => setSkills(r.skills || []))
    } catch (ex: any) { setErr(ex?.message || 'Failed to save skill') }
  }
  function runSkill(s: AgentSkill) {
    const prompt = `Use playbook "${s.slug}" for this operator request:\n${input.trim()}`.trim()
    send(prompt)
  }
  async function editSkill(slug: string) {
    try { const s = await agentApi.skill(slug); setSkillModal({ mode: 'edit', name: s.name || '', slug: s.slug, content: s.content || '' }) } catch { /* ignore */ }
  }

  // drag the left edge to resize the half-screen panel; value is remembered
  function startResize(e: React.MouseEvent) {
    e.preventDefault()
    const startX = e.clientX
    const startW = dockRef.current?.getBoundingClientRect().width ?? window.innerWidth * 0.5
    setResizing(true)
    const onMove = (ev: MouseEvent) => {
      setHalfWidth(Math.min(maxWidth(), Math.max(MIN_W, startW + (startX - ev.clientX))))
    }
    const onUp = () => {
      setResizing(false)
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      setHalfWidth((w) => { if (w != null) localStorage.setItem(LS_WIDTH, String(Math.round(w))); return w })
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }
  function resetWidth() { setHalfWidth(null); localStorage.removeItem(LS_WIDTH) }

  const empty = msgs.length === 0

  const composer = (
    <div className="agent-composer">
      {imageB64 && (
        <div className="ag-img-preview">
          <img src={`data:image/jpeg;base64,${imageB64}`} alt="attached" />
          <button className="ag-img-clear" onClick={() => setImageB64(null)} title="Remove"><IconX size={13} /></button>
        </div>
      )}
      <div className="agent-input">
        <textarea rows={1} placeholder="Ask the agent…" value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
          disabled={busy} />
        {busy ? (
          <button className="agent-send agent-stop" title="Stop current request" onClick={() => abortRef.current?.abort()}>
            <IconPlayerStop size={18} />
          </button>
        ) : (
          <button className="agent-send" title="Send" disabled={!input.trim() && !imageB64} onClick={() => send(input)}>
            <IconArrowUp size={18} />
          </button>
        )}
      </div>
      <div className="agent-chips">
        <label className="agent-chip agent-chip-file"><IconPhoto size={13} /> Image
          <input type="file" accept="image/*" style={{ display: 'none' }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.currentTarget.value = '' }} />
        </label>
      </div>
    </div>
  )

  const strip = (
    <div className="ag-strip">
      {/* New session */}
      <button className="ag-newbtn" onClick={newSession} title="Start a new session">
        <IconPlus size={15} /> New session
      </button>

      {/* Session history */}
      <div className="ag-drop">
        <button className={`ag-drop-btn ${openMenu === 'history' ? 'on' : ''}`}
          onClick={() => { const n = openMenu === 'history' ? null : 'history'; if (n) refreshSessions(); setOpenMenu(n) }}>
          <IconHistory size={14} /> History <span className="ag-drop-count">{sessions.length}</span> <IconChevronDown size={13} />
        </button>
        {openMenu === 'history' && (
          <div className="ag-drop-pop">
            <div className="ag-drop-head"><span>Sessions</span>
              <button className="ag-mini-btn" title="New session" onClick={() => { setOpenMenu(null); newSession() }}><IconPlus size={15} /></button>
            </div>
            <div className="ag-drop-list">
              {sessions.length === 0 && <div className="ag-empty">No sessions yet</div>}
              {sessions.map((s) => (
                <div key={s.id} className={`ag-session ${s.id === curSession ? 'active' : ''}`} onClick={() => { setOpenMenu(null); openSession(s.id) }}>
                  <div className="ag-session-title">{sessionLabel(s)}</div>
                  <div className="ag-session-meta">
                    <span>{s.message_count || 0} msg{s.updated_at ? ` · ${fmtTs(s.updated_at)}` : ''}</span>
                    <button className="ag-session-del" title="Delete" onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }}><IconTrash size={13} /></button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Operator Mode — forces the agent to act via tools & drive the console */}
      <button
        className={`ag-op-toggle ${operatorMode ? 'on' : ''}`}
        onClick={() => setOperatorMode((v) => !v)}
        title={operatorMode ? 'Operator Mode ON — agent is forced to use tools and drive the console' : 'Operator Mode OFF — agent answers freely'}
      >
        <IconDeviceGamepad2 size={14} /> Operator {operatorMode ? 'ON' : 'OFF'}
      </button>

      {/* Skills */}
      <div className="ag-drop">
        <button className={`ag-drop-btn ${openMenu === 'skills' ? 'on' : ''}`} onClick={() => setOpenMenu(openMenu === 'skills' ? null : 'skills')}>
          <IconWand size={14} /> Skills <span className="ag-drop-count">{skills.length}</span> <IconChevronDown size={13} />
        </button>
        {openMenu === 'skills' && (
          <div className="ag-drop-pop">
            <div className="ag-drop-head"><span>Skills</span>
              {canManageSkills && <button className="ag-mini-btn" title="New skill" onClick={() => { setOpenMenu(null); setSkillModal({ mode: 'create', name: '', slug: '', content: '' }) }}><IconPlus size={15} /></button>}
            </div>
            <div className="ag-drop-list">
              {skills.length === 0 && <div className="ag-empty">No skills</div>}
              {skills.map((s) => (
                <div key={s.slug} className="ag-skill">
                  <button className="ag-skill-run" onClick={() => { setOpenMenu(null); runSkill(s) }} title={s.summary || s.slug}>
                    <IconChevronRight size={13} /> {s.name || s.slug}
                  </button>
                  {canManageSkills && <button className="ag-mini-btn" title="Edit" onClick={() => { setOpenMenu(null); editSkill(s.slug) }}><IconPencil size={13} /></button>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Model */}
      <div className="ag-drop">
        <button className={`ag-drop-btn ${openMenu === 'model' ? 'on' : ''}`} onClick={() => setOpenMenu(openMenu === 'model' ? null : 'model')}>
          <IconCpu size={14} /> <span className="ag-drop-model">{model || 'default'}</span> <IconChevronDown size={13} />
        </button>
        {openMenu === 'model' && (
          <div className="ag-drop-pop">
            <div className="ag-drop-head"><span>Agent model</span></div>
            <div className="ag-model">
              <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="default" title={modelDefault ? `Default: ${modelDefault}` : ''} disabled={busy || !canManageModels} />
              {canManageModels && <button className="ag-mini-btn wide" disabled={busy} onClick={() => agentApi.saveConfig(model.trim()).then((c) => { setModel(c.model || model); setOpenMenu(null) }).catch(() => setErr('Failed to set model'))}>Apply</button>}
            </div>
            {modelDefault && <div className="ag-drop-note">Default: {modelDefault}</div>}
          </div>
        )}
      </div>

      {/* Video streams */}
      <div className="ag-drop">
        <button className={`ag-drop-btn ${openMenu === 'streams' ? 'on' : ''}`} onClick={() => setOpenMenu(openMenu === 'streams' ? null : 'streams')}>
          <IconVideo size={14} /> Streams <span className="ag-drop-count">{streams.length}</span> <IconChevronDown size={13} />
        </button>
        {openMenu === 'streams' && (
          <div className="ag-drop-pop">
            <div className="ag-drop-head"><span>Video streams</span>
              <button className="ag-mini-btn" title="Refresh" onClick={refreshStreams}><IconChevronRight size={15} /></button>
            </div>
            <div className="ag-drop-list">
              {streams.length === 0 && <div className="ag-empty">No active streams</div>}
              {streams.map((s, i) => {
                const on = !!(s.running ?? s.active ?? s.is_running)
                const warn = !!(s.error || s.warning)
                const name = `CH ${s.channel_id ?? s.channel ?? s.id ?? '?'}${s.model ? ` · ${s.model}` : ''}`
                return (
                  <div key={i} className="ag-probe">
                    <span className={`ag-probe-dot ${warn ? 'warn' : on ? 'on' : 'off'}`} />
                    <span className="ag-probe-name">{name}</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )

  const chat = (
    <section className="ag-chat">
      {strip}
      <div className="ag-messages" ref={bodyRef}>
        {empty && (
          <>
            <div className="agent-hello">
              <div className="agent-badge lg"><IconSparkles size={22} /></div>
              <div style={{ fontSize: 15, fontWeight: 600 }}>EVA Agent</div>
              <div className="brand-sub" style={{ maxWidth: 300 }}>
                Ask about video summaries, VLM alerts, coverage gaps, archive evidence and live frames. When it searches, watch it drive the console.
              </div>
            </div>
            <div className="agent-suggest">
              {SUGGESTIONS.map((s) => <button key={s} className="agent-chip" onClick={() => send(s)}>{s}</button>)}
            </div>
          </>
        )}

        {msgs.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <div className="chat-role">{m.role === 'user' ? 'Operator' : 'EVA Agent'}</div>
            {m.role === 'user' ? (
              <div className="chat-bubble">
                {m.imageB64 && <img className="chat-img" src={`data:image/jpeg;base64,${m.imageB64}`} alt="attached" />}
                {m.text}
              </div>
            ) : (
              <div className="chat-assistant">
                {m.text
                  ? <div className="chat-bubble md" dangerouslySetInnerHTML={{ __html: renderMarkdown(m.text) }} />
                  : m.streaming && <div className="chat-status"><span className="action-dots"><i /><i /><i /></span>{m.status || 'Thinking…'}</div>}

                {((m.actions?.length ?? 0) > 0 || (m.notes?.length ?? 0) > 0) && (
                  <details className="ag-trace" open>
                    <summary>Research trace · {(m.actions?.length ?? 0) + (m.notes?.length ?? 0)} step{((m.actions?.length ?? 0) + (m.notes?.length ?? 0)) === 1 ? '' : 's'}</summary>
                    <div className="ag-trace-body">
                      {m.notes?.map((n) => (
                        <div key={n.id} className="ag-note"><span className="ag-note-badge">In progress</span>{n.message}</div>
                      ))}
                      {m.actions?.map((a) => (
                        <ActionCard key={a.id} action={a} onThumb={(url, title) => setLightbox({ url, title })} onApply={applyPlan} />
                      ))}
                    </div>
                  </details>
                )}
                {m.error && <div className="chat-error"><IconAlertTriangle size={14} /> {m.error}</div>}
              </div>
            )}
          </div>
        ))}
        {err && <div className="chat-error"><IconAlertTriangle size={14} /> {err}</div>}
      </div>
      {composer}
    </section>
  )

  return (
    <div ref={dockRef} className={`agent-dock ${open ? 'open' : ''} ${full ? 'full' : ''} ${resizing ? 'resizing' : ''}`} data-agent
      style={open && !full ? { width: effHalfWidth(halfWidth) } : undefined}>
      {open && !full && <div className="agent-resize" onMouseDown={startResize} title="Drag to resize" />}
      <div className="agent-head">
        <div className="agent-head-title">
          <span className="agent-badge"><IconSparkles size={15} /></span>
          <div>
            <div className="agent-title">EVA Agent</div>
            <div className="brand-sub">Watches {channels.length} channels · operates the console for you</div>
          </div>
        </div>
        <div className="agent-head-btns">
          <button className="modal-close" onClick={onToggleFull} title={full ? 'Exit full screen' : 'Open full screen'}>
            {full ? <IconMinimize size={17} /> : <IconMaximize size={17} />}
          </button>
          <button className="modal-close" onClick={resetWidth} disabled={halfWidth == null} title="Reset to default width">
            <IconArrowAutofitWidth size={17} />
          </button>
          <button className="modal-close" onClick={onClose} title="Close agent"><IconX size={18} /></button>
        </div>
      </div>

      <div className={`ag-cols ${full ? 'full' : ''}`}>
        {full && (
          <aside className="ag-rail">
            <div className="ag-rail-head"><span>Sessions</span></div>
            <button className="ag-newbtn full" onClick={newSession} title="Start a new session">
              <IconPlus size={15} /> New session
            </button>
            <div className="ag-session-list">
              {sessions.length === 0 && <div className="ag-empty">No sessions yet</div>}
              {sessions.map((s) => (
                <div key={s.id} className={`ag-session ${s.id === curSession ? 'active' : ''}`} onClick={() => openSession(s.id)}>
                  <div className="ag-session-title">{s.title || 'Untitled session'}</div>
                  <div className="ag-session-meta">
                    <span>{s.message_count || 0} msg</span>
                    <button className="ag-session-del" title="Delete" onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }}><IconTrash size={13} /></button>
                  </div>
                </div>
              ))}
            </div>
          </aside>
        )}

        {chat}
      </div>

      {lightbox && (
        <div className="ag-lightbox" onClick={() => setLightbox(null)}>
          <img src={lightbox.url} alt={lightbox.title} onClick={(e) => e.stopPropagation()} />
          <div className="ag-lightbox-cap">{lightbox.title}</div>
        </div>
      )}

      {skillModal && canManageSkills && (
        <div className="scrim" onClick={() => setSkillModal(null)}>
          <div className="modal" style={{ maxWidth: 560 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="modal-title">{skillModal.mode === 'create' ? 'Create skill' : 'Edit skill'}</div>
              <button className="modal-close" onClick={() => setSkillModal(null)}><IconX size={18} /></button>
            </div>
            <div className="modal-body">
              <div className="wform">
                <div className="wfield"><label>Name</label>
                  <input value={skillModal.name} onChange={(e) => setSkillModal({ ...skillModal, name: e.target.value })} placeholder="Skill name" />
                </div>
                <div className="wfield"><label>Slug</label>
                  <input value={skillModal.slug} disabled={skillModal.mode === 'edit'}
                    onChange={(e) => setSkillModal({ ...skillModal, slug: e.target.value.replace(/[^a-z0-9_-]/gi, '-').toLowerCase() })} placeholder="slug" />
                </div>
                <div className="wfield"><label>Content</label>
                  <textarea rows={8} value={skillModal.content} onChange={(e) => setSkillModal({ ...skillModal, content: e.target.value })}
                    placeholder={'# Skill title\nInstructions…'} style={{ resize: 'vertical', fontFamily: 'inherit' }} />
                </div>
                <button className="btn primary" style={{ justifyContent: 'center' }} onClick={saveSkill}>Save skill</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
