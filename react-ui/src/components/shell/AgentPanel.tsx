import { useRef, useState, useEffect, useCallback } from 'react'
import {
  IconSparkles, IconX, IconArrowUp,
  IconPlus, IconTrash, IconPaperclip, IconAlertTriangle, IconPencil, IconChevronRight,
  IconChevronDown, IconWand, IconCpu, IconVideo, IconDeviceGamepad2, IconArrowAutofitWidth, IconHistory,
  IconMaximize, IconMinimize, IconPlayerStop,
} from '@tabler/icons-react'
import type { Channel, ArchiveFilters } from '../../api/types'
import { agentSubmissionText, streamAgent, agentApi, type AgentEvent, type AgentSession, type AgentSkill } from '../../api/agent'
import {
  buildAgentConsoleContext,
  normalizeConsoleUiEffects,
  type ConsoleUiEffect,
} from '../../ui-effects/consoleEffects'
import { renderMarkdown } from '../agent/markdown'
import { ActionCard, type ToolAction } from '../agent/ActionCard'
import {
  maxTranscriptActionId,
  restoreAgentTranscript,
} from '../agent/agentTranscript'
import { PromptAssist } from '../agent/PromptAssist'
import {
  AGENT_WIDTH_PRESET_STORAGE_KEY,
  MIN_AGENT_WIDTH,
  agentDragGeometry,
  archiveColumnsForAgentWidth,
  agentWidthPresets,
  closestAgentWidthPresetIndex,
  maxAgentPanelWidth,
  nextAgentWidthPresetIndex,
} from './agentWidthPresets'

export interface AgentAction { name: string; args: any; done: boolean; error?: string; result?: any }

const LS_SESSION = 'evs_agent_session_id'
const LS_WIDTH = 'evs_agent_half_width'
const validPresetIndex = (index: number, count: number) => Math.max(0, Math.min(index, count - 1))
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

function ResearchTrace({
  message,
  onThumb,
  onApply,
}: {
  message: Msg
  onThumb: (url: string, title: string) => void
  onApply: (action: ToolAction) => void
}) {
  // The operator owns this disclosure state. Streaming updates must not force
  // a trace open or closed after the user has touched it.
  const [traceOpen, setTraceOpen] = useState(true)
  const stepCount = (message.actions?.length ?? 0) + (message.notes?.length ?? 0)
  return (
    <details
      className="ag-trace"
      open={traceOpen}
      onToggle={(event) => setTraceOpen(event.currentTarget.open)}
    >
      <summary>Research trace · {stepCount} step{stepCount === 1 ? '' : 's'}</summary>
      <div className="ag-trace-body">
        {message.notes?.map((note) => (
          <div key={note.id} className="ag-note"><span className="ag-note-badge">In progress</span>{note.message}</div>
        ))}
        {message.actions?.map((action) => (
          <ActionCard key={action.id} action={action} onThumb={onThumb} onApply={onApply} />
        ))}
      </div>
    </details>
  )
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

export function AgentPanel({
  open, full, onClose, onToggleFull, section, channels, archiveFilters, onUiEffects, onBusyChange,
  onLayoutPresetChange, onLayoutPresetCommit, canManageModels, canManageSkills,
}: {
  open: boolean
  full: boolean
  onClose: () => void
  onToggleFull: () => void
  section: string
  channels: Channel[]
  archiveFilters?: ArchiveFilters | null
  onUiEffects: (effects: ConsoleUiEffect[], result: unknown) => void
  onBusyChange?: (busy: boolean) => void
  onLayoutPresetChange?: (archiveColumns: number) => void
  onLayoutPresetCommit?: (archiveColumns: number) => void
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
  const [widthPresetIndex, setWidthPresetIndex] = useState(() => {
    const presets = agentWidthPresets(window.innerWidth)
    const storedValue = localStorage.getItem(AGENT_WIDTH_PRESET_STORAGE_KEY)
    const storedIndex = Number(storedValue)
    if (storedValue != null && Number.isInteger(storedIndex) && storedIndex >= 0 && storedIndex < presets.length) return storedIndex
    const legacyWidth = Number(localStorage.getItem(LS_WIDTH))
    return isFinite(legacyWidth) && legacyWidth >= MIN_AGENT_WIDTH
      ? closestAgentWidthPresetIndex(legacyWidth, presets)
      : 0
  })
  const [halfWidth, setHalfWidth] = useState(() => (
    agentWidthPresets(window.innerWidth)[widthPresetIndex].width
  ))
  const [resizing, setResizing] = useState(false)

  const sessionRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const dockRef = useRef<HTMLDivElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const composerRef = useRef<HTMLTextAreaElement>(null)
  const idRef = useRef(1)
  const loadedRef = useRef(false)

  useEffect(() => { bodyRef.current?.scrollTo({ top: 1e9, behavior: 'smooth' }) }, [msgs, busy])
  useEffect(() => { onBusyChange?.(busy) }, [busy, onBusyChange])
  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => {
    const presets = agentWidthPresets(window.innerWidth)
    const normalizedIndex = validPresetIndex(widthPresetIndex, presets.length)
    const preset = presets[normalizedIndex]
    if (normalizedIndex !== widthPresetIndex) setWidthPresetIndex(normalizedIndex)
    onLayoutPresetChange?.(preset.archiveColumns)
    onLayoutPresetCommit?.(preset.archiveColumns)
  }, [widthPresetIndex, onLayoutPresetChange, onLayoutPresetCommit])

  // Preserve the selected layout when moving between Full HD / 2K or changing OS scaling.
  useEffect(() => {
    const onResize = () => {
      const presets = agentWidthPresets(window.innerWidth)
      const normalizedIndex = validPresetIndex(widthPresetIndex, presets.length)
      const preset = presets[normalizedIndex]
      if (normalizedIndex !== widthPresetIndex) {
        setWidthPresetIndex(normalizedIndex)
        localStorage.setItem(AGENT_WIDTH_PRESET_STORAGE_KEY, String(normalizedIndex))
      }
      setHalfWidth(preset.width)
      onLayoutPresetChange?.(preset.archiveColumns)
      onLayoutPresetCommit?.(preset.archiveColumns)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [widthPresetIndex, onLayoutPresetChange, onLayoutPresetCommit])

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
      const restored = restoreAgentTranscript(data.messages || [])
      idRef.current = Math.max(idRef.current, maxTranscriptActionId(restored) + 1)
      setMsgs(restored)
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

    const consoleContext = buildAgentConsoleContext(section, archiveFilters)

    try {
      await streamAgent(q, {
        sessionId: sessionRef.current,
        imageB64: img,
        operatorMode,
        consoleContext,
        signal: ctrl.signal,
      }, (e: AgentEvent) => {
        if ((e.type === 'session' || e.type === 'done') && e.session_id) {
          sessionRef.current = e.session_id; setCurSession(e.session_id); localStorage.setItem(LS_SESSION, e.session_id)
        }
        if ((e.type === 'text' || e.type === 'token') && e.content) patchLast((m) => ({ ...m, text: m.text + e.content, status: '' }))
        else if (e.type === 'tool_call') {
          patchLast((m) => ({ ...m, status: `Running ${labelTool(e.name || '')}…` }))
        } else if (e.type === 'tool_result') {
          const planId = e.result?.approval?.plan_id || (e.result?.status === 'preview' ? e.result?.plan_id : null) || null
          const act: ToolAction = { id: ++idRef.current, name: e.name || '', result: e.result, error: e.error, planId }
          patchLast((m) => ({ ...m, actions: [...(m.actions || []), act], status: '' }))
          const effects = normalizeConsoleUiEffects(e.ui_effects)
          if (effects.length) onUiEffects(effects, e.result)
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
      const effects = normalizeConsoleUiEffects(res.ui_effects)
      if (effects.length) onUiEffects(effects, res.result)
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

  function applyWidthPreset(index: number) {
    const presets = agentWidthPresets(window.innerWidth)
    const normalizedIndex = validPresetIndex(index, presets.length)
    const preset = presets[normalizedIndex]
    setWidthPresetIndex(normalizedIndex)
    setHalfWidth(preset.width)
    localStorage.setItem(AGENT_WIDTH_PRESET_STORAGE_KEY, String(normalizedIndex))
    localStorage.setItem(LS_WIDTH, String(preset.width))
    onLayoutPresetChange?.(preset.archiveColumns)
    onLayoutPresetCommit?.(preset.archiveColumns)
  }

  // Dragging is allowed, but release snaps to the nearest stable Full HD / 2K layout.
  function startResize(e: React.MouseEvent) {
    e.preventDefault()
    const startX = e.clientX
    const startW = dockRef.current?.getBoundingClientRect().width ?? window.innerWidth * 0.5
    let latestWidth = startW
    setResizing(true)
    const onMove = (ev: MouseEvent) => {
      latestWidth = Math.min(
        maxAgentPanelWidth(window.innerWidth),
        Math.max(MIN_AGENT_WIDTH, startW + (startX - ev.clientX)),
      )
      const presets = agentWidthPresets(window.innerWidth)
      const { layoutWidth } = agentDragGeometry(latestWidth, presets)
      setHalfWidth(latestWidth)
      onLayoutPresetChange?.(archiveColumnsForAgentWidth(layoutWidth, window.innerWidth))
    }
    const onUp = () => {
      setResizing(false)
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      const presets = agentWidthPresets(window.innerWidth)
      applyWidthPreset(closestAgentWidthPresetIndex(latestWidth, presets))
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }
  function cycleWidthPreset() {
    const presets = agentWidthPresets(window.innerWidth)
    const normalizedIndex = validPresetIndex(widthPresetIndex, presets.length)
    applyWidthPreset(nextAgentWidthPresetIndex(normalizedIndex, presets.length))
  }

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
        <label className="agent-clip" title="Attach an image">
          <IconPaperclip size={17} />
          <input type="file" accept="image/*" style={{ display: 'none' }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.currentTarget.value = '' }} />
        </label>
        <textarea ref={composerRef} rows={1} placeholder="Ask the agent…" value={input}
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
      {/* on a fresh session the prompt assist lives up in the centre instead (see empty state) */}
      {!empty && <PromptAssist channels={channels} value={input} onChange={setInput} inputRef={composerRef} />}
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
            <div className="agent-assist-center">
              <PromptAssist channels={channels} value={input} onChange={setInput} inputRef={composerRef} />
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
                  <ResearchTrace
                    message={m}
                    onThumb={(url, title) => setLightbox({ url, title })}
                    onApply={applyPlan}
                  />
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

  const widthPresets = agentWidthPresets(window.innerWidth)
  const activeWidthPresetIndex = validPresetIndex(widthPresetIndex, widthPresets.length)
  const activeWidthPreset = widthPresets[activeWidthPresetIndex]
  const nextWidthPreset = widthPresets[nextAgentWidthPresetIndex(activeWidthPresetIndex, widthPresets.length)]
  const dragGeometry = agentDragGeometry(halfWidth, widthPresets)

  return (
    <div ref={dockRef} className={`agent-dock ${open ? 'open' : ''} ${full ? 'full' : ''} ${resizing ? 'resizing' : ''}`} data-agent
      style={open && !full ? { width: halfWidth, marginLeft: -dragGeometry.overlayWidth } : undefined}>
      {open && !full && (
        <div className="agent-resize" onMouseDown={startResize}
          title="Drag to resize; release snaps to the nearest fixed layout" />
      )}
      <div className="agent-head">
        <div className="agent-head-title">
          <span className="agent-badge"><IconSparkles size={15} /></span>
          <div>
            <div className="agent-title">EVA Agent</div>
          </div>
        </div>
        <div className="agent-head-btns">
          <button className="modal-close" onClick={onToggleFull} title={full ? 'Exit full screen' : 'Open full screen'}>
            {full ? <IconMinimize size={17} /> : <IconMaximize size={17} />}
          </button>
          <button className="modal-close" onClick={cycleWidthPreset} disabled={full}
            title={`Reset to default width · ${activeWidthPreset.label}; click for ${nextWidthPreset.label}`}>
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
