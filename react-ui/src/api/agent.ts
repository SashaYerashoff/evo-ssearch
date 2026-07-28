// SSE streaming client + REST for the EVA agent (ported from the original UI).
// The chat endpoint needs a POST body, so EventSource can't be used — we read the
// fetch stream manually and parse `data: {json}\n\n` frames.
import { api } from './client'

export interface AgentEvent {
  type: 'session' | 'text' | 'token' | 'tool_call' | 'tool_result' | 'tool_progress' | 'tool_start'
    | 'heartbeat' | 'tool_budget' | 'context_budget' | 'error' | 'done'
  session_id?: string
  content?: string
  name?: string
  args?: any
  result?: any
  error?: string
  message?: string
  [k: string]: any
}

function getCookie(name: string): string | null {
  const m = document.cookie.match(new RegExp('(?:^|; )' + name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1') + '=([^;]*)'))
  return m ? decodeURIComponent(m[1]) : null
}

export function agentSubmissionText(message: string, imageB64?: string | null): string {
  const text = String(message || '').trim()
  if (text) return text
  return imageB64 ? 'Describe this image.' : ''
}

/** Incremental SSE parser supporting LF/CRLF, split chunks and an unterminated EOF frame. */
export class AgentSseParser {
  private buffer = ''

  push(chunk: string, final = false): AgentEvent[] {
    this.buffer += chunk
    const events: AgentEvent[] = []
    let separator = this.buffer.match(/\r?\n\r?\n/)
    while (separator?.index !== undefined) {
      const frame = this.buffer.slice(0, separator.index)
      this.buffer = this.buffer.slice(separator.index + separator[0].length)
      const event = this.parseFrame(frame)
      if (event) events.push(event)
      separator = this.buffer.match(/\r?\n\r?\n/)
    }
    if (final) {
      const event = this.parseFrame(this.buffer)
      this.buffer = ''
      if (event) events.push(event)
    }
    return events
  }

  private parseFrame(frame: string): AgentEvent | null {
    const payload = frame
      .split(/\r?\n/)
      .filter((line) => line === 'data' || line.startsWith('data:'))
      .map((line) => line === 'data' ? '' : line.slice(5).replace(/^ /, ''))
      .join('\n')
      .trim()
    if (!payload) return null
    try {
      const parsed = JSON.parse(payload)
      return parsed && typeof parsed === 'object' ? parsed as AgentEvent : null
    } catch {
      return null
    }
  }
}

export async function streamAgent(
  message: string,
  opts: { sessionId?: string | null; imageB64?: string | null; operatorMode?: boolean; signal?: AbortSignal },
  onEvent: (e: AgentEvent) => void,
): Promise<void> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const csrf = getCookie('eva_csrf')
  if (csrf) headers['X-CSRF-Token'] = csrf

  const res = await fetch('/agent/chat', {
    method: 'POST',
    headers,
    credentials: 'include',
    signal: opts.signal,
    body: JSON.stringify({
      message,
      session_id: opts.sessionId || undefined,
      image_b64: opts.imageB64 || undefined,
      operator_mode: opts.operatorMode || undefined,
    }),
  })
  if (!res.ok || !res.body) {
    const txt = await res.text().catch(() => '')
    throw new Error(`Agent HTTP ${res.status}${txt ? ` · ${txt.slice(0, 120)}` : ''}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  const parser = new AgentSseParser()
  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      for (const event of parser.push(decoder.decode(), true)) onEvent(event)
      break
    }
    for (const event of parser.push(decoder.decode(value, { stream: true }))) onEvent(event)
  }
}

// ---- REST ----------------------------------------------------------------

export interface AgentSession { id: string; title?: string; updated_at?: string | number; message_count?: number }
export interface AgentStoredMsg { role: 'user' | 'assistant'; content: string; created_at?: string | number }
export interface AgentSkill { slug: string; name?: string; summary?: string; content?: string; path?: string }
export interface AgentConfig { model?: string; default_model?: string; source?: string }

export const agentApi = {
  sessions: (): Promise<{ sessions: AgentSession[] }> => api.get('/agent/sessions'),
  session: (id: string): Promise<{ messages: AgentStoredMsg[] }> => api.get(`/agent/session/${id}`),
  deleteSession: (id: string): Promise<any> => api.del(`/agent/session/${id}`),
  getConfig: (): Promise<AgentConfig> => api.get('/agent/config'),
  saveConfig: (model: string): Promise<AgentConfig & { error?: string }> => api.postJson('/agent/config', { model }),
  skills: (): Promise<{ skills: AgentSkill[]; error?: string }> => api.get('/agent/skills'),
  skill: (slug: string): Promise<AgentSkill> => api.get(`/agent/skills/${slug}`),
  createSkill: (b: { name: string; slug: string; content: string }): Promise<{ skill?: AgentSkill; error?: string }> =>
    api.postJson('/agent/skills/create', b),
  updateSkill: (slug: string, b: { name: string; slug: string; content: string }): Promise<{ skill?: AgentSkill; error?: string }> =>
    api.postJson(`/agent/skills/${slug}`, b),
  executePlan: (planId: string, sessionId: string | null): Promise<{ success: boolean; result?: any; error?: string }> =>
    api.postJson(`/agent/action-plans/${planId}/execute`, { session_id: sessionId || undefined }),
  streams: (): Promise<{ video_streams?: any[]; desired_video_missing?: any[]; error?: string }> =>
    api.get('/luxriot/streams'),
}

// Resolve a thumbnail/preview URL for an evidence item (mirrors the original agentImageUrlForItem).
export function agentImageUrl(item: any): string {
  if (!item || typeof item !== 'object') return ''
  const direct = item.image_url || item.imageUrl || item.url
  if (direct) return String(direct)
  const p = item.image_path || item.path
  if (p && String(p).startsWith('/')) return `/detections/image?image_path=${encodeURIComponent(String(p))}`
  const thumb = String(item.thumbnail || item.thumbnail_b64 || '').trim()
  if (thumb) return /^data:image\//i.test(thumb) ? thumb : `data:image/jpeg;base64,${thumb}`
  const id = item.id || item.detection_id
  if (id != null) return `/detections/thumbnail/${id}`
  return ''
}
