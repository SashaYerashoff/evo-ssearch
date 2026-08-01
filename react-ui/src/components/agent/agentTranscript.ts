import type { AgentStoredMsg } from '../../api/agent'
import type { ToolAction } from './ActionCard'

export interface RestoredAgentMessage {
  role: 'user' | 'assistant'
  text: string
  ts?: number
  actions?: ToolAction[]
}

export function restoreAgentTranscript(messages: AgentStoredMsg[]): RestoredAgentMessage[] {
  const restored: RestoredAgentMessage[] = []
  const callOwners = new Map<string, { index: number; name: string }>()
  let actionId = 0

  for (const message of messages || []) {
    if (message.role === 'user') {
      restored.push({
        role: 'user',
        text: String(message.content || ''),
        ts: timestampOf(message.created_at),
      })
      continue
    }
    if (message.role === 'assistant') {
      const index = restored.push({
        role: 'assistant',
        text: String(message.content || ''),
        ts: timestampOf(message.created_at),
        actions: [],
      }) - 1
      for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
        const callId = String(call?.id || '').trim()
        const name = String(call?.function?.name || call?.name || '').trim()
        if (callId && name) callOwners.set(callId, { index, name })
      }
      continue
    }
    if (message.role === 'tool') {
      const owner = callOwners.get(String(message.tool_call_id || ''))
      const index = owner?.index ?? ensureAssistant(restored, message.created_at)
      const result = parsedToolResult(message.tool_result ?? message.content)
      const name = String(message.tool_name || owner?.name || 'tool').trim()
      const planId = result?.approval?.plan_id
        || (result?.status === 'preview' ? result?.plan_id : null)
        || null
      appendAction(restored, index, {
        id: ++actionId,
        name,
        result,
        planId,
        error: typeof result?.error === 'string' ? result.error : undefined,
      })
      continue
    }
    if (message.role === 'system' && message.tool_name === 'action_receipt') {
      const receipt = parsedToolResult(message.tool_result ?? message.content)
      const index = lastAssistant(restored) ?? ensureAssistant(restored, message.created_at)
      appendAction(restored, index, {
        id: ++actionId,
        name: String(receipt?.tool || 'action_receipt'),
        result: receipt,
        applied: String(receipt?.status || '').toLowerCase() === 'applied',
      })
    }
  }

  return restored.filter((message) => (
    message.role === 'user' || message.text.trim() || (message.actions?.length ?? 0) > 0
  ))
}

export function maxTranscriptActionId(messages: RestoredAgentMessage[]): number {
  return messages.reduce(
    (maximum, message) => Math.max(
      maximum,
      ...(message.actions || []).map((action) => Number(action.id) || 0),
    ),
    0,
  )
}

function ensureAssistant(messages: RestoredAgentMessage[], createdAt?: string | number): number {
  const existing = lastAssistant(messages)
  if (existing != null) return existing
  return messages.push({
    role: 'assistant',
    text: '',
    ts: timestampOf(createdAt),
    actions: [],
  }) - 1
}

function lastAssistant(messages: RestoredAgentMessage[]): number | null {
  for (let index = messages.length - 1; index >= 0; index--) {
    if (messages[index].role === 'assistant') return index
  }
  return null
}

function appendAction(
  messages: RestoredAgentMessage[],
  index: number,
  action: ToolAction,
) {
  const message = messages[index]
  messages[index] = {
    ...message,
    actions: [...(message.actions || []), action],
  }
}

function parsedToolResult(value: unknown): any {
  if (typeof value !== 'string') return value && typeof value === 'object' ? value : {}
  try {
    const parsed = JSON.parse(value)
    return parsed && typeof parsed === 'object' ? parsed : { value: parsed }
  } catch {
    return value.trim() ? { message: value.slice(0, 4_096) } : {}
  }
}

function timestampOf(value?: string | number): number | undefined {
  if (value == null || value === '') return undefined
  const numeric = Number(value)
  if (Number.isFinite(numeric)) return numeric < 1e12 ? numeric * 1000 : numeric
  const parsed = new Date(String(value)).getTime()
  return Number.isFinite(parsed) ? parsed : undefined
}
