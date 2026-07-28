import { api } from './client'
import type { AuthUser } from './types'

export function mapUser(u: any): AuthUser {
  const list = (value: unknown): unknown[] => Array.isArray(value) ? value : value == null ? [] : [value]
  return {
    id: String(u?.id || ''),
    username: String(u?.username || ''),
    displayName: u?.displayName ?? u?.display_name,
    tenantId: String(u?.tenantId ?? u?.tenant_id ?? ''),
    roles: list(u?.roles).map((role) => String(role).trim().toLowerCase()).filter(Boolean),
    permissions: list(u?.permissions).map((permission) => String(permission).trim().toLowerCase()).filter(Boolean),
    allowedChannelIds: list(u?.allowedChannelIds ?? u?.allowed_channel_ids).map((id) => String(id).trim()).filter(Boolean),
    currentSessionId: u?.currentSessionId ?? u?.current_session_id,
  }
}

export async function login(username: string, password: string): Promise<AuthUser> {
  const res = await api.postJson('/auth/login', { username, password })
  return mapUser({ ...res.user, currentSessionId: res.sessionId })
}

export async function me(): Promise<AuthUser | null> {
  try {
    const res = await api.get('/auth/me')
    return res?.user ? mapUser({ ...res.user, currentSessionId: res.sessionId }) : null
  } catch {
    return null
  }
}

export async function logout(): Promise<void> {
  try { await api.postJson('/auth/logout', {}) } catch { /* ignore */ }
}
