import { api } from './client'
import type { AuthUser } from './types'

function mapUser(u: any): AuthUser {
  return {
    id: u.id,
    username: u.username,
    displayName: u.displayName,
    tenantId: u.tenantId,
    roles: u.roles || [],
    permissions: u.permissions || [],
    allowedChannelIds: u.allowedChannelIds || [],
  }
}

export async function login(username: string, password: string): Promise<AuthUser> {
  const res = await api.postJson('/auth/login', { username, password })
  return mapUser(res.user)
}

export async function me(): Promise<AuthUser | null> {
  try {
    const res = await api.get('/auth/me')
    return res?.user ? mapUser(res.user) : null
  } catch {
    return null
  }
}

export async function logout(): Promise<void> {
  try { await api.postJson('/auth/logout', {}) } catch { /* ignore */ }
}
