import { api } from './client'

export type Settings = Record<string, any>

export interface SettingsPrecedence {
  declared_file_matches_project?: boolean
  declared_config_env_file?: string | null
  different_process_and_file_keys?: string[]
  source_confidence?: string
  config_source_status?: 'undeclared' | 'declared_pending_or_overridden' | 'declared_aligned'
  persistence_source?: string | null
  running_source?: string
  write_allowed?: boolean
  write_block_reason?: string | null
}

export interface SettingsSaveResult {
  success: boolean
  message?: string
  error?: string
  warning?: string
  appliedFields?: string[]
  runtimeAppliedFields?: string[]
  restartRequiredFields?: string[]
  writtenEnvKeys?: string[]
  pendingOrOverriddenKeys?: string[]
  envFile?: string
  precedence?: SettingsPrecedence
}

export interface AuditEvent {
  timestamp?: string
  action?: string
  actor_user_id?: string
  target_type?: string
  target_id?: string
  result?: string
  channel_id?: number | null
  request_id?: string
  details?: any
}

export function normalizeAuditEvent(raw: any): AuditEvent {
  return {
    timestamp: raw?.timestamp ?? raw?.occurredAt,
    action: raw?.action,
    actor_user_id: raw?.actor_user_id ?? raw?.actorUserId,
    target_type: raw?.target_type ?? raw?.targetType,
    target_id: raw?.target_id ?? raw?.targetId,
    result: raw?.result,
    channel_id: raw?.channel_id ?? raw?.channelId ?? null,
    request_id: raw?.request_id ?? raw?.requestId,
    details: raw?.details,
  }
}

export function buildAuditQuery(
  filters: Record<string, string>,
  cursor?: string | null,
): Record<string, string | undefined> {
  const query: Record<string, string | undefined> = {
    limit: filters.limit || '50',
    cursor: cursor || undefined,
  }
  for (const [key, value] of Object.entries(filters)) {
    if (key !== 'limit' && value) query[key] = value
  }
  return query
}

export interface ArchiveCapacitySummary {
  dailyFrameRows: number | null
  retainedFrameRows: number | null
  totalBytes: number | null
  currentRows: number | null
}

function finiteNumber(value: any): number | null {
  if (value === null || value === undefined || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function normalizeArchiveCapacity(raw: any): ArchiveCapacitySummary {
  return {
    dailyFrameRows: finiteNumber(raw?.estimate?.daily?.frame_rows),
    retainedFrameRows: finiteNumber(raw?.estimate?.retained?.frame_rows),
    totalBytes: finiteNumber(raw?.estimate?.bytes?.total),
    currentRows: finiteNumber(raw?.current?.row_count),
  }
}

export function normalizeRevokedSessions(raw: any): { success: boolean; revoked_count: number } {
  return {
    success: !!raw?.success,
    revoked_count: Math.max(0, finiteNumber(raw?.revoked_count ?? raw?.revokedSessions) ?? 0),
  }
}

export function buildArchiveCapacityQuery(includeCurrent = false): Record<string, string> {
  return { include_current: String(includeCurrent) }
}

export interface AuthUserRow {
  user_id: string
  username: string
  display_name?: string
  roles?: string[]
  allowed_channel_ids?: number[] | string
  is_active?: boolean
  created_at?: string
  sessions?: { session_id: string; created_at?: string; last_active_at?: string }[]
}

export interface UserInput {
  username?: string
  password?: string
  displayName?: string
  display_name?: string
  roles?: string[]
  allowedChannelIds?: number[] | string
  allowed_channel_ids?: number[] | string
  isActive?: boolean
  is_active?: boolean
}

export interface AuthSessionRow {
  id: string
  userId?: string
  username?: string
  createdAt?: string | null
  lastSeenAt?: string | null
  expiresAt?: string | null
  revokedAt?: string | null
  revokeReason?: string | null
  clientIp?: string | null
  userAgent?: string | null
}

export const settingsApi = {
  get: (): Promise<{ success: boolean; settings: Settings }> => api.get('/settings'),
  save: (patch: Settings): Promise<SettingsSaveResult> => api.postJson('/settings', patch),
  archiveCapacity: (includeCurrent = false): Promise<any> =>
    api.get('/settings/archive_capacity', buildArchiveCapacityQuery(includeCurrent)),
  getEnv: (): Promise<{ success?: boolean; envText?: string; envVariables?: Record<string, string>; count?: number; envFile?: string; precedence?: SettingsPrecedence }> => api.get('/settings/env'),
  saveEnv: (envText: string): Promise<SettingsSaveResult & { count?: number }> => api.postJson('/settings/env', { envText }),
  audit: async (params: Record<string, unknown>): Promise<{ success: boolean; events?: AuditEvent[]; nextCursor?: string | null; error?: string }> => {
    const response = await api.get('/audit/events', params)
    return { ...response, events: (response?.events || []).map(normalizeAuditEvent) }
  },
  lmModels: (): Promise<any> => api.get('/lm/models'),
  // users
  users: (includeInactive = true): Promise<{ users?: AuthUserRow[]; error?: string }> => api.get('/auth/users', { includeInactive: String(includeInactive) }),
  roles: (): Promise<{ roles?: { name: string; description?: string }[] }> => api.get('/auth/roles'),
  createUser: (b: UserInput): Promise<{ success: boolean; user?: AuthUserRow; error?: string }> => api.postJson('/auth/users', b),
  updateUser: (id: string, b: UserInput): Promise<{ success: boolean; user?: AuthUserRow; error?: string }> => api.patch(`/auth/users/${id}`, b),
  revokeSessions: async (id: string): Promise<{ success: boolean; revoked_count: number }> =>
    normalizeRevokedSessions(await api.postJson(`/auth/users/${id}/revoke-sessions`, {})),
  sessions: (userId?: string, activeOnly = true): Promise<{ success: boolean; sessions?: AuthSessionRow[] }> =>
    api.get('/auth/sessions', { userId, activeOnly: String(activeOnly) }),
  revokeSession: (sessionId: string): Promise<{ success: boolean; revoked?: boolean }> =>
    api.postJson(`/auth/sessions/${sessionId}/revoke`, { reason: 'admin_revoked' }),
}
