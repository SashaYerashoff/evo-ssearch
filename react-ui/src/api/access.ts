import type { AuthUser, Channel } from './types'

export const PERMISSION = {
  streamsView: 'streams:view',
  detectionsView: 'detections:view',
  reportsView: 'reports:view',
  agentUse: 'agent:use',
  probesRun: 'probes:run',
  bookmarksCreate: 'bookmarks:create',
  incidentsManage: 'incidents:manage',
  probesManage: 'probes:manage',
  promptsManage: 'prompts:manage',
  modelsManage: 'models:manage',
  captureManage: 'capture:manage',
  diagnosticsView: 'diagnostics:view',
  usersManage: 'users:manage',
  settingsView: 'settings:view',
  settingsManage: 'settings:manage',
  auditView: 'audit:view',
  dataExport: 'data:export',
} as const

export type PermissionKey = (typeof PERMISSION)[keyof typeof PERMISSION]

export function hasPermission(user: AuthUser | null | undefined, permission: PermissionKey): boolean {
  return !!user?.permissions?.some((item) => String(item).trim().toLowerCase() === permission)
}

export function hasAnyPermission(
  user: AuthUser | null | undefined,
  permissions: readonly PermissionKey[],
): boolean {
  return permissions.some((permission) => hasPermission(user, permission))
}

export function hasAllChannelScope(user: AuthUser | null | undefined): boolean {
  return !!user?.allowedChannelIds?.some((id) => String(id).trim() === '*')
}

export function canAccessChannel(user: AuthUser | null | undefined, channelId: number | string): boolean {
  if (!user) return false
  if (hasAllChannelScope(user)) return true
  const target = String(channelId)
  return user.allowedChannelIds.some((id) => String(id) === target)
}

export function filterAllowedChannels(user: AuthUser | null | undefined, channels: Channel[]): Channel[] {
  return channels.filter((channel) => canAccessChannel(user, channel.id))
}

export function parseChannelSelection(value: string): '*' | number[] {
  const text = String(value || '').trim()
  if (text === '*') return '*'
  if (!text) return []
  return [...new Set(text
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((id) => Number.isInteger(id) && id > 0))]
}

export function unknownChannelIds(selection: '*' | number[], channels: Channel[]): number[] {
  if (selection === '*') return []
  const known = new Set(channels.map((channel) => channel.id))
  return selection.filter((id) => !known.has(id))
}

export function toggleChannelSelection(
  selection: '*' | number[],
  channelId: number,
  channels: Channel[],
): number[] {
  const selected = new Set(selection === '*' ? channels.map((channel) => channel.id) : selection)
  if (selected.has(channelId)) selected.delete(channelId)
  else selected.add(channelId)
  return [...selected].sort((a, b) => a - b)
}

export function canOpenSettings(user: AuthUser | null | undefined): boolean {
  return hasAnyPermission(user, [
    PERMISSION.settingsView,
    PERMISSION.settingsManage,
    PERMISSION.usersManage,
    PERMISSION.auditView,
    PERMISSION.diagnosticsView,
  ])
}

export type AccessibleSection = 'home' | 'archive' | 'video' | 'monitoring' | 'agent'

export function canViewSection(user: AuthUser | null | undefined, section: AccessibleSection): boolean {
  if (section === 'home') return true
  if (section === 'archive') return hasPermission(user, PERMISSION.detectionsView)
  if (section === 'video') return hasPermission(user, PERMISSION.streamsView)
  if (section === 'monitoring') {
    return hasPermission(user, PERMISSION.streamsView) && hasPermission(user, PERMISSION.reportsView)
  }
  return hasPermission(user, PERMISSION.agentUse)
}

export type SettingsTabKind = 'settings' | 'users' | 'audit' | 'env' | 'diagnostics'

export function canViewSettingsTab(user: AuthUser | null | undefined, kind: SettingsTabKind): boolean {
  if (kind === 'users') return hasPermission(user, PERMISSION.usersManage)
  if (kind === 'audit') return hasPermission(user, PERMISSION.auditView) && hasAllChannelScope(user)
  if (kind === 'env') return hasPermission(user, PERMISSION.settingsManage)
  if (kind === 'diagnostics') return hasPermission(user, PERMISSION.diagnosticsView)
  return hasPermission(user, PERMISSION.settingsView)
}
