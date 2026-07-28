export interface AuthUser {
  id: string
  username: string
  displayName?: string
  tenantId: string
  roles: string[]
  permissions: string[]
  allowedChannelIds: string[]
  currentSessionId?: string
}

export interface Channel {
  id: number
  title: string
  guid?: string
  server?: number
}

/** Unified detection view-model (normalized from list + search rows). */
export interface Detection {
  key: string
  id: number | string | null
  channelId: number | null
  channelTitle?: string
  probeId?: string | null
  probeName: string
  source: string        // probe | vlm_summary | vlm_alert
  sourceLabel: string
  severity: string
  posScore?: number | null
  negScore?: number | null
  margin?: number | null
  matchPct?: number | null    // 0..100
  tsMs: number | null
  thumbnail?: string | null   // base64 (no data: prefix)
  imageRef?: string | null    // server path for full-res / describe
  raw: any
}

export interface ArchiveFilters {
  channelId?: string
  source?: string
  probeId?: string
  hours?: string
  sinceMs?: string
  untilMs?: string
  sortBy?: string
  rows?: string
}
