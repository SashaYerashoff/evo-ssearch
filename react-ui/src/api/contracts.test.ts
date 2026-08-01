import { describe, expect, it } from 'vitest'
import { AgentSseParser, agentSubmissionText } from './agent'
import {
  canAccessChannel,
  canOpenSettings,
  canViewSection,
  canViewSettingsTab,
  filterAllowedChannels,
  hasPermission,
  parseChannelSelection,
  PERMISSION,
  toggleChannelSelection,
  unknownChannelIds,
} from './access'
import { mapUser } from './auth'
import { apiErrorMessage, shouldAttachCsrf } from './client'
import {
  buildArchiveFilterPayload,
  buildArchiveListQuery,
  buildArchiveSearchPayload,
  batchFrameNumber,
  falsePositiveExportUrl,
  fullDetectionImageSrc,
  normalizeDetection,
} from './detections'
import { authorizeProbeInput, probeMutationRequiresBookmarkPermission, probeRangeDurationMs } from './probes'
import {
  normalizeArchiveCapacity,
  normalizeAuditEvent,
  normalizeRevokedSessions,
  buildAuditQuery,
  buildArchiveCapacityQuery,
} from './settings'
import {
  attentionStreamUrl,
  buildSummaryBookmarkInput,
  buildCaptureInput,
  buildPromptSettingsPayload,
  buildSessionQuery,
  buildSummaryFeedQuery,
  fullLiveMediaUrl,
  recentFrameUrl,
} from './video'
import { API_PREFIXES } from '../../proxy-config'

describe('React/backend contract normalizers', () => {
  it('maps the camelCase audit response to the UI model', () => {
    expect(normalizeAuditEvent({
      occurredAt: '2026-07-24T12:00:00Z',
      actorUserId: 'user-1',
      targetType: 'probe',
      targetId: 'probe-7',
      channelId: 7,
      requestId: 'req-1',
      action: 'probes.save',
      result: 'success',
    })).toEqual({
      timestamp: '2026-07-24T12:00:00Z',
      actor_user_id: 'user-1',
      target_type: 'probe',
      target_id: 'probe-7',
      channel_id: 7,
      request_id: 'req-1',
      action: 'probes.save',
      result: 'success',
      details: undefined,
    })
  })

  it('serializes an audit cursor without dropping active filters', () => {
    expect(buildAuditQuery({
      result: 'denied',
      action: 'auth.login',
      actor_user_id: '',
      channel_id: '7',
      request_id: '',
      limit: '25',
    }, 'cursor-2')).toEqual({
      limit: '25',
      cursor: 'cursor-2',
      result: 'denied',
      action: 'auth.login',
      channel_id: '7',
    })
  })

  it('reads the nested archive capacity response', () => {
    expect(normalizeArchiveCapacity({
      estimate: {
        daily: { frame_rows: 1200 },
        retained: { frame_rows: 54000 },
        bytes: { total: 1048576 },
      },
      current: { row_count: 321 },
    })).toEqual({
      dailyFrameRows: 1200,
      retainedFrameRows: 54000,
      totalBytes: 1048576,
      currentRows: 321,
    })
    expect(normalizeArchiveCapacity(null)).toEqual({
      dailyFrameRows: null,
      retainedFrameRows: null,
      totalBytes: null,
      currentRows: null,
    })
  })

  it('keeps expensive current archive statistics opt-in', () => {
    expect(buildArchiveCapacityQuery()).toEqual({ include_current: 'false' })
    expect(buildArchiveCapacityQuery(true)).toEqual({ include_current: 'true' })
  })

  it('maps revokedSessions to the UI count', () => {
    expect(normalizeRevokedSessions({ success: true, revokedSessions: 3 })).toEqual({
      success: true,
      revoked_count: 3,
    })
  })

  it('computes probe duration from the backend timestamp pair', () => {
    expect(probeRangeDurationMs({ time_range_ms: [1000, 5500] })).toBe(4500)
    expect(probeRangeDurationMs({ first_timestamp_ms: 2000, last_timestamp_ms: 7000 })).toBe(5000)
    expect(probeRangeDurationMs({ time_range_ms: null })).toBeNull()
  })

  it('serializes the L0 history boundary', () => {
    expect(buildSessionQuery(7, { limit: 60, from_ts: 1234 })).toEqual({
      channel_id: '7',
      limit: '60',
      run: undefined,
      from_ts: 1234,
      to_ts: undefined,
    })
  })

  it('requests the bounded video-summary feed instead of internal diagnostics', () => {
    expect(buildSummaryFeedQuery(7, { limit: 240, from_ts: 1234 })).toEqual({
      channel_id: '7',
      limit: '240',
      run: undefined,
      from_ts: 1234,
      to_ts: undefined,
      view: 'feed',
    })
  })

  it('keeps model-view preview fresh across dense capture windows', () => {
    expect(recentFrameUrl(112, 7)).toBe(
      '/luxriot/recent_frame/112?stream=mainStream&fallback=snapshot&mode=latest&max_age_sec=60&_=7',
    )
  })

  it('uses the credential-safe bounded broker for opt-in full live', () => {
    expect(fullLiveMediaUrl(112, 9)).toBe(
      '/luxriot/media/live/112?stream=mainStream&request=9',
    )
  })

  it('uses the shared EVA attention ring for the model-view stream', () => {
    expect(attentionStreamUrl(112, 8)).toBe(
      '/luxriot/attention_stream/112?max_age_sec=60&request=8',
    )
  })

  it('builds the same bounded L0 bookmark payload as the legacy console', () => {
    expect(buildSummaryBookmarkInput({
      channel_id: 112,
      created_at: 1_785_000_000,
      summary: 'Person entered\nMore evidence',
    })).toEqual({
      channel_id: 112,
      title: 'Live summary: Person entered',
      description: 'Person entered\nMore evidence',
      severity: 'normal',
      state: 'new',
      timestamp_ms: 1_785_000_000_000,
    })
  })

  it('starts video capture from persistent channel prompt settings', () => {
    expect(buildCaptureInput(7, {
      batch: '12',
      every: '5',
      model: '  auto ',
    })).toEqual({
      channel_id: 7,
      batch_size: 12,
      interval_sec: 5,
      model: 'auto',
    })
  })

  it('supplies a useful prompt for image-only agent messages', () => {
    expect(agentSubmissionText('', 'base64-image')).toBe('Describe this image.')
    expect(agentSubmissionText('  inspect this  ', 'base64-image')).toBe('inspect this')
    expect(agentSubmissionText('', null)).toBe('')
  })

  it('parses split SSE frames, CRLF, malformed input and the EOF remainder', () => {
    const parser = new AgentSseParser()
    expect(parser.push('data: {"type":"to')).toEqual([])
    expect(parser.push('ken","content":"a"}\r\n\r\n')).toEqual([
      { type: 'token', content: 'a' },
    ])
    expect(parser.push('data: not-json\n\n')).toEqual([])
    expect(parser.push(': heartbeat\n\ndata: {"type":"done"}', true)).toEqual([
      { type: 'done' },
    ])
  })

  it('proxies the audit endpoint in Vite development', () => {
    expect(API_PREFIXES).toContain('/audit')
  })

  it('uses the same archive filters for list, text and image contracts', () => {
    const filters = {
      channelId: '7',
      source: 'probe',
      probeId: 'probe-1',
      hours: '72',
      sortBy: 'time',
      rows: '36',
    }
    const common = {
      channel_id: '7',
      channel_ids: undefined,
      source: 'probe',
      probe_id: 'probe-1',
      hours: 72,
      since_ms: undefined,
      until_ms: undefined,
    }
    expect(buildArchiveFilterPayload(filters)).toEqual(common)
    expect(buildArchiveListQuery(filters, 36)).toEqual({
      ...common,
      limit: 36,
      offset: 36,
    })
    expect(buildArchiveSearchPayload(filters)).toEqual({
      ...common,
      limit: 36,
      sort_by: 'time',
    })
  })

  it('uses an absolute archive range without leaking a preset and scopes probe ids', () => {
    expect(buildArchiveFilterPayload({
      source: 'vlm_summary',
      probeId: 'must-not-leak',
      hours: '24',
      sinceMs: '1000',
      untilMs: '9000',
    })).toEqual({
      channel_id: undefined,
      channel_ids: undefined,
      source: 'vlm_summary',
      probe_id: undefined,
      hours: undefined,
      since_ms: '1000',
      until_ms: '9000',
    })
  })

  it('keeps multi-stream archive scope across list and semantic search', () => {
    expect(buildArchiveFilterPayload({
      channelIds: ['9', '7', '9'],
      source: 'semantic_snapshot',
      hours: '6',
    })).toEqual({
      channel_id: undefined,
      channel_ids: ['9', '7'],
      source: 'semantic_snapshot',
      probe_id: undefined,
      hours: 6,
      since_ms: undefined,
      until_ms: undefined,
    })
  })

  it('generates a stable fallback detection key', () => {
    const raw = {
      source: 'vlm_summary',
      channel_id: 7,
      timestamp_ms: 1234,
      image_path: '/archive/frame.jpg',
    }
    expect(normalizeDetection(raw).key).toBe(normalizeDetection({ ...raw }).key)
  })

  it('routes a full archive image through the guarded backend endpoint', () => {
    const detection = normalizeDetection({
      id: 42,
      source: 'probe',
      image_path: String.raw`D:\archive\camera 7\frame.jpg`,
      thumbnail: 'preview',
    })
    expect(fullDetectionImageSrc(detection)).toBe(
      '/detections/image?image_path=D%3A%5Carchive%5Ccamera%207%5Cframe.jpg',
    )
  })

  it('orders batch frames by stored snapshot index and scopes feedback exports', () => {
    expect(batchFrameNumber(normalizeDetection({
      source: 'vlm_summary',
      payload: { snapshot_index: 6 },
    }))).toBe(6)
    expect(falsePositiveExportUrl('xml', 112)).toBe(
      '/reports/false-positives/export?format=xml&hours=24&channel_id=112',
    )
  })

  it('normalizes auth aliases and channel ids before authorization checks', () => {
    const user = mapUser({
      id: 'u1',
      username: 'pilot',
      tenant_id: 'tenant-1',
      display_name: 'Pilot',
      roles: ['VIEWER'],
      permissions: ['STREAMS:VIEW', 'DETECTIONS:VIEW'],
      allowed_channel_ids: [7, '8'],
      current_session_id: 'session-1',
    })
    expect(user).toMatchObject({
      tenantId: 'tenant-1',
      displayName: 'Pilot',
      roles: ['viewer'],
      permissions: ['streams:view', 'detections:view'],
      allowedChannelIds: ['7', '8'],
      currentSessionId: 'session-1',
    })
    expect(canAccessChannel(user, 7)).toBe(true)
    expect(canAccessChannel(user, 9)).toBe(false)
    expect(filterAllowedChannels(user, [
      { id: 7, title: 'Allowed' },
      { id: 9, title: 'Denied' },
    ])).toEqual([{ id: 7, title: 'Allowed' }])
  })

  it('applies wildcard, navigation and Settings permission rules', () => {
    const admin = mapUser({
      id: 'admin',
      username: 'admin',
      permissions: Object.values(PERMISSION),
      allowedChannelIds: ['*'],
    })
    expect(hasPermission(admin, PERMISSION.usersManage)).toBe(true)
    expect(canViewSection(admin, 'monitoring')).toBe(true)
    expect(canOpenSettings(admin)).toBe(true)
    expect(canViewSettingsTab(admin, 'audit')).toBe(true)

    const scopedAudit = mapUser({
      id: 'audit',
      username: 'audit',
      permissions: ['audit:view'],
      allowedChannelIds: [7],
    })
    expect(canOpenSettings(scopedAudit)).toBe(true)
    expect(canViewSettingsTab(scopedAudit, 'audit')).toBe(false)
  })

  it('validates explicit user channel selections', () => {
    const selection = parseChannelSelection('7, 8, invalid, -1')
    expect(selection).toEqual([7, 8])
    expect(unknownChannelIds(selection, [{ id: 7, title: 'Known' }])).toEqual([8])
    expect(unknownChannelIds('*', [])).toEqual([])
  })

  it('turns wildcard channel scope into all channels except the unchecked one', () => {
    const channels = [
      { id: 7, title: 'One' },
      { id: 8, title: 'Two' },
      { id: 9, title: 'Three' },
    ]
    expect(toggleChannelSelection('*', 8, channels)).toEqual([7, 9])
    expect(toggleChannelSelection([7, 9], 8, channels)).toEqual([7, 8, 9])
  })

  it('produces useful JSON and non-JSON API errors', () => {
    expect(apiErrorMessage(403, { error: 'channel access denied' })).toBe('channel access denied')
    expect(apiErrorMessage(401, '')).toBe('Authentication required')
    expect(apiErrorMessage(500, '<html>unavailable</html>')).toBe('<html>unavailable</html>')
  })

  it('attaches CSRF to JSON, multipart and DELETE mutations', () => {
    expect(shouldAttachCsrf('GET')).toBe(false)
    expect(shouldAttachCsrf('POST')).toBe(true)
    expect(shouldAttachCsrf('PATCH')).toBe(true)
    expect(shouldAttachCsrf('DELETE')).toBe(true)
  })

  it('does not submit bookmark prompt fields without permission', () => {
    expect(buildPromptSettingsPayload({
      stream_system_prompt: 'Observe',
      json_alert_prompt: 'Emit JSON',
      bookmark_enabled: true,
      bookmark_cooldown_sec: 8,
    }, 7, false)).toEqual({
      channel_id: 7,
      stream_system_prompt: 'Observe',
    })
  })

  it('does not submit probe bookmark fields without permission', () => {
    expect(authorizeProbeInput({
      id: 'probe-1',
      name: 'Door',
      severity: 'high',
      bookmark: true,
      bookmark_cooldown_sec: 8,
      bookmark_dedupe_window_sec: 20,
    }, false)).toEqual({
      id: 'probe-1',
      name: 'Door',
      severity: 'high',
    })
    expect(probeMutationRequiresBookmarkPermission({ bookmark: true }, false)).toBe(true)
    expect(probeMutationRequiresBookmarkPermission({ bookmark: false }, false)).toBe(false)
    expect(probeMutationRequiresBookmarkPermission({ bookmark: true }, true)).toBe(false)
  })
})
