import { useMemo, useState } from 'react'
import { agentImageUrl } from '../../api/agent'

export interface ToolAction {
  id: number
  name: string
  result?: any
  error?: string
  planId?: string | null
  applying?: boolean
  applied?: boolean
}

const LABELS: Record<string, string> = {
  search_archive: 'Search archive', search_detections: 'Search archive', search_folder: 'Search frames',
  get_detections: 'Detections', get_detection_summary: 'Detection summary', list_channels: 'Channels',
  list_video_summary_channels: 'Video channel coverage',
  list_probes: 'Probes', survey_channels: 'Channel survey', create_probe: 'Create probe', update_probe: 'Update probe',
  delete_probes: 'Delete probes', deploy_summary: 'Deploy summary', describe_frame: 'Describe frame',
  get_video_summaries: 'Video summaries', count_video_summary_events: 'Summary events',
  track_visual_state_transitions: 'State transitions', create_bookmark: 'Bookmark', generate_report: 'Report',
  normalize_time_window: 'Time window', build_research_batch: 'Research batch',
  get_prompt_settings: 'VLM prompt layers', update_prompt_settings: 'Update VLM prompts',
  start_deployment: 'Protocol Deploy inventory', configure_deployment: 'Protocol Deploy draft',
  survey_deployment: 'Protocol Deploy survey', apply_deployment_plan: 'Protocol Deploy approval',
}

function label(name: string) { return (LABELS[name] || name.replace(/_/g, ' ')).toUpperCase() }

// find the most likely array of evidence items in a tool result
function itemsOf(result: any): any[] {
  if (!result || typeof result !== 'object') return []
  for (const k of ['results', 'detections', 'frames', 'evidence_frames', 'boundary_frames', 'candidate_frames', 'items', 'matches']) {
    if (Array.isArray(result[k]) && result[k].length) return result[k]
  }
  return []
}

function scoreOf(item: any): string {
  const s = item?.similarity ?? item?.score ?? item?.match ?? item?.matchPct
  if (s == null) return ''
  const n = Number(s)
  if (!isFinite(n)) return ''
  return n <= 1 ? `${Math.round(n * 100)}%` : String(Math.round(n))
}

// scalar key/value rows worth showing for generic results / approvals
function entriesOf(obj: any): [string, string][] {
  if (!obj || typeof obj !== 'object') return []
  const skip = new Set(['thumbnail', 'thumbnail_b64', 'image', 'image_b64', 'results', 'detections', 'frames', 'approval'])
  const out: [string, string][] = []
  for (const [k, v] of Object.entries(obj)) {
    if (skip.has(k)) continue
    if (v == null) continue
    if (typeof v === 'object') {
      if (Array.isArray(v)) { if (v.length && typeof v[0] !== 'object') out.push([k, v.join(', ')]) }
      continue
    }
    out.push([k, String(v)])
    if (out.length >= 8) break
  }
  return out
}

export interface ActionTable {
  title: string
  columns: string[]
  rows: string[][]
}

function textValue(value: unknown, fallback = '—'): string {
  if (value == null || value === '') return fallback
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (Array.isArray(value)) return value.map((item) => textValue(item, '')).filter(Boolean).join(', ') || fallback
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    return Object.entries(record)
      .slice(0, 4)
      .map(([key, item]) => `${key.replace(/_/g, ' ')} ${textValue(item, '')}`)
      .join(' · ') || fallback
  }
  return String(value)
}

function row(values: unknown[]): string[] {
  return values.map((value) => textValue(value))
}

function DeploymentInventoryCard({ result, onSend }: { result: any; onSend: (message: string) => void }) {
  const channels = Array.isArray(result?.ui_available_channels) && result.ui_available_channels.length
    ? result.ui_available_channels
    : (Array.isArray(result?.available_channels) ? result.available_channels : [])
  const cap = Math.max(1, Math.min(8, Number(result?.target_channel_count) || 8))
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<number[]>([])
  const [groups, setGroups] = useState<Record<number, string>>({})
  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    if (!needle) return channels
    return channels.filter((channel: any) => (
      String(channel?.id ?? '').includes(needle)
      || String(channel?.title || channel?.name || '').toLocaleLowerCase().includes(needle)
    ))
  }, [channels, query])

  function toggle(channelId: number) {
    setSelected((current) => current.includes(channelId)
      ? current.filter((item) => item !== channelId)
      : current.length < cap ? [...current, channelId] : current)
  }

  function submit() {
    if (!selected.length) return
    const groupClauses = Object.entries(groups)
      .map(([channelId, group]) => ({ channelId: Number(channelId), group: group.trim() }))
      .filter((item) => selected.includes(item.channelId) && item.group)
      .map((item) => `group ${item.group}: ${item.channelId}`)
    onSend([
      `Continue Protocol Deploy ${result?.deployment_id || ''}. Select channels ${selected.join(', ')}`,
      ...groupClauses,
    ].filter(Boolean).join('; '))
  }

  return (
    <div className="ag-deploy-inventory">
      <div className="ag-approval-head">
        <div>
          <div className="ag-approval-kick">Protocol Deploy · channel scope</div>
          <div className="ag-approval-title">Choose up to {cap} of {result?.available_channel_count ?? channels.length} visible channels</div>
        </div>
        <span className="ag-approval-status">{selected.length}/{cap}</span>
      </div>
      <div className="ag-deploy-search"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter by channel ID or name…" /></div>
      <div className="ag-deploy-channel-list">
        {filtered.map((channel: any) => {
          const channelId = Number(channel?.id)
          const checked = selected.includes(channelId)
          return (
            <div className={`ag-deploy-channel-row ${checked ? 'selected' : ''}`} key={String(channel?.id)}>
              <label>
                <input type="checkbox" checked={checked} disabled={!checked && selected.length >= cap} onChange={() => toggle(channelId)} />
                <span><b>#{channelId}</b> {channel?.title || channel?.name || 'Untitled channel'}</span>
              </label>
              {checked && <input className="ag-deploy-group-input" value={groups[channelId] || ''} onChange={(event) => setGroups((current) => ({ ...current, [channelId]: event.target.value }))} placeholder="optional group" />}
            </div>
          )
        })}
      </div>
      <div className="ag-approval-note">Groups are optional. Channels without a group will be commissioned one by one.</div>
      <div className="ag-approval-foot"><button className="ag-apply" disabled={!selected.length} onClick={submit}>Survey selected channels</button></div>
    </div>
  )
}

function DeploymentSurveyCard({ result, onSend }: { result: any; onSend: (message: string) => void }) {
  const surveys = Array.isArray(result?.surveys) ? result.surveys : []
  const groups = Array.isArray(result?.groups) ? result.groups : []
  const selectedIds = (result?.selected_channel_ids || []).map(Number)
  const groupedIds = new Set<number>(groups.flatMap((group: any) => (
    Array.isArray(group?.channel_ids) ? group.channel_ids.map(Number) : []
  )))
  const scopes = [
    ...groups.map((group: any) => ({
      name: String(group?.name || 'group'),
      channelIds: (group?.channel_ids || []).map(Number),
    })),
    ...selectedIds
      .filter((channelId: number) => !groupedIds.has(channelId))
      .map((channelId: number) => ({ name: `channel_${channelId}`, channelIds: [channelId] })),
  ]

  function requestProposal(scope: { name: string; channelIds: number[] }) {
    onSend(
      `Continue Protocol Deploy ${result?.deployment_id || ''}. Draft grounded default alerts for group ${scope.name}, channels ${scope.channelIds.join(', ')}, using only its recorded scene survey. Include expected routine, visible alert criteria and severity, novelty sensitivity, and only useful optional counters. Do not apply anything yet.`,
    )
  }

  function chooseNoAlerts(scope: { name: string; channelIds: number[] }) {
    const clauses = scope.channelIds.map((channelId) => `CH ${channelId}: no default alerts`)
    onSend(`Continue Protocol Deploy ${result?.deployment_id || ''}. ${clauses.join('; ')}. Keep ordinary L0 descriptions and continue to the next scope.`)
  }

  return (
    <div className="ag-deploy-survey">
      <div className="ag-approval-head">
        <div>
          <div className="ag-approval-kick">Protocol Deploy · scene survey</div>
          <div className="ag-approval-title">Choose alert policy scope by scope</div>
        </div>
        <span className="ag-approval-status">{surveys.length} sampled</span>
      </div>
      <div className="ag-deploy-survey-example">
        <b>Good alert description</b>
        <span>“Alert HIGH when a sailing vessel visibly enters the cargo fairway on a converging path; do not alert on ordinary parallel passage or PTZ scene changes.”</span>
      </div>
      <div className="ag-deploy-scope-list">
        {scopes.map((scope) => (
          <section className="ag-deploy-scope" key={`${scope.name}-${scope.channelIds.join('-')}`}>
            <div className="ag-deploy-scope-head"><b>{scope.name}</b><span>CH {scope.channelIds.join(', ')}</span></div>
            {surveys.filter((survey: any) => scope.channelIds.includes(Number(survey?.channel_id))).map((survey: any) => (
              <div className="ag-deploy-fingerprint" key={String(survey?.channel_id)}>
                <b>#{survey?.channel_id} {survey?.title || ''}</b>
                <span>{survey?.error || survey?.scene_fingerprint || 'No usable scene fingerprint yet.'}</span>
              </div>
            ))}
            <div className="ag-deploy-review-actions">
              <button className="ag-mini-btn" onClick={() => chooseNoAlerts(scope)}>No default alerts</button>
              <button className="ag-apply" onClick={() => requestProposal(scope)}>Draft alerts for this scope</button>
            </div>
          </section>
        ))}
      </div>
      <div className="ag-approval-note">You can also describe the desired alerts in chat. EVA keeps every scope as a draft until the final deployment card is applied.</div>
    </div>
  )
}

function DeploymentApprovalCard({ action, onApply, onSend }: {
  action: ToolAction
  onApply: (action: ToolAction) => void
  onSend: (message: string) => void
}) {
  const result = action.result || {}
  const diff = result.diff || {}
  const channels = Array.isArray(result.per_channel) ? result.per_channel : []
  const groups = Array.isArray(result.groups) ? result.groups : []
  const probes = Array.isArray(result.proposed_probes) ? result.proposed_probes : []
  const counters = Array.isArray(result.proposed_counted_states) ? result.proposed_counted_states : []
  const groupedIds = new Set<number>(groups.flatMap((group: any) => (
    Array.isArray(group?.channel_ids) ? group.channel_ids.map(Number) : []
  )))
  const scopes = [
    ...groups.map((group: any) => ({
      name: String(group?.name || 'group'),
      channelIds: (group?.channel_ids || []).map(Number),
    })),
    ...channels
      .map((channel: any) => Number(channel?.channel_id))
      .filter((channelId: number) => !groupedIds.has(channelId))
      .map((channelId: number) => ({ name: `channel_${channelId}`, channelIds: [channelId] })),
  ]
  const reviewSteps = scopes.flatMap((scope) => [
    { kind: 'policy' as const, scope },
    { kind: 'probes' as const, scope },
  ])
  const [stepIndex, setStepIndex] = useState(0)
  const finalStep = stepIndex >= reviewSteps.length
  const step = finalStep ? null : reviewSteps[stepIndex]
  const scopeChannels = new Set(step?.scope.channelIds || [])
  const scopePolicies = channels.filter((channel: any) => scopeChannels.has(Number(channel?.channel_id)))
  const scopeProbes = probes.filter((probe: any) => scopeChannels.has(Number(probe?.channel_id)))
  const scopeCounters = counters.filter((counter: any) => scopeChannels.has(Number(counter?.channel_id)))

  function removeScopeProbes() {
    const ids = step?.scope.channelIds || []
    onSend(
      `Continue Protocol Deploy ${result.deployment_id}. For channels ${ids.join(', ')}, keep the proposed VLM alert policies but remove all attention probes and counters; regenerate preview.`,
    )
  }

  return (
    <div className="ag-approval ag-deployment-approval">
      <div className="ag-approval-head">
        <div>
          <div className="ag-approval-kick">Operator review required</div>
          <div className="ag-approval-title">PROTOCOL DEPLOY · {result.deployment_id || 'draft'}</div>
        </div>
        <span className={`ag-approval-status ${action.applied ? 'ok' : ''}`}>{action.applied ? 'Applied' : finalStep ? 'Ready to apply' : `${stepIndex + 1}/${reviewSteps.length}`}</span>
      </div>
      {!finalStep && step && (
        <div className="ag-deploy-review-step">
          <div className="ag-deploy-review-title">{step.scope.name} · channels {step.scope.channelIds.join(', ')}</div>
          {step.kind === 'policy' ? (
            <>
              <div className="ag-approval-kick">Proposed VLM alert policy</div>
              {scopePolicies.map((channel: any) => (
                <details className="ag-deployment-channel" open key={String(channel?.channel_id)}>
                  <summary>CH {channel?.channel_id}</summary>
                  <pre>{String(channel?.alert_policy_preview || 'No default alert policy proposed.')}</pre>
                </details>
              ))}
              <div className="ag-approval-note">If the wording or severity is wrong, describe the correction in chat. EVA will invalidate this card and generate a new one.</div>
            </>
          ) : (
            <>
              <div className="ag-approval-kick">Proposed attention probes and counters</div>
              {!scopeProbes.length && !scopeCounters.length && <div className="ag-approval-note">No vector probes or counted-state metrics are proposed for this scope.</div>}
              {scopeProbes.map((probe: any) => (
                <div className="ag-deploy-proposal" key={`${probe.channel_id}-${probe.name}`}>
                  <b>CH {probe.channel_id} · {probe.name}</b>
                  <span>P: {textValue(probe.positives)} · N: {textValue(probe.negatives)} · floor {textValue(probe.pos_floor)} · margin {textValue(probe.margin)} · {probe.embedding_backend || 'embedding space pending'} · {probe.severity || 'normal'}</span>
                </div>
              ))}
              {scopeCounters.map((counter: any) => (
                <div className="ag-deploy-proposal" key={String(counter.id)}><b>Counter · {counter.name}</b><span>{counter.counter_mode} · {counter.count_transition} · duration {counter.duration_state}</span></div>
              ))}
            </>
          )}
          <div className="ag-approval-foot ag-deploy-review-actions">
            <button className="ag-mini-btn" disabled={stepIndex === 0} onClick={() => setStepIndex((value) => Math.max(0, value - 1))}>Back</button>
            {step.kind === 'probes' && (scopeProbes.length > 0 || scopeCounters.length > 0) && <button className="ag-mini-btn" onClick={removeScopeProbes}>Reject probes</button>}
            <button className="ag-apply" onClick={() => setStepIndex((value) => value + 1)}>{step.kind === 'policy' ? 'Policy looks right' : 'Accept and continue'}</button>
          </div>
        </div>
      )}
      {finalStep && (
        <>
          <div className="ag-fields">
            <div className="ag-field"><span className="ag-field-k">Channels</span><span className="ag-field-v">{textValue(diff.channel_ids)}</span></div>
            <div className="ag-field"><span className="ag-field-k">Policies</span><span className="ag-field-v">{textValue(diff.alert_policy_count)}</span></div>
            <div className="ag-field"><span className="ag-field-k">Attention probes</span><span className="ag-field-v">{textValue(diff.probe_count)}</span></div>
            <div className="ag-field"><span className="ag-field-k">Counters</span><span className="ag-field-v">{textValue(diff.counted_state_count)}</span></div>
          </div>
          <div className="ag-approval-note">This is the only mutating step. Nothing changes until Apply deployment succeeds.</div>
          {action.error && <div className="ag-card-err">{action.error}</div>}
          {!action.applied && action.planId && <div className="ag-approval-foot ag-deploy-review-actions"><button className="ag-mini-btn" onClick={() => setStepIndex(Math.max(0, reviewSteps.length - 1))}>Back</button><button className="ag-apply" disabled={action.applying} onClick={() => onApply(action)}>{action.applying ? 'Applying deployment…' : 'Apply deployment'}</button></div>}
        </>
      )}
    </div>
  )
}

/** Closed, bounded renderers for the high-volume EVA tools used in ordinary operator flows. */
export function actionTables(name: string, result: any): ActionTable[] {
  if (!result || typeof result !== 'object') return []

  if (name === 'list_channels') {
    const rows = Array.isArray(result.channels) ? result.channels : []
    return [{
      title: `${result.count ?? rows.length} visible channels`,
      columns: ['Channel', 'Title', 'State'],
      rows: rows.slice(0, 12).map((item: any) => row([
        `#${item.id ?? item.channel_id ?? '?'}`,
        item.title,
        item.status ?? (item.enabled === false ? 'disabled' : 'enabled'),
      ])),
    }]
  }

  if (name === 'list_video_summary_channels') {
    const rows = Array.isArray(result.candidate_channels) ? result.candidate_channels : []
    return [{
      title: `${result.active_count ?? rows.length} active · ${result.error_count ?? 0} problems`,
      columns: ['Channel', 'VLM / coverage', 'Signals'],
      rows: rows.slice(0, 12).map((item: any) => row([
        `#${item.channel_id ?? '?'} ${item.title || ''}`.trim(),
        `${item.running || item.runtime_running ? 'running' : 'idle'} · ${item.coverage_status || 'coverage unknown'}`,
        `${item.summary_count ?? 0} summaries · ${item.alert_total ?? 0} alerts${item.dropped_frames ? ` · ${item.dropped_frames} dropped` : ''}`,
      ])),
    }]
  }

  if (name === 'survey_channels') {
    const rows = Array.isArray(result.channels) ? result.channels : []
    return [{
      title: `${rows.length} channel scene surveys`,
      columns: ['Channel', 'Samples', 'Observed scene'],
      rows: rows.slice(0, 8).map((item: any) => row([
        `#${item.channel_id ?? '?'} ${item.title || ''}`.trim(),
        item.sample_count,
        item.error || item.survey,
      ])),
    }]
  }

  if (name === 'list_probes') {
    const rows = Array.isArray(result.probes) ? result.probes : []
    return [{
      title: `${result.count ?? rows.length} probes`,
      columns: ['Probe', 'Channel / state', 'Thresholds / hits'],
      rows: rows.slice(0, 12).map((item: any) => row([
        item.name || item.id,
        `#${item.channel_id ?? '?'} · ${item.enabled === false ? 'disabled' : 'enabled'} · ${item.severity || 'info'}`,
        `P ≥ ${item.pos_floor ?? '—'} · M ≥ ${item.margin ?? '—'} · ${item.hit_count_24h ?? 0} hits/24h`,
      ])),
    }]
  }

  if (name === 'get_video_summaries') {
    const rows = Array.isArray(result.entries) ? result.entries : []
    return [{
      title: `${result.count ?? rows.length} ${result.depth || 'video'} summaries · ${result.coverage?.status || 'coverage unknown'}`,
      columns: ['Window', 'Material', 'Summary'],
      rows: rows.slice(0, 5).map((item: any) => row([
        item.time || item.window_start || item.window_end_time,
        `${item.level || result.depth || 'L0'} · ${item.frame_count ?? 0} frames · ${item.alert_total ?? 0} alerts`,
        item.summary,
      ])),
    }]
  }

  if (name === 'count_video_summary_events') {
    const events = Array.isArray(result.transition_events) ? result.transition_events : []
    const timeline = Array.isArray(result.timeline_samples) ? result.timeline_samples : []
    const tables: ActionTable[] = [{
      title: `${result.event_total ?? events.length} counted events · ${textValue(result.coverage?.status || result.coverage)}`,
      columns: ['Transition', 'Time', 'Evidence'],
      rows: events.slice(0, 12).map((item: any) => row([
        item.type || item.basis,
        item.time || item.window_start,
        item.summary || `${item.previous_state || 'unknown'} → event`,
      ])),
    }]
    if (timeline.length) {
      tables.push({
        title: 'State timeline sample',
        columns: ['Time', 'State', 'Evidence'],
        rows: timeline.slice(0, 8).map((item: any) => row([
          item.time || item.window_start,
          item.state,
          item.summary,
        ])),
      })
    }
    return tables
  }

  if (name === 'get_prompt_settings' || name === 'update_prompt_settings') {
    const current = result.current && typeof result.current === 'object' ? result.current : result
    const health = current.prompt_health || {}
    return [{
      title: `VLM prompt scope · ${current.scope || (current.channel_id ? `channel ${current.channel_id}` : 'global')}`,
      columns: ['Layer', 'State', 'Preview'],
      rows: [
        row(['L0 live role', 'editable', current.stream_system_prompt || current.L0_live_prompt]),
        row(['Alert criteria', health.needs_migration ? 'migration suggested' : 'separate', current.alert_policy_prompt]),
        row(['BATCH_STATE_JSON', 'system contract', current.json_alert_prompt]),
      ],
    }]
  }

  if (name === 'generate_report' && result.report_type === 'false_positives') {
    const rows = Array.isArray(result.reason_counts) ? result.reason_counts : []
    return [{
      title: `${textValue(result.period, 'Selected period')} false-positive feedback`,
      columns: ['Reason', 'Count'],
      rows: rows.slice(0, 8).map((item: any) => row([item.reason_label || item.reason_code, item.count])),
    }]
  }

  return []
}

export function ActionCard({ action, onThumb, onApply, onSend }: {
  action: ToolAction
  onThumb: (url: string, title: string) => void
  onApply: (a: ToolAction) => void
  onSend: (message: string) => void
}) {
  const { name, result, error } = action
  const items = itemsOf(result)
  const isApproval = !!action.applied || !!action.planId || result?.status === 'preview'
  const describeImg = name === 'describe_frame' ? agentImageUrl(result) : ''
  const text = result?.description || result?.summary || result?.note || result?.text || result?.message
  const tables = actionTables(name, result)

  if (name === 'start_deployment' && result?.stage === 'inventory') {
    return <DeploymentInventoryCard result={result} onSend={onSend} />
  }

  if (name === 'survey_deployment' && result?.stage === 'surveyed') {
    return <DeploymentSurveyCard result={result} onSend={onSend} />
  }

  if (isApproval) {
    if (name === 'apply_deployment_plan' && result?.status === 'preview') {
      return <DeploymentApprovalCard action={action} onApply={onApply} onSend={onSend} />
    }
    const fields = entriesOf(result?.approval || result?.preview || result)
    return (
      <div className="ag-approval">
        <div className="ag-approval-head">
          <div>
            <div className="ag-approval-kick">{action.applied ? 'Probe action receipt' : 'Operator approval required'}</div>
            <div className="ag-approval-title">{label(name)}</div>
          </div>
          <span className={`ag-approval-status ${action.applied ? 'ok' : ''}`}>{action.applied ? 'Applied' : 'Preview'}</span>
        </div>
        {fields.length > 0 && (
          <div className="ag-fields">
            {fields.map(([k, v]) => <div key={k} className="ag-field"><span className="ag-field-k">{k}</span><span className="ag-field-v">{v}</span></div>)}
          </div>
        )}
        {error && <div className="ag-card-err">{error}</div>}
        {!action.applied && action.planId && (
          <div className="ag-approval-foot">
            <button className="ag-apply" disabled={action.applying} onClick={() => onApply(action)}>
              {action.applying ? 'Applying…' : 'Apply'}
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="ag-card">
      <div className="ag-card-head">◆ {label(name)}</div>
      <div className="ag-card-body">
        {error && <div className="ag-card-err">{error}</div>}
        {describeImg && (
          <div className="ag-thumb solo" onClick={() => onThumb(describeImg, label(name))}>
            <img src={describeImg} alt="frame" loading="lazy" />
          </div>
        )}
        {items.length > 0 && (
          <div className="ag-grid">
            {items.slice(0, 8).map((it, i) => {
              const url = agentImageUrl(it)
              const sc = scoreOf(it)
              const title = it.filename || it.name || it.source || `#${it.id ?? it.detection_id ?? i}`
              return (
                <div key={i} className="ag-thumb" onClick={() => url && onThumb(url, String(title))}>
                  {url ? <img src={url} alt={String(title)} loading="lazy" /> : <span className="ag-thumb-miss">no image</span>}
                  {sc && <span className="ag-score">{sc}</span>}
                </div>
              )
            })}
          </div>
        )}
        {text && <div className="ag-card-text">{String(text)}</div>}
        {tables.map((table, tableIndex) => (
          <section className="ag-tool-table" key={`${table.title}-${tableIndex}`}>
            <div className="ag-tool-table-title">{table.title}</div>
            <div className="ag-tool-table-scroll">
              <div className="ag-tool-table-grid" style={{ '--ag-columns': table.columns.length } as React.CSSProperties}>
                {table.columns.map((column) => <b key={column}>{column}</b>)}
                {table.rows.flatMap((values, rowIndex) => values.map((value, columnIndex) => (
                  <span key={`${rowIndex}-${columnIndex}`} title={value}>{value}</span>
                )))}
              </div>
            </div>
          </section>
        ))}
        {!items.length && !describeImg && !text && !tables.length && (
          <div className="ag-fields">
            {entriesOf(result).map(([k, v]) => <div key={k} className="ag-field"><span className="ag-field-k">{k}</span><span className="ag-field-v">{v}</span></div>)}
          </div>
        )}
      </div>
    </div>
  )
}
