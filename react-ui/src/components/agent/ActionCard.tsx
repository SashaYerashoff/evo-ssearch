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

export function ActionCard({ action, onThumb, onApply }: {
  action: ToolAction
  onThumb: (url: string, title: string) => void
  onApply: (a: ToolAction) => void
}) {
  const { name, result, error } = action
  const items = itemsOf(result)
  const isApproval = !!action.applied || !!action.planId || result?.status === 'preview'
  const describeImg = name === 'describe_frame' ? agentImageUrl(result) : ''
  const text = result?.description || result?.summary || result?.note || result?.text || result?.message
  const tables = actionTables(name, result)

  if (isApproval) {
    if (name === 'apply_deployment_plan' && result?.status === 'preview') {
      const diff = result?.diff || {}
      const channels = Array.isArray(result?.per_channel) ? result.per_channel : []
      return (
        <div className="ag-approval ag-deployment-approval">
          <div className="ag-approval-head">
            <div>
              <div className="ag-approval-kick">Operator approval required</div>
              <div className="ag-approval-title">PROTOCOL DEPLOY · {result?.deployment_id || 'draft'}</div>
            </div>
            <span className={`ag-approval-status ${action.applied ? 'ok' : ''}`}>{action.applied ? 'Applied' : 'Preview only'}</span>
          </div>
          <div className="ag-fields">
            <div className="ag-field"><span className="ag-field-k">Channels</span><span className="ag-field-v">{textValue(diff.channel_ids)}</span></div>
            <div className="ag-field"><span className="ag-field-k">Policies</span><span className="ag-field-v">{textValue(diff.alert_policy_count)}</span></div>
            <div className="ag-field"><span className="ag-field-k">Attention probes</span><span className="ag-field-v">{textValue(diff.probe_count)}</span></div>
            <div className="ag-field"><span className="ag-field-k">Counters</span><span className="ag-field-v">{textValue(diff.counted_state_count)}</span></div>
          </div>
          {channels.map((channel: any) => (
            <details className="ag-deployment-channel" key={String(channel?.channel_id)}>
              <summary>CH {channel?.channel_id} · review proposed alert policy</summary>
              <pre>{String(channel?.alert_policy_preview || 'No default alert policy proposed.')}</pre>
            </details>
          ))}
          <div className="ag-approval-note">Nothing changes until Apply succeeds. To revise the draft, describe the channel and correction in chat.</div>
          {error && <div className="ag-card-err">{error}</div>}
          {!action.applied && action.planId && (
            <div className="ag-approval-foot">
              <button className="ag-apply" disabled={action.applying} onClick={() => onApply(action)}>
                {action.applying ? 'Applying deployment…' : 'Apply deployment'}
              </button>
            </div>
          )}
        </div>
      )
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
