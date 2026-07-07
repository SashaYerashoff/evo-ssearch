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
  list_probes: 'Probes', survey_channels: 'Channel survey', create_probe: 'Create probe', update_probe: 'Update probe',
  delete_probes: 'Delete probes', deploy_summary: 'Deploy summary', describe_frame: 'Describe frame',
  get_video_summaries: 'Video summaries', count_video_summary_events: 'Summary events',
  track_visual_state_transitions: 'State transitions', create_bookmark: 'Bookmark', generate_report: 'Report',
  normalize_time_window: 'Time window', build_research_batch: 'Research batch',
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

export function ActionCard({ action, onThumb, onApply }: {
  action: ToolAction
  onThumb: (url: string, title: string) => void
  onApply: (a: ToolAction) => void
}) {
  const { name, result, error } = action
  const items = itemsOf(result)
  const isApproval = !!action.planId || result?.status === 'preview'
  const describeImg = name === 'describe_frame' ? agentImageUrl(result) : ''
  const text = result?.description || result?.summary || result?.note || result?.text || result?.message

  if (isApproval) {
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
        {!items.length && !describeImg && !text && (
          <div className="ag-fields">
            {entriesOf(result).map(([k, v]) => <div key={k} className="ag-field"><span className="ag-field-k">{k}</span><span className="ag-field-v">{v}</span></div>)}
          </div>
        )}
      </div>
    </div>
  )
}
