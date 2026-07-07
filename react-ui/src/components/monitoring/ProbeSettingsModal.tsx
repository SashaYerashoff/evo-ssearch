import { useState } from 'react'
import { IconX, IconPlus, IconTrash, IconPhoto, IconDeviceFloppy } from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import type { Probe, ProbeInput } from '../../api/probes'

const SEVERITIES = ['info', 'low', 'normal', 'high', 'critical']

interface Draft {
  id?: string
  name: string
  channel_id?: number
  enabled: boolean
  pairs: { pos: string; neg: string }[]
  pos_floor: number
  margin: number
  bookmark: boolean
  severity: string
  cooldown: number
  dedupe: number
  imgData: string | null
  imgName: string
  imgFloor: number
  imgEnabled: boolean
}

function fromProbe(p: Probe | null, channels: Channel[]): Draft {
  // positives[]/negatives[] are authoritative — zip them into editable pairs
  const pos = p?.positives || []
  const neg = p?.negatives || []
  const len = Math.max(pos.length, neg.length, 1)
  const pairs = Array.from({ length: len }, (_, i) => ({ pos: pos[i] || '', neg: neg[i] || '' }))
  return {
    id: p?.id,
    name: p?.name || '',
    channel_id: p?.channel_id ?? channels[0]?.id,
    enabled: p?.enabled ?? true,
    pairs,
    pos_floor: p?.pos_floor ?? 0.2,
    margin: p?.margin ?? 0.05,
    bookmark: p?.bookmark ?? false,
    severity: p?.severity || 'info',
    cooldown: p?.bookmark_cooldown_sec ?? 8,
    dedupe: p?.bookmark_dedupe_window_sec ?? 20,
    imgData: p?.image_probe?.data ?? null,
    imgName: p?.image_probe?.name || '',
    imgFloor: p?.image_probe?.pos_floor ?? 0.7,
    imgEnabled: p?.image_probe?.enabled ?? false,
  }
}

export function ProbeSettingsModal({ probe, channels, busy, onClose, onSave }: {
  probe: Probe | null
  channels: Channel[]
  busy: boolean
  onClose: () => void
  onSave: (input: ProbeInput) => void
}) {
  const [d, setD] = useState<Draft>(() => fromProbe(probe, channels))
  const set = (p: Partial<Draft>) => setD((x) => ({ ...x, ...p }))
  const setPair = (i: number, p: Partial<{ pos: string; neg: string }>) =>
    setD((x) => ({ ...x, pairs: x.pairs.map((pr, j) => (j === i ? { ...pr, ...p } : pr)) }))

  function onFile(f: File) {
    const r = new FileReader()
    r.onload = () => { const s = String(r.result || ''); set({ imgData: s.includes(',') ? s.split(',')[1] : s, imgName: f.name, imgEnabled: true }) }
    r.readAsDataURL(f)
  }

  function save() {
    const positives = d.pairs.map((p) => p.pos.trim()).filter(Boolean)
    const negatives = d.pairs.map((p) => p.neg.trim()).filter(Boolean)
    const input: ProbeInput = {
      id: d.id,
      name: d.name.trim() || 'Untitled probe',
      channel_id: d.channel_id,
      enabled: d.enabled,
      pairs: d.pairs.filter((p) => p.pos.trim() || p.neg.trim()).map((p) => ({ positive: p.pos.trim(), negative: p.neg.trim() })),
      positives, negatives,
      pos_floor: d.pos_floor, margin: d.margin, top_k: 6,
      window_sec: probe?.window_sec ?? 300,
      severity: d.severity,
      bookmark: d.bookmark,
      bookmark_cooldown_sec: d.cooldown,
      bookmark_dedupe_window_sec: d.dedupe,
      image_probe: d.imgData ? { data: d.imgData, name: d.imgName, pos_floor: d.imgFloor, enabled: d.imgEnabled } : null,
    }
    onSave(input)
  }

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal mon-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">{d.id ? 'CLIP probe settings' : 'New CLIP probe'}</div>
          <button className="modal-close" onClick={onClose}><IconX size={18} /></button>
        </div>
        <div className="modal-body">
          <div className="wform">
            <div className="wgrid">
              <div className="wfield"><label>Probe name</label>
                <input value={d.name} onChange={(e) => set({ name: e.target.value })} placeholder="Descriptive name" autoFocus />
              </div>
              <div className="wfield"><label>Channel</label>
                <select value={d.channel_id ?? ''} onChange={(e) => set({ channel_id: Number(e.target.value) })}>
                  {channels.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
                </select>
              </div>
            </div>

            <label className="mon-check"><input type="checkbox" checked={d.enabled} onChange={(e) => set({ enabled: e.target.checked })} /> Enabled</label>

            <div className="mon-heading">Text probe thresholds</div>
            <div className="wgrid">
              <div className="wfield"><label>Positive floor</label>
                <input type="number" step="0.01" value={d.pos_floor} onChange={(e) => set({ pos_floor: Number(e.target.value) })} />
              </div>
              <div className="wfield"><label>Margin</label>
                <input type="number" step="0.01" min="0" value={d.margin} onChange={(e) => set({ margin: Number(e.target.value) })} />
              </div>
            </div>

            <div className="mon-heading">Text pairs</div>
            <div className="mon-pairs">
              {d.pairs.map((pr, i) => (
                <div key={i} className="mon-pair">
                  <input placeholder="Positive prompt" value={pr.pos} onChange={(e) => setPair(i, { pos: e.target.value })} />
                  <input placeholder="Negative prompt" value={pr.neg} onChange={(e) => setPair(i, { neg: e.target.value })} />
                  <button className="mon-icobtn danger" title="Remove" disabled={d.pairs.length === 1}
                    onClick={() => set({ pairs: d.pairs.filter((_, j) => j !== i) })}><IconTrash size={14} /></button>
                </div>
              ))}
              <button className="mon-btn" onClick={() => set({ pairs: [...d.pairs, { pos: '', neg: '' }] })}><IconPlus size={14} /> Add pair</button>
            </div>

            <div className="mon-heading">Bookmarks</div>
            <div className="wgrid">
              <label className="mon-check"><input type="checkbox" checked={d.bookmark} onChange={(e) => set({ bookmark: e.target.checked })} /> Make bookmarks</label>
              <div className="wfield"><label>Severity</label>
                <select value={d.severity} onChange={(e) => set({ severity: e.target.value })}>
                  {SEVERITIES.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>
            {d.bookmark && (
              <div className="wgrid">
                <div className="wfield"><label>Cooldown (s)</label>
                  <input type="number" step="0.5" min="0" value={d.cooldown} onChange={(e) => set({ cooldown: Number(e.target.value) })} />
                </div>
                <div className="wfield"><label>Dedup window (s)</label>
                  <input type="number" step="0.5" min="0.5" value={d.dedupe} onChange={(e) => set({ dedupe: Number(e.target.value) })} />
                </div>
              </div>
            )}

            <div className="mon-heading">Image probe</div>
            <div className="mon-img-row">
              <label className="mon-btn"><IconPhoto size={14} /> {d.imgName || 'Choose image'}
                <input type="file" accept="image/*" style={{ display: 'none' }}
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.currentTarget.value = '' }} />
              </label>
              {d.imgData && (
                <>
                  <img className="mon-img-preview" src={`data:image/jpeg;base64,${d.imgData}`} alt="probe" />
                  <label className="mon-check"><input type="checkbox" checked={d.imgEnabled} onChange={(e) => set({ imgEnabled: e.target.checked })} /> Enabled</label>
                  <div className="wfield" style={{ maxWidth: 130 }}><label>Min match</label>
                    <input type="number" step="0.01" min="0" max="1" value={d.imgFloor} onChange={(e) => set({ imgFloor: Number(e.target.value) })} />
                  </div>
                  <button className="mon-icobtn danger" title="Clear image" onClick={() => set({ imgData: null, imgName: '', imgEnabled: false })}><IconTrash size={14} /></button>
                </>
              )}
            </div>

            <div className="mon-modal-actions">
              <button className="mon-btn" onClick={onClose}>Close</button>
              <button className="mon-btn accent" disabled={busy} onClick={save}><IconDeviceFloppy size={15} /> Save probe</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
