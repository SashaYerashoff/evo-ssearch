import { useEffect, useState } from 'react'
import { IconX, IconDeviceFloppy } from '@tabler/icons-react'
import { buildPromptSettingsPayload, videoApi, type PromptSettings } from '../../api/video'

const TABS: { key: string; label: string }[] = [
  { key: 'stream', label: 'Stream' }, { key: 'alerts', label: 'Alerts' },
  { key: 'L1', label: 'L1 · minutes' }, { key: 'L2', label: 'L2 · hours' }, { key: 'L3', label: 'L3 · days' },
  { key: 'json', label: 'JSON alert' },
]

export function PromptSettingsModal({
  channelId,
  canCreateBookmarks,
  onClose,
}: {
  channelId: number
  canCreateBookmarks: boolean
  onClose: () => void
}) {
  const [s, setS] = useState<PromptSettings>({ rollup_prompts: {} })
  const [tab, setTab] = useState('stream')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [loadErr, setLoadErr] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const tabs = canCreateBookmarks ? TABS : TABS.filter((item) => item.key !== 'json')

  useEffect(() => {
    let alive = true
    setLoaded(false)
    setLoadErr(null)
    setS({ rollup_prompts: {} })
    videoApi.getPromptSettings(channelId)
      .then((d) => {
        if (!alive) return
        setS({ ...d, rollup_prompts: d.rollup_prompts || {} })
        setLoaded(true)
      })
      .catch((error: any) => {
        if (!alive) return
        setLoadErr(error?.message || 'Saved prompts and alerts could not be loaded.')
        setLoaded(true)
      })
    return () => { alive = false }
  }, [channelId])

  const getVal = () => {
    if (tab === 'stream') return s.stream_system_prompt || ''
    if (tab === 'alerts') return s.alert_policy_prompt || ''
    if (tab === 'json') return s.json_alert_prompt || ''
    return s.rollup_prompts?.[tab as 'L1' | 'L2' | 'L3'] || ''
  }
  const setVal = (v: string) => {
    if (tab === 'stream') setS({ ...s, stream_system_prompt: v })
    else if (tab === 'alerts') setS({ ...s, alert_policy_prompt: v })
    else if (tab === 'json') setS({ ...s, json_alert_prompt: v })
    else setS({ ...s, rollup_prompts: { ...s.rollup_prompts, [tab]: v } })
  }

  async function save() {
    setBusy(true); setErr(null)
    try {
      const payload = buildPromptSettingsPayload(s, channelId, canCreateBookmarks)
      await videoApi.savePromptSettings(payload)
      onClose()
    }
    catch (e: any) { setErr(e?.message || 'Save failed') } finally { setBusy(false) }
  }

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 680 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">System prompt settings · ch {channelId}</div>
          <button className="modal-close" onClick={onClose}><IconX size={18} /></button>
        </div>
        <div className="modal-body">
          <div className="vid-tabs">
            {tabs.map((t) => <button key={t.key} className={`vid-tab ${tab === t.key ? 'on' : ''}`} onClick={() => setTab(t.key)}>{t.label}</button>)}
          </div>
          {!loaded ? (
            <div className="vid-prompt-loading" role="status"><span className="spinner" /> Loading saved prompts and alerts…</div>
          ) : loadErr ? (
            <div className="chat-error" role="alert">{loadErr} Close this window and try again; nothing was overwritten.</div>
          ) : (
            <textarea className="vid-prompt-area" value={getVal()} onChange={(e) => setVal(e.target.value)}
              placeholder="Prompt for this layer…" rows={12} />
          )}
          {loaded && !loadErr && canCreateBookmarks && <div className="vid-bookmark-row">
            <label className="mon-check"><input type="checkbox" checked={!!s.bookmark_enabled} onChange={(e) => setS({ ...s, bookmark_enabled: e.target.checked })} /> Make bookmarks on alerts</label>
            <div className="wfield" style={{ maxWidth: 150 }}><label>Cooldown (s)</label>
              <input type="number" step="0.5" min="0" value={s.bookmark_cooldown_sec ?? 5} onChange={(e) => setS({ ...s, bookmark_cooldown_sec: Number(e.target.value) })} />
            </div>
          </div>}
          {err && <div className="chat-error">{err}</div>}
          <div className="mon-modal-actions">
            <button className="mon-btn" onClick={onClose}>Close</button>
            <button className="mon-btn accent" disabled={busy || !loaded || !!loadErr} onClick={save}><IconDeviceFloppy size={15} /> Save</button>
          </div>
        </div>
      </div>
    </div>
  )
}
