import { useEffect, useState, useCallback } from 'react'
import { IconX, IconDeviceFloppy, IconHelpCircle, IconRotate, IconSearch } from '@tabler/icons-react'
import type { AuthUser, Channel } from '../../api/types'
import { canViewSettingsTab, hasPermission, PERMISSION } from '../../api/access'
import { normalizeArchiveCapacity, settingsApi, type Settings } from '../../api/settings'
import { TABS, WRITABLE_KEYS, SEVERITY_KEYS, type FieldDef } from './defs'
import { EnvTab } from './EnvTab'
import { AuditTab } from './AuditTab'
import { UsersTab } from './UsersTab'
import { DiagnosticsTab } from './DiagnosticsTab'
import { Dropdown } from '../shell/Dropdown'
import { AppearanceModal } from '../appearance/AppearanceModal'
import { buildSettingsPatch } from './settingsPatch'

const DEFAULTS: Settings = {
  host: '0.0.0.0', port: 5000, debug: false,
  minResults: 3, maxResults: 48, defaultResults: 12,
  embedder: 'clip', clipModel: 'ViT-B/32', dinoModel: 'dinov3_vitb16', dinoEmbedDim: 1280, dinoWeightsPath: '',
  fusionEnabled: false, fusionAlpha: 0.7, batchSize: 32, thumbnailQuality: 85,
  vlmBaseUrl: '', vlmModel: '', vlmApiKey: '', vlmTimeout: 600,
  agentBaseUrl: '', agentModel: '', agentApiKey: '', agentTimeout: 600,
  rerankEnabled: false, rerankTopK: 50, segmentsEnabled: false, segmentMinPatches: 3, maxCommentLength: 100, maxFileSize: 50,
  luxriotBaseUrl: '', luxriotUsername: '', luxriotPassword: '', luxriotDefaultChannelId: 1,
  luxriotSnapshotInterval: 5, luxriotSnapshotMaxEdge: 800, luxriotMaxBufferFrames: 180,
  luxriotSummaryRetentionDays: 7, luxriotSummaryHistoryLimit: 10080, luxriotAutoBookmarks: false,
  probeBookmarkCooldownSec: 8, probeBookmarkDedupeWindowSec: 20, probeBookmarkSimHigh: 0.985,
  probeBookmarkMarginDelta: 0.08, probeBookmarkScoreDelta: 0.08, probeBookmarkMaxFrameGap: 8,
  luxriotSeverityMap: { info: 'info', low: 'low', normal: 'normal', high: 'high', critical: 'critical' },
  archiveRetentionEnabled: true, archiveRowRetentionDays: 90, archiveThumbnailRetentionDays: 14, archiveMaxRecords: 5000000,
  archiveEstimateChannels: 50, archiveEstimateFramesPerBatch: 4, archiveEstimateAvgJpegKb: 100, archiveEstimateProbeRecordsPerChannelDay: 250,
}

function FieldRow({ f, value, disabled, onChange }: { f: FieldDef; value: any; disabled?: boolean; onChange: (v: any) => void }) {
  const label = <span className="set-label">{f.label}</span>
  if (f.type === 'checkbox') {
    return <label className="set-row set-check"><input type="checkbox" disabled={disabled} checked={!!value} onChange={(e) => onChange(e.target.checked)} />{label}</label>
  }
  if (f.type === 'range') {
    return (
      <div className="set-row">
        {label}
        <div className="set-range-wrap">
          <input type="range" disabled={disabled} min={f.min} max={f.max} step={f.step} value={Number(value ?? 0)} onChange={(e) => onChange(Number(e.target.value))} />
          <span className="set-range-val">{value}</span>
        </div>
        {f.note && <div className="set-note">{f.note}</div>}
      </div>
    )
  }
  return (
    <div className="set-row">
      {label}
      {f.type === 'select'
        ? <Dropdown disabled={disabled} value={String(value ?? '')} onChange={onChange} options={(f.options || []).map((o) => ({ value: o.v, label: o.label || o.v }))} />
        : <input type={f.type === 'number' ? 'number' : f.type === 'password' ? 'password' : 'text'}
            disabled={disabled}
            min={f.min} max={f.max} step={f.step}
            value={f.type === 'password' ? (value ?? '') : (value ?? '')}
            placeholder={f.type === 'password' ? '•••••• (unchanged)' : ''}
            onChange={(e) => onChange(f.type === 'number' ? (e.target.value === '' ? '' : Number(e.target.value)) : e.target.value)} />}
      {f.note && <div className="set-note">{f.note}</div>}
    </div>
  )
}

export function SettingsModal({
  user,
  channels,
  onRefreshChannels,
  showIncidents,
  onShowIncidentsChange,
  onClose,
}: {
  user: AuthUser
  channels: Channel[]
  onRefreshChannels?: () => Promise<void> | void
  showIncidents: boolean
  onShowIncidentsChange: (enabled: boolean) => void
  onClose: () => void
}) {
  const [s, setS] = useState<Settings>({})
  const [tab, setTab] = useState('server')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [dirtyKeys, setDirtyKeys] = useState<Set<string>>(() => new Set())
  const [status, setStatus] = useState<{ msg: string; ok: boolean } | null>(null)
  const [settingsLoadError, setSettingsLoadError] = useState('')
  const [capacity, setCapacity] = useState<any>(null)
  const [capacityLoading, setCapacityLoading] = useState(false)
  const [capacityError, setCapacityError] = useState('')
  const [search, setSearch] = useState('')
  const canReadSettings = canViewSettingsTab(user, 'settings')
  const canManageSettings = hasPermission(user, PERMISSION.settingsManage)
  const canReadCapacity = hasPermission(user, PERMISSION.diagnosticsView)

  const mergeSettings = useCallback((settings?: Settings) => ({
    ...DEFAULTS,
    ...(settings || {}),
    luxriotSeverityMap: {
      ...DEFAULTS.luxriotSeverityMap,
      ...(settings?.luxriotSeverityMap || {}),
    },
  }), [])

  const loadCapacity = useCallback(async (includeCurrent = false) => {
    if (!canReadCapacity) return
    setCapacityLoading(true)
    setCapacityError('')
    try {
      const result = await settingsApi.archiveCapacity(includeCurrent)
      if (!result?.success) throw new Error(result?.error || 'Capacity estimate failed')
      setCapacity(result)
    } catch (error: any) {
      setCapacityError(error?.message || 'Capacity estimate unavailable')
    } finally {
      setCapacityLoading(false)
    }
  }, [canReadCapacity])

  useEffect(() => {
    if (canReadSettings) {
      settingsApi.get()
        .then((r) => {
          if (!r?.success || !r.settings) throw new Error('Settings response is incomplete')
          setS(mergeSettings(r.settings))
          setDirtyKeys(new Set())
          setSettingsLoadError('')
        })
        .catch((error: any) => {
          const message = error?.message || 'Failed to load settings'
          setS(mergeSettings())
          setSettingsLoadError(message)
          setStatus({ msg: message, ok: false })
        })
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
    void loadCapacity(false)
  }, [canReadSettings, loadCapacity, mergeSettings])

  const patch = useCallback((k: string, v: any) => {
    setS((x) => ({ ...x, [k]: v }))
    setDirtyKeys((current) => new Set(current).add(k))
  }, [])
  const patchSev = (k: string, v: string) => {
    setS((x) => ({ ...x, luxriotSeverityMap: { ...(x.luxriotSeverityMap || {}), [k]: v } }))
    setDirtyKeys((current) => new Set(current).add('luxriotSeverityMap'))
  }

  async function save() {
    setSaving(true); setStatus(null)
    try {
      const payload = buildSettingsPatch(s, dirtyKeys, WRITABLE_KEYS)
      if (!Object.keys(payload).length) throw new Error('No settings changed')
      if (Number(s.minResults) > Number(s.maxResults)) throw new Error('Min results must not exceed max results')
      if (Number(s.defaultResults) < Number(s.minResults) || Number(s.defaultResults) > Number(s.maxResults)) {
        throw new Error('Default results must be between min and max results')
      }
      const r = await settingsApi.save(payload)
      if (!r.success) throw new Error(r.error || 'Save failed')
      setS((current) => ({ ...current, luxriotPassword: '', vlmApiKey: '', agentApiKey: '' }))
      setDirtyKeys(new Set())
      const pending = r.pendingOrOverriddenKeys || []
      const restartFields = r.restartRequiredFields || []
      const sourceUnknown = r.precedence?.declared_file_matches_project === false
      const detail = restartFields.length
        ? ` Restart required for: ${restartFields.join(', ')}.`
        : pending.length
          ? ` ${pending.length} persisted environment value${pending.length === 1 ? '' : 's'} differ from the startup environment; runtime-safe fields were applied.`
        : ''
      const sourceWarning = sourceUnknown
        ? ' Service env ownership is not declared; verify the systemd EnvironmentFile before restart.'
        : ''
      setStatus({ msg: `${r.message || 'Settings saved.'}${detail}${sourceWarning}`, ok: !sourceUnknown })
    } catch (e: any) { setStatus({ msg: e?.message || 'Save failed', ok: false }) }
    finally { setSaving(false) }
  }
  function reset() {
    setS((x) => ({ ...x, ...DEFAULTS, luxriotSeverityMap: { ...DEFAULTS.luxriotSeverityMap } }))
    setDirtyKeys(new Set(WRITABLE_KEYS.filter((key) => !['luxriotPassword', 'vlmApiKey', 'agentApiKey'].includes(key))))
    setStatus({ msg: 'Reverted to defaults — press Save settings to persist.', ok: true })
  }

  const q = search.trim().toLowerCase()
  const fieldMatch = (label: string) => label.toLowerCase().includes(q)
  const tabHasMatch = (t: (typeof TABS)[number]) => {
    if (!q) return true
    if (t.label.toLowerCase().includes(q)) return true
    if (t.custom) return `${t.custom} ${t.searchTerms || ''}`.includes(q)
    return !!t.sections?.some((sec) => sec.title.toLowerCase().includes(q) || sec.fields?.some((f) => fieldMatch(f.label)))
  }
  const tabKind = (custom?: string) => custom === 'users'
    ? 'users'
    : custom === 'audit'
      ? 'audit'
      : custom === 'env'
        ? 'env'
        : custom === 'diagnostics'
          ? 'diagnostics'
          : 'settings'
  const permittedTabs = TABS.filter((candidate) => (
    candidate.custom === 'appearance'
    || candidate.custom === 'features'
    || canViewSettingsTab(user, tabKind(candidate.custom))
  ))
  const visibleTabs = permittedTabs.filter(tabHasMatch)
  const activeId = visibleTabs.some((t) => t.id === tab) ? tab : (visibleTabs[0]?.id ?? tab)
  const activeTab = permittedTabs.find((t) => t.id === activeId) ?? permittedTabs[0]
  const localPreferenceTab = activeTab?.custom === 'appearance' || activeTab?.custom === 'features'
  const capacitySummary = normalizeArchiveCapacity(capacity)
  const sourceDeclared = s.envPrecedence?.declared_file_matches_project === true
  const sourceWritable = s.envPrecedence?.write_allowed !== false
  const pendingSourceKeys = s.envPrecedence?.different_process_and_file_keys || []

  return (
    <div className="scrim" onClick={onClose}>
      <div className="modal set-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">Settings</div>
            <div className="brand-sub">Tune appearance, feature visibility, runtime, ranking, Luxriot integration, and environment.</div>
          </div>
          <div className="set-actions">
            <a
              className="mon-btn"
              href="/ui-assets/quick-start.html"
              target="_blank"
              rel="noreferrer"
              title="Open the EVA AI operator quick-start guide"
            >
              <IconHelpCircle size={15} /> Operator guide
            </a>
            <button className="modal-close" onClick={onClose}><IconX size={18} /></button>
          </div>
        </div>

        <div className="set-body">
          <aside className="set-nav">
            <div className="set-search">
              <IconSearch size={15} />
              <input placeholder="Search settings…" value={search} onChange={(e) => setSearch(e.target.value)} />
            </div>
            <div className="set-navlist">
              {visibleTabs.map((t) => (
                <button key={t.id} className={`set-navitem ${activeId === t.id ? 'on' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
              ))}
              {visibleTabs.length === 0 && <div className="set-note" style={{ padding: '10px 12px' }}>No matches.</div>}
            </div>
          </aside>

          <div className="set-content">
            {loading && (
              <div className="set-loading">
                <div className="spinner" />
                <div>
                  <b>Loading runtime settings…</b>
                  <span>Archive capacity is loaded separately and cannot block these controls.</span>
                </div>
              </div>
            )}
            {!loading && settingsLoadError && !localPreferenceTab && (
              <div className="set-load-error">{settingsLoadError}. Showing safe defaults; saving is disabled until the live configuration can be read.</div>
            )}
            {!loading && !settingsLoadError && !localPreferenceTab && (
              <div className={`set-source-state ${sourceDeclared ? (pendingSourceKeys.length ? 'pending' : 'aligned') : 'unknown'}`}>
                {sourceDeclared ? (
                  <>
                    <b>Configuration source:</b> <code>{s.envPrecedence?.persistence_source || s.envFile}</code>.
                    {pendingSourceKeys.length
                      ? ` ${pendingSourceKeys.length} persisted value${pendingSourceKeys.length === 1 ? '' : 's'} differ from the startup environment; runtime-safe fields may already be active, while restart-only fields still need a restart.`
                      : ' The running process was started with this declared file; there are no detected startup/file differences.'}
                  </>
                ) : (
                  <>
                    <b>Configuration source is not declared.</b> A secure deployment cannot save settings until <code>EVOSSEARCH_CONFIG_ENV_FILE</code> identifies the service-owned file.
                  </>
                )}
              </div>
            )}
            {!loading && activeId === 'models' && (
              <div className="set-note" style={{ marginBottom: 12 }}>
                Editing active profiles: VLM <b>{s.vlmProfileId || 'default'}</b>, Agent <b>{s.agentProfileId || 'default'}</b>. Other configured inference profiles are preserved.
              </div>
            )}
            {!loading && !activeTab && (
              <div className="set-load-error">
                This account can open Settings but has no readable settings tab. Ask an administrator to grant settings:view, users:manage, or audit:view.
              </div>
            )}
            {!loading && activeTab?.custom === 'env' && <EnvTab />}
            {!loading && activeTab?.custom === 'audit' && <AuditTab />}
            {!loading && activeTab?.custom === 'diagnostics' && <DiagnosticsTab />}
            {!loading && activeTab?.custom === 'appearance' && <AppearanceModal embedded onClose={onClose} />}
            {!loading && activeTab?.custom === 'features' && (
              <div className="set-section set-feature-section">
                <h3>
                  Operator features
                  <span className={`set-exp ${showIncidents ? 'on' : ''}`}>feature in progress</span>
                </h3>
                <p className="set-section-help">
                  Keep developing operator surfaces visible while validating them, or hide them on this workstation if their output does not match operational expectations.
                </p>
                <label className="set-row set-check set-feature-toggle">
                  <input
                    type="checkbox"
                    checked={showIncidents}
                    onChange={(event) => onShowIncidentsChange(event.target.checked)}
                  />
                  <span>
                    <b>Show incidents (FiP)</b>
                    <small>Show Incident Review in the Video workspace.</small>
                  </span>
                </label>
                <div className="set-feature-advisory">
                  This preference applies immediately in this browser. Turning it off hides and stops the Incident Review UI, but does not delete incident history or disable backend incident processing.
                </div>
              </div>
            )}
            {!loading && activeTab?.custom === 'users' && (
              <UsersTab
                currentUserId={user.id}
                currentSessionId={user.currentSessionId}
                channels={channels}
                onRefreshChannels={onRefreshChannels}
              />
            )}
            {!loading && activeTab && !activeTab.custom && activeTab.sections?.map((sec, i) => {
              const titleMatch = !q || sec.title.toLowerCase().includes(q)
              if (sec.kind) {
                if (!titleMatch) return null
                return (
                  <div key={i} className="set-section">
                    <h3>{sec.title}</h3>
                    {sec.help && <p className="set-section-help">{sec.help}</p>}
                    {sec.kind === 'severity' ? (
                      <div className="set-sev">
                        {SEVERITY_KEYS.map((k) => (
                          <div key={k} className="set-row"><span className="set-label">{k}</span>
                            <input disabled={!canManageSettings} value={s.luxriotSeverityMap?.[k] ?? k} onChange={(e) => patchSev(k, e.target.value)} /></div>
                        ))}
                      </div>
                    ) : (
                      <div className="set-capacity-wrap">
                        <div className="set-capacity">
                          {capacity ? (
                            <>
                              <div><span>Daily frame rows</span><b>{fmt(capacitySummary.dailyFrameRows)}</b></div>
                              <div><span>Retained frame rows</span><b>{fmt(capacitySummary.retainedFrameRows)}</b></div>
                              <div><span>Estimated storage</span><b>{fmtBytes(capacitySummary.totalBytes)}</b></div>
                              <div>
                                <span>Current archive rows</span>
                                <b>{capacitySummary.currentRows == null ? 'Not scanned' : fmt(capacitySummary.currentRows)}</b>
                              </div>
                            </>
                          ) : <div className="set-note">Capacity estimate unavailable.</div>}
                        </div>
                        <div className="set-capacity-actions">
                          <button className="mon-btn" disabled={capacityLoading || !canReadCapacity} onClick={() => void loadCapacity(true)}>
                            <IconRotate size={14} /> {capacityLoading ? 'Reading archive…' : 'Scan current archive'}
                          </button>
                          <span>
                            Estimates load immediately. The current-row scan is manual because a large PostgreSQL archive may take several seconds.
                          </span>
                        </div>
                        {capacityError && <div className="set-load-error">{capacityError}</div>}
                      </div>
                    )}
                  </div>
                )
              }
              const fields = (sec.fields || []).filter((f) => !q || titleMatch || fieldMatch(f.label))
              if (!fields.length) return null
              return (
                <div key={i} className="set-section">
                  <h3>{sec.title} {sec.experimental && (
                    <span className={`set-exp ${s.experimentalEmbeddersEnabled ? 'on' : ''}`}
                      title={s.experimentalEmbeddersEnabled
                        ? 'Experimental embedders are enabled (EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED=true)'
                        : 'Ignored while experimental embedders are disabled — set EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED=true and restart'}>
                      experimental · {s.experimentalEmbeddersEnabled ? 'on' : 'off'}
                    </span>
                  )}</h3>
                  {sec.help && <p className="set-section-help">{sec.help}</p>}
                  <div className="set-fields">
                    {fields.map((f) => <FieldRow key={f.key} f={f} value={s[f.key]} disabled={!canManageSettings || !sourceWritable} onChange={(v) => patch(f.key, v)} />)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {activeTab?.custom !== 'appearance' && (
          <div className="set-footer">
            <div className={`set-status ${status ? (status.ok ? 'ok' : 'err') : ''}`}>{status?.msg || ''}</div>
            {activeTab && !activeTab.custom && canManageSettings && !loading && !settingsLoadError && (
              <div className="set-actions">
                <button className="mon-btn" disabled={!sourceWritable} onClick={reset}><IconRotate size={15} /> Reset to defaults</button>
                <button className="mon-btn accent" disabled={!sourceWritable || saving || dirtyKeys.size === 0} onClick={save}><IconDeviceFloppy size={15} /> {saving ? 'Saving…' : `Save settings${dirtyKeys.size ? ` (${dirtyKeys.size})` : ''}`}</button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function fmt(n: any): string {
  if (n === null || n === undefined || n === '') return '—'
  const v = Number(n)
  return isFinite(v) ? v.toLocaleString() : '—'
}

function fmtBytes(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  let value = Math.max(0, n)
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++ }
  return `${value.toLocaleString(undefined, { maximumFractionDigits: value >= 100 ? 0 : 1 })} ${units[unit]}`
}
