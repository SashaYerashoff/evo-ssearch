import { useEffect, useMemo, useState } from 'react'
import {
  IconCheck,
  IconColorSwatch,
  IconDeviceFloppy,
  IconRestore,
  IconTrash,
  IconTypography,
  IconX,
} from '@tabler/icons-react'
import {
  DEFAULT_APPEARANCE,
  CUSTOM_APPEARANCE_PRESETS_STORAGE_KEY,
  THEME_PRESETS,
  contrastRatio,
  contrastText,
  getThemePreset,
  hasCustomColors,
  normalizeHex,
  normalizeSavedAppearancePresets,
  resolveThemePalette,
  type AppearancePreferences,
  type CustomColorKey,
  type SavedAppearancePreset,
} from '../../appearance/appearance'
import { useAppearance } from '../../appearance/AppearanceProvider'
import { useI18n, type UiLanguage } from '../../i18n/I18nProvider'

const COLOR_FIELDS: ReadonlyArray<{ key: CustomColorKey; label: string; help: string }> = [
  { key: 'canvas', label: 'Canvas', help: 'Application background' },
  { key: 'surface', label: 'Surfaces', help: 'Panels and modal bodies' },
  { key: 'control', label: 'Controls', help: 'Fields and secondary buttons' },
  { key: 'text', label: 'Text', help: 'Primary readable content' },
  { key: 'accent', label: 'Accent', help: 'Primary actions and focus' },
]

function readCustomPresets(): SavedAppearancePreset[] {
  try {
    return normalizeSavedAppearancePresets(JSON.parse(window.localStorage.getItem(CUSTOM_APPEARANCE_PRESETS_STORAGE_KEY) || '[]'))
  } catch {
    return []
  }
}

export function AppearanceModal({ onClose }: { onClose: () => void }) {
  const { language, setLanguage, t } = useI18n()
  const {
    savedPreferences,
    previewPreferences,
    cancelPreview,
    commitPreferences,
  } = useAppearance()
  const [draft, setDraft] = useState<AppearancePreferences>(() => ({
    ...savedPreferences,
    overrides: { ...savedPreferences.overrides },
  }))
  const [customPresets, setCustomPresets] = useState<SavedAppearancePreset[]>(readCustomPresets)
  const [customPresetName, setCustomPresetName] = useState('')
  const palette = useMemo(() => resolveThemePalette(draft), [draft])
  const contrastWarnings = useMemo(() => {
    const warnings: string[] = []
    if (contrastRatio(palette.text, palette.canvas) < 4.5) warnings.push('Text needs more contrast against the canvas.')
    if (contrastRatio(palette.text, palette.surface) < 4.5) warnings.push('Text needs more contrast against panel surfaces.')
    if (contrastRatio(palette.accent, palette.surface) < 3) warnings.push('Accent is too close to the panel color.')
    if (contrastRatio(contrastText(palette.accent), palette.accent) < 4.5) warnings.push('Accent cannot support readable primary buttons.')
    return warnings
  }, [palette])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        cancelPreview()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cancelPreview, onClose])

  function update(next: AppearancePreferences) {
    setDraft(next)
    previewPreferences(next)
  }

  function closeWithoutSaving() {
    cancelPreview()
    onClose()
  }

  function apply() {
    commitPreferences(draft)
    onClose()
  }

  function choosePreset(preset: AppearancePreferences['preset']) {
    update({ ...draft, preset, overrides: {} })
  }

  function setColor(key: CustomColorKey, value: string) {
    const color = normalizeHex(value)
    if (!color) return
    update({ ...draft, overrides: { ...draft.overrides, [key]: color } })
  }

  function resetColors() {
    update({ ...draft, overrides: {} })
  }

  function persistCustomPresets(next: SavedAppearancePreset[]) {
    const normalized = normalizeSavedAppearancePresets(next)
    setCustomPresets(normalized)
    try {
      window.localStorage.setItem(CUSTOM_APPEARANCE_PRESETS_STORAGE_KEY, JSON.stringify(normalized))
    } catch {
      // Presets remain available for this browser session when storage is locked.
    }
  }

  function saveCustomPreset() {
    const name = customPresetName.replace(/\s+/g, ' ').trim().slice(0, 48)
    if (!name || contrastWarnings.length) return
    const existing = customPresets.find((preset) => preset.name.toLocaleLowerCase() === name.toLocaleLowerCase())
    const saved: SavedAppearancePreset = {
      id: existing?.id || `custom-${Date.now().toString(36)}`,
      name,
      preferences: { ...draft, overrides: { ...draft.overrides } },
    }
    persistCustomPresets(existing
      ? customPresets.map((preset) => preset.id === existing.id ? saved : preset)
      : [...customPresets, saved])
    setCustomPresetName('')
  }

  function loadCustomPreset(preset: SavedAppearancePreset) {
    update({ ...preset.preferences, overrides: { ...preset.preferences.overrides } })
  }

  return (
    <div className="scrim appearance-scrim" onClick={closeWithoutSaving}>
      <div
        className="modal appearance-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="appearance-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-head appearance-head">
          <div>
            <div className="modal-title" id="appearance-title">{t('appearance.title')}</div>
            <div className="brand-sub">Choose a balanced preset, then tune its operational palette.</div>
          </div>
          <button className="modal-close" onClick={closeWithoutSaving} aria-label="Close appearance settings">
            <IconX size={18} />
          </button>
        </div>

        <div className="appearance-body">
          <section className="appearance-section">
            <div className="appearance-section-head">
              <div>
                <h3>{t('appearance.language')}</h3>
                <p>{t('appearance.languageHelp')}</p>
              </div>
            </div>
            <OptionGroup
              label={t('appearance.language')}
              value={language}
              options={[
                { value: 'en', label: t('appearance.english') },
                { value: 'lv', label: t('appearance.latvian') },
              ]}
              onChange={(value) => setLanguage(value as UiLanguage)}
            />
          </section>

          <section className="appearance-section">
            <div className="appearance-section-head">
              <div>
                <h3><IconColorSwatch size={16} /> Theme preset</h3>
                <p>Changing preset clears color overrides but keeps density and motion preferences.</p>
              </div>
              {hasCustomColors(draft) && <span className="appearance-custom-badge">customized</span>}
            </div>
            <div className="theme-preset-grid">
              {THEME_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  className={`theme-preset ${draft.preset === preset.id ? 'on' : ''}`}
                  onClick={() => choosePreset(preset.id)}
                  aria-pressed={draft.preset === preset.id}
                >
                  <span className="theme-preset-swatches" aria-hidden="true">
                    <i style={{ background: preset.palette.canvas }} />
                    <i style={{ background: preset.palette.surface }} />
                    <i style={{ background: preset.palette.control }} />
                    <i style={{ background: preset.palette.accent }} />
                  </span>
                  <span className="theme-preset-copy">
                    <b>{preset.label}</b>
                    <small>{preset.description}</small>
                  </span>
                  {draft.preset === preset.id && <IconCheck className="theme-preset-check" size={17} />}
                </button>
              ))}
            </div>
            <div className="appearance-custom-presets">
              <div className="appearance-custom-save">
                <input
                  value={customPresetName}
                  maxLength={48}
                  placeholder="Custom preset name"
                  onChange={(event) => setCustomPresetName(event.target.value)}
                  onKeyDown={(event) => { if (event.key === 'Enter') saveCustomPreset() }}
                />
                <button className="btn" disabled={!customPresetName.trim() || contrastWarnings.length > 0} onClick={saveCustomPreset}>
                  <IconDeviceFloppy size={14} /> Save custom preset
                </button>
              </div>
              {customPresets.length > 0 && (
                <div className="appearance-custom-list">
                  {customPresets.map((preset) => (
                    <div key={preset.id}>
                      <button onClick={() => loadCustomPreset(preset)}>{preset.name}</button>
                      <button className="appearance-custom-delete" onClick={() => persistCustomPresets(customPresets.filter((item) => item.id !== preset.id))} aria-label={`Delete ${preset.name}`}><IconTrash size={13} /></button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          <div className="appearance-columns">
            <section className="appearance-section">
              <div className="appearance-section-head">
                <div>
                  <h3>Palette tuning</h3>
                  <p>Only semantic colors are exposed; contrast states are derived automatically.</p>
                </div>
                <button className="appearance-reset" disabled={!hasCustomColors(draft)} onClick={resetColors}>
                  <IconRestore size={14} /> Preset colors
                </button>
              </div>
              <div className="appearance-color-grid">
                {COLOR_FIELDS.map((field) => {
                  const value = draft.overrides[field.key] || palette[field.key]
                  return (
                    <ColorField
                      key={field.key}
                      label={field.label}
                      help={field.help}
                      value={value}
                      customized={Boolean(draft.overrides[field.key])}
                      onChange={(color) => setColor(field.key, color)}
                    />
                  )
                })}
              </div>
              {contrastWarnings.length > 0 && (
                <div className="appearance-contrast-warning" role="alert">
                  <b>Contrast check</b>
                  {contrastWarnings.map((warning) => <span key={warning}>{warning}</span>)}
                </div>
              )}
            </section>

            <section className="appearance-section">
              <div className="appearance-section-head">
                <div>
                  <h3>Controls</h3>
                  <p>Geometry may adapt to an operator station; typography remains fixed.</p>
                </div>
              </div>
              <OptionGroup
                label="Shape"
                value={draft.shape}
                options={[
                  { value: 'precise', label: 'Precise' },
                  { value: 'balanced', label: 'Balanced' },
                  { value: 'soft', label: 'Soft' },
                ]}
                onChange={(shape) => update({ ...draft, shape: shape as AppearancePreferences['shape'] })}
              />
              <OptionGroup
                label="Density"
                value={draft.density}
                options={[
                  { value: 'compact', label: 'Compact' },
                  { value: 'comfortable', label: 'Comfortable' },
                ]}
                onChange={(density) => update({ ...draft, density: density as AppearancePreferences['density'] })}
              />
              <OptionGroup
                label="Motion"
                value={draft.motion}
                options={[
                  { value: 'system', label: 'System' },
                  { value: 'full', label: 'Full' },
                  { value: 'reduced', label: 'Reduced' },
                ]}
                onChange={(motion) => update({ ...draft, motion: motion as AppearancePreferences['motion'] })}
              />

              <div className="typography-lock">
                <div className="typography-lock-icon"><IconTypography size={19} /></div>
                <div>
                  <b>Fixed EVA typography</b>
                  <span>Noto Sans for reading · Noto Sans Mono for telemetry and identifiers</span>
                  <div className="type-sample">
                    <strong>Scene activity</strong>
                    <code>CH 900001 · P 0.82 · 21:34:09</code>
                  </div>
                </div>
              </div>
            </section>
          </div>

          <section className="appearance-preview" style={{
            '--preview-canvas': palette.canvas,
            '--preview-surface': palette.surface,
            '--preview-control': palette.control,
            '--preview-text': palette.text,
            '--preview-accent': palette.accent,
          } as React.CSSProperties}>
            <div className="appearance-preview-title">Live preview · {getThemePreset(draft.preset).label}</div>
            <div className="appearance-preview-shell">
              <span className="appearance-preview-rail" />
              <div className="appearance-preview-panel">
                <span>VLM attention</span>
                <b>8 channels · regulated</b>
                <button>Open live feed</button>
              </div>
              <div className="appearance-preview-panel muted">
                <span>Agent</span>
                <b>Ready for operator request</b>
                <div className="appearance-preview-input">Ask EVA AI…</div>
              </div>
            </div>
          </section>
        </div>

        <div className="appearance-footer">
          <button
            className="btn"
            onClick={() => update({ ...DEFAULT_APPEARANCE, overrides: {} })}
          >
            Reset all
          </button>
          <div>
            <button className="btn" onClick={closeWithoutSaving}>Cancel</button>
            <button
              className="btn primary"
              disabled={contrastWarnings.length > 0}
              title={contrastWarnings.length > 0 ? 'Resolve palette contrast warnings before applying.' : ''}
              onClick={apply}
            >
              Apply appearance
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ColorField({
  label,
  help,
  value,
  customized,
  onChange,
}: {
  label: string
  help: string
  value: string
  customized: boolean
  onChange: (value: string) => void
}) {
  const [text, setText] = useState(value)

  useEffect(() => setText(value), [value])

  function commitText() {
    const normalized = normalizeHex(text)
    if (normalized) {
      setText(normalized)
      onChange(normalized)
    } else {
      setText(value)
    }
  }

  return (
    <label className={`appearance-color ${customized ? 'customized' : ''}`}>
      <input
        className="appearance-color-picker"
        type="color"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-label={`${label} color`}
      />
      <span>
        <b>{label}</b>
        <small>{help}</small>
      </span>
      <input
        className="appearance-color-hex"
        value={text}
        maxLength={7}
        spellCheck={false}
        onChange={(event) => setText(event.target.value)}
        onBlur={commitText}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault()
            commitText()
          }
        }}
        aria-label={`${label} hex value`}
      />
    </label>
  )
}

function OptionGroup({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: ReadonlyArray<{ value: string; label: string }>
  onChange: (value: string) => void
}) {
  return (
    <div className="appearance-option-row">
      <span>{label}</span>
      <div className="appearance-segment">
        {options.map((option) => (
          <button
            key={option.value}
            className={option.value === value ? 'on' : ''}
            onClick={() => onChange(option.value)}
            aria-pressed={option.value === value}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}
