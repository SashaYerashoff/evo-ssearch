import {
  IconAlertTriangle,
  IconChevronsDown,
  IconChevronsUp,
  IconDroplet,
  IconEye,
  IconFileDescription,
  IconPlayerPlay,
  IconPlayerStop,
  IconReload,
  IconSettings,
  IconVideo,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import type { ReactNode } from 'react'
import { Dropdown } from '../shell/Dropdown'
import { ToolbarActionMenu } from '../shell/ToolbarActionMenu'
import { ToolTabs } from '../shell/ToolTabs'
import type { SummaryPeriod, SummaryResolution } from './summaryView'
import type { IncidentPeriod } from '../incidents/IncidentReview'
import { useI18n, type TranslationKey } from '../../i18n/I18nProvider'

const BATCHES = ['4', '8', '12', '16']
export const PERIODS: Array<{ v: SummaryPeriod; label: string }> = [
  { v: 'live', label: 'Live' },
  { v: 'today', label: 'Today' },
  { v: 'yesterday', label: 'Yesterday' },
  { v: 'day_before_yesterday', label: 'Day before yesterday' },
  { v: '7d', label: 'Last 7 days' },
  { v: '30d', label: 'Last 30 days' },
  { v: 'custom', label: 'Custom range…' },
]
export const RESOLUTIONS: Array<{ v: SummaryResolution; label: string }> = [
  { v: 'AUTO', label: 'Auto' },
  { v: 'L0', label: 'Observations' },
  { v: 'L1', label: '15 minute summaries' },
  { v: 'L2', label: '1 hour summaries' },
  { v: 'L3', label: '8 hour summaries' },
]
const PERIOD_LABEL_KEYS: Record<SummaryPeriod, TranslationKey> = {
  live: 'period.live', today: 'period.today', yesterday: 'period.yesterday',
  day_before_yesterday: 'period.dayBefore', '7d': 'period.last7d',
  '30d': 'period.last30d', custom: 'period.custom',
}
const RESOLUTION_LABEL_KEYS: Record<SummaryResolution, TranslationKey> = {
  AUTO: 'resolution.auto', L0: 'resolution.observations', L1: 'resolution.l1',
  L2: 'resolution.l2', L3: 'resolution.l3',
}

export type VideoWorkspaceTab = 'review' | 'incidents' | 'settings'

export function visibleVideoWorkspaceTabs(showIncidents: boolean): VideoWorkspaceTab[] {
  return showIncidents ? ['review', 'incidents', 'settings'] : ['review', 'settings']
}

export function resolveVideoWorkspaceTab(
  active: VideoWorkspaceTab,
  showIncidents: boolean,
): VideoWorkspaceTab {
  return visibleVideoWorkspaceTabs(showIncidents).includes(active) ? active : 'review'
}

export function StreamControl(p: {
  navigation?: ReactNode
  channels: Channel[]
  activeTab: VideoWorkspaceTab
  onTab: (tab: VideoWorkspaceTab) => void
  settingsChannelId: number | null
  onSettingsChannel: (id: number) => void
  reviewChannelId: number | null
  onReviewChannel: (id: number) => void
  onReload: () => void
  batch: string; onBatch: (v: string) => void
  every: string; onEvery: (v: string) => void
  canCapture: boolean
  canManagePrompts: boolean
  capturing: boolean; busy: boolean
  onStart: () => void; onStop: () => void; onFlush: () => void
  onPromptSettings: () => void
  period: SummaryPeriod; onPeriod: (v: SummaryPeriod) => void
  resolution: SummaryResolution; onResolution: (v: SummaryResolution) => void
  customFrom: string; onCustomFrom: (v: string) => void
  customTo: string; onCustomTo: (v: string) => void
  onApplyCustom: () => void
  onRefreshFeed: () => void
  live: boolean; onToggleLive: () => void
  summaryCount: number
  allSummariesCollapsed: boolean
  onToggleAllSummaries: () => void
  onOpenPreview: () => void
  settingsDirty: boolean
  onDiscardSettings: () => void
  incidentChannelId: string
  onIncidentChannel: (id: string) => void
  incidentPeriod: IncidentPeriod
  onIncidentPeriod: (period: IncidentPeriod) => void
  incidentLoading: boolean
  onRefreshIncidents: () => void
  showIncidents: boolean
}) {
  const { t } = useI18n()
  const periods = PERIODS.map((item) => ({ ...item, label: t(PERIOD_LABEL_KEYS[item.v]) }))
  const resolutions = RESOLUTIONS.map((item) => ({ ...item, label: t(RESOLUTION_LABEL_KEYS[item.v]) }))
  return (
    <ToolTabs
      tabs={[
        { id: 'review', icon: <IconFileDescription size={13} />, label: t('video.review') },
        ...(visibleVideoWorkspaceTabs(p.showIncidents).includes('incidents') ? [{
          id: 'incidents',
          icon: <IconAlertTriangle size={13} />,
          label: t('incident.review'),
          badge: 'FiP',
        }] : []),
        { id: 'settings', icon: <IconVideo size={13} />, label: t('video.settings') },
      ]}
      active={p.activeTab}
      onSelect={(id) => p.onTab(id as VideoWorkspaceTab)}
      leading={p.navigation}
      reserveLeading
    >
      {p.activeTab === 'settings' ? (
        <div className="vid-settings-toolbar">
          <div className="toolbar-scroll-rail vid-settings-scroll">
          <section className="vid-control-group source">
            <div className="vid-control-group-title">{t('video.source')}</div>
            <div className="wfield ch"><label>{t('video.channel')}</label>
              <div className="vid-row">
                <Dropdown variant="chip" value={String(p.settingsChannelId ?? '')} onChange={(v) => p.onSettingsChannel(Number(v))}
                  options={p.channels.map((c) => ({ value: String(c.id), label: c.title }))} />
              </div>
            </div>
          </section>
          <section className="vid-control-group sampling">
            <div className="vid-control-group-title">{t('video.sampling')}</div>
            <div className="vid-control-fields">
              <div className="wfield batch"><label>{t('video.batch')}</label>
                <Dropdown variant="chip" value={p.batch} onChange={p.onBatch} options={BATCHES.map((b) => ({ value: b, label: b }))} />
              </div>
              <div className="wfield xs"><label>{t('video.every')}</label>
                <input type="number" min={0.2} max={300} step={0.1} value={p.every} onChange={(e) => p.onEvery(e.target.value)} />
              </div>
            </div>
          </section>
          </div>
          <section className="vid-control-group actions">
            <div className="vid-control-group-title">{t('video.runtime')}</div>
            <div className="vid-tb-actions">
              <ToolbarActionMenu actions={[
                {
                  id: 'reload-channels', label: t('video.reloadChannels'), icon: <IconReload size={15} />,
                  onSelect: p.onReload,
                },
                ...(p.canCapture && p.capturing ? [{
                  id: 'stop', label: t('video.stop'), icon: <IconPlayerStop size={15} />,
                  onSelect: p.onStop, disabled: p.busy, danger: true,
                }] : []),
                ...(p.canCapture ? [{
                  id: 'flush', label: t('video.flush'), icon: <IconDroplet size={15} />,
                  onSelect: p.onFlush, disabled: p.busy || !p.capturing,
                }] : []),
                ...(p.canManagePrompts ? [{
                  id: 'prompts', label: t('video.prompts'), icon: <IconSettings size={15} />,
                  onSelect: p.onPromptSettings,
                }] : []),
                ...(p.settingsDirty ? [{
                  id: 'discard', label: t('video.discard'), icon: <IconReload size={15} />,
                  onSelect: p.onDiscardSettings, disabled: p.busy,
                }] : []),
              ]} />
              {p.canCapture && (p.capturing
                ? <button className="mon-btn accent vid-toggle" disabled={p.busy || !p.settingsDirty} onClick={p.onStart} title="Restart this stream with the edited sampling and inference settings"><IconPlayerPlay size={15} /> {t('video.applyChanges')}</button>
                : <button className="mon-btn accent vid-toggle" disabled={p.busy} onClick={p.onStart}><IconPlayerPlay size={15} /> {t('video.start')}</button>)}
            </div>
          </section>
        </div>
      ) : p.activeTab === 'incidents' ? (
        <div className="vid-incident-toolbar incident-review-filters">
          <div className="vid-incident-tab-note">
            <IconAlertTriangle size={16} />
            <span><b>Feature in progress.</b> Disable “Show incidents (FiP)” in Settings → Features if the result is not operationally useful.</span>
          </div>
          <label>
            {t('video.channel')}
            <Dropdown
              value={p.incidentChannelId}
              onChange={p.onIncidentChannel}
              options={[
                { value: 'all', label: t('incident.allChannels') },
                ...p.channels.map((channel) => ({ value: String(channel.id), label: channel.title })),
              ]}
            />
          </label>
          <label>
            {t('video.period')}
            <Dropdown
              value={p.incidentPeriod}
              onChange={(value) => p.onIncidentPeriod(value as IncidentPeriod)}
              options={[
                { value: '24h', label: t('incident.last24h') },
                { value: '7d', label: t('period.last7d') },
                { value: '30d', label: t('period.last30d') },
                { value: 'all', label: t('incident.allTime') },
              ]}
            />
          </label>
          <button className="mon-btn" onClick={p.onRefreshIncidents} disabled={p.incidentLoading || !p.incidentChannelId}>
            <IconReload size={14} /> {p.incidentLoading ? t('status.checking') : t('video.refresh')}
          </button>
        </div>
      ) : (
        <div className="vid-lens-stack">
          <div className="vid-tb-row vid-lens-row">
            <div className="toolbar-scroll-rail vid-review-scroll">
            <div className="wfield ch"><label>{t('video.channel')}</label>
              <div className="vid-row">
                <Dropdown variant="chip" value={String(p.reviewChannelId ?? '')} onChange={(v) => p.onReviewChannel(Number(v))}
                  options={p.channels.map((c) => ({ value: String(c.id), label: c.title }))} />
              </div>
            </div>
            <div className="wfield hist"><label>{t('video.period')}</label>
              <Dropdown
                variant="chip"
                value={p.period}
                onChange={(value) => p.onPeriod(value as SummaryPeriod)}
                options={periods.map((item) => ({ value: item.v, label: item.label }))}
              />
            </div>
            <div className="wfield resolution"><label>{t('video.resolution')}</label>
              <Dropdown
                variant="chip"
                value={p.resolution}
                onChange={(value) => p.onResolution(value as SummaryResolution)}
                options={resolutions.map((item) => ({ value: item.v, label: item.label }))}
              />
            </div>
            </div>
            <div className="vid-tb-actions">
              <ToolbarActionMenu actions={[
                { id: 'reload-channels', label: t('video.reloadChannels'), icon: <IconReload size={15} />, onSelect: p.onReload },
                { id: 'refresh', label: t('video.refresh'), icon: <IconReload size={15} />, onSelect: p.onRefreshFeed },
                {
                  id: 'collapse',
                  label: p.allSummariesCollapsed ? t('video.expandAll') : t('video.collapseAll'),
                  icon: p.allSummariesCollapsed ? <IconChevronsDown size={15} /> : <IconChevronsUp size={15} />,
                  onSelect: p.onToggleAllSummaries,
                  disabled: !p.summaryCount,
                },
                {
                  id: 'preview', label: t('video.openPreview'), icon: <IconEye size={15} />,
                  onSelect: p.onOpenPreview, disabled: p.reviewChannelId == null,
                },
                ...(p.period === 'custom' ? [{
                  id: 'apply-range', label: t('video.applyChanges'), icon: <IconReload size={15} />,
                  onSelect: p.onApplyCustom, disabled: !p.customFrom || !p.customTo,
                }] : []),
              ]} />
              <button className="mon-btn accent vid-toggle" onClick={p.onToggleLive}><IconPlayerPlay size={14} /> {p.live ? t('video.liveOn') : t('video.liveOff')}</button>
            </div>
          </div>
          {p.period === 'custom' && (
            <div className="vid-lens-custom">
              <div className="wfield"><label>{t('video.from')}</label>
                <input type="datetime-local" value={p.customFrom} onChange={(event) => p.onCustomFrom(event.target.value)} />
              </div>
              <div className="wfield"><label>{t('video.to')}</label>
                <input type="datetime-local" value={p.customTo} onChange={(event) => p.onCustomTo(event.target.value)} />
              </div>
            </div>
          )}
        </div>
      )}
    </ToolTabs>
  )
}
