import {
  IconAlertTriangle,
  IconArrowBackUp,
  IconArrowLeft,
  IconCheck,
  IconChevronsDown,
  IconChevronsUp,
  IconDroplet,
  IconEye,
  IconFileDescription,
  IconPlayerPlay,
  IconPlayerStop,
  IconRefresh,
  IconReload,
  IconSettings,
  IconVideo,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import type { ReactNode } from 'react'
import { Dropdown } from '../shell/Dropdown'
import { ToolTabs } from '../shell/ToolTabs'
import { IcoBtn } from '../shell/IcoBtn'
import type { SummaryPeriod, SummaryResolution } from './summaryView'
import type { IncidentPeriod } from '../incidents/IncidentReview'
import { useI18n, type TranslationKey } from '../../i18n/I18nProvider'

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

export function describeVlmSampling(
  batchRaw: string | number,
  intervalRaw: string | number,
  maxSelectedRaw: string | number,
  maxImagesRaw: string | number = maxSelectedRaw,
): { compressed: boolean; label: string } {
  const batch = Math.max(1, Number(batchRaw) || 1)
  const interval = Math.max(0.2, Number(intervalRaw) || 0.2)
  const maxSelected = Math.max(1, Number(maxSelectedRaw) || 8)
  const maxImages = Math.max(1, Number(maxImagesRaw) || 8)
  const visible = Math.min(batch, maxSelected)
  const windowSec = Math.max(0.2, batch * interval)
  const window = Number.isInteger(windowSec) ? String(windowSec) : windowSec.toFixed(1)
  const caps = maxSelected === maxImages
    ? `hard cap ${maxImages}`
    : `selection budget ${maxSelected} · hard cap ${maxImages}`
  return batch > maxSelected
    ? {
        compressed: true,
        label: `VLM sees 4–${visible} chronological images from ${batch} captured observations · bounded attention selection with temporal context backfill · ${caps} · seals by ~${window}s`,
      }
    : {
        compressed: false,
        label: `VLM sees 4–${Math.max(4, visible)} chronological images · attention-ranked with temporal context backfill · ${caps} · seals by ~${window}s`,
      }
}

export function StreamControl(p: {
  navigation?: ReactNode
  channels: Channel[]
  activeTab: VideoWorkspaceTab
  onTab: (tab: VideoWorkspaceTab) => void
  /** Leaves the stream workspace and shows the channel list again. */
  onBackToList: () => void
  channelId: number | null
  onChannel: (id: number) => void
  onReload: () => void
  batch: string; onBatch: (v: string) => void
  allowedBatchSizes: string[]
  maxSelectedFrames: number
  maxVlmImages: number
  every: string; onEvery: (v: string) => void
  /* Routing is still applied on start (VideoScreen sends `model: routingSelector`);
     the picker is currently not exposed in the toolbar. */
  routingSelector: string
  onRoutingSelector: (v: string) => void
  routingOptions: Array<{ value: string; label: string }>
  canCapture: boolean
  canManagePrompts: boolean
  samplingReady: boolean
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
      <div className="vid-unified-toolbar" role="group" aria-label="Video workspace controls">
        <section className="vid-unified-strip vid-camera-strip">
          <button type="button" className="vid-back" onClick={p.onBackToList}
            title="Back to all channels" aria-label="Back to all channels">
            <IconArrowLeft size={22} />
          </button>
          <h2 className="vid-open-channel">
            <IconVideo size={18} />
            <span>{p.channels.find((channel) => channel.id === p.channelId)?.title ?? t('video.channel')}</span>
          </h2>
        </section>

        {p.activeTab === 'review' && (
          <section className="vid-unified-strip vid-controls-strip">
            <div className="vid-lens-stack">
              <div className="vid-tb-row vid-lens-row">
                <div className="toolbar-scroll-rail vid-review-scroll">
                  <div className="wfield hist">
                    <Dropdown variant="chip" title={t('video.period')} value={p.period}
                      onChange={(value) => p.onPeriod(value as SummaryPeriod)}
                      options={periods.map((item) => ({ value: item.v, label: item.label }))} />
                  </div>
                  <div className="wfield resolution">
                    <Dropdown variant="chip" title={t('video.resolution')} value={p.resolution}
                      onChange={(value) => p.onResolution(value as SummaryResolution)}
                      options={resolutions.map((item) => ({ value: item.v, label: item.label }))} />
                  </div>
                </div>
                <button className="mon-btn accent vid-toggle vid-primary" onClick={p.onToggleLive}>
                  <IconPlayerPlay size={14} /> {p.live ? t('video.liveOn') : t('video.liveOff')}
                </button>
                <div className="vid-tb-actions">
                  <IcoBtn title={t('video.reloadChannels')} onClick={p.onReload}><IconReload size={16} /></IcoBtn>
                  <IcoBtn title={t('video.refresh')} onClick={p.onRefreshFeed}><IconRefresh size={16} /></IcoBtn>
                  <IcoBtn
                    title={p.allSummariesCollapsed ? t('video.expandAll') : t('video.collapseAll')}
                    onClick={p.onToggleAllSummaries}
                    disabled={!p.summaryCount}
                  >
                    {p.allSummariesCollapsed ? <IconChevronsDown size={16} /> : <IconChevronsUp size={16} />}
                  </IcoBtn>
                  <IcoBtn title={t('video.openPreview')} onClick={p.onOpenPreview} disabled={p.channelId == null}>
                    <IconEye size={16} />
                  </IcoBtn>
                  {p.period === 'custom' && (
                    <IcoBtn title={t('video.applyChanges')} onClick={p.onApplyCustom} disabled={!p.customFrom || !p.customTo}>
                      <IconCheck size={16} />
                    </IcoBtn>
                  )}
                </div>
              </div>
              {p.period === 'custom' && (
                <div className="vid-lens-custom">
                  <div className="wfield"><input aria-label={t('video.from')} title={t('video.from')} type="datetime-local" value={p.customFrom} onChange={(event) => p.onCustomFrom(event.target.value)} /></div>
                  <div className="wfield"><input aria-label={t('video.to')} title={t('video.to')} type="datetime-local" value={p.customTo} onChange={(event) => p.onCustomTo(event.target.value)} /></div>
                </div>
              )}
            </div>
          </section>
        )}

        {p.activeTab === 'incidents' && p.showIncidents && (
          <section className="vid-unified-strip vid-controls-strip">
            <div className="vid-incident-toolbar incident-review-filters">
              <div className="wfield vid-incident-period">
                <Dropdown variant="chip" title={t('video.period')} value={p.incidentPeriod}
                onChange={(value) => p.onIncidentPeriod(value as IncidentPeriod)}
                options={[
                  { value: '24h', label: t('incident.last24h') },
                  { value: '7d', label: t('period.last7d') },
                  { value: '30d', label: t('period.last30d') },
                  { value: 'all', label: t('incident.allTime') },
                ]} />
              </div>
              <div className="vid-tb-actions">
                <IcoBtn
                  title={p.incidentLoading ? t('status.checking') : t('video.refresh')}
                  onClick={p.onRefreshIncidents}
                  disabled={p.incidentLoading || p.channelId == null}
                >
                  <IconRefresh size={16} />
                </IcoBtn>
              </div>
            </div>
          </section>
        )}

        {p.activeTab === 'settings' && (
          <section className="vid-unified-strip vid-controls-strip">
            <div className="vid-settings-toolbar">
              {p.canCapture && (p.capturing
                ? <button className="mon-btn accent vid-toggle vid-primary" disabled={p.busy || !p.settingsDirty || !p.samplingReady} onClick={p.onStart} title="Restart this stream with the edited sampling and inference settings"><IconPlayerPlay size={15} /> {t('video.applyChanges')}</button>
                : <button className="mon-btn accent vid-toggle vid-primary" disabled={p.busy || !p.samplingReady} onClick={p.onStart}><IconPlayerPlay size={15} /> {t('video.start')}</button>)}
              <div className="vid-tb-actions">
                <IcoBtn title={t('video.reloadChannels')} onClick={p.onReload}><IconReload size={16} /></IcoBtn>
                {p.canCapture && p.capturing && (
                  <IcoBtn title={t('video.stop')} onClick={p.onStop} disabled={p.busy} danger><IconPlayerStop size={16} /></IcoBtn>
                )}
                {p.canCapture && (
                  <IcoBtn title={t('video.flush')} onClick={p.onFlush} disabled={p.busy || !p.capturing}><IconDroplet size={16} /></IcoBtn>
                )}
                {p.canManagePrompts && (
                  <IcoBtn title={t('video.prompts')} onClick={p.onPromptSettings}><IconSettings size={16} /></IcoBtn>
                )}
                {p.settingsDirty && (
                  <IcoBtn title={t('video.discard')} onClick={p.onDiscardSettings} disabled={p.busy}><IconArrowBackUp size={16} /></IcoBtn>
                )}
              </div>
            </div>
          </section>
        )}
      </div>
    </ToolTabs>
  )
}
