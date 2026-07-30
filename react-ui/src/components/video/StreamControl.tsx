import { useState } from 'react'
import {
  IconChevronsDown,
  IconChevronsUp,
  IconDroplet,
  IconFileDescription,
  IconPlayerPlay,
  IconPlayerStop,
  IconReload,
  IconSettings,
  IconVideo,
} from '@tabler/icons-react'
import type { Channel } from '../../api/types'
import { Dropdown, type DropOption } from '../shell/Dropdown'
import { ToolTabs } from '../shell/ToolTabs'
import type { SummaryPeriod, SummaryResolution } from './summaryView'

const BATCHES = ['4', '8', '12', '16', '24', '32']
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

export function StreamControl(p: {
  channels: Channel[]
  channelId: number | null
  onChannel: (id: number) => void
  onReload: () => void
  batch: string; onBatch: (v: string) => void
  every: string; onEvery: (v: string) => void
  model: string; onModel: (v: string) => void; modelOptions: DropOption[]
  prompt: string; onPrompt: (v: string) => void
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
  onCollapseAll: () => void
  onExpandAll: () => void
}) {
  const [tab, setTab] = useState<'stream' | 'lens'>('stream')

  const chTitle = p.channels.find((c) => c.id === p.channelId)?.title || '—'
  const streamSummary = `${chTitle} · batch ${p.batch} · ${p.every}s · ${p.capturing ? 'capturing' : 'idle'}`
  const lensSummary = [
    PERIODS.find((item) => item.v === p.period)?.label || 'Live',
    RESOLUTIONS.find((item) => item.v === p.resolution)?.label || 'Auto',
    p.live ? 'live on' : 'live off',
  ].join(' · ')

  return (
    <ToolTabs
      tabs={[
        { id: 'stream', icon: <IconVideo size={13} />, label: 'Live stream control', summary: streamSummary },
        { id: 'lens', icon: <IconFileDescription size={13} />, label: 'Summary lens', summary: lensSummary },
      ]}
      active={tab}
      onSelect={(id) => setTab(id as 'stream' | 'lens')}
    >
      {tab === 'stream' ? (
        <div className="vid-tb-row">
          <div className="wfield ch"><label>Channel</label>
            <div className="vid-row">
              <Dropdown value={String(p.channelId ?? '')} onChange={(v) => p.onChannel(Number(v))}
                options={p.channels.map((c) => ({ value: String(c.id), label: c.title }))} />
              <button className="mon-icobtn" title="Reload channels" onClick={p.onReload}><IconReload size={15} /></button>
            </div>
          </div>
          <div className="wfield batch"><label>Batch</label>
            <Dropdown value={p.batch} onChange={p.onBatch} options={BATCHES.map((b) => ({ value: b, label: b }))} />
          </div>
          <div className="wfield xs"><label>Every (s)</label>
            <input type="number" min={0.2} max={300} step={0.1} value={p.every} onChange={(e) => p.onEvery(e.target.value)} />
          </div>
          <div className="wfield model"><label>Live model</label>
            <Dropdown value={p.model} onChange={p.onModel} options={p.modelOptions} />
          </div>
          <div className="wfield prompt"><label>Live prompt</label>
            <input value={p.prompt} onChange={(e) => p.onPrompt(e.target.value)} placeholder="Describe ongoing activity…" />
          </div>
          <div className="vid-tb-actions">
            {p.canCapture && (p.capturing
              ? <button className="mon-btn danger vid-toggle" disabled={p.busy} onClick={p.onStop}><IconPlayerStop size={15} /> Stop summaries</button>
              : <button className="mon-btn accent vid-toggle" disabled={p.busy} onClick={p.onStart}><IconPlayerPlay size={15} /> Start summaries</button>)}
            {p.canCapture && <button className="mon-btn" disabled={p.busy || !p.capturing} onClick={p.onFlush}><IconDroplet size={15} /> Flush</button>}
            {p.canManagePrompts && <button className="mon-btn" onClick={p.onPromptSettings} title="System prompt settings"><IconSettings size={15} /> Prompt</button>}
          </div>
        </div>
      ) : (
        <div className="vid-lens-stack">
          <div className="vid-tb-row vid-lens-row">
            <div className="wfield hist"><label>Period</label>
              <Dropdown
                value={p.period}
                onChange={(value) => p.onPeriod(value as SummaryPeriod)}
                options={PERIODS.map((item) => ({ value: item.v, label: item.label }))}
              />
            </div>
            <div className="wfield resolution"><label>Resolution</label>
              <Dropdown
                value={p.resolution}
                onChange={(value) => p.onResolution(value as SummaryResolution)}
                options={RESOLUTIONS.map((item) => ({ value: item.v, label: item.label }))}
              />
            </div>
            <div className="vid-tb-actions">
              <button className="mon-btn" onClick={p.onRefreshFeed}><IconReload size={14} /> Refresh</button>
              <button className={`mon-btn ${p.live ? 'accent' : ''}`} onClick={p.onToggleLive}><IconPlayerPlay size={14} /> {p.live ? 'Live on' : 'Live off'}</button>
              <button className="mon-btn" disabled={!p.summaryCount} onClick={p.onCollapseAll}>
                <IconChevronsUp size={14} /> Collapse all
              </button>
              <button className="mon-btn" disabled={!p.summaryCount} onClick={p.onExpandAll}>
                <IconChevronsDown size={14} /> Expand all
              </button>
            </div>
          </div>
          {p.period === 'custom' && (
            <div className="vid-lens-custom">
              <div className="wfield"><label>From</label>
                <input type="datetime-local" value={p.customFrom} onChange={(event) => p.onCustomFrom(event.target.value)} />
              </div>
              <div className="wfield"><label>To</label>
                <input type="datetime-local" value={p.customTo} onChange={(event) => p.onCustomTo(event.target.value)} />
              </div>
              <button className="mon-btn" onClick={p.onApplyCustom}><IconReload size={14} /> Apply</button>
            </div>
          )}
        </div>
      )}
    </ToolTabs>
  )
}
