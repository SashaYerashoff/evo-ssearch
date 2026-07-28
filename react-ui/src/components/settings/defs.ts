import type { Settings } from '../../api/settings'

export type FieldType = 'text' | 'number' | 'password' | 'checkbox' | 'select' | 'range'

export interface FieldDef {
  key: string
  label: string
  type: FieldType
  min?: number
  max?: number
  step?: number
  options?: { v: string; label?: string }[]
  note?: string
  experimental?: boolean
  showIf?: (s: Settings) => boolean
}

export interface Section { title: string; help?: string; fields?: FieldDef[]; kind?: 'severity' | 'capacity'; experimental?: boolean }
export interface TabDef { id: string; label: string; custom?: 'env' | 'audit' | 'users'; sections?: Section[] }

const CLIP_MODELS = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'].map((v) => ({ v }))

// keys writable via POST /settings (guards the save payload)
export const WRITABLE_KEYS: string[] = [
  'host', 'port', 'debug',
  'minResults', 'maxResults', 'defaultResults',
  'embedder', 'clipModel', 'dinoModel', 'dinoEmbedDim', 'dinoWeightsPath', 'indexMode',
  'fusionEnabled', 'fusionAlpha', 'batchSize', 'thumbnailQuality',
  'rerankEnabled', 'rerankTopK', 'segmentsEnabled', 'segmentMinPatches', 'maxCommentLength', 'maxFileSize',
  'luxriotBaseUrl', 'luxriotUsername', 'luxriotPassword', 'luxriotDefaultChannelId',
  'luxriotSnapshotInterval', 'luxriotSnapshotMaxEdge', 'luxriotMaxBufferFrames',
  'luxriotSummaryRetentionDays', 'luxriotSummaryHistoryLimit', 'luxriotAutoBookmarks',
  'probeBookmarkCooldownSec', 'probeBookmarkDedupeWindowSec', 'probeBookmarkSimHigh',
  'probeBookmarkMarginDelta', 'probeBookmarkScoreDelta', 'probeBookmarkMaxFrameGap',
  'luxriotSeverityMap',
  'archiveRetentionEnabled', 'archiveRowRetentionDays', 'archiveThumbnailRetentionDays', 'archiveMaxRecords',
  'archiveEstimateChannels', 'archiveEstimateFramesPerBatch', 'archiveEstimateAvgJpegKb', 'archiveEstimateProbeRecordsPerChannelDay',
]

export const TABS: TabDef[] = [
  {
    id: 'server', label: 'Server',
    sections: [{
      title: 'Server configuration', help: 'Core HTTP runtime used by the local Flask control plane. Changes need a server restart.',
      fields: [
        { key: 'host', label: 'Host', type: 'text' },
        { key: 'port', label: 'Port', type: 'number', min: 1000, max: 65535 },
        { key: 'debug', label: 'Debug mode', type: 'checkbox' },
      ],
    }],
  },
  {
    id: 'search', label: 'Search',
    sections: [{
      title: 'Search configuration', help: 'Baseline retrieval counts for archive and agent-assisted search flows.',
      fields: [
        { key: 'minResults', label: 'Min results', type: 'number', min: 1 },
        { key: 'maxResults', label: 'Max results', type: 'number', min: 1 },
        { key: 'defaultResults', label: 'Default results', type: 'number', min: 1 },
      ],
    }],
  },
  {
    id: 'models', label: 'Models',
    sections: [
      {
        title: 'Backend & model', help: 'Embedding backend and thumbnail quality. DINO / fusion are experimental.',
        fields: [
          { key: 'embedder', label: 'Backend', type: 'select', options: [{ v: 'clip' }, { v: 'dino' }, { v: 'fusion' }], note: 'dino / fusion require experimental embedders' },
          { key: 'clipModel', label: 'CLIP model', type: 'select', options: CLIP_MODELS },
          { key: 'batchSize', label: 'Batch size', type: 'number', min: 1, max: 128 },
          { key: 'thumbnailQuality', label: 'Thumbnail quality', type: 'range', min: 50, max: 100, step: 1 },
        ],
      },
      {
        title: 'DINO', help: 'Secondary DINOv3 encoder — used only when backend is DINO or fusion.', experimental: true,
        fields: [
          { key: 'dinoModel', label: 'DINO model', type: 'text' },
          { key: 'dinoEmbedDim', label: 'DINO embedding dim', type: 'number', min: 128, max: 4096 },
          { key: 'dinoWeightsPath', label: 'DINO weights path', type: 'text' },
        ],
      },
      {
        title: 'Fusion', help: 'Weighted CLIP+DINO blend — implemented but off unless experimental embedders are enabled.', experimental: true,
        fields: [
          { key: 'fusionEnabled', label: 'Fusion enabled', type: 'checkbox' },
          { key: 'fusionAlpha', label: 'Fusion alpha (CLIP↔DINO)', type: 'range', min: 0, max: 1, step: 0.05 },
        ],
      },
    ],
  },
  {
    id: 'advanced', label: 'Advanced',
    sections: [
      {
        title: 'Reranking & segments', help: 'Re-score / segment passes — implemented but disabled unless experimental embedders are on.', experimental: true,
        fields: [
          { key: 'rerankEnabled', label: 'Rerank enabled', type: 'checkbox' },
          { key: 'rerankTopK', label: 'Rerank top-K', type: 'number', min: 1, max: 500 },
          { key: 'segmentsEnabled', label: 'Segment embeddings', type: 'checkbox' },
          { key: 'segmentMinPatches', label: 'Min segment patches', type: 'number', min: 1, max: 256 },
        ],
      },
      {
        title: 'Limits',
        fields: [
          { key: 'maxCommentLength', label: 'Max comment length', type: 'number', min: 50, max: 2000 },
          { key: 'maxFileSize', label: 'Max file size (MB)', type: 'number', min: 1, max: 500 },
        ],
      },
    ],
  },
  {
    id: 'luxriot', label: 'Luxriot',
    sections: [
      {
        title: 'Integration', help: 'Live Luxriot Evo source. Password is write-only (leave blank to keep current).',
        fields: [
          { key: 'luxriotBaseUrl', label: 'Base URL', type: 'text' },
          { key: 'luxriotUsername', label: 'Username', type: 'text' },
          { key: 'luxriotPassword', label: 'Password', type: 'password' },
          { key: 'luxriotDefaultChannelId', label: 'Default channel ID', type: 'number', min: 1 },
          { key: 'luxriotSnapshotInterval', label: 'Snapshot interval (s)', type: 'number', min: 1, max: 300 },
          { key: 'luxriotSnapshotMaxEdge', label: 'Snapshot max edge (px)', type: 'number', min: 640, max: 1600 },
          { key: 'luxriotMaxBufferFrames', label: 'Max buffer frames', type: 'number', min: 12, max: 2000 },
          { key: 'luxriotSummaryRetentionDays', label: 'Description retention (days)', type: 'number', min: 0, max: 3650, step: 0.5 },
          { key: 'luxriotSummaryHistoryLimit', label: 'Description cap / channel', type: 'number', min: 40, max: 1000000, step: 100 },
        ],
      },
      {
        title: 'Bookmark & probe behaviour',
        fields: [
          { key: 'luxriotAutoBookmarks', label: 'Auto-bookmark alerts', type: 'checkbox' },
          { key: 'probeBookmarkCooldownSec', label: 'Probe bookmark cooldown (s)', type: 'number', min: 0, step: 0.5 },
          { key: 'probeBookmarkDedupeWindowSec', label: 'Probe dedupe window (s)', type: 'number', min: 0.5, step: 0.5 },
          { key: 'probeBookmarkSimHigh', label: 'Probe similarity high', type: 'number', min: 0.5, max: 0.9999, step: 0.0001 },
          { key: 'probeBookmarkMarginDelta', label: 'Probe margin delta', type: 'number', min: 0, step: 0.01 },
          { key: 'probeBookmarkScoreDelta', label: 'Probe score delta', type: 'number', min: 0, step: 0.01 },
          { key: 'probeBookmarkMaxFrameGap', label: 'Probe max frame gap', type: 'number', min: 1, step: 1 },
        ],
      },
      { title: 'Severity mapping', help: 'Map internal severity labels to display labels.', kind: 'severity' },
      {
        title: 'Archive capacity',
        fields: [
          { key: 'archiveRetentionEnabled', label: 'Archive retention enabled', type: 'checkbox' },
          { key: 'archiveRowRetentionDays', label: 'Frame rows (days)', type: 'number', min: 0, max: 3650, step: 1 },
          { key: 'archiveThumbnailRetentionDays', label: 'DB previews (days)', type: 'number', min: 0, max: 3650, step: 1 },
          { key: 'archiveMaxRecords', label: 'Max frame records', type: 'number', min: 1000, max: 500000000, step: 1000 },
          { key: 'archiveEstimateChannels', label: 'Planned channels', type: 'number', min: 1, max: 10000, step: 1 },
          { key: 'archiveEstimateFramesPerBatch', label: 'Frames / batch', type: 'number', min: 0, max: 32, step: 0.5 },
          { key: 'archiveEstimateAvgJpegKb', label: 'Avg JPEG (KB)', type: 'number', min: 1, max: 5000, step: 5 },
          { key: 'archiveEstimateProbeRecordsPerChannelDay', label: 'Probe rows / channel / day', type: 'number', min: 0, max: 100000, step: 10 },
        ],
      },
      { title: 'Estimated capacity', kind: 'capacity' },
    ],
  },
  { id: 'users', label: 'Users', custom: 'users' },
  { id: 'audit', label: 'Audit', custom: 'audit' },
  { id: 'environment', label: 'Environment', custom: 'env' },
]

export const SEVERITY_KEYS = ['info', 'low', 'normal', 'high', 'critical'] as const
