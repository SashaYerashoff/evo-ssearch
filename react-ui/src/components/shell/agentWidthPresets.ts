export const AGENT_WIDTH_PRESET_STORAGE_KEY = 'evs_agent_width_preset'
export const MIN_AGENT_WIDTH = 400
export const MIN_CONSOLE_WIDTH = 640
export const ARCHIVE_GRID_GAP = 20
export const ARCHIVE_GRID_COLUMNS = 6
export const ARCHIVE_GRID_CHROME = 92 // 60px rail + 16px horizontal content padding per side

export interface AgentWidthPreset {
  profile: 'full-hd' | '2k'
  label: string
  width: number
  archiveColumns: number
}

export function agentLayoutViewportWidth(
  physicalViewportWidth: number,
  interfaceScale = 1,
): number {
  const safeScale = Number.isFinite(interfaceScale) && interfaceScale > 0 ? interfaceScale : 1
  return physicalViewportWidth / safeScale
}

const FULL_HD_SPECS = [
  { hiddenColumns: 2, archiveColumns: 4 },
  { hiddenColumns: 3, archiveColumns: 3 },
] as const

const TWO_K_SPECS = [
  { hiddenColumns: 1, archiveColumns: 5 },
  { hiddenColumns: 2, archiveColumns: 4 },
  { hiddenColumns: 3, archiveColumns: 3 },
] as const

export function maxAgentPanelWidth(viewportWidth: number): number {
  return Math.max(MIN_AGENT_WIDTH, viewportWidth - MIN_CONSOLE_WIDTH)
}

export function agentWidthPresets(
  viewportWidth: number,
): AgentWidthPreset[] {
  const isTwoK = viewportWidth >= 2300
  const profile = isTwoK ? '2k' : 'full-hd'
  const specs = isTwoK ? TWO_K_SPECS : FULL_HD_SPECS
  const maxWidth = maxAgentPanelWidth(viewportWidth)
  const slotWidth = archiveGridSlotWidth(viewportWidth)

  return specs.map(({ hiddenColumns, archiveColumns }, index) => ({
    profile,
    label: `${profile === '2k' ? '2K' : 'Full HD'} · layout ${index + 1} · ${archiveColumns} cards`,
    width: Math.floor(Math.max(MIN_AGENT_WIDTH, Math.min(slotWidth * hiddenColumns, maxWidth))),
    archiveColumns,
  }))
}

export function archiveGridSlotWidth(viewportWidth: number): number {
  const contentWidth = Math.max(0, viewportWidth - ARCHIVE_GRID_CHROME)
  const cardsWidth = contentWidth - ARCHIVE_GRID_GAP * (ARCHIVE_GRID_COLUMNS - 1)
  return cardsWidth / ARCHIVE_GRID_COLUMNS + ARCHIVE_GRID_GAP
}

export function archiveColumnsForAgentWidth(width: number, viewportWidth: number): number {
  const contentWidth = Math.max(0, viewportWidth - ARCHIVE_GRID_CHROME - width)
  const columns = Math.floor(
    (contentWidth + ARCHIVE_GRID_GAP) / Math.max(1, archiveGridSlotWidth(viewportWidth)),
  )
  return Math.max(1, Math.min(ARCHIVE_GRID_COLUMNS, columns))
}

export function closestAgentWidthPresetIndex(
  width: number,
  presets: AgentWidthPreset[],
): number {
  if (!presets.length) return 0
  return presets.reduce(
    (best, preset, index) => (
      Math.abs(preset.width - width) < Math.abs(presets[best].width - width) ? index : best
    ),
    0,
  )
}

export function nextAgentWidthPresetIndex(index: number, count: number): number {
  return count > 0 ? (index + 1) % count : 0
}

export function agentDragGeometry(
  visualWidth: number,
  presets: AgentWidthPreset[],
): { layoutWidth: number; overlayWidth: number } {
  const maximumLayoutWidth = presets.length
    ? presets[presets.length - 1].width
    : MIN_AGENT_WIDTH
  const layoutWidth = Math.min(visualWidth, maximumLayoutWidth)
  return {
    layoutWidth,
    overlayWidth: Math.max(0, visualWidth - layoutWidth),
  }
}
