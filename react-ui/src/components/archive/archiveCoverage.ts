import type { Channel } from '../../api/types'
import type { ArchiveSearchCoverage } from '../../api/detections'

function ids(value: unknown): number[] {
  if (!Array.isArray(value)) return []
  return Array.from(new Set(value.map(Number).filter((item) => Number.isInteger(item) && item > 0)))
}

function count(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 0
}

function names(channelIds: number[], channels: Channel[]): string {
  const byId = new Map(channels.map((channel) => [channel.id, channel.title]))
  return channelIds.map((id) => byId.get(id) || `#${id}`).join(', ')
}

function embeddingLabel(coverage: ArchiveSearchCoverage): string {
  const space = coverage.embedding_space
  if (!space || typeof space !== 'object') return 'current embedding epoch'
  const backend = String(space.backend || '').trim()
  const model = String(space.model || '').trim()
  if (backend.toLowerCase().includes('siglip')) return 'current SigLIP2 epoch'
  return model || backend || 'current embedding epoch'
}

export function archiveCoverageMessages(
  coverage: ArchiveSearchCoverage | null,
  channels: Channel[],
): string[] {
  if (!coverage) return []
  const messages: string[] = []
  const failed = ids(coverage.failed_channel_ids)
  const searched = ids(coverage.searched_channel_ids)
  if (failed.length) {
    messages.push(
      searched.length
        ? `Partial results: ${names(failed, channels)} could not be fully searched for this period. Available evidence from ${names(searched, channels)} is shown.`
        : `No results: ${names(failed, channels)} could not be searched for this period.`,
    )
  }

  const failedSources = Array.isArray(coverage.failed_sources)
    ? coverage.failed_sources.map(String).filter(Boolean)
    : []
  if (failedSources.length) {
    messages.push(`Partial results: unavailable archive sources were skipped (${failedSources.join(', ')}).`)
  }

  const excludedVectors = count(coverage.embedding_space_excluded_vectors)
  if (excludedVectors) {
    messages.push(
      `Embedding transition: ${excludedVectors.toLocaleString()} legacy or mismatched vectors were excluded; ranking uses the ${embeddingLabel(coverage)} only.`,
    )
  }

  const missingVisuals = count(coverage.visual_evidence_excluded)
  if (missingVisuals) {
    messages.push(`${missingVisuals.toLocaleString()} archive rows without a usable snapshot were omitted.`)
  }
  return messages
}
