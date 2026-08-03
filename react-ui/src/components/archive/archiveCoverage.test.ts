import { describe, expect, it } from 'vitest'
import { archiveCoverageMessages } from './archiveCoverage'

const channels = [
  { id: 7, title: 'Gate' },
  { id: 8, title: 'Beach' },
]

describe('archive coverage messages', () => {
  it('explains partial channel coverage and embedding transitions', () => {
    expect(archiveCoverageMessages({
      partial: true,
      failed_channel_ids: [8],
      searched_channel_ids: [7],
      embedding_space: { backend: 'siglip2' },
      embedding_space_excluded_vectors: 42,
      visual_evidence_excluded: 2,
    }, channels)).toEqual([
      'Partial results: Beach could not be fully searched for this period. Available evidence from Gate is shown.',
      'Embedding transition: 42 legacy or mismatched vectors were excluded; ranking uses the current SigLIP2 epoch only.',
      '2 archive rows without a usable snapshot were omitted.',
    ])
  })

  it('stays silent for complete current-epoch coverage', () => {
    expect(archiveCoverageMessages({ status: 'complete' }, channels)).toEqual([])
  })
})
