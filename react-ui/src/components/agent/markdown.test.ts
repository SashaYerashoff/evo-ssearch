import { describe, expect, it } from 'vitest'

import { renderMarkdown } from './markdown'

describe('renderMarkdown', () => {
  it('does not preserve blank source lines between rendered blocks', () => {
    expect(renderMarkdown('First paragraph\n\nSecond paragraph')).toBe(
      '<div class="md-p">First paragraph</div><div class="md-p">Second paragraph</div>',
    )
  })

  it('renders markdown separators as compact horizontal rules', () => {
    expect(renderMarkdown('Before\n\n---\n\nAfter')).toBe(
      '<div class="md-p">Before</div><hr class="md-hr"><div class="md-p">After</div>',
    )
  })

  it('keeps source line breaks inside one paragraph block', () => {
    expect(renderMarkdown('First line\nsecond line\nthird line')).toBe(
      '<div class="md-p">First line<br>second line<br>third line</div>',
    )
  })

  it('keeps list and table markup adjacent without whitespace text nodes', () => {
    const html = renderMarkdown('- one\n- two\n\n| A | B |\n|---|---|\n| 1 | 2 |')

    expect(html).toContain('<ul class="md-ul"><li>one</li><li>two</li></ul>')
    expect(html).toContain('<table class="md-table">')
    expect(html).not.toContain('</ul>\n')
  })
})
