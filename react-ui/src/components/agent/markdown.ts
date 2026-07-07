// Minimal, safe markdown-to-HTML for agent responses.
// Escapes first, then applies a small subset: code, bold, italic, links, headings, lists, line breaks.

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

export function renderMarkdown(src: string): string {
  if (!src) return ''
  // Pull out fenced code blocks first.
  const blocks: string[] = []
  let text = src.replace(/```([\s\S]*?)```/g, (_m, code) => {
    blocks.push(`<pre class="md-pre"><code>${esc(String(code).replace(/^\n/, ''))}</code></pre>`)
    return `@@CODE_BLOCK_${blocks.length - 1}@@`
  })

  text = esc(text)

  // Inline code.
  text = text.replace(/`([^`]+)`/g, (_m, c) => `<code class="md-code">${c}</code>`)
  // Bold / italic.
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  text = text.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
  // Links [t](u).
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')

  const lines = text.split('\n')
  const out: string[] = []
  let inList = false
  const closeList = () => {
    if (inList) {
      out.push('</ul>')
      inList = false
    }
  }
  const isSep = (s: string) => /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(s) && s.includes('-')
  const cells = (s: string) => s.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim())

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trimEnd()
    // GFM table: header row + separator row + body rows.
    if (line.includes('|') && i + 1 < lines.length && isSep(lines[i + 1])) {
      closeList()
      const head = cells(line)
      let j = i + 2
      const rows: string[][] = []
      while (j < lines.length && lines[j].includes('|') && lines[j].trim() !== '') {
        rows.push(cells(lines[j]))
        j++
      }
      out.push(`<table class="md-table"><thead><tr>${head.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${
        rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')
      }</tbody></table>`)
      i = j - 1
      continue
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/)
    const li = line.match(/^[-*]\s+(.*)$/)
    if (h) {
      closeList()
      const lvl = h[1].length
      out.push(`<div class="md-h md-h${lvl}">${h[2]}</div>`)
    } else if (li) {
      if (!inList) {
        out.push('<ul class="md-ul">')
        inList = true
      }
      out.push(`<li>${li[1]}</li>`)
    } else if (line === '') {
      closeList()
      out.push('')
    } else {
      out.push(`<div class="md-p">${line}</div>`)
    }
  }
  closeList()
  let html = out.join('\n')
  // Restore code blocks.
  html = html.replace(/@@CODE_BLOCK_(\d+)@@/g, (_m, i) => blocks[Number(i)] || '')
  return html
}
