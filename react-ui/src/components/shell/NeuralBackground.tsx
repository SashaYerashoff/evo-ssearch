import { useEffect, useRef } from 'react'

/**
 * Ambient neural-network backdrop for the whole app.
 *
 * A slow, calm field of "neurons" wired into a fixed network. Luminous dots fly in
 * from beyond the screen edge and random-walk the mesh — at every neuron they pick a
 * fresh wire — until they wander back to a border and slip off-screen. At most a
 * handful travel at once. Behind them, huge aurora colour fields breathe for depth.
 * Rendered on a single transparent <canvas> behind all content, so the cosmic backdrop
 * and the glassy panels still show through. Kept deliberately dim and unhurried: it
 * should read as atmosphere, never as motion competing with the UI.
 *
 * Honours the app's motion kill-switch (noAnim) and prefers-reduced-motion by
 * painting one static frame and stopping the loop.
 */

interface Node {
  bx: number; by: number      // anchor position
  x: number; y: number        // live position (anchor + gentle wander)
  r: number
  px: number; py: number      // wander phase
  ax: number; ay: number      // wander amplitude
  sx: number; sy: number      // wander speed (rad/s)
  tint: number                // 0..1 → cyan→violet blend
}
interface Edge { a: number; b: number; len: number }
// a dot walking the mesh. a→b is the current segment; a node index of -1 means the
// endpoint is a fixed off-screen point (entry point, or the exit it's flying out to).
interface Walker {
  aNode: number; bNode: number
  ax: number; ay: number; bx: number; by: number
  seglen: number; t: number
  edge: number                 // wire index of the current segment, or -1 (entry/exit leg)
  from: number                 // neuron we arrived from (avoid instant backtrack)
  hops: number
  exiting: boolean
  tint: number
}
interface Aurora { ax: number; ay: number; sx: number; sy: number; rad: number; col: readonly number[] }

const CYAN = [90, 224, 220] as const
const VIOLET = [150, 130, 255] as const
const BLUE = [106, 168, 255] as const
const mix = (t: number) => [
  Math.round(CYAN[0] + (VIOLET[0] - CYAN[0]) * t),
  Math.round(CYAN[1] + (VIOLET[1] - CYAN[1]) * t),
  Math.round(CYAN[2] + (VIOLET[2] - CYAN[2]) * t),
] as const

const CELL = 168          // grid spacing for node placement
const MAX_NODES = 66      // budget; the grid is scaled to fit this while covering the whole screen
const LINK_DIST = 300     // longest wire we'll draw (roomier → more branching paths)
const MAX_LINKS = 4       // nearest neighbours per node → more forks for the walkers

// travelling dots: they fly in from off-screen, random-walk the mesh (picking a fresh
// wire at every neuron), and eventually leave past the edge. At most MAX_WALKERS at once.
const MAX_WALKERS = 10
const WALK_SPEED = 55      // px per second — slow & calm
const SPAWN_INTERVAL = 0.7 // seconds between new dots entering (while below the cap)
const EXIT_CHANCE = 0.45   // at a border neuron, chance to head off-screen instead of turning
const MAX_HOPS = 16        // safety: force a dot off-screen after this many turns
const BORDER_MARGIN = 150  // a neuron this close to an edge is a valid entry / exit point
const WANDER = 0.42        // global multiplier on how fast neurons drift

// aurora — very slow breathing colour fields painted BEHIND the mesh, for depth
const AURORA_COLORS = [CYAN, VIOLET, BLUE, [120, 150, 245]] as const
const AURORA_N = 5
const AURORA_SPEED = 0.05  // fraction of a lissajous cycle per second-ish — extremely slow
const AURORA_ALPHA = 0.08  // per-blob opacity (normal blending — overlaps stay bounded)
export function NeuralBackground({ noAnim = false }: { noAnim?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const cv = canvas       // stable non-null alias for use inside closures
    const c2 = ctx
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    const still = noAnim || reduced

    let w = 0, h = 0
    let nodes: Node[] = []
    let edges: Edge[] = []
    let adj: number[][] = []       // node → incident edge indices
    let walkers: Walker[] = []
    let border: number[] = []      // indices of neurons near the screen edge (entry/exit)
    let isBorder: boolean[] = []
    let auroras: Aurora[] = []
    let raf = 0
    let last = 0
    let spawnAcc = 0
    // deterministic-ish PRNG so the layout is stable within a mount
    let seed = 1
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed / 0x7fffffff
    }

    function build() {
      w = window.innerWidth
      h = window.innerHeight
      const pixelRatio = window.devicePixelRatio || 1
      cv.width = Math.round(w * pixelRatio)
      cv.height = Math.round(h * pixelRatio)
      cv.style.width = w + 'px'
      cv.style.height = h + 'px'
      c2.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)

      // scatter neurons on a jittered grid so they're evenly spread but organic.
      // if the ideal grid exceeds the node budget, grow the cell instead of cutting
      // rows off the bottom — the field must cover the whole screen, top to bottom.
      seed = 1
      nodes = []
      let cols = Math.max(2, Math.round(w / CELL))
      let rows = Math.max(2, Math.round(h / CELL))
      if (cols * rows > MAX_NODES) {
        const s = Math.sqrt((cols * rows) / MAX_NODES)
        cols = Math.max(2, Math.round(cols / s))
        rows = Math.max(2, Math.round(rows / s))
      }
      const gx = w / cols, gy = h / rows
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const bx = gx * (c + 0.5) + (rnd() - 0.5) * gx * 0.7
          const by = gy * (r + 0.5) + (rnd() - 0.5) * gy * 0.7
          nodes.push({
            bx, by, x: bx, y: by,
            r: 1.2 + rnd() * 1.6,
            px: rnd() * Math.PI * 2, py: rnd() * Math.PI * 2,
            ax: 5 + rnd() * 9, ay: 5 + rnd() * 9,
            sx: 0.02 + rnd() * 0.03, sy: 0.02 + rnd() * 0.03,   // slowed drift
            tint: rnd(),
          })
        }
      }

      // wire each neuron to its nearest few — a stable, real-looking mesh
      edges = []
      const seen = new Set<string>()
      for (let i = 0; i < nodes.length; i++) {
        const near = nodes
          .map((n, j) => ({ j, d: Math.hypot(n.bx - nodes[i].bx, n.by - nodes[i].by) }))
          .filter((o) => o.j !== i && o.d < LINK_DIST)
          .sort((p, q) => p.d - q.d)
          .slice(0, MAX_LINKS)
        for (const { j, d } of near) {
          const key = i < j ? `${i}-${j}` : `${j}-${i}`
          if (seen.has(key)) continue
          seen.add(key)
          edges.push({ a: i, b: j, len: d })
        }
      }
      adj = nodes.map(() => [])
      edges.forEach((e, idx) => { adj[e.a].push(idx); adj[e.b].push(idx) })

      // neurons close to a screen edge are the doorways where dots enter and leave
      isBorder = nodes.map((n) => n.bx < BORDER_MARGIN || n.bx > w - BORDER_MARGIN || n.by < BORDER_MARGIN || n.by > h - BORDER_MARGIN)
      border = []
      isBorder.forEach((b, i) => { if (b) border.push(i) })
      walkers = []

      // a few big, soft colour blobs that drift on slow lissajous paths behind the mesh
      auroras = []
      for (let i = 0; i < AURORA_N; i++) {
        auroras.push({
          ax: rnd() * Math.PI * 2, ay: rnd() * Math.PI * 2,
          sx: AURORA_SPEED * (0.6 + rnd() * 0.8), sy: AURORA_SPEED * (0.6 + rnd() * 0.8),
          rad: Math.max(w, h) * (0.30 + rnd() * 0.24),
          col: AURORA_COLORS[i % AURORA_COLORS.length],
        })
      }
    }

    function paintAuroras(time: number) {
      for (const au of auroras) {
        const cx = w * 0.5 + Math.cos(au.ax + time * au.sx * 6.283) * w * 0.42
        const cy = h * 0.5 + Math.sin(au.ay + time * au.sy * 6.283) * h * 0.46
        const gradient = c2.createRadialGradient(cx, cy, 0, cx, cy, au.rad)
        gradient.addColorStop(0, `rgba(${au.col[0]},${au.col[1]},${au.col[2]},${AURORA_ALPHA})`)
        gradient.addColorStop(1, `rgba(${au.col[0]},${au.col[1]},${au.col[2]},0)`)
        c2.fillStyle = gradient
        c2.beginPath()
        c2.arc(cx, cy, au.rad, 0, Math.PI * 2)
        c2.fill()
      }
    }

    // a point safely off-screen, in line with a border neuron (its entry/exit doorway)
    function outPoint(i: number) {
      const n = nodes[i]
      const dx = n.bx - w / 2, dy = n.by - h / 2
      const d = Math.hypot(dx, dy) || 1
      // comfortably past the nearest edge (must exceed BORDER_MARGIN so entries really
      // start off-screen) yet short enough that dots come into view quickly
      return { x: n.bx + (dx / d) * 260, y: n.by + (dy / d) * 260 }
    }
    const neighborsOf = (i: number) => adj[i].map((ei) => ({ node: edges[ei].a === i ? edges[ei].b : edges[ei].a, edge: ei }))

    // spawn a dot flying in from off-screen toward a random border neuron
    function spawnWalker() {
      if (walkers.length >= MAX_WALKERS || !border.length) return
      const b = border[(rnd() * border.length) | 0]
      const op = outPoint(b), nb = nodes[b]
      walkers.push({
        aNode: -1, bNode: b, ax: op.x, ay: op.y, bx: nb.bx, by: nb.by,
        seglen: Math.max(1, Math.hypot(nb.bx - op.x, nb.by - op.y)), t: 0,
        edge: -1, from: -1, hops: 0, exiting: false, tint: nb.tint,
      })
    }

    // dot reached neuron `cur` → pick a fresh wire, or leave the screen
    function nextSegment(wk: Walker) {
      const cur = wk.bNode
      wk.hops++
      let nbs = neighborsOf(cur).filter((o) => o.node !== wk.from)
      if (!nbs.length) nbs = neighborsOf(cur)
      // leave the screen at a border neuron (but not instantly on entry — walk a bit first)
      const leave = wk.hops >= MAX_HOPS || !nbs.length || (wk.hops >= 2 && isBorder[cur] && rnd() < EXIT_CHANCE)
      const nc = nodes[cur]
      if (leave) {
        const op = outPoint(cur)
        wk.aNode = cur; wk.from = cur; wk.bNode = -1
        wk.ax = nc.x; wk.ay = nc.y; wk.bx = op.x; wk.by = op.y
        wk.edge = -1; wk.exiting = true
        wk.seglen = Math.max(1, Math.hypot(op.x - nc.x, op.y - nc.y)); wk.t = 0
        return
      }
      const c = nbs[(rnd() * nbs.length) | 0], nn = nodes[c.node]
      wk.aNode = cur; wk.from = cur; wk.bNode = c.node; wk.edge = c.edge; wk.exiting = false
      wk.seglen = Math.max(1, Math.hypot(nn.bx - nc.bx, nn.by - nc.by)); wk.t = 0
    }

    function draw(now: number) {
      const dt = Math.min(0.05, last ? (now - last) / 1000 : 0.016)
      last = now
      const time = now / 1000

      c2.clearRect(0, 0, w, h)

      paintAuroras(time)

      // wander the neurons a touch (topology stays; wires just breathe)
      if (!still) {
        for (const n of nodes) {
          n.x = n.bx + Math.cos(n.px + time * n.sx * 6.283 * WANDER) * n.ax
          n.y = n.by + Math.sin(n.py + time * n.sy * 6.283 * WANDER) * n.ay
        }
      }

      // which wires currently carry a dot (drawn a little brighter)
      const hot = new Set<number>()
      for (const wk of walkers) if (wk.edge >= 0) hot.add(wk.edge)

      // wires
      c2.lineWidth = 1
      for (let i = 0; i < edges.length; i++) {
        const e = edges[i]
        const a = nodes[e.a], b = nodes[e.b]
        const fade = 1 - e.len / LINK_DIST            // shorter = clearer
        const base = 0.05 + fade * 0.07
        const alpha = hot.has(i) ? base + 0.10 : base
        const t = (nodes[e.a].tint + nodes[e.b].tint) / 2
        const [r, g, bl] = mix(t)
        c2.strokeStyle = `rgba(${r},${g},${bl},${alpha})`
        c2.beginPath()
        c2.moveTo(a.x, a.y)
        c2.lineTo(b.x, b.y)
        c2.stroke()
      }

      // neurons
      for (const n of nodes) {
        const [r, g, bl] = mix(n.tint)
        c2.fillStyle = `rgba(${r},${g},${bl},0.34)`
        c2.beginPath()
        c2.arc(n.x, n.y, n.r, 0, Math.PI * 2)
        c2.fill()
      }

      // travelling dots — fly in, random-walk the mesh, fly out. Luminous, additive glow.
      if (!still) {
        c2.globalCompositeOperation = 'lighter'
        for (let i = walkers.length - 1; i >= 0; i--) {
          const wk = walkers[i]
          wk.t += (WALK_SPEED * dt) / wk.seglen
          const ax = wk.aNode >= 0 ? nodes[wk.aNode].x : wk.ax
          const ay = wk.aNode >= 0 ? nodes[wk.aNode].y : wk.ay
          const bx = wk.bNode >= 0 ? nodes[wk.bNode].x : wk.bx
          const by = wk.bNode >= 0 ? nodes[wk.bNode].y : wk.by
          const k = Math.min(1, wk.t)
          const x = ax + (bx - ax) * k, y = ay + (by - ay) * k
          // an exiting dot is done the moment it clears the viewport — frees the slot
          if (wk.exiting && (x < -20 || x > w + 20 || y < -20 || y > h + 20)) { walkers.splice(i, 1); continue }
          const [r, g, bl] = mix(wk.tint)
          const glow = c2.createRadialGradient(x, y, 0, x, y, 9)
          glow.addColorStop(0, `rgba(${r},${g},${bl},0.55)`)
          glow.addColorStop(1, `rgba(${r},${g},${bl},0)`)
          c2.fillStyle = glow
          c2.beginPath()
          c2.arc(x, y, 9, 0, Math.PI * 2)
          c2.fill()
          c2.fillStyle = `rgba(${Math.min(255, r + 60)},${Math.min(255, g + 40)},${bl},0.9)`
          c2.beginPath()
          c2.arc(x, y, 1.5, 0, Math.PI * 2)
          c2.fill()

          if (wk.t >= 1) {
            if (wk.exiting) walkers.splice(i, 1)   // flew off the screen — gone
            else nextSegment(wk)                   // reached a neuron — pick the next wire
          }
        }
        c2.globalCompositeOperation = 'source-over'

        // trickle new dots in from the edges, up to the on-screen cap
        spawnAcc += dt
        while (spawnAcc >= SPAWN_INTERVAL) { spawnAcc -= SPAWN_INTERVAL; spawnWalker() }
      }

      if (!still) raf = requestAnimationFrame(draw)
    }

    let resizeTimer = 0
    const rebuild = () => {
      window.clearTimeout(resizeTimer)
      resizeTimer = window.setTimeout(() => {
        build()
        last = 0
        draw(performance.now())          // always paint one frame immediately
        if (!still && !raf && !document.hidden) raf = requestAnimationFrame(draw)
      }, 150)
    }
    const onVisibility = () => {
      if (document.hidden) {
        if (raf) cancelAnimationFrame(raf)
        raf = 0
      } else if (!still && !raf) {
        last = 0
        raf = requestAnimationFrame(draw)
      }
    }

    build()
    draw(performance.now())              // immediate first frame, no rAF wait
    if (!still && !document.hidden) raf = requestAnimationFrame(draw)
    window.addEventListener('resize', rebuild)
    document.addEventListener('visibilitychange', onVisibility)
    // ResizeObserver catches layout size changes the window 'resize' event misses
    const ro = new ResizeObserver(rebuild)
    ro.observe(document.documentElement)

    return () => {
      if (raf) cancelAnimationFrame(raf)
      window.clearTimeout(resizeTimer)
      window.removeEventListener('resize', rebuild)
      document.removeEventListener('visibilitychange', onVisibility)
      ro.disconnect()
    }
  }, [noAnim])

  return <canvas ref={ref} className="neural-bg" aria-hidden="true" />
}
