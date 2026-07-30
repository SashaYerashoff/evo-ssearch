import { useEffect, useRef, useState } from 'react'

// Home / hero screen — an artificial eye augmented by a human-shaped neural brain.
// Mini-game: metadata frames drift slowly toward the eye; hovering the mouse near a
// frame "analyses" it — it accelerates into the pupil. Each catch bumps the HUD
// counter and fires a packet up a nerve thread + an excitation cascade in the brain.

// neural mesh laid over a human brain in left-facing profile
const RING: [number, number][] = [
  [440, 280], [470, 205], [510, 148], [575, 120], [648, 122], [712, 135],
  [768, 175], [790, 225], [775, 285], [740, 320], [700, 355], [645, 352],
  [580, 342], [510, 325], [462, 305],
]
const CORE: [number, number][] = [
  [540, 238], [600, 214], [662, 238], [568, 296], [634, 296], [700, 280], [600, 258],
]
const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 0],
  [15, 16], [16, 17], [15, 21], [16, 21], [17, 21], [18, 21], [19, 21], [20, 17], [18, 19], [19, 20], [20, 21],
  [15, 1], [15, 2], [15, 3], [16, 3], [16, 4], [16, 5], [17, 5], [17, 6], [17, 7], [20, 7], [20, 8], [20, 9], [19, 10], [19, 11], [18, 12], [18, 13], [15, 14], [18, 14],
]
// deep-structure nodes appended AFTER ring+core (keeps existing indices/NERVES valid),
// adds interior density so the brain reads as a real mesh, not a sparse web
const EXTRA: [number, number][] = [
  [500, 252], [520, 292], [480, 272], [555, 258], [560, 200],
  [612, 236], [610, 340], [600, 312], [640, 190], [690, 210],
  [720, 258], [740, 300], [660, 322], [583, 300], [636, 262],
  [512, 224], [700, 240], [548, 224], [672, 300], [590, 178],
]
const RING_N = RING.length
const CORE_N = CORE.length
const NODES = [...RING, ...CORE, ...EXTRA]

const distN = (a: [number, number], b: [number, number]) => Math.hypot(a[0] - b[0], a[1] - b[1])
// fine synaptic web: connect every node to its k nearest neighbours (organic density)
function nearestEdges(nodes: [number, number][], k: number): [number, number][] {
  const seen = new Set<string>()
  const out: [number, number][] = []
  nodes.forEach((n, i) => {
    nodes.map((m, j) => ({ j, d: distN(n, m) })).filter((o) => o.j !== i).sort((a, b) => a.d - b.d).slice(0, k)
      .forEach((o) => { const key = i < o.j ? `${i}-${o.j}` : `${o.j}-${i}`; if (!seen.has(key)) { seen.add(key); out.push([Math.min(i, o.j), Math.max(i, o.j)]) } })
  })
  return out
}
const MESH_EDGES = nearestEdges(NODES, 3)
// tiny satellite dots hung off deep nodes for texture
const MICRO: [number, number][] = EXTRA.map(([x, y], i) => [x + (i % 2 ? 9 : -9), y + (i % 3 ? -7 : 8)] as [number, number])

const BRAIN_CX = 610
const BRAIN_CY = 245
const BRAIN_SCALE = 1.22
const BRAIN_TRANSFORM = `translate(${BRAIN_CX} ${BRAIN_CY}) scale(${BRAIN_SCALE}) translate(${-BRAIN_CX} ${-BRAIN_CY})`
const brainPoint = ([x, y]: [number, number]): [number, number] => [
  BRAIN_CX + (x - BRAIN_CX) * BRAIN_SCALE,
  BRAIN_CY + (y - BRAIN_CY) * BRAIN_SCALE,
]
const BRAIN_NODES = NODES.map(brainPoint)
const BRAIN_OUTLINE = `M 440 280
  C 415 230, 445 163, 510 145
  C 542 116, 612 110, 655 127
  C 715 116, 772 148, 786 205
  C 801 250, 788 290, 756 312
  C 776 330, 764 356, 728 361
  C 704 372, 676 363, 667 344
  C 636 356, 600 352, 570 341
  C 532 331, 500 326, 480 318
  C 463 312, 449 297, 440 280 Z`

// optic-nerve threads; each STARTS behind the eyelid (hidden by the opaque iris,
// so it looks like it slips out from behind the eye) and ends exactly on a neuron.
const NERVE_SPECS: { start: [number, number]; node: number; bend: number }[] = [
  { start: [535, 555], node: 13, bend: -34 },
  { start: [566, 548], node: 18, bend: -18 },
  { start: [596, 544], node: 12, bend: -4 },
  { start: [626, 548], node: 11, bend: 18 },
  { start: [657, 555], node: 10, bend: 34 },
]
const NERVES: { path: string; node: number }[] = NERVE_SPECS.map(({ start, node, bend }) => {
  const tip = BRAIN_NODES[node]
  return {
    node,
    path: `M ${start[0]} ${start[1]} C ${start[0] + bend} 500, ${tip[0] - bend} ${tip[1] + 72}, ${tip[0]} ${tip[1]}`,
  }
})
const dist = (a: [number, number], b: [number, number]) => Math.hypot(a[0] - b[0], a[1] - b[1])

const EYE: [number, number] = [600, 590]
const EYE_SCALE = 0.78
const EYE_TRANSFORM = `translate(${EYE[0]} ${EYE[1]}) scale(${EYE_SCALE}) translate(-600 -560)`
const EYE_RX = 248 * EYE_SCALE
const EYE_RY = 108 * EYE_SCALE
const IRIS_SPOKES = Array.from({ length: 48 }, (_, i) => {
  const angle = (i / 48) * Math.PI * 2
  const jitter = (i % 3) * 2
  return {
    x1: 600 + Math.cos(angle) * (49 + jitter),
    y1: 560 + Math.sin(angle) * (49 + jitter),
    x2: 600 + Math.cos(angle) * (96 - (i % 4) * 2),
    y2: 560 + Math.sin(angle) * (96 - (i % 4) * 2),
  }
})
// spawn ring pushed out past the canvas edges — frames appear far away and travel in;
// top spawns sit at the corners so their paths skirt the brain instead of crossing it
// spawn ring on the FAR perimeter of the expanded viewBox (-700..1900 × -280..1080),
// so frames appear well past the real screen edges and travel a long way to the eye
const SPAWNS: [number, number][] = [
  [-660, -100], [-660, 400], [-660, 900],
  [-200, -240], [600, -240], [1400, -240],
  [1860, 0], [1860, 500], [1860, 1000],
  [100, 1040], [700, 1040], [1300, 1040],
]
const LABELS = ['person · 0.97', 'vehicle · 0.91', 'face · 0.88', 'bag · 0.76', 'person · 0.93', 'plate · 0.85', 'helmet · 0.82', 'bicycle · 0.84', 'dog · 0.79', 'truck · 0.90']
const DETS = [
  { x: -20, y: -14, w: 20, h: 26 }, { x: -30, y: -6, w: 34, h: 16 }, { x: -8, y: -14, w: 18, h: 18 },
  { x: 2, y: -10, w: 16, h: 20 }, { x: -16, y: -16, w: 18, h: 30 }, { x: -26, y: -2, w: 30, h: 12 },
]

const BASE_RATE = 0.064      // doubled: ~15s edge → eye
const PULL_RATE = 0.6        // doubled game pull speed
const PULL_RADIUS = 90       // direct mouse proximity also counts
const MAX_FRAMES = 14
// line-of-sight beam: a cone from the eye toward the cursor
const BEAM_START = 90        // begins just outside the smaller iris
const BEAM_HALF_W = 21       // half-width at the start
const BEAM_SPREAD = Math.tan((8 * Math.PI) / 180)  // widens ~8° per side
const fmtTimer = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

interface FlyFrame { id: number; sx: number; sy: number; p: number; label: string; det: { x: number; y: number; w: number; h: number }; bad?: boolean; cx?: number; cy?: number }
interface BrainEvent { id: number; nerve: number; casc: { ni: number; hop: number }[]; bad?: boolean }
interface EvapPart { px: number; py: number; ex: number; ey: number; s: number; rot: number; delay: number; fill: string }
interface Evap { id: number; x: number; y: number; parts: EvapPart[] }
interface MetadataStreamSnapshot {
  version: 1
  serverStartedAtMs: number | null
  savedAtMs: number
  nextId: number
  spawnDelayMs: number
  frames: FlyFrame[]
  gameActive: boolean
  eyePresses: number
  count: number
  secs: number
}

const rnd = (a: number, b: number) => a + Math.random() * (b - a)
const EVAP_FILLS = ['#38e0d4', '#00d68f', '#7af7e6']
const STREAM_STORAGE_KEY = 'eva.home.metadata-stream.v1'
const SERVER_START_TOLERANCE_MS = 2500
const AVERAGE_SPAWN_MS = 875

const BAD_LABELS = ['CORRUPT · ??', 'signal lost', 'decode err', 'malformed', 'ERR · 0x1F']
function makeFlyFrame(id: number, p = 0): FlyFrame {
  const sp = SPAWNS[Math.floor(Math.random() * SPAWNS.length)]
  const bad = Math.random() < 0.08   // rare harmful frame
  return {
    id,
    sx: sp[0] + rnd(-30, 30),
    sy: sp[1] + rnd(-25, 25),
    p,
    bad,
    label: bad ? BAD_LABELS[Math.floor(Math.random() * BAD_LABELS.length)] : LABELS[Math.floor(Math.random() * LABELS.length)],
    det: DETS[Math.floor(Math.random() * DETS.length)],
  }
}

function seedServerStream(serverStartedAtMs: number, nowMs = Date.now()): FlyFrame[] {
  const uptimeMs = Math.max(0, nowMs - serverStartedAtMs)
  const count = Math.min(MAX_FRAMES, Math.floor(uptimeMs / AVERAGE_SPAWN_MS) + 1)
  const phase = (uptimeMs % AVERAGE_SPAWN_MS) / AVERAGE_SPAWN_MS
  return Array.from({ length: count }, (_, index) => {
    const spacing = 0.82 / Math.max(1, count)
    const p = Math.min(0.9, 0.03 + (index + phase) * spacing)
    return makeFlyFrame(index + 1, p)
  })
}

function readStreamSnapshot(): MetadataStreamSnapshot | null {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STREAM_STORAGE_KEY) || 'null')
    if (
      !parsed || parsed.version !== 1 || !Array.isArray(parsed.frames)
      || !Number.isFinite(parsed.savedAtMs) || !Number.isFinite(parsed.nextId)
    ) return null
    const elapsedSec = Math.max(0, Date.now() - parsed.savedAtMs) / 1000
    const frames = parsed.frames
      .filter((frame: FlyFrame) => (
        Number.isFinite(frame?.id) && Number.isFinite(frame?.sx)
        && Number.isFinite(frame?.sy) && Number.isFinite(frame?.p)
        && typeof frame?.label === 'string' && frame?.det
      ))
      .slice(0, MAX_FRAMES)
      .map((frame: FlyFrame) => ({ ...frame, p: frame.p + elapsedSec * BASE_RATE }))
      .filter((frame: FlyFrame) => frame.p < 0.96)
    return {
      version: 1,
      serverStartedAtMs: Number.isFinite(parsed.serverStartedAtMs) ? parsed.serverStartedAtMs : null,
      savedAtMs: parsed.savedAtMs,
      nextId: Math.max(1, parsed.nextId),
      spawnDelayMs: Math.max(0, Number(parsed.spawnDelayMs) - elapsedSec * 1000 || 0),
      frames,
      gameActive: Boolean(parsed.gameActive),
      eyePresses: Math.max(0, Math.min(5, Number(parsed.eyePresses) || 0)),
      count: Math.max(0, Number(parsed.count) || 0),
      secs: Math.max(0, Number(parsed.secs) || 0) + (parsed.gameActive ? Math.floor(elapsedSec) : 0),
    }
  } catch {
    return null
  }
}

// pixel-dissolve burst: little squares scatter outward/up and fade
function makeEvap(id: number, x: number, y: number): Evap {
  const parts: EvapPart[] = []
  for (let i = 0; i < 12; i++) {
    const a = rnd(0, Math.PI * 2), d = rnd(24, 64)
    parts.push({
      px: rnd(-30, 30), py: rnd(-16, 16),
      ex: Math.cos(a) * d, ey: Math.sin(a) * d - rnd(14, 30),
      s: rnd(3.5, 8), rot: rnd(-160, 160), delay: rnd(0, 0.16),
      fill: EVAP_FILLS[i % EVAP_FILLS.length],
    })
  }
  return { id, x, y, parts }
}

// random 4–8 neurons near the arrival node, staggered by distance ring
function makeCascade(nerveIdx: number): { ni: number; hop: number }[] {
  const tip = NODES[NERVES[nerveIdx].node] as [number, number]
  const ranked = NODES.map((n, ni) => ({ ni, d: dist(n as [number, number], tip) })).sort((a, b) => a.d - b.d)
  const pool = ranked.slice(0, 13)
  const target = 4 + Math.floor(Math.random() * 5)
  const chosen = new Set<number>([ranked[0].ni])
  let guard = 0
  while (chosen.size < target && guard++ < 80) chosen.add(pool[Math.floor(Math.random() * pool.length)].ni)
  return [...chosen].map((ni) => ({ ni, hop: Math.round(ranked.find((r) => r.ni === ni)!.d / 70) }))
}

export function HomeScreen({
  active = true,
  serverStartedAtMs = null,
}: {
  active?: boolean
  serverStartedAtMs?: number | null
}) {
  const [initialSnapshot] = useState(readStreamSnapshot)
  const svgRef = useRef<SVGSVGElement>(null)
  const pupilRef = useRef<SVGGElement>(null)
  const framesRef = useRef<FlyFrame[]>(initialSnapshot?.frames ?? [])
  const frameEls = useRef(new Map<number, SVGGElement>())
  const mouse = useRef<{ x: number; y: number } | null>(null)
  const idRef = useRef(initialSnapshot?.nextId ?? 1)
  const spawnAt = useRef(performance.now() + (initialSnapshot?.spawnDelayMs ?? 0))
  const beamRef = useRef<SVGPathElement>(null)
  // the GAME never survives a reload/navigation — only the ambient frame stream does
  const eyePressRef = useRef(0)
  const blinkTimerRef = useRef<number | null>(null)
  const serverBootRef = useRef<number | null>(initialSnapshot?.serverStartedAtMs ?? null)
  const gameStateRef = useRef({ gameActive: false, eyePresses: 0, count: 0, secs: 0 })
  const [, force] = useState(0)
  const [events, setEvents] = useState<BrainEvent[]>([])
  const [evaps, setEvaps] = useState<Evap[]>([])
  const [gameActive, setGameActive] = useState(false)
  const [eyePresses, setEyePresses] = useState(0)
  const [blinkTick, setBlinkTick] = useState(0)
  const [manualBlink, setManualBlink] = useState(false)
  // Game score — hidden until the five-click eye activation.
  const [count, setCount] = useState(0)
  const [secs, setSecs] = useState(0)

  // leaving the Home screen (navigate away) ends the game
  useEffect(() => {
    if (active) return
    eyePressRef.current = 0
    setEyePresses(0); setGameActive(false); setCount(0); setSecs(0)
  }, [active])

  useEffect(() => {
    gameStateRef.current = { gameActive, eyePresses, count, secs }
  }, [gameActive, eyePresses, count, secs])

  useEffect(() => {
    if (serverStartedAtMs == null) return
    const sameServer = serverBootRef.current != null
      && Math.abs(serverBootRef.current - serverStartedAtMs) <= SERVER_START_TOLERANCE_MS
    serverBootRef.current = serverStartedAtMs
    if (sameServer && framesRef.current.length) return

    framesRef.current = seedServerStream(serverStartedAtMs)
    idRef.current = framesRef.current.length + 1
    spawnAt.current = performance.now() + rnd(450, 1300)
    frameEls.current.clear()
    if (initialSnapshot && !sameServer) {
      eyePressRef.current = 0
      setEyePresses(0)
      setGameActive(false)
      setCount(0)
      setSecs(0)
    }
    force((value) => value + 1)
  }, [initialSnapshot, serverStartedAtMs])

  useEffect(() => {
    const persist = () => {
      try {
        const game = gameStateRef.current
        const snapshot: MetadataStreamSnapshot = {
          version: 1,
          serverStartedAtMs: serverBootRef.current,
          savedAtMs: Date.now(),
          nextId: idRef.current,
          spawnDelayMs: Math.max(0, spawnAt.current - performance.now()),
          frames: framesRef.current.map((frame) => ({ ...frame, det: { ...frame.det } })),
          ...game,
        }
        window.localStorage.setItem(STREAM_STORAGE_KEY, JSON.stringify(snapshot))
      } catch {
        // Storage can be unavailable in privacy mode; the in-memory stream still continues.
      }
    }
    const timer = window.setInterval(persist, 500)
    window.addEventListener('pagehide', persist)
    return () => {
      persist()
      window.clearInterval(timer)
      window.removeEventListener('pagehide', persist)
    }
  }, [])

  useEffect(() => {
    if (!gameActive) return
    const t = window.setInterval(() => setSecs((s) => s + 1), 1000)
    return () => window.clearInterval(t)
  }, [gameActive])

  useEffect(() => () => {
    if (blinkTimerRef.current != null) window.clearTimeout(blinkTimerRef.current)
  }, [])

  const pressEye = () => {
    setBlinkTick((tick) => tick + 1)
    setManualBlink(true)
    if (blinkTimerRef.current != null) window.clearTimeout(blinkTimerRef.current)
    blinkTimerRef.current = window.setTimeout(() => {
      setManualBlink(false)
      blinkTimerRef.current = null
    }, 420)

    if (gameActive) return
    eyePressRef.current = Math.min(5, eyePressRef.current + 1)
    setEyePresses(eyePressRef.current)
    if (eyePressRef.current === 5) {
      setCount(0)
      setSecs(0)
      setGameActive(true)
    }
  }

  // mouse → pupil tracking + svg-space coords for the attraction check
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const svg = svgRef.current, g = pupilRef.current
      if (!svg) return
      const r = svg.getBoundingClientRect()
      const s = Math.min(r.width / 2600, r.height / 1360)   // meet: fit, matches preserveAspectRatio + viewBox
      const offX = r.left + (r.width - 2600 * s) / 2
      const offY = r.top + (r.height - 1360 * s) / 2
      mouse.current = { x: -700 + (e.clientX - offX) / s, y: -280 + (e.clientY - offY) / s }
      if (g) {
        const dx = Math.max(-16, Math.min(16, (mouse.current.x - EYE[0]) / 18))
        const dy = Math.max(-16, Math.min(16, (mouse.current.y - EYE[1]) / 18))
        g.style.transform = `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px)`
      }
    }
    const reset = () => { mouse.current = null; if (pupilRef.current) pupilRef.current.style.transform = 'translate(0px, 0px)' }
    window.addEventListener('mousemove', onMove)
    document.addEventListener('mouseleave', reset)
    return () => { window.removeEventListener('mousemove', onMove); document.removeEventListener('mouseleave', reset) }
  }, [])

  // game loop
  useEffect(() => {
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const list = framesRef.current

      // spawn up to MAX_FRAMES, with breathing room between spawns
      if (now >= spawnAt.current && list.length < MAX_FRAMES) {
        list.push(makeFlyFrame(idRef.current++))
        spawnAt.current = now + rnd(450, 1300)
        force((n) => n + 1)
      }

      // the eye's line of sight — a cone from the pupil toward the cursor
      let ux = 0, uy = 0, sighted = false
      const beam = beamRef.current
      if (gameActive && mouse.current) {
        const dx = mouse.current.x - EYE[0], dy = mouse.current.y - EYE[1]
        const L = Math.hypot(dx, dy)
        if (L > 24) {
          sighted = true
          ux = dx / L; uy = dy / L
          if (beam) {
            const px = -uy, py = ux
            const end = 1500, w1 = BEAM_HALF_W + end * BEAM_SPREAD
            const ax = EYE[0] + ux * BEAM_START + px * BEAM_HALF_W, ay = EYE[1] + uy * BEAM_START + py * BEAM_HALF_W
            const bx = EYE[0] + ux * end + px * w1, by = EYE[1] + uy * end + py * w1
            const cx = EYE[0] + ux * end - px * w1, cy = EYE[1] + uy * end - py * w1
            const dxp = EYE[0] + ux * BEAM_START - px * BEAM_HALF_W, dyp = EYE[1] + uy * BEAM_START - py * BEAM_HALF_W
            beam.setAttribute('d', `M ${ax.toFixed(1)} ${ay.toFixed(1)} L ${bx.toFixed(1)} ${by.toFixed(1)} L ${cx.toFixed(1)} ${cy.toFixed(1)} L ${dxp.toFixed(1)} ${dyp.toFixed(1)} Z`)
            beam.style.opacity = '1'
          }
        }
      }
      if (!sighted && beam) beam.style.opacity = '0'

      const caught: FlyFrame[] = []
      for (const f of list) {
        const x = f.sx + (EYE[0] - f.sx) * f.p
        const y = f.sy + (EYE[1] - f.sy) * f.p
        // "the eye looks at it": inside the sight cone, or right under the cursor
        let seen = false
        if (gameActive && mouse.current) {
          seen = Math.hypot(mouse.current.x - x, mouse.current.y - y) < PULL_RADIUS
          if (!seen && sighted) {
            const vx = x - EYE[0], vy = y - EYE[1]
            const proj = vx * ux + vy * uy
            if (proj > BEAM_START * 0.8) seen = Math.abs(vx * uy - vy * ux) < BEAM_HALF_W + proj * BEAM_SPREAD
          }
        }
        f.p += (seen ? PULL_RATE : BASE_RATE) * dt
        const el = frameEls.current.get(f.id)
        if (el) {
          // gentle shrink while drifting, hard suck-in over the last stretch
          const k = f.p < 0.86 ? 1 - f.p * 0.12 : Math.max(0.06, 0.9 - ((f.p - 0.86) / 0.14) * 0.84)
          const op = f.p < 0.04 ? f.p / 0.04 : f.p > 0.92 ? Math.max(0, 1 - (f.p - 0.92) / 0.08) : 1
          el.setAttribute('transform', `translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${k.toFixed(3)})`)
          el.style.opacity = op.toFixed(2)
          el.classList.toggle('analyzing', seen)
          // detection reveals itself mid-flight on its own, or instantly under the eye's gaze
          el.classList.toggle('det-on', seen || f.p > 0.5)
        }
        // caught the moment it touches the eye's edge (almond ≈ ellipse, padded for frame size)
        const nx = (x - EYE[0]) / EYE_RX, ny = (y - EYE[1]) / EYE_RY
        if (nx * nx + ny * ny <= 1 || f.p >= 1) { f.cx = x; f.cy = y; caught.push(f) }
      }

      if (caught.length) {
        framesRef.current = list.filter((f) => !caught.includes(f))
        // score + dissolve only for good frames consumed during an active game
        if (gameActive) {
          const good = caught.filter((f) => !f.bad)
          if (good.length) setCount((c) => c + good.length)
          for (const f of good) {
            const evap = makeEvap(idRef.current++, f.cx ?? EYE[0], f.cy ?? EYE[1])
            setEvaps((es) => [...es, evap])
            window.setTimeout(() => setEvaps((es) => es.filter((e) => e.id !== evap.id)), 1400)
          }
        }
        // brain impulse fires for EVERY metadata frame — game or not, watched or not.
        // bad frames throw a red glitch ripple across the whole brain instead of a cascade.
        for (const f of caught) {
          const nerve = Math.floor(Math.random() * NERVES.length)
          const ev: BrainEvent = { id: idRef.current++, nerve, casc: makeCascade(nerve), bad: f.bad }
          setEvents((es) => [...es, ev])
          window.setTimeout(() => setEvents((es) => es.filter((e) => e.id !== ev.id)), f.bad ? 1700 : 4500)
        }
        force((n) => n + 1)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [gameActive])

  return (
    <div className={`home-screen ${active ? '' : 'home-screen-hidden'}`} aria-hidden={!active}>
      <svg ref={svgRef} viewBox="-700 -280 2600 1360" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img"
        aria-label="EVA AI — artificial eye augmented by a neural human brain">
        {/* HUD — session timer + analysed counter, right above the brain */}
        {gameActive && (
          <text className="sp-hud" x="600" y="62" textAnchor="middle">
            <tspan className="hud-timer">{fmtTimer(secs)}</tspan>
            <tspan dx="18" className="hud-label">METADATA ANALYSED</tspan>
            <tspan dx="12" className="hud-count">{count}</tspan>
          </text>
        )}
        <defs>
          <linearGradient id="sp-line" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#38e0d4" />
            <stop offset="1" stopColor="#00d68f" />
          </linearGradient>
          <linearGradient id="sp-brain-fill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#172a4c" stopOpacity="0.82" />
            <stop offset="0.48" stopColor="#10253e" stopOpacity="0.62" />
            <stop offset="1" stopColor="#071a2c" stopOpacity="0.9" />
          </linearGradient>
          <radialGradient id="sp-brain-core" cx="0.52" cy="0.5" r="0.58">
            <stop offset="0" stopColor="#00d68f" stopOpacity="0.18" />
            <stop offset="0.58" stopColor="#38e0d4" stopOpacity="0.07" />
            <stop offset="1" stopColor="#38e0d4" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="sp-iris" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor="#0af5d0" />
            <stop offset="0.35" stopColor="#0a8f86" />
            <stop offset="0.75" stopColor="#0c3550" />
            <stop offset="1" stopColor="#0a1c33" />
          </radialGradient>
          <radialGradient id="sp-pupil" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor="#031018" />
            <stop offset="0.75" stopColor="#04222e" />
            <stop offset="1" stopColor="#0a4a4e" />
          </radialGradient>
          <filter id="sp-glow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="6" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <clipPath id="sp-brain-clip">
            <path d={BRAIN_OUTLINE} />
          </clipPath>
          <clipPath id="sp-eyeclip">
            <path key={`clip-lid-${blinkTick}`} className={`sp-lid ${manualBlink ? 'sp-manual-lid' : ''}`}
              d="M 380 560 Q 600 402 820 560 Q 600 718 380 560 Z" />
          </clipPath>
        </defs>

        {/* holographic registration field behind the neural assembly */}
        <g className="sp-backdrop" stroke="rgba(150,180,245,0.10)" fill="none">
          <ellipse cx="605" cy="340" rx="285" ry="270" />
          <ellipse cx="605" cy="340" rx="380" ry="350" strokeDasharray="3 10" />
          <ellipse cx="605" cy="340" rx="475" ry="430" strokeDasharray="1 14" />
          <path d="M 600 48 V 736 M 168 342 H 1032" strokeDasharray="2 16" opacity="0.5" />
          <path d="M 600 70 l -7 14 h 14 Z M 600 714 l -7 -14 h 14 Z" fill="rgba(56,224,212,0.18)" stroke="none" />
        </g>

        {/* ===== human brain, left-facing profile ===== */}
        <g className="sp-brain" transform={BRAIN_TRANSFORM} filter="url(#sp-glow)">
          <path d={BRAIN_OUTLINE}
            fill="url(#sp-brain-fill)" stroke="rgba(56,224,212,0.76)" strokeWidth="2.2" />
          <g clipPath="url(#sp-brain-clip)">
            <ellipse cx="608" cy="246" rx="185" ry="142" fill="url(#sp-brain-core)" />
            <g className="sp-brain-scanlines" stroke="rgba(122,247,230,0.075)" strokeWidth="0.7">
              {Array.from({ length: 17 }, (_, i) => (
                <line key={i} x1="420" y1={126 + i * 14} x2="798" y2={126 + i * 14} />
              ))}
            </g>
            <path className="sp-brain-scanband" d="M 420 190 H 804 V 230 H 420 Z" fill="rgba(56,224,212,0.035)" />
          </g>
          <path d="M 468 292 C 510 272, 560 268, 596 282 C 620 291, 640 297, 656 296"
            stroke="rgba(56,224,212,0.45)" strokeWidth="1.6" fill="none" />
          <g stroke="rgba(56,224,212,0.35)" strokeWidth="1.4" fill="none">
            <path d="M 490 195 C 510 210, 505 230, 525 240" />
            <path d="M 555 150 C 570 175, 560 195, 580 215" />
            <path d="M 635 135 C 640 165, 625 185, 640 210" />
            <path d="M 705 155 C 700 185, 715 200, 705 230" />
            <path d="M 750 205 C 738 230, 752 250, 740 275" />
            <path d="M 520 300 C 545 310, 570 308, 590 318" />
          </g>
          {/* cerebellum striations */}
          <g stroke="rgba(56,224,212,0.4)" strokeWidth="1.2" fill="none">
            <path d="M 676 350 C 690 342, 712 340, 730 346" />
            <path d="M 682 358 C 696 351, 714 349, 728 354" />
            <path d="M 688 366 C 700 360, 716 358, 726 362" />
          </g>
          {/* extra gyri folds — denser cortical texture */}
          <g stroke="rgba(56,224,212,0.28)" strokeWidth="1.2" fill="none">
            <path d="M 528 170 C 548 186, 542 208, 560 222" />
            <path d="M 596 132 C 606 156, 598 176, 612 196" />
            <path d="M 672 128 C 678 158, 664 176, 678 200" />
            <path d="M 740 168 C 730 192, 744 210, 734 236" />
            <path d="M 470 240 C 492 250, 500 270, 522 278" />
            <path d="M 560 260 C 584 268, 606 262, 628 272" />
            <path d="M 632 210 C 646 226, 638 246, 654 262" />
            <path d="M 700 250 C 690 272, 704 288, 694 310" />
            <path d="M 458 214 C 476 222, 482 238, 474 254" />
            <path d="M 492 164 C 510 174, 518 190, 510 208" />
            <path d="M 536 128 C 552 142, 546 158, 558 172" />
            <path d="M 624 118 C 632 140, 624 154, 636 170" />
            <path d="M 708 142 C 694 160, 706 176, 696 194" />
            <path d="M 766 196 C 750 214, 762 234, 748 248" />
            <path d="M 758 268 C 742 280, 746 298, 728 308" />
            <path d="M 492 286 C 508 292, 514 310, 532 314" />
            <path d="M 548 282 C 558 300, 550 314, 566 328" />
            <path d="M 622 286 C 610 306, 622 320, 614 338" />
            <path d="M 676 278 C 662 296, 674 312, 662 330" />
            <path d="M 716 214 C 698 222, 704 242, 686 252" />
          </g>
          {/* anatomical depth: longitudinal fissure, corpus callosum, thalamus and brain stem */}
          <g className="sp-brain-anatomy" fill="none">
            <path d="M 610 120 C 600 150, 614 174, 602 202 C 594 222, 606 242, 596 266"
              stroke="rgba(122,247,230,0.34)" strokeWidth="1.4" strokeDasharray="4 5" />
            <path d="M 520 250 C 552 218, 620 210, 676 236 C 650 224, 616 238, 590 254 C 566 270, 540 270, 520 250 Z"
              fill="rgba(0,214,143,0.055)" stroke="rgba(0,214,143,0.32)" strokeWidth="1.5" />
            <ellipse cx="612" cy="264" rx="33" ry="21"
              fill="rgba(56,224,212,0.045)" stroke="rgba(122,247,230,0.32)" strokeWidth="1.2" />
            <path d="M 644 304 C 634 326, 636 352, 622 382 C 616 394, 612 402, 606 412"
              stroke="rgba(56,224,212,0.6)" strokeWidth="5" opacity="0.22" strokeLinecap="round" />
            <path d="M 644 304 C 634 326, 636 352, 622 382 C 616 394, 612 402, 606 412"
              stroke="rgba(122,247,230,0.58)" strokeWidth="1.5" strokeLinecap="round" />
          </g>
          {/* inner contour for depth */}
          <path d="M 470 278 C 452 234, 480 182, 536 172 C 566 150, 620 148, 656 164
                   C 704 156, 750 184, 760 226 C 770 262, 758 292, 732 308"
            stroke="rgba(56,224,212,0.18)" strokeWidth="1" fill="none" />
          <path d="M 662 350 C 652 368, 640 382, 628 398"
            stroke="rgba(56,224,212,0.6)" strokeWidth="2.2" fill="none" strokeLinecap="round" />

          {/* fine synaptic web (nearest-neighbour) under the structural mesh */}
          <g stroke="rgba(56,224,212,0.16)" strokeWidth="0.8" fill="none">
            {MESH_EDGES.map(([a, b], i) => (
              <line key={`m${i}`} className={i % 8 === 0 ? 'sp-neural-edge' : undefined}
                x1={NODES[a][0]} y1={NODES[a][1]} x2={NODES[b][0]} y2={NODES[b][1]} />
            ))}
          </g>
          {/* structural mesh */}
          <g stroke="url(#sp-line)" strokeWidth="1.3" opacity="0.5" fill="none">
            {EDGES.map(([a, b], i) => (
              <line key={i} x1={NODES[a][0]} y1={NODES[a][1]} x2={NODES[b][0]} y2={NODES[b][1]} />
            ))}
          </g>
          {/* satellite micro-dots for texture */}
          {MICRO.map(([x, y], i) => (
            <circle key={`mi${i}`} cx={x} cy={y} r={1.3} fill="rgba(122,247,230,0.5)" />
          ))}
          {/* neurons: ring (bright), core (large), deep (small dim) */}
          {NODES.map(([x, y], i) => {
            const isRing = i < RING_N
            const isCore = i >= RING_N && i < RING_N + CORE_N
            return (
              <g key={i}>
                {isCore && <circle className="sp-node-halo" cx={x} cy={y} r="10" fill="none" stroke="rgba(0,214,143,0.18)" />}
                <circle className={`sp-node n${i % 6}`} cx={x} cy={y}
                  r={isRing ? 3.6 : isCore ? 5.2 : 2.4}
                  fill={isRing ? '#38e0d4' : isCore ? '#00d68f' : 'rgba(122,247,230,0.7)'} />
              </g>
            )
          })}
        </g>

        {/* optic-nerve threads */}
        <g className="sp-flow" fill="none" opacity="0.75">
          {NERVES.map((n, i) => (
            <path key={i} d={n.path} stroke="url(#sp-line)" strokeWidth="1.6" strokeDasharray="6 8" />
          ))}
        </g>

        {/* line-of-sight beam — the eye's capture cone, follows the cursor */}
        <path ref={beamRef} className="sp-beam" style={{ opacity: 0 }}
          fill="rgba(56,224,212,0.05)" stroke="rgba(56,224,212,0.22)" strokeWidth="1.2" strokeDasharray="7 9" />

        {/* ===== game: metadata frames drifting toward the eye ===== */}
        {framesRef.current.map((f) => (
          <g key={f.id} className={`sp-frame ${f.bad ? 'bad' : ''}`}
            transform={`translate(${f.sx} ${f.sy})`} style={{ opacity: 0 }}
            ref={(el) => { if (el) frameEls.current.set(f.id, el); else frameEls.current.delete(f.id) }}>
            <rect className="fr" x={-38} y={-22} width={76} height={44} rx={4} />
            <path d="M -38 -14 h 6 M -30 -22 v 6 M 38 14 h -6 M 30 22 v -6" stroke="rgba(56,224,212,0.9)" strokeWidth="1.4" fill="none" />
            <g className="det-g">
              <rect className="det" x={f.det.x} y={f.det.y} width={f.det.w} height={f.det.h} />
              <text x={f.det.x} y={f.det.y - 4}>{f.label}</text>
            </g>
          </g>
        ))}

        {/* ===== artificial eye — lids blink, iris/pupil stay round underneath ===== */}
        <g
          className={`sp-eye sp-eye-trigger ${gameActive ? 'game-active' : ''}`}
          transform={EYE_TRANSFORM}
          role="button"
          tabIndex={0}
          aria-label={gameActive ? 'EVA eye, game active' : `EVA eye, activation ${eyePresses} of 5`}
          onClick={pressEye}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              pressEye()
            }
          }}
        >
          <path key={`eye-lid-${blinkTick}`} className={`sp-lid ${manualBlink ? 'sp-manual-lid' : ''}`}
            d="M 380 560 Q 600 402 820 560 Q 600 718 380 560 Z"
            fill="rgba(14,24,48,0.55)" stroke="url(#sp-line)" strokeWidth="2" filter="url(#sp-glow)" />
          <g clipPath="url(#sp-eyeclip)">
            <ellipse cx="600" cy="560" rx="224" ry="142" fill="rgba(108,177,194,0.075)" />
            <ellipse cx="600" cy="560" rx="204" ry="126" fill="none"
              stroke="rgba(122,247,230,0.13)" strokeWidth="1.2" strokeDasharray="2 10" />
            <circle cx="600" cy="560" r="104" fill="url(#sp-iris)" />
            <g className="sp-iris-fibers" stroke="rgba(122,247,230,0.34)" strokeWidth="1">
              {IRIS_SPOKES.map((spoke, i) => (
                <line key={i} x1={spoke.x1} y1={spoke.y1} x2={spoke.x2} y2={spoke.y2} />
              ))}
            </g>
            <circle className="sp-iris-ring" cx="600" cy="560" r="104" fill="none"
              stroke="#38e0d4" strokeWidth="4" strokeDasharray="18 10" opacity="0.75" />
            <circle className="sp-iris-counter-ring" cx="600" cy="560" r="91" fill="none"
              stroke="rgba(122,247,230,0.42)" strokeWidth="1.4" strokeDasharray="2 8" />
            <circle cx="600" cy="560" r="76" fill="none" stroke="rgba(10,245,208,0.5)" strokeWidth="7" strokeDasharray="3.5 15" />
            <circle cx="600" cy="560" r="59" fill="none" stroke="rgba(10,245,208,0.28)" strokeWidth="1.2" />
            <g ref={pupilRef} className="sp-pupil">
              <circle cx="600" cy="560" r="44" fill="url(#sp-pupil)" stroke="rgba(10,245,208,0.6)" strokeWidth="1.5" />
              <circle cx="600" cy="560" r="35" fill="none" stroke="rgba(56,224,212,0.23)" strokeWidth="1" strokeDasharray="3 5" />
              <circle cx="600" cy="560" r="10" fill="#0af5d0" opacity="0.9" filter="url(#sp-glow)" />
              <ellipse cx="576" cy="535" rx="8" ry="5" fill="rgba(230,255,252,0.72)" transform="rotate(-34 576 535)" />
              <circle cx="622" cy="582" r="3" fill="rgba(230,255,252,0.35)" />
            </g>
            <path className="sp-scan" d="M 600 428 A 132 132 0 0 1 732 560" fill="none"
              stroke="#00d68f" strokeWidth="3" strokeLinecap="round" filter="url(#sp-glow)" />
          </g>
        </g>
        <g transform={EYE_TRANSFORM} stroke="rgba(56,224,212,0.55)" strokeWidth="2" fill="none">
          <path d="M 372 540 v -20 h 20" /> <path d="M 828 540 v -20 h -20" />
          <path d="M 372 580 v 20 h 20" /> <path d="M 828 580 v 20 h -20" />
          <path d="M 350 560 h 14 M 836 560 h 14 M 600 390 v 14 M 600 716 v 14"
            strokeWidth="1.2" opacity="0.45" />
        </g>

        {/* ===== evaporation: the caught frame dissolves into little squares ===== */}
        {evaps.map((ev) => (
          <g key={ev.id} transform={`translate(${ev.x.toFixed(1)} ${ev.y.toFixed(1)})`}>
            {ev.parts.map((p, i) => (
              <rect key={i} className="sp-evap" x={p.px - p.s / 2} y={p.py - p.s / 2} width={p.s} height={p.s}
                fill={p.fill} rx={1}
                style={{ '--ex': `${p.ex.toFixed(1)}px`, '--ey': `${p.ey.toFixed(1)}px`, '--erot': `${p.rot.toFixed(0)}deg`, animationDelay: `${p.delay.toFixed(2)}s` } as React.CSSProperties} />
            ))}
          </g>
        ))}

        {/* ===== catch events: packet climbs a nerve, then the cascade (or bad glitch) rolls out ===== */}
        {events.map((ev) => {
          const nerve = NERVES[ev.nerve]
          const tip = BRAIN_NODES[nerve.node]
          if (ev.bad) {
            // harmful frame → red data spike + a glitch ripple across the WHOLE brain
            return (
              <g key={ev.id}>
                <circle className="sp-up-once" style={{ offsetPath: `path('${nerve.path}')` }}
                  r={5} fill="#ff5a6a" filter="url(#sp-glow)" />
                <ellipse className="sp-glitch-ring" cx={612} cy={244} rx={234} ry={164}
                  fill="rgba(255,70,90,0.05)" stroke="rgba(255,90,106,0.55)" strokeWidth={1.6} />
                <ellipse className="sp-glitch-ring two" cx={612} cy={244} rx={234} ry={164}
                  fill="none" stroke="rgba(255,120,134,0.4)" strokeWidth={1.2} />
                {BRAIN_NODES.map(([x, y], i) => (
                  <circle key={i} className="sp-glitch-node" cx={x} cy={y} r={i < RING_N ? 4 : 3}
                    fill="#ff6a78" style={{ animationDelay: `${(1.0 + distN(BRAIN_NODES[i], tip) / 520).toFixed(2)}s` }} />
                ))}
              </g>
            )
          }
          return (
            <g key={ev.id}>
              <circle className="sp-up-once" style={{ offsetPath: `path('${nerve.path}')` }}
                r={4.5} fill="#0af5d0" filter="url(#sp-glow)" />
              <circle className="sp-wave-once" style={{ animationDelay: '1.05s' }}
                cx={tip[0]} cy={tip[1]} r={105} fill="none" stroke="rgba(56,224,212,0.5)" strokeWidth={1.6} />
              {ev.casc.map(({ ni, hop }) => (
                <circle key={ni} className="sp-casc-once" style={{ animationDelay: `${(1.05 + hop * 0.3).toFixed(2)}s` }}
                  cx={BRAIN_NODES[ni][0]} cy={BRAIN_NODES[ni][1]} r={ni < RING_N ? 4.5 : 6} fill="#7af7e6" />
              ))}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
