import { useState } from 'react'
import { IconBolt } from '@tabler/icons-react'
import { probesApi, type Benchmark } from '../../api/probes'

// Explicit diagnostic action; kept out of the operational status chrome.
export function BenchmarkButton() {
  const [bench, setBench] = useState<Benchmark | null>(null)
  const [busy, setBusy] = useState(false)

  async function run() {
    if (busy) return
    setBusy(true)
    try { const b = await probesApi.bench(16); if (!(b as any).error) setBench(b) }
    catch { /* ignore */ }
    finally { setBusy(false) }
  }

  return (
    <button
      className={`bench-btn ${busy ? 'busy' : ''} ${bench ? 'done' : ''}`}
      onClick={run}
      disabled={busy}
      title="Run embedding throughput benchmark (GPU embed fps)"
    >
      <IconBolt size={14} />
      {busy ? 'Benchmarking…' : bench ? <span className="mono">~{bench.approx_fps} fps · {bench.device}</span> : 'Benchmark'}
    </button>
  )
}
