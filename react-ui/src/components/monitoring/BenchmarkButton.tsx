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

  const deviceLabel = String(bench?.device_name || bench?.device || '')
    .replace(/^NVIDIA\s+/i, '')

  return (
    <button
      className={`bench-btn ${busy ? 'busy' : ''} ${bench ? 'done' : ''}`}
      onClick={run}
      disabled={busy}
      title={bench
        ? `Encoder ${bench.encoder_fps ?? bench.approx_fps} fps · effective ${bench.effective_fps ?? bench.approx_fps} fps · lock wait ${bench.average_lock_wait_ms ?? 0} ms · warm-up wait ${bench.warmup_lock_wait_ms ?? 0} ms · ${bench.device_name || bench.device}${bench.truncated ? ' · time-budget reached' : ''}`
        : 'Run synchronized embedding benchmark (encoder throughput vs shared-runtime wait)'}
    >
      <IconBolt size={14} />
      {busy
        ? 'Benchmarking…'
        : bench
          ? <span className="mono">{bench.encoder_fps ?? bench.approx_fps} enc / {bench.effective_fps ?? bench.approx_fps} eff · {deviceLabel}</span>
          : 'Benchmark'}
    </button>
  )
}
