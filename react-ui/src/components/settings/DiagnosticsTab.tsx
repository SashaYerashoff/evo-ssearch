import { IconActivityHeartbeat } from '@tabler/icons-react'
import { BenchmarkButton } from '../monitoring/BenchmarkButton'

export function DiagnosticsTab() {
  return (
    <div className="set-section diagnostics-tab">
      <h3><IconActivityHeartbeat size={16} /> Runtime diagnostics</h3>
      <p className="set-section-help">
        Measure the active embedding backend on this host. The benchmark is explicit and does not alter stream settings.
      </p>
      <div className="diagnostics-action">
        <div>
          <b>Embedding throughput</b>
          <span>Runs a short 16-frame probe batch and reports approximate frames per second and device.</span>
        </div>
        <BenchmarkButton />
      </div>
    </div>
  )
}
