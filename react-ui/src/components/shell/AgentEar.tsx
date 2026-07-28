import { IconSparkles } from '@tabler/icons-react'

export function AgentEar({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <div className="agent-ear-wrap">
      <button
        className={`agent-ear ${open ? 'on' : ''}`}
        data-agent
        title={open ? 'Close agent' : 'Ask EVA agent'}
        onClick={onToggle}
      >
        <IconSparkles size={15} stroke={2} />
        <span className="txt">AGENT</span>
      </button>
    </div>
  )
}
