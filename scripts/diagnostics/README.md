# Live appliance diagnostics

These scripts are read-only operator/engineering diagnostics for an installed
EVA appliance. The shell wrappers discover the application directory and
environment file from `eva-ai.service`; they do not depend on a developer
Desktop or on the factory-lab filesystem layout.

- `live_l0_e2e_report.sh --minutes 20` reports end-to-end durable L0 latency.
- `live_l0_trace.sh --channel 112 --minutes 15` prints a secret-free recent L0
  trace for one channel.

Run them with sufficient permission to read the service environment and EVA
database. They do not print credentials, prompts, or image data.
