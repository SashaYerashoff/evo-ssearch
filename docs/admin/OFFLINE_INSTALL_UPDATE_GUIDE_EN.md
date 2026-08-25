# EVA AI 0.8.7 Offline Installation and Update Guide

This guide covers the universal offline bundles for a fresh EVA AI installation
and for an in-place update of an existing EVA AI server. Keep the complete bundle
directory intact. Do not copy only `START_EVA_AI.sh`.

## 1. What must be ready before you start

### Target computer

- Root or `sudo` access.
- The bundle matching the target architecture and operating system:
  - **x64:** Ubuntu 26.04 LTS amd64 for a fresh installation. The same x64
    bundle can update supported EVA installations on Ubuntu 24.04 or 26.04
    whose existing EVA virtual environment uses CPython 3.12, 3.13, or 3.14.
  - **NVIDIA GB10 / Spark-class ARM64:** Ubuntu 24.04 LTS arm64.
- Working `python3` from the Ubuntu installation. Do not preinstall Python ML
  packages: the bundle carries the reviewed application and inference
  dependencies.
- An NVIDIA GPU visible to the operating system. The x64 local-VLM profile is
  intended for a modern NVIDIA GPU with at least 12 GB VRAM. The ARM64 bundle is
  built for NVIDIA GB10 / Spark-class hardware.
- At least approximately 48 GiB free for the x64 bundle target or 70 GiB for
  the ARM64 target. The installer calculates the exact requirement before it
  changes the host.
- Correct system time and access to the local network.

### Luxriot Evo

Luxriot Evo must already be running on the same reachable network. Have these
values ready:

- Evo IP address or URL, including a non-default port when applicable;
- Evo administrator username;
- Evo administrator password.

Example:

```text
Evo URL:      http://192.168.1.100:8080
Evo username: admin
Evo password: <the actual Evo administrator password>
```

A bare address such as `192.168.1.100` is accepted and means
`http://192.168.1.100:8080`. Using the complete URL is clearer. Do not put the
username or password inside the URL.

### Optional existing VLM inference

If you do not want the installer to deploy the bundled local vLLM, prepare an
OpenAI-compatible multimodal endpoint reachable from the EVA host. It must
provide `/v1/models` and multimodal chat completions and accept the exact model
identifier entered during installation.

Example:

```text
Base URL: http://192.168.1.110:8080/v1
Model:    Qwen/Qwen3-VL-4B-Instruct
```

The endpoint should accept up to eight images per request. The current
interactive fresh installer does not ask for an external VLM API key, so the
endpoint must be reachable without authentication during installation. If the
endpoint requires a key, stop and contact EVA engineering instead of editing
the generated environment file by hand.

Selecting an external VLM skips the bundled local Qwen/vLLM service. It does
**not** move semantic probes off the EVA host: SigLIP2 still uses the local
NVIDIA GPU.

## 2. What a fresh installation deploys

The default installation deploys and configures:

- EVA AI backend and React operator interface;
- PostgreSQL, the EVA database, roles, and schema migrations;
- local SigLIP2 semantic embedding for probes and archive search;
- systemd services, an nginx HTTPS endpoint, and a locally generated TLS
  certificate;
- Qwen3-VL-4B served by local vLLM when local VLM is selected;
- on x64 only, optional CPU Qwen3.5-9B-MTP served by llama.cpp for preemptible
  deep review;
- health/readiness checks, a VLM vision canary, installation journal, and a
  secret-free deployment report.

The x64 local profile uses Qwen3-VL-4B-Instruct AWQ. The GB10 ARM64 profile uses
the pinned NVIDIA container runtime and Qwen3-VL-4B online FP8.

## 3. Copy and start the bundle

Copy the complete architecture-specific directory from the USB drive or over
SSH. Open a terminal inside that directory and run:

```bash
sudo ./START_EVA_AI.sh
```

The launcher verifies the complete payload before making changes. On a USB
drive this can take several minutes; progress is printed as verified GiB and a
percentage. Entering the sudo password and then seeing verification progress is
normal.

The launcher prints one detected mode:

- `INSTALL` — no completed EVA installation was detected;
- `RESUME` — a previous fresh installation stopped during a recorded phase;
- `UPDATE` — an existing completed EVA installation was detected.

Do not force `INSTALL` over an existing deployment. If the detected mode is not
what you expected, stop and investigate before approving changes.

## 4. Fresh-install prompts and example answers

### Luxriot Evo connection

```text
Press Enter when Evo is connected and you know its credentials... <Enter>
Luxriot Evo IP address or URL: http://192.168.1.100:8080
Luxriot Evo username: admin
Luxriot Evo password: <hidden input>
```

The installer performs an authenticated read-only Evo check before crossing the
mutation boundary. A failed check stops the installation without accepting an
unverified connection.

### Filesystem layout

```text
Use this layout? [Y/n]: Y
```

The defaults are:

```text
application and inference: /opt/eva-ai
runtime data:              /var/lib/eva-ai
configuration:             /etc/eva-ai
```

Use custom paths only when the site has an intentional storage layout and the
paths are backed up and monitored.

### Inference placement: install the bundled local VLM

Choose this on a dedicated EVA GPU server when no reviewed inference endpoint
already exists:

```text
Install and run the VLM on this computer? [Y/n]: Y
```

The installer deploys Qwen3-VL-4B and vLLM from the offline payload. The first
model start can take several minutes. Messages such as `Waiting for Local VLM`
are normal while the endpoint is loading, provided the remaining time continues
to update and it eventually reports the model ready and the vision smoke test
passed.

### Inference placement: use an existing VLM

Choose this when an already tested OpenAI-compatible multimodal endpoint will
serve EVA:

```text
Install and run the VLM on this computer? [Y/n]: n
External OpenAI-compatible VLM URL: http://192.168.1.110:8080/v1
External VLM model id [qwen/qwen3-vl-4b]: Qwen/Qwen3-VL-4B-Instruct
```

The URL is the API base ending in `/v1`, not the server health URL. The model ID
must match a model returned by the endpoint's `/v1/models` response. The
installer proves model and vision readiness before declaring EVA ready.

### Optional x64 deep review

On x64:

```text
Install the CPU Qwen3.5-9B-MTP endpoint for preemptible L3 review? [Y/n]: Y
```

Select `n` to omit it. You may then enter an external deep-review endpoint or
leave that field empty to disable deep review. The compact ARM64 bundle does not
install the x64 CPU llama.cpp payload.

### Site time and quiet window

Use an IANA timezone name for the physical site:

```text
Site timezone [Europe/Riga]: Asia/Tbilisi
Configure a quiet window for 9B consolidation now? [y/N]: n
```

If enabled, enter valid 24-hour times such as `01:00` and `05:00`.

### First EVA administrator

This creates an EVA user; it is separate from the Evo administrator:

```text
Admin username [admin]: admin
Admin display name [EVA Administrator]: Site Administrator
Admin password: <hidden input, at least 12 characters>
Confirm admin password: <repeat it>
```

## 5. What to expect during installation

The exact phases differ by architecture, but the normal sequence includes:

1. offline payload verification;
2. host, disk, architecture, Evo, GPU, and database preflight;
3. explicit installation plan and approval;
4. offline operating-system packages and runtime preparation;
5. application, Python environments, PostgreSQL, and migrations;
6. local inference installation when selected;
7. configuration, systemd, nginx/TLS, and administrator creation;
8. VLM model readiness and a real image-understanding smoke test;
9. EVA `/health` and `/ready` checks;
10. a final deployment report and the HTTPS operator URL.

Large package extraction, CUDA runtime preparation, and first model loading are
not instant. Wait while periodic progress or `Waiting for ... (Ns remaining)`
messages continue. Do not interrupt a healthy phase merely because GPU usage is
temporarily zero.

After success, open:

```text
https://<EVA-server-IP>/
```

The initial TLS certificate is locally generated, so the browser will require
the site's normal trust/import procedure.

## 6. Updating an existing EVA installation

Use the bundle matching the host architecture:

```bash
cd /path/to/EVA-AI-0.8.7-OFFLINE-ARCH-COMMIT
sudo ./START_EVA_AI.sh --mode update
```

Some pilot Evo servers intentionally use a weak value such as `123` as their
real administrator password. The update preflight treats common weak values as
placeholders and stops by default. If, and only if, the saved value is the real
site credential, rerun with explicit live credential verification:

```bash
sudo ./START_EVA_AI.sh --mode update --verify-luxriot-credential
```

This flag does not waive the check. It permits the value only after a live,
authenticated, read-only Evo `/channels` request succeeds. A failed request
still blocks the update without changing the installation. Do not use the flag
to bypass an unknown or unconfirmed credential.

An update preserves the existing environment, Evo credentials, users, probes,
channel settings, inference profiles and routing, runtime state, and archive
database. It backs up PostgreSQL and application/configuration state before
migrations and code replacement.

The updater first prints `UPDATE PREFLIGHT (read-only)`. Review its final
summary. Enter `y` at this prompt only when there is no `FAIL`:

```text
Apply the reviewed update, database backup and migrations now? [y/N]: y
```

Warnings must be read, but not every warning is fatal. For example, no recent
stream baseline is expected when no channel was running before the update. A
database visibility, migration identity, insufficient disk, missing dependency,
or incompatible host error must not be waived.

If the updater asks for a migration DSN, use the approved privileged PostgreSQL
migration identity for that site. The value is not displayed or saved. Do not
substitute the ordinary restricted EVA runtime DSN and never paste either DSN
into chat.

For an installer-managed local PostgreSQL appliance, the updater does not need
a permanent superuser credential. Before approval it proves the local
PostgreSQL peer-auth path without lasting changes. After approval it creates a
random, process-only migration login with full-site RLS visibility, limits its
password lifetime to two hours, uses it for the preservation manifest, backup,
and migrations, and removes it before handoff. It also repairs older
installer-managed migrator and full-site backup roles so their declared
cross-tenant purpose is not silently filtered by RLS. Tenant-scoped runtime
roles are not changed. The terminal should print:

```text
Local PostgreSQL migration preflight: OK.
Temporary local migration identity created...
Temporary local migration identity removed.
```

If cleanup cannot remove the temporary identity, the updater reports a warning;
its random password still expires automatically. Send that warning to EVA
engineering rather than attempting manual role or database cleanup.

The successful terminal report should show:

```text
EVA service: ACTIVE
EVA readiness: READY
UI updated and running: YES (React)
Migrations successful: YES (<expected revision>)
Luxriot Evo: REACHABLE
```

Configured inference profiles should report `READY`. Previously active summary
streams should resume and produce a post-update record. The exact secret-free
report is saved at:

```text
/var/lib/eva-ai-installer/last-deployment-report.txt
/var/lib/eva-ai-installer/last-deployment-report.json
```

Do not run a manual rollback merely because a semantic model is still warming
up after the migrated application is healthy. Follow the status printed by the
updater; it distinguishes automatic rollback from a recoverable post-update
acceptance failure.

## 7. Short acceptance check

Run:

```bash
sudo systemctl is-active eva-ai
curl -fsS http://127.0.0.1:5000/health
curl -fsS http://127.0.0.1:5000/ready
curl -kI https://127.0.0.1/
sudo cat /var/lib/eva-ai-installer/last-deployment-report.txt
```

When a local VLM was selected:

```bash
sudo systemctl is-active eva-vllm
curl -fsS http://127.0.0.1:1234/v1/models
sudo systemctl is-active eva-vlm-vision-watchdog.timer
```

Then log in, configure or confirm one test channel, and verify:

- live preview updates;
- a video description reaches L0;
- a semantic probe produces a current pulse;
- existing users, probes, summaries, and archive search are present after an
  update.

## 8. When to ask an engineering agent for help

Stop and collect diagnostics when any of these occurs:

- bundle verification reports a checksum mismatch;
- the detected `INSTALL`, `RESUME`, or `UPDATE` mode is unexpected;
- update preflight ends with `FAIL`;
- the installer prints `INSTALLER ERROR` or `DEPLOYMENT ERROR`;
- the final report is `FAIL`;
- no progress or updated waiting message appears for more than ten minutes;
- `/health`, `/ready`, the VLM model endpoint, or the HTTPS UI remains
  unavailable after the installer has finished;
- an update completes but users, probes, channel configuration, or archive data
  is missing.

A final report of `WARN` is not automatically a failure. Read the warning. For
example, `no recent video-summary records` is expected before any channel has
been configured.

Do not run individual migration commands, edit `.env`, drop a database, delete
installer state, or invoke a rollback suggested from incomplete context. The
fresh installer is journaled and normally resumes safely when its main launcher
is run again after the diagnosed cause is fixed.

Collect the following secret-free evidence:

```bash
cd /path/to/the/same/offline-bundle
sudo ./START_EVA_AI.sh --mode report | tee ~/eva-deployment-report.txt
sudo eva-ai-doctor --output ~/eva-ai-doctor.json
sudo systemctl --failed --no-pager | tee ~/eva-failed-units.txt
sudo systemctl status eva-ai eva-vllm --no-pager -l | tee ~/eva-service-status.txt
sudo journalctl -u eva-ai -u eva-vllm -n 200 --no-pager | tee ~/eva-service-journal.txt
sudo cp /var/lib/eva-ai-installer/install-state.json ~/eva-install-state.json
sudo chown "$USER":"$USER" ~/eva-install-state.json
```

If `eva-ai-doctor` or report mode is not yet installed, note that fact and send
the other files plus the final terminal output. Never attach
`/etc/eva-ai/eva-ai.env`; it contains credentials.

Use a request such as:

```text
I am installing/updating EVA AI from the offline bundle
EVA-AI-0.8.7-OFFLINE-<X64|ARM64>-<commit>.

Host: <Ubuntu version, architecture, GPU>
Detected mode: <INSTALL|RESUME|UPDATE>
Last successful installer phase: <phase or unknown>
First error: <copy the first error line>
Current result: <PASS|WARN|FAIL|installer stopped>

I attached the secret-free deployment report, doctor report, failed units,
service status, journal, installer state, and the final terminal section.
Please identify the failed contract first. Do not reset the database, edit the
environment file, or roll back until the existing backup/mutation state and
data-preservation boundary are established.
```

Passwords, API keys, database DSNs, cookies, and the contents of
`/etc/eva-ai/eva-ai.env` must never be included in an agent request or support
ticket.
