# EVA AI Bug Report and AI-Assisted Engineering Handoff

## Message to the engineering team

Please treat every EVA AI installation or runtime failure as an engineering
incident. A verbal summary such as “it does not work”, a cropped screenshot, or
a paraphrased error is not actionable and must not be the only report.

When a problem occurs:

1. Preserve the first failure and its complete terminal output.
2. Give the error, this template, and the local system context to your AI coding
   assistant.
3. Let the assistant diagnose the evidence. It may apply a narrow, reversible
   fix only when the cause is supported by evidence and the operator approves
   any destructive or security-sensitive action.
4. If the issue is fixed, return the completed report including exactly what
   changed and how the result was verified.
5. If the issue is not fixed, return a complete engineering handoff using the
   same form. Do not replace it with an oral retelling.

Never include passwords, API keys, session cookies, access tokens, or database
credentials. Do not reset EVA, drop a database, delete its data, roll back, or
repeat the installer blindly merely to make the original evidence disappear.

## Prompt for the local AI coding assistant

Copy the following prompt to the assistant together with the completed facts
and the full terminal trace:

> Diagnose this EVA AI installation, update, or runtime failure as an
> evidence-driven engineering incident. Inspect the local system and the
> extracted offline bundle before proposing a change. Do not guess, hide the
> first error, silently change inference models, reset EVA, delete data, drop
> databases, roll back, reinstall, or expose secrets. Prefer read-only checks
> first. If a narrow and reversible fix is justified, explain the evidence,
> record every command and changed file, preserve a rollback path, apply it only
> within the operator's authorization, and verify the actual end-to-end result.
> Then complete the report below. If you cannot safely finish, stop and produce
> a complete handoff for the EVA AI engineering team, including the strongest
> supported cause, eliminated hypotheses, remaining blocker, and exact next
> check. Do not return only a conversational summary.

## Required report

### 1. Incident identity

- Report title:
- Reporter and organization:
- Site/customer:
- Date, local time, and timezone:
- Hostname:
- Operational impact:
- Is this a fresh install, resume, update, rollback, or running-system issue?

### 2. Release identity

- Bundle filename:
- Bundle source commit (from `SOURCE_REVISION.json`):
- Bundle SHA-256 result:
- Previous EVA version, if updating:
- Expected EVA version:
- Installation directory:
- Configuration file path:
- Systemd service names:

### 3. Host and topology

- OS and release:
- Architecture (`x86_64` or `aarch64`):
- Kernel:
- CPU and RAM:
- GPU(s), VRAM, NVIDIA driver, and CUDA version:
- Free disk space on the application, data, backup, and bundle filesystems:
- Luxriot Evo address and reachability, with credentials redacted:
- Local or external VLM topology, models, addresses, and ports:
- Any proxy, firewall, container, or unusual filesystem layout:

### 4. Expected and actual behavior

- What should have happened?
- What happened instead?
- First visible failure time:
- Is the issue reproducible? If yes, list the exact minimal steps:
- What still works?
- What is unavailable or degraded?

### 5. Exact evidence

- Exact command that failed:
- Exit code:
- First error, copied verbatim:
- Complete terminal trace attached: yes/no
- EVA deployment report `.txt` attached: yes/no
- EVA deployment report `.json` attached: yes/no
- Relevant service status and journal attached: yes/no
- Screenshots or screen recording attached, if this is a UI issue: yes/no
- Relevant timestamps, channel IDs, batch IDs, probe IDs, or request IDs:

Do not crop away commands, timestamps, warnings immediately before the error,
or the final exit status. Redact only secrets.

### 6. Read-only diagnosis

- Strongest supported cause:
- Evidence supporting it:
- Alternative hypotheses checked:
- Hypotheses eliminated and why:
- Data, database, configuration, or service state at risk:
- Is the existing installation still recoverable without reset? Why?

### 7. Work performed by the local engineer or AI assistant

For every attempted change, record:

- Hypothesis being tested:
- Exact command or action:
- Files changed, with diff or before/after values:
- Packages, images, models, roles, migrations, or services affected:
- Backup or rollback path created first:
- Result and new evidence:

If nothing was changed, write `No mutations performed`.

### 8. Verification

- `eva-ai.service` active: yes/no
- EVA `/health` result:
- EVA `/ready` result:
- Database schema revision:
- UI reachable and login verified: yes/no
- Luxriot Evo reachable from EVA: yes/no
- VLM health and one real vision result: pass/fail/not tested
- SigLIP2 backend, CUDA device, dtype, and real forward: pass/fail/not tested
- Live stream preview: pass/fail/not tested
- One live semantic probe with changing P/N/M pulse: pass/fail/not tested
- Existing users, probes, settings, summaries, alerts, and archive preserved:
- Full deployment report result: PASS/WARN/FAIL
- Remaining warnings or regressions:

### 9. Final disposition

Choose exactly one:

- **FIXED AND VERIFIED** — include the permanent fix, evidence, and any patch or
  commit that must be merged into future bundles.
- **WORKAROUND ONLY** — include limitations, rollback, and the required
  permanent change.
- **NOT FIXED — ENGINEERING HANDOFF** — include the precise blocker, best next
  diagnostic step, files/logs attached, and the decision or access required
  from the EVA AI engineering team.

### 10. Attachments checklist

- [ ] Complete original terminal trace
- [ ] Deployment report text and JSON
- [ ] Relevant journal excerpt with timestamps
- [ ] Bundle `SOURCE_REVISION.json` and checksum result
- [ ] Redacted configuration facts relevant to the failure
- [ ] Hardware and GPU report
- [ ] Screenshots or recording when applicable
- [ ] Patch/diff and exact commands when anything was changed
- [ ] Verification evidence after the change

## Useful evidence commands

Run only the commands relevant to the incident. These are diagnostic; they do
not reset EVA or modify its database.

```bash
cd /path/to/extracted/EVA-AI-bundle
sudo ./START_EVA_AI.sh --mode report

sudo systemctl status \
  eva-ai.service eva-vllm.service eva-deep-review.service \
  --no-pager --full

sudo journalctl \
  -u eva-ai.service -u eva-vllm.service -u eva-deep-review.service \
  --since "30 minutes ago" --no-pager

curl -fsS http://127.0.0.1:5000/health
curl -fsS http://127.0.0.1:5000/ready

uname -a
cat /etc/os-release
nvidia-smi
df -h
```

Before sharing the output, remove secrets but keep commands, paths, service
names, timestamps, versions, IP addresses/ports needed to understand the site
topology, and the complete error chain.
