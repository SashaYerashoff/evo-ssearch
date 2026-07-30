# EVA AI β 0.8.4-r4 offline field update

Use this guide for an existing EVA AI installation whose database is already at
schema `20260614_0006`. The update is code-only: it does not migrate or replace
the database and it preserves runtime data, models, configuration, and the
existing Python environment.

## Files on the USB drive

Copy both files. Do not rename either one:

- `eva-ai-0.8.4-r4-offline.tar.gz`
- `eva-ai-0.8.4-r4-offline.tar.gz.sha256`

Do not reuse an older archive named only `eva-ai-0.8.4-offline.tar.gz`.

## Before the update

The existing service must be working before it is upgraded. On a standard
system installation:

```bash
sudo systemctl is-active eva-ai
curl -sS http://127.0.0.1:5000/ready | jq -c '{status,version,required}'
```

Continue only when the service is `active` and `status` is `ready`. Start the
configured agent LM/LM Studio model before running the updater. If it is down,
the updater cannot verify agent availability or context.

## Copy, verify, and start

Open a terminal as the normal site user, not as root. Replace `/media/USB` with
the actual USB mount path:

```bash
mkdir -p ~/eva-ai-update-r4
cp /media/USB/eva-ai-0.8.4-r4-offline.tar.gz ~/eva-ai-update-r4/
cp /media/USB/eva-ai-0.8.4-r4-offline.tar.gz.sha256 ~/eva-ai-update-r4/
cd ~/eva-ai-update-r4
sha256sum -c eva-ai-0.8.4-r4-offline.tar.gz.sha256
tar -xzf eva-ai-0.8.4-r4-offline.tar.gz
cd eva-ai-0.8.4-r4-offline
./update.sh
```

Run `./update.sh` without `sudo`. It detects user or system systemd and asks for
the sudo password only when the system service requires it.

## Expected preflight

For a standard system service, expect lines equivalent to:

```text
Mode: system systemd
Config source: systemd EnvironmentFiles
OK: selected config matches the active runtime endpoints
OK: dependencies are unchanged and existing .venv is healthy
OK: database schema is already 20260614_0006
```

Any `STOP:` before `Type UPDATE` means nothing was installed and the running
service was not stopped. Save the complete terminal output and contact the
responsible developer. Do not edit configuration, copy a DSN, or improvise a
different install command.

If the updater requests `FORCE-CONTEXT`, record the reported context and call
the responsible developer before accepting a reduced value. If it requests
`FORCE-UNKNOWN-CONTEXT`, prefer stopping and fixing/starting LM Studio. Never
accept an unknown context just to get past preflight.

When all checks pass, type exactly:

```text
UPDATE
```

For `Restart eva-ai.service now? [Y/n]:`, press Enter. Do not close the terminal
or press Ctrl+C after confirming the update.

## Success

The final output must include:

```text
OK: EVA AI β 0.8.4 is up and running
```

It also prints the service, URL, agent-context state, and backup directory. If
post-start verification fails, the updater automatically restores the previous
code and configuration. Keep the full output and do not rerun immediately.

## Minimum post-update smoke test

1. Refresh the EVA AI page and log in.
2. Confirm `/ready` reports `status=ready` and `version=β 0.8.4`.
3. Open Video and confirm enabled channels restore within about two minutes.
4. Confirm live previews update; a disconnected channel must show signal loss,
   not a stale frame.
5. Open a stored alert frame and an archive video segment.
6. Ask the agent for runtime status. The answer must finish and use the actual
   channel/model state.
7. Ask the agent to review one named channel for the last hour. It must provide
   evidence, state coverage, and finish with a useful conclusion.

Commands for the final runtime check:

```bash
sudo systemctl is-active eva-ai
curl -sS http://127.0.0.1:5000/ready | jq -c '{status,version,required}'
```

For diagnostics requested by the developer:

```bash
bash repo/scripts/client_diagnostics.sh > diag.txt
```

Do not send `/etc/eva-ai/eva-ai.env`; it contains credentials and database
connection information.
