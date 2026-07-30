# Port appliance installation recovery

The installer is safe to rerun after correcting a reported failure. Completed
phases are replayed and revalidated; the journal is diagnostic state, not a
reason to skip work after a power loss.

## First response

```bash
sudo eva-ai-doctor --output /tmp/eva-ai-doctor.json
sudo cat /var/lib/eva-ai-installer/install-state.json
sudo systemctl --failed --no-pager
```

The diagnostic JSON contains presence/status information but no Evo, database,
administrator or API secrets. Copy that file off the appliance for support.

If the package has not yet been installed, run the bundled copy directly:

```bash
sudo python3 ./repo/scripts/eva_appliance_doctor.py \
  --output /tmp/eva-ai-doctor.json
```

## Relevant logs

```bash
sudo journalctl -u eva-vllm -u eva-deep-review -u eva-ai \
  --since=-30min --no-pager
```

Do not paste `/etc/eva-ai/eva-ai.env` into chat or tickets. The doctor reports
which required keys exist without emitting their values.

## Recovery invariants

- The APT repository used by `_apt` must be below
  `/var/cache/eva-ai-offline-apt/repos`, never on removable media.
- `/etc/eva-ai/eva-ai.env` is written atomically only after migrations and
  least-privilege database logins succeed.
- Auth, archive and inference queue tenant UUIDs must be present and identical.
- Named-user auth and secure cookies are mandatory; the legacy admin token is
  removed during configuration rendering.
- The administrator is bootstrapped before EVA starts.
- Local and external inference endpoints must expose the configured model from
  `/v1/models`.
- Installation succeeds only when EVA returns `status=ready`, not merely when
  Gunicorn returns a liveness response.

If a driver install requests a reboot, reboot and rerun the same installer
command. Do not manually construct a partial environment or launch model
commands in an interactive shell; systemd and the final environment are the
deployment source of truth.
