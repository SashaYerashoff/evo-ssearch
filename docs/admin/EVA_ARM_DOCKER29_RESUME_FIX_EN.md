# EVA AI ARM64 Docker 29 resume hotfix

This hotfix is only for the interrupted ASUS GX10 installation made from:

```text
EVA-AI-0.8.7-OFFLINE-ARM64-9063d2f
```

Use it only when the installer journal reports:

```text
status: failed
failed phase: spark_container_runtime
```

The hotfix supports both Docker's classic image store and the Docker 29
containerd image store. It does not change Docker configuration, replace the
NVIDIA driver, edit the original bundle, reset PostgreSQL, or delete data.

Copy both supplied files to the ASUS host:

```text
EVA_ARM_DOCKER29_RESUME_FIX_9063d2f.sh
EVA_ARM_DOCKER29_RESUME_FIX_9063d2f.sh.sha256
```

Verify and start it:

```bash
cd ~/Downloads
sha256sum -c EVA_ARM_DOCKER29_RESUME_FIX_9063d2f.sh.sha256

sudo ./EVA_ARM_DOCKER29_RESUME_FIX_9063d2f.sh \
  ~/Downloads/EVA-AI-0.8.7-OFFLINE-ARM64-9063d2f
```

The script refuses to run unless all of these facts match the incident:

- the bundle source commit is exactly `9063d2f` and was built from a clean tree;
- the original affected installer has its expected SHA-256;
- the installer journal belongs to the same bundle and stopped at
  `spark_container_runtime`;
- the loaded image tag is ARM64 and resolves to either the pinned OCI config
  digest or its pinned OCI manifest digest.

The installation questions are asked again. Enter the same Evo address,
credentials, filesystem layout, inference selection, timezone, and EVA
administrator values used for the interrupted attempt.

After `INSTALLATION COMPLETE`, generate the normal report:

```bash
cd ~/Downloads/EVA-AI-0.8.7-OFFLINE-ARM64-9063d2f
sudo ./START_EVA_AI.sh --mode report | tee ~/eva-deployment-report.txt
```

If the hotfix stops, do not reset the host or edit Docker. Send EVA engineering:

```bash
sudo cp /var/lib/eva-ai-installer/install-state.json ~/eva-install-state.json
sudo chown "$USER":"$USER" ~/eva-install-state.json
journalctl -u eva-ai -u eva-vllm --no-pager -n 300 \
  > ~/eva-services-journal.txt
```

Include the new terminal output, `eva-install-state.json`, and
`eva-services-journal.txt` in the incident handoff.
