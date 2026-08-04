# EVA AI universal offline deployment

This USB supports two paths from the same entry point:

- **Fresh install** on Ubuntu 24.04 LTS: PostgreSQL, EVA AI, React UI, local
  Qwen VLM inference, optional CPU deep review, SigLIP2, systemd and TLS.
- **In-place update** of an existing EVA AI server: the site configuration and
  external inference topology are preserved, PostgreSQL is backed up, Alembic
  migrations are applied through `20260801_0011`, the React UI is replaced,
  services are restarted, and previously active summary streams must resume.

## Start

Open a terminal in the USB bundle directory and run:

```bash
sudo ./START_EVA_AI.sh
```

The script detects whether EVA AI is already installed and prints `INSTALL` or
`UPDATE` before changing anything. To force an explicit path:

```bash
sudo ./START_EVA_AI.sh --mode install
sudo ./START_EVA_AI.sh --mode update
```

Every update runs a read-only preflight first and asks for final confirmation.
If the existing configuration does not contain a privileged migration DSN, the
script asks for one without displaying or saving it. The ordinary EVA runtime
database login is deliberately not accepted as a migration identity.

## Successful update handoff

The terminal ends with an English acceptance report containing:

- kernel version and detected NVIDIA GPU;
- EVA service/version and React UI status;
- `Migrations successful: YES (20260801_0011)`;
- Luxriot Evo reachability;
- every configured Agent/VLM inference profile;
- recent video-summary channel activity;
- confirmation that streams active before the update produced new records
  after the restart.

The same secret-free report is saved as:

```text
/var/lib/eva-ai-installer/last-deployment-report.txt
/var/lib/eva-ai-installer/last-deployment-report.json
```

The update backup and rollback command are printed by the installer and stored
under `/var/backups/eva-ai`. Do not delete a backup until operators have checked
the live streams and archive search.

## Georgia profile

The updater never replaces existing model URLs, API keys, profile IDs, channel
routing, Luxriot credentials, tenant IDs or retention settings. This is the
expected path for the Georgia deployment with external VLM servers and many
channels. The post-update stream check compares the site with its own live
pre-update baseline rather than assuming an eight-channel appliance.

## Fresh single-GPU profile

The fresh installer asks for the Evo URL and credentials, paths, local versus
external inference, quiet-window deep review, timezone and first EVA
administrator. The bundled local profile is intended for a modern NVIDIA GPU
with at least 12 GB VRAM; RTX 5070 Ti is supported by the included open NVIDIA
driver/CUDA/PyTorch payload. After installation, open the printed HTTPS URL and
configure the required channels in EVA.

## Diagnostics

```bash
sudo ./START_EVA_AI.sh --mode report
sudo eva-ai-doctor --output /tmp/eva-ai-doctor.json
sudo systemctl --failed --no-pager
```

Never paste `/etc/eva-ai/eva-ai.env` into chat or a support ticket. It contains
client secrets; both diagnostic reports are designed to omit them.
