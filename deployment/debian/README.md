# EVA AI appliance packaging boundary

The Debian package is deliberately a small, stable bootstrap layer. It installs
`eva-ai-install` and `eva-ai-doctor`; its `postinst` creates only the root-owned
installer state directory. It does not prompt, migrate PostgreSQL, create an
administrator, start EVA, or download anything.

The versioned offline bundle remains the payload and contains:

- the exact EVA application source;
- the offline Python wheelhouse and constraints;
- the local APT repository;
- Qwen VLM/deep-review weights, CLIP weights and llama.cpp source;
- a manifest and checksums.

This split is intentional. Putting large model files or site secrets in the
bootstrap package would duplicate tens of gigabytes during `dpkg` upgrades and
make maintainer-script failures difficult to recover from.

Install and configure from removable media:

```bash
sudo dpkg -i installer-deb/eva-ai-appliance-installer_*_amd64.deb
sudo eva-ai-install --bundle-root "$PWD"
```

The direct `./install.sh` path remains equivalent for recovery. The installer
stages APT content on the local disk, writes a crash-safe phase journal at
`/var/lib/eva-ai-installer/install-state.json`, and can be rerun after a reboot
or corrected preflight failure.

For repeated site provisioning, use one-line root-readable secret files instead
of putting passwords in the process list:

```bash
sudo eva-ai-install --bundle-root /srv/eva-bundle --non-interactive \
  --evo-url http://evo.example --evo-username eva \
  --evo-password-file /run/secrets/evo-password \
  --admin-username admins \
  --admin-password-file /run/secrets/eva-admin-password \
  --quiet-window-start 01:00 --quiet-window-end 05:00
```

Future repository distribution can publish this same package through APT while
serving the content-addressed payload from an ISO, local mirror, or asset
package. Database migration and first-site configuration must remain explicit
`eva-ai-install`/upgrade operations, never implicit `postinst` side effects.
