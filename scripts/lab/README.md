# Disposable lab and rehearsal helpers

These scripts are intentionally host-specific. They are not field installers
and must not be shipped as generic client recovery commands.

- `georgia_upgrade_recover.sh` restores the pinned local Georgia 0.8.1
  rehearsal baseline.
- `georgia_upgrade_test.sh` runs the pinned local Georgia upgrade rehearsal.
- `reset_factory_x64.sh` retires only the disposable x64 factory appliance
  under `/mnt/eva-llamacpp-lab/factory-x64` after an explicit confirmation.

Review their pinned paths and artifact identities before every use.
