# EVA AI offline update: 01 - prepare USB media and Linux basics

Goal: prepare the USB drive on Windows, find it on the Linux client machine,
mount it if needed, and run the basic terminal commands required for the patch.

Target release: `β 0.8.3`  
Schema head: `20260614_0006`  
Database migration for `β 0.8.2.1 -> β 0.8.3`: **none**

Do not copy or share `/etc/eva-ai/eva-ai.env`. It may contain passwords, DSNs,
and internal network addresses.

## 1. Files required on the USB drive

The USB drive must contain two files:

```text
eva-ai-patch-0.8.3-YYYYMMDD-*.tar.gz
eva-ai-patch-0.8.3-YYYYMMDD-*.tar.gz.sha256
```

The project engineer will provide the current bundle link:
`[FIELD: bundle link]`.

If the link points to a `.tar.gz`, there must also be a matching `.sha256`
file with the same name plus `.sha256`.

## 2. Prepare the USB drive on Windows

1. Insert the USB drive.
2. Copy the `.tar.gz` and `.tar.gz.sha256` files to the USB drive.
3. Do not unpack the archive on Windows. Copy the archive as-is.
4. Do not rename the files.
5. Use "Safely Remove Hardware" before unplugging the drive.

Optional SHA256 check in PowerShell:

```powershell
cd E:\
Get-FileHash .\eva-ai-patch-0.8.3-*.tar.gz -Algorithm SHA256
type .\eva-ai-patch-0.8.3-*.tar.gz.sha256
```

The hash from `Get-FileHash` must match the hash in the `.sha256` file. If it
does not match, do not use this copy.

Recommended USB filesystem: `exFAT` or `NTFS`. Linux permissions are preserved
inside the `tar.gz`; they do not depend on the USB filesystem.

## 3. Find the USB drive on Linux

Insert the USB drive and run:

```bash
lsblk -f
```

Look for a `part` device with filesystem `exfat`, `ntfs`, `vfat`, or `ext4`,
and with the expected USB size.

Example:

```text
sdb
└─sdb1 exfat EVA_USB  64G  /media/luxriot/EVA_USB
```

In this example:

- disk: `/dev/sdb`;
- USB partition: `/dev/sdb1`;
- mounted path: `/media/luxriot/EVA_USB`.

Mount the partition, for example `/dev/sdb1`, not the whole disk `/dev/sdb`.

## 4. If the USB drive was mounted automatically

Check the files:

```bash
ls -lh /media/$USER/*/
find /media/$USER -maxdepth 2 -name 'eva-ai-patch-0.8.3-*.tar.gz*' -ls
```

Copy the archive locally:

```bash
mkdir -p ~/eva-ai-patch
cp /media/$USER/EVA_USB/eva-ai-patch-0.8.3-*.tar.gz* ~/eva-ai-patch/
cd ~/eva-ai-patch
sha256sum -c eva-ai-patch-0.8.3-*.tar.gz.sha256
```

Expected result:

```text
eva-ai-patch-...tar.gz: OK
```

## 5. If the USB drive was not mounted automatically

Find the partition:

```bash
lsblk -f
```

Create a mount point:

```bash
sudo mkdir -p /mnt/eva-usb
```

Mount the partition. Replace `/dev/sdX1` with the real partition from
`lsblk -f`.

```bash
sudo mount /dev/sdX1 /mnt/eva-usb
ls -lh /mnt/eva-usb
```

Copy the archive locally:

```bash
mkdir -p ~/eva-ai-patch
cp /mnt/eva-usb/eva-ai-patch-0.8.3-*.tar.gz* ~/eva-ai-patch/
cd ~/eva-ai-patch
sha256sum -c eva-ai-patch-0.8.3-*.tar.gz.sha256
```

Unmount after the work:

```bash
cd ~
sudo umount /mnt/eva-usb
```

If `umount` reports `target is busy`, some terminal is still inside
`/mnt/eva-usb`. Run `cd ~` in all terminals and repeat.

## 6. Unpack the bundle

After SHA256 verification:

```bash
cd ~/eva-ai-patch
tar -xzf eva-ai-patch-0.8.3-*.tar.gz
cd eva-ai-patch-0.8.3-*
cat manifest.txt
```

The manifest should show:

```text
version=β 0.8.3
```

If the manifest says `wheelhouse=included`, the bundle includes offline Python
packages. If it says `wheelhouse=not_included`, installation may still work by
reusing the existing `.venv`, but that is a risk for a fully offline site.

## 7. Terminal cheat sheet

### Navigation

```bash
pwd                 # current directory
ls                  # list files
ls -lh              # list files with sizes
ls -la              # include hidden files
cd /path/to/dir     # change directory
cd ~                # go home
cd -                # return to previous directory
```

### Copying and reading

```bash
cp source target        # copy file
cp -a source target     # copy preserving attributes
mkdir -p dir            # create directory
cat file                # print file
sed -n '1,80p' file     # first 80 lines
less file               # scroll file, quit with q
```

### Archives and checksums

```bash
tar -tzf file.tar.gz | head     # list archive contents
tar -xzf file.tar.gz            # unpack
sha256sum -c file.tar.gz.sha256 # verify checksum
```

### Disks and space

```bash
lsblk -f            # disks, partitions, mount points
df -h               # free space
du -sh path         # size of file/directory
```

### EVA AI service

```bash
systemctl status eva-ai --no-pager -l
sudo systemctl stop eva-ai
sudo systemctl start eva-ai
sudo systemctl restart eva-ai
journalctl -u eva-ai -n 120 --no-pager
```

### HTTP checks

```bash
curl -sS http://127.0.0.1:5000/health
curl -sS http://127.0.0.1:5000/ready | jq
```

For local HTTPS with a self-signed certificate:

```bash
curl -k -sS https://127.0.0.1:5443/health
```

### Useful keys

```text
Tab       autocomplete command/path
Up/Down   command history
Ctrl+C    stop current command
Ctrl+L    clear screen
q         quit less/journalctl
```

## 8. Next step

After copying and unpacking the bundle, continue with:

```text
readiness/OFFLINE_USB_02_PREFLIGHT_DECISION_EN.md
```

