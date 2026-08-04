#!/usr/bin/env bash
set -Eeuo pipefail

# Final, destructive network handoff for the measured Ventspils appliance.
# Removes the temporary office connection, saved Wi-Fi and developer SSH key,
# then restores the preserved client-side static Ethernet profile.

TARGET_PROFILE="${EVA_HANDOFF_NETWORK_PROFILE:-netplan-eno1}"
TARGET_ADDRESS="${EVA_HANDOFF_NETWORK_ADDRESS:-192.168.210.80/24}"
OFFICE_PROFILE="${EVA_HANDOFF_OFFICE_PROFILE:-eva-office}"
WIFI_PROFILE="${EVA_HANDOFF_WIFI_PROFILE:-krn-wifi-pub}"
SERVICE_USER="${EVA_HANDOFF_SERVICE_USER:-admins}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ ${EUID} -eq 0 ]] || die "run with sudo from the appliance console"
command -v nmcli >/dev/null || die "nmcli is not installed"
nmcli connection show "${TARGET_PROFILE}" >/dev/null 2>&1 \
  || die "preserved target profile ${TARGET_PROFILE} does not exist"

actual_address="$(nmcli -g ipv4.addresses connection show "${TARGET_PROFILE}" | head -n 1)"
[[ "${actual_address}" == "${TARGET_ADDRESS}" ]] \
  || die "target profile address is ${actual_address:-unset}, expected ${TARGET_ADDRESS}"

office_uuid="$(nmcli -g connection.uuid connection show "${OFFICE_PROFILE}" 2>/dev/null || true)"
wifi_uuid="$(nmcli -g connection.uuid connection show "${WIFI_PROFILE}" 2>/dev/null || true)"

printf 'Preparing preserved profile %s (%s)...\n' "${TARGET_PROFILE}" "${TARGET_ADDRESS}"
nmcli connection modify "${TARGET_PROFILE}" \
  connection.autoconnect yes connection.autoconnect-priority 200
if [[ -n "${office_uuid}" ]]; then
  nmcli connection modify "${OFFICE_PROFILE}" connection.autoconnect no
fi

if [[ -n "${wifi_uuid}" ]]; then
  nmcli connection delete "${WIFI_PROFILE}"
fi
nmcli radio wifi off || true

printf 'Removing developer SSH authorization and transient client state...\n'
rm -f -- \
  "/home/${SERVICE_USER}/.ssh/authorized_keys" \
  "/home/${SERVICE_USER}/.ssh/known_hosts" \
  "/home/${SERVICE_USER}/.ssh/known_hosts.old" \
  "/home/${SERVICE_USER}/.bash_history"
find /var/lib/NetworkManager -maxdepth 1 -type f -name '*.lease' -delete 2>/dev/null || true
resolvectl flush-caches || true

printf 'Switching to the client network. Remote office SSH will disconnect now.\n'
nmcli --wait 25 connection up "${TARGET_PROFILE}"

if [[ -n "${office_uuid}" ]]; then
  nmcli connection delete "${OFFICE_PROFILE}"
fi

# NetworkManager's netplan backend normally removes these on profile deletion.
# Delete only files carrying the exact discarded UUIDs if stale copies remain.
for stale_uuid in "${office_uuid}" "${wifi_uuid}"; do
  [[ -n "${stale_uuid}" ]] || continue
  find /etc/netplan -maxdepth 1 -type f -name "*${stale_uuid}*" -delete
done
nmcli connection reload
netplan generate
ip neigh flush all 2>/dev/null || true

active_profile="$(nmcli -g GENERAL.CONNECTION device show eno1)"
active_address="$(ip -4 -o address show dev eno1 | awk '{print $4}' | head -n 1)"
[[ "${active_profile}" == "${TARGET_PROFILE}" ]] \
  || die "eno1 activated ${active_profile:-nothing}, expected ${TARGET_PROFILE}"
[[ "${active_address}" == "${TARGET_ADDRESS}" ]] \
  || die "eno1 address is ${active_address:-unset}, expected ${TARGET_ADDRESS}"

rm -f -- \
  "/home/${SERVICE_USER}/Desktop/FINALIZE VENTSPILS NETWORK.txt" \
  "/home/${SERVICE_USER}/eva-finalize-handoff-network.sh"

printf '\nNetwork handoff complete.\n'
printf 'Active profile: %s\n' "${active_profile}"
printf 'Address: %s\n' "${active_address}"
printf 'Office profile: removed\n'
printf 'Saved Wi-Fi: removed; radio off\n'
printf 'Developer SSH key: removed\n'
