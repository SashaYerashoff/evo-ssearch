# EVA AI Security Roadmap to 1.0

Last updated: 2026-08-24

Status: living engineering roadmap. It is expected to change as pilots expose
new risks. An item in this document is planned work, not evidence that a security
control already exists and not a contractual commitment to a customer.

## Scope boundary

This roadmap covers EVA AI, a separate product/integration developed by its
dedicated team within Riot Engineering Ltd.: application code, deployment
bundles, migrations, configuration, and engineering release gates. It does not
cover the internal implementation or operational security of the separately
developed Luxriot Evo VMS/NVR. EVA AI uses the same partner-facing HTTP API made
available to other integration partners and has no privileged internal coupling.
Customer networks, host operations, PKI, backups, and endpoint security remain
deployment responsibilities unless explicitly brought into EVA AI scope.

## Status vocabulary

- **Accepted** — engineering agrees that the work belongs in the 1.0 release
  path, but implementation may not have started.
- **In progress** — implementation or validation is active.
- **Release gate** — 1.0 must not be declared ready without the stated evidence.
- **Later** — useful work that is not currently a 1.0 gate.

## Security and release track

| ID | Work item | Current 0.8.7 state | Target / evidence of completion | Status |
| -- | --------- | ------------------- | ------------------------------- | ------ |
| SEC-001 | Authenticated vulnerability assessment and penetration test | No authenticated scan or penetration test has been performed. The 0.8.7 deployment is a trusted pilot, not a 1.0 security release. | Run an authenticated assessment against a representative deployed appliance before 1.0; retain scope, tool/version, date, findings, remediation decisions, and a clean verification pass. Arrange an independent penetration test or explicitly record the accepted test scope and provider. | Release gate |
| SEC-002 | Automated security scanning | CI has functional tests and build/integrity checks, but no dependency, SAST, secret, container, or authenticated DAST gate. | Add reproducible dependency, static, secret, and shipped-container scans; define severity handling, false-positive recording, and the release-blocking threshold. Keep authenticated deployed-application testing as a separate control. | Accepted |
| SEC-003 | Audit retention and export | `audit.events` is append-only at the application-role boundary and currently remains until an authorized database operation removes it. There is no explicit audit retention policy. | Choose a documented, customer-configurable retention policy; define the safe default, legal hold/export behavior, backup interaction, pruning authority, audit trail, and migration/restore tests. Do not silently introduce deletion into an update. | Accepted |
| SEC-004 | Encryption for off-host service links | Browser traffic is HTTPS in the appliance, but EVA currently permits HTTP Evo/VLM endpoints and does not require PostgreSQL TLS. Luxriot Evo currently exposes the ordinary partner API used by EVA AI; its server-side transport implementation is owned by the Evo team. Trusted-LAN HTTP is used in pilots as a temporary operational exception. | Inventory every link and credential; support and validate HTTPS/PostgreSQL TLS on EVA-owned links; make insecure non-loopback links explicit and visible rather than accidental. Prefer and validate HTTPS for the Evo partner API when the Evo product exposes it. Define whether 1.0 rejects remaining insecure links or requires a deliberate site exception with documented compensating controls. Preserve offline/on-prem operation. | Accepted |
| SEC-005 | Local-password defense in depth | Supported create/reset/install paths require 12 characters; Argon2id is used; composition, history, and expiry rules are absent; low-level `bootstrap_admin()` only rejects an empty password. | Enforce the same minimum at the repository boundary and all supported callers. Keep passphrases and arbitrary character classes supported; do not add forced composition or periodic expiry without a demonstrated requirement. Decide and test a generous input-size safety bound. Treat external IdP support as a separate roadmap capability. | Accepted |
| SEC-006 | Security findings and customer testing process | The repository contains no approved vulnerability-disclosure promise or customer penetration-testing rules of engagement. | Organizational Security/Legal defines authorization, disclosure, NDA, safe-harbor, scope, stop conditions, incident contact, and remediation communication. EVA AI Engineering supplies technical scope and fixes but does not unilaterally make the contractual commitment. | Accepted; organizational dependency |

## Decision log

### 2026-08-24

- Confirmed that no authenticated vulnerability scan or penetration test has yet
  been performed; these belong to the path from trusted pilot 0.8.7 to 1.0.
- Confirmed that audit retention has no separate engineering policy today; the
  current implementation remains the source of truth until a safe retention
  design is accepted.
- Confirmed that unencrypted EVA-owned off-host links are not the desired final
  security posture. The Luxriot Evo integration uses the ordinary partner HTTP
  API supplied by the Evo product; EVA AI should prefer HTTPS when that API
  exposes it but does not own its server-side implementation. Trusted-LAN HTTP
  remains a practical pilot configuration requiring network protection.
- Accepted a local-password direction of a 12-character minimum, Argon2id, and no
  forced composition/history/periodic-expiry policy; repository-level
  defense-in-depth remains planned.

## Maintenance rule

Update this roadmap when a risk is accepted, a release gate changes, or evidence
is produced. Completed work must link to implementation/tests or an approved
assessment artifact. Never convert a planned row into a HECVAT `Yes` answer until
the control is actually implemented and verified.
