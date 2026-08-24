# EVA AI HECVAT Repository Audit

Audit date: 2026-08-24

Repository snapshot: `e4f8b5e9964edaadcaf79d3909a524a7ff62064f` (`main`)

Product/deployment scope: EVA AI — a separate product/integration developed by a
dedicated team within Riot Engineering Ltd., primarily customer-hosted on-premises

## HECVAT quick-entry matrix

Use the **Answer** column for the workbook and the **Paste-ready explanation** as
the short supplementary comment. `Requires organizational confirmation` must be
completed by an authorized owner of the responding/contracting organization; do
not silently convert it to `Yes` or `No`. Detailed evidence and caveats follow
the table.

| # | Control | Answer | Paste-ready explanation | Answer owner |
| - | ------- | ------ | ----------------------- | ------------ |
| 25 | AAAI-04 | Yes | EVA-local passwords require at least 12 characters and use Argon2id. No maximum, composition, history, periodic expiry, administrator-configurable password policy, or external IdP is currently implemented. Login throttling applies after repeated failures. | EVA AI Engineering |
| 27 | AAAI-06 | N/A | EVA is an on-premises product using local named users and does not implement InCommon, eduGAIN, SAML, OIDC, OAuth IdP, or LDAP federation. Any organizational membership answer must be confirmed separately. | EVA AI Engineering / responding organization |
| 30 | AAAI-09 | Yes | EVA records successful and failed login, throttling, logout, authorization decisions, protected actions, administration, and agent tool actions with identity, IP, timestamp, tenant, result, and request context in an append-only, hash-chained PostgreSQL audit log. No audit-retention policy or explicit authentication-mechanism field is currently defined. | EVA AI Engineering / Customer DBA |
| 36 | DATA-02 | No | The appliance protects browser traffic with HTTPS, but EVA does not require encryption for every off-host inference, PostgreSQL, or partner API link. Luxriot Evo's partner API transport is owned by the Evo team; EVA should select HTTPS if that API exposes it. | EVA AI Engineering / Customer infrastructure / Luxriot Evo team |
| 39 | DATA-05 | Requires organizational confirmation | EVA data remains primarily in customer-controlled PostgreSQL and local filesystems; continuous video remains in the separately supplied Luxriot Evo system. Contract-termination access, assistance, complete export, retention, and deletion commitments require contractual confirmation. | Legal/Privacy of responding organization / Customer / Luxriot Evo owner for Evo data |
| 42 | DATA-08 | N/A | EVA does not define or control physical archival media. Physical protection of customer servers, disks, removable media, and backups is a customer infrastructure control; Luxriot Evo recording media is outside EVA AI's scope. | Customer infrastructure / Luxriot Evo owner |
| 46 | FIDP-02 | Requires organizational confirmation | EVA documents service ports and uses secure local bindings but does not apply firewall rules or implement a formal firewall change-management process. | Customer IT/Security / responding organization |
| 47 | FIDP-03 | No | EVA does not bundle or integrate a network IDS/IPS or SIEM/syslog-forwarding control. Application audit and health logging are not NIDS. | Customer IT/Security |
| 48 | FIDP-04 | No | EVA does not install or integrate a HIDS/EDR product. Watchdogs, readiness checks, and bundle integrity checks are not host intrusion detection. | Customer IT/Security |
| 50 | PPPR-01 | Yes | EVA has a controlled offline install/update process with checksums, preflight, database backup, transactional migrations, preservation checks, rollback, health/readiness verification, and release notes. Corporate patch cadence, vulnerability SLA, and notification policy require separate confirmation. | EVA AI Engineering / organizational Product, Security, and Operations |
| 51 | PPPR-02 | Requires organizational confirmation | EVA provides RBAC, channel scoping, PostgreSQL RLS, audit, configurable retention, local data residency, and approval gates, but compliance with a customer's specific policy requires contractual review and control mapping. | Legal/Privacy/Security of responding organization / Customer governance |
| 52 | PPPR-03 | Requires organizational confirmation | Source code cannot establish the contracting entity, governing law, jurisdiction, or deployment-specific legal obligations. | Legal/Management of responding organization / Customer Legal |
| 53 | VULN-01 | No | No authenticated vulnerability scan or penetration test has yet been performed, and committed CI has no authenticated DAST or automated dependency/SAST/container/secret-scanning gate. These are planned release gates toward 1.0, not current 0.8.7 controls. | EVA AI Engineering / organizational IT Security |
| 54 | VULN-02 | Requires organizational confirmation | No repository-backed policy promises vulnerability or penetration-test reports to customers. Availability, NDA, redaction, and disclosure channel require authorization. | IT Security / Legal of responding organization |
| 55 | VULN-03 | Requires organizational confirmation | The repository neither authorizes nor prohibits customer scanning or penetration testing. Written authorization and rules of engagement are required. | IT Security / Legal of responding organization / Customer Security |
| 56 | HIPA-01 | N/A | The scoped on-premises public-space video product does not claim to process PHI and makes no HIPAA/HITECH claim. Workforce training and any future PHI applicability require organizational confirmation. | Legal/Privacy/HR of responding organization |

## Scope and interpretation

This report answers only from implementation, configuration, migrations,
deployment assets, tests, and procedures committed in the repository snapshot
above. It does not establish Riot Engineering Ltd or a distributor/contracting
entity's corporate policy, contractual commitments, staff training,
certifications, or controls operated on a customer's network.
Explicit decisions recorded under "Items requiring human confirmation" are
identified as EVA AI Engineering confirmations supplied during this audit; they
are not misrepresented as repository evidence or as already implemented controls.

The proposed answers are intended for the supplementary product detail behind a
HECVAT response. Where the main HECVAT workbook marks a control N/A because EVA
is not a cloud service, that deployment-model answer remains valid. The findings
below explain what the on-premises product itself does.

### Responsibility vocabulary

- **EVA AI / EVA AI Engineering** means the separate product/integration and its
  dedicated engineering team within Riot Engineering Ltd. Product behavior,
  repository code, migrations, installers, configuration validation, and
  technical documentation are in this audit's engineering scope.
- **Luxriot Evo / Luxriot Evo team** means the separately developed VMS/NVR and
  its owning team. EVA AI integrates through the same partner-facing HTTP API
  made available to other integration partners; it has no privileged internal
  API or ownership of that server implementation. EVA AI consumes configured API
  and media surfaces, but does not own Evo's authentication implementation,
  continuous-recording store, retention, availability, export, deletion,
  physical media, or security controls. Those
  properties are outside this repository and must not be attributed to EVA AI.
- **Riot Engineering Ltd organizational functions** means Legal, Privacy,
  Management, corporate IT/Security, Product, Services, and Support commitments
  that cannot be established from product code. If a distributor, reseller, or
  other contracting entity submits the questionnaire, its own authorized owners
  must also confirm the claims made in its name.
- **Customer infrastructure** means the customer's hosts, network, firewalls,
  PKI, database operations, backups, endpoint protection, and configured external
  inference systems. These are deployment controls, not EVA AI software claims.

Whenever this report mentions data remaining in Luxriot Evo, that is a statement
about the integration boundary, not a claim that EVA AI Engineering implements
or guarantees Evo behavior.

Negative findings were checked with repository-wide searches for the named
standards and products (including federation protocols, IDS/HIDS products,
security scanners, HIPAA/HITECH, penetration-testing policy, and vulnerability
report disclosure). Absence from this repository does not prove absence from
Riot Engineering Ltd or a responding/contracting organization's company-wide
processes; it means the repository cannot be cited as evidence for them.

Assessment terms:

- **Yes** — the repository establishes the material product control, subject to
  any stated boundary or caveat.
- **No** — the product does not implement the full control as framed. A customer
  infrastructure control may still satisfy it in a deployment.
- **N/A** — the control does not apply to the on-premises product as described.
- **Requires organizational confirmation** — code cannot establish the requested
  legal, personnel, or operational commitment.

## #25 — AAAI-04 — Password restrictions / limitations

**Assessment:** Yes

**Confidence:** High

**Proposed HECVAT response:**

> EVA supports locally managed named-user passwords. Supported provisioning and
> reset paths require at least 12 characters; no explicit maximum length,
> uppercase, lowercase, numeric, special-character, password-history, reuse, or
> password-expiration rule is implemented. Ordinary user create/reset rejects a
> NUL character but otherwise has no explicit character-class restriction.
> Passwords are stored using Argon2id. Login attempts are throttled per tenant,
> normalized username, and source IP: five failures in a five-minute window cause
> a 15-minute lockout. These policy values are not configurable in the current
> administrator UI or environment configuration. EVA does not currently delegate
> authentication to an external identity provider.

**Evidence:**

- `security/postgres_identity.py:273-355,357-452,1165-1167` —
  `PostgresIdentityRepository.create_user()`, `update_user()`, and
  `_require_password_strength()` enforce 12 or more characters and reject NUL for
  ordinary account creation and password reset.
- `scripts/bootstrap_admin.py:34-62` and
  `scripts/install_port_appliance.py:1398-1408` — the supported first-admin CLI
  and appliance installer also require 12 or more characters and confirmation.
- `security/passwords.py:22-79` — `Argon2idPasswordHasher` uses Argon2id with
  time cost 3, memory cost 65,536 KiB, parallelism 4, 32-byte hash, and 16-byte
  salt; empty passwords are rejected.
- `security/throttling.py:11-23,72-137` — the default throttle policy is five
  attempts in 300 seconds and a 900-second lockout; success clears the throttle.
- `security/http_auth.py:120-165,215-216` — login applies the throttle before
  password verification and keys it by tenant, lowercase username, and client IP.
- `security/postgres_throttling.py:16-99` and
  `migrations/versions/20260610_0004_iam_admin_and_throttle.py:24-64` — throttle
  state is durable, shared, tenant-isolated PostgreSQL state.
- `migrations/versions/20260609_0001_secure_foundation.py:93-145` — user rows
  contain the password hash and `password_changed_at`; sessions contain expiry,
  revocation, client IP, and user agent.
- `config.py:371-398` — session lifetimes are configurable and bounded, but no
  password-policy settings exist.
- `react-ui/src/components/settings/UsersTab.tsx:119-140,218-225` — administrators
  can create/reset passwords, but the UI exposes no password-policy configuration.

**Responsibility boundary:**

EVA AI Engineering owns the local password implementation. Customer
administrators own account provisioning and credential handling. Riot Engineering
Ltd organizational IT/Security would own any corporate password standard or
future external-identity integration.

**Gaps or caveats:**

- No password history/reuse prevention or password expiration is implemented.
- No explicit maximum length or composition policy is implemented. A customer
  cannot configure these requirements without a product change or external IdP.
- The lower-level `PostgresIdentityRepository.bootstrap_admin()` method at
  `security/postgres_identity.py:97-113` checks only that the password is non-empty.
  The shipped CLI and installer enforce 12 characters before calling it, but the
  repository layer is not defense-in-depth against another direct caller.
- `password_changed_at` is recorded but is not used to enforce expiry.
- EVA AI Engineering has accepted the current password direction for the pilot:
  a 12-character minimum with Argon2id and no forced character composition,
  history, or periodic expiry. Repository-level defense-in-depth and an input
  safety bound are tracked in `docs/security/ROADMAP_TO_1_0.md`; planned work is
  not part of this assessment.

## #27 — AAAI-06 — InCommon / eduGAIN federation

**Assessment:** N/A

**Confidence:** High

**Proposed HECVAT response:**

> EVA is an on-premises product and does not currently implement InCommon,
> eduGAIN, SAML, OpenID Connect, OAuth identity-provider login, LDAP, or generic
> external IdP federation. Authentication uses EVA-local named users and sessions.
> Membership in InCommon or eduGAIN, if the questionnaire still requires an
> organizational answer, must be confirmed separately by the
> responding/contracting organization's management or IT.

**Evidence:**

- `security/http_auth.py:53-88,100-190` — the authentication interface and service
  authenticate a username/password against EVA's identity repository and create
  an EVA session.
- `security/postgres_identity.py:84-95,273-355` — identities and password hashes
  are persisted in EVA's PostgreSQL IAM schema.
- `docs/admin/admin_guide.md:21-33` — the documented model is local named users,
  role/channel grants, password reset, and session revocation.
- Repository-wide search at the audited commit found no implementation or
  configuration reference for InCommon, eduGAIN, SAML, OIDC, OAuth/OAuth2, LDAP,
  generic external IdP, or identity-provider metadata.

**Responsibility boundary:**

EVA AI Engineering owns product federation capabilities. InCommon/eduGAIN
membership is a responding/contracting-organization matter. A customer's IdP and
federation configuration would be shared with customer IT if a future integration
were added.

**Gaps or caveats:**

There is no current federation capability. Confirm whether the workbook expects
an organizational membership answer even though the offering is on-premises.

## #30 — AAAI-09 — Authentication and audit logging

**Assessment:** Yes

**Confidence:** High

**Proposed HECVAT response:**

> EVA records successful and failed login attempts, throttled attempts, logout,
> authorization and CSRF denials, protected route access, user/session
> administration, security-sensitive mutations, and agent tool actions. Audit
> records include a UTC timestamp, request ID, authenticated user and roles when
> available, tenant, source IP, action, target, channel, result, and redacted safe
> details. They are stored in a tenant-isolated PostgreSQL append-only table with
> a per-tenant SHA-256 hash chain. Successful logins identify the resulting
> session; failed anonymous logins intentionally do not store the attempted
> username. The repository does not define an audit-retention period or an
> explicit authentication-mechanism field.

**Evidence:**

- `security/audit.py:13-92,111-192` — sensitive detail fields are recursively
  redacted, and `AuditEvent`/`AuditEventBuilder` require timezone-aware timestamp,
  source IP, action, target, result, identity/roles/tenant/request context, and
  optional channel.
- `oldapp.py:1203-1228` — `_write_security_audit()` derives the source IP and
  persists the event through the PostgreSQL writer.
- `oldapp.py:1312-1443` — `_session_guard()` audits unauthenticated requests,
  rejected CSRF, permission/channel denial, and successful authorization; audit
  failure is fail-closed for protected requests.
- `oldapp.py:9389-9508,9872-9900` — `/auth/login` audits missing credentials,
  throttling, invalid credentials, and success; `/auth/logout` audits completion.
- `oldapp.py:9589-9626,9629-9745,9786-9815` — user creation/update, role/channel
  changes, session revocation, and their completion results are protected and
  audited.
- `oldapp.py:1275-1305` and `agent_security/audit.py:9-49` — each agent tool audit
  includes phase, operation, risk, required permission, normalized-argument hash,
  result code, duration, actor roles, tenant, request, session context, and source
  IP; it is linked to the durable audit event.
- `migrations/versions/20260609_0001_secure_foundation.py:286-317` —
  `agent.tool_runs` persists tool, actor, permission decision, argument hash,
  duration, result class, and audit-event reference.
- `migrations/versions/20260609_0001_secure_foundation.py:383-440,603-625` —
  `audit.events` persists the audit fields, is indexed by tenant/time and
  actor/time, rejects UPDATE/DELETE through an append-only trigger, and grants the
  audit writer INSERT rather than mutation rights.
- `security/postgres_audit.py:167-277` and
  `migrations/versions/20260727_0010_audit_hash_chain.py:24-54` — the durable
  writer serializes each tenant chain and stores a SHA-256 hash linked to the
  preceding event.
- `security/postgres_audit_reader.py:221-330` and `oldapp.py:9822-9869` — a
  protected, tenant-scoped, paginated reader supports operational review.
- `docs/admin/admin_guide.md:46-50` — sensitive endpoints and agent tool calls are
  documented as visible through the administrative audit surface.

**Responsibility boundary:**

EVA implements event generation, authorization context, redaction, persistence,
and the audit reader. Customer database/backup administrators own storage
capacity, database backup, access to host logs, and any SIEM export. Luxriot
Security owns any required retention policy and review procedure.

**Gaps or caveats:**

- No audit-specific retention duration or pruning policy was found. The table is
  append-only at the application role boundary, so it persists until a separately
  authorized database/retention operation is defined.
- Failed unauthenticated login records include timestamp, source IP, action,
  result, and reason but not the attempted username or a user/session ID.
- The local password mechanism is implied by `auth.login`; there is no dedicated
  `authentication_mechanism` audit field.
- The central guard covers protected/sensitive routes and explicit completion
  events, not a claim that every non-sensitive read or UI interaction is logged.
- No separate operational retention rule currently overrides the implementation:
  audit rows remain until an authorized database operation removes them. A safe,
  customer-configurable retention/export design is tracked for 1.0 in
  `docs/security/ROADMAP_TO_1_0.md`.

## #36 — DATA-02 — Encryption of sensitive data in transit

**Assessment:** No

**Confidence:** High

**Proposed HECVAT response:**

> EVA's offline appliance protects operator browser traffic with Nginx HTTPS and
> redirects HTTP to HTTPS using a locally generated certificate. The application
> itself listens on loopback HTTP behind that proxy, and the agent's SSE stream
> uses the same browser HTTPS boundary. However, EVA does not require TLS for all
> service-to-service links: EVA's Luxriot Evo and inference endpoint settings
> accept HTTP or HTTPS, and PostgreSQL encryption depends on the supplied libpq
> DSN. This describes EVA AI's client-side enforcement only; it does not assert
> what transport modes or controls Luxriot Evo itself provides. The bundled
> single-host services use loopback connections. EVA AI should select HTTPS for
> the Evo partner API when the Evo product exposes it; otherwise the site must
> protect the HTTP integration with an appropriately isolated trusted network or
> VPN. Off-host database and inference links similarly require PostgreSQL TLS,
> HTTPS, VPN/mTLS gateway, or an appropriately isolated trusted network.
> EVA should not be represented as enforcing end-to-end encryption for every
> deployment.

**Evidence:**

- `scripts/install_port_appliance.py:3035-3111` — the appliance creates a
  3,072-bit RSA/SHA-256 certificate valid for 825 days, protects the private key
  with mode `0600`, redirects port 80 to HTTPS, listens on 443 with Nginx, and
  proxies to `http://127.0.0.1:5000`.
- `docs/install/deployment_guide.md:89-110` and
  `docs/install/production_settings.md:7-19,21-43` — production documentation
  requires browser-facing TLS but explicitly states that Gunicorn is HTTP and a
  lab can run HTTP-only.
- `oldapp.py:9389-9506` — authenticated browser sessions use Secure (when
  configured), HttpOnly, SameSite=Strict session cookies and a CSRF cookie.
- `oldapp.py:22902-22920` and `react-ui/src/api/agent.ts:1-3` — agent output is
  same-origin HTTP Server-Sent Events over a streamed POST, not a separate
  WebSocket transport; it inherits the browser/proxy scheme.
- `config.py:452-464,623-626` — default LM and Luxriot URLs are HTTP; API keys and
  Luxriot credentials are configuration values.
- `oldapp.py:6040-6048,6132-6142,22147-22170` — inference calls send an optional
  bearer key and visual/text payload to the configured URL; Settings accepts both
  `http` and `https` schemes.
- `luxriot_connector.py:941-979` — the EVA AI integration client uses HTTP Digest
  authentication against the configured partner API base URL. Digest avoids
  sending the cleartext password but does not encrypt media or API payload when
  the URL is HTTP. This is evidence about EVA AI's client, not the Evo server
  implementation.
- `eva_db/settings.py:116-128,183-201` and `eva_db/pool.py:135-152` — the database
  layer accepts a PostgreSQL URI/libpq conninfo and passes it to psycopg without
  adding or requiring an SSL mode. Customer-provided `sslmode` parameters can be
  honored by libpq.
- `scripts/install_port_appliance.py:2289-2305,2623-2630,2671-2690,2790-2802` —
  the single-host appliance binds PostgreSQL, local VLM, deep review, and EVA
  application links to `127.0.0.1` and uses unencrypted loopback protocols.
- `docs/architecture/security_threat_model.md:24-39` — the documented trust model
  assumes a closed network for Evo/LM links and TLS at the browser boundary.

**Responsibility boundary:**

EVA AI/appliance deployment owns browser TLS termination and safe cookie behavior.
Customer infrastructure owns certificate trust/replacement and protection of
off-host PostgreSQL, inference, and inter-node links. The Luxriot Evo team owns
the transport capabilities exposed by its partner API. EVA AI Engineering owns
selection and validation of the configured API scheme and should use HTTPS if the
partner API provides it; Evo's server-side transport controls remain outside this
repository.

**Gaps or caveats:**

- TLS is not mandatory for Evo, LM/VLM, distributed inference, or remote database
  connections. Bearer credentials and customer images/text would be exposed to a
  network observer if an off-host HTTP endpoint were used.
- The generated certificate is locally self-signed; trust distribution and
  replacement with a customer PKI certificate are operational responsibilities.
- The generated Nginx block relies on the operating system's TLS defaults and
  does not pin protocol versions, cipher suites, HSTS, or mTLS.
- No separate WebSocket implementation was found; browser streaming is SSE and
  ordinary same-origin API/media traffic.
- Trusted-LAN HTTP is a practical 0.8.7 pilot configuration, not the intended
  final security posture. TLS enforcement or an explicit, visible site exception
  is tracked for 1.0 in `docs/security/ROADMAP_TO_1_0.md`; this plan does not
  change the current `No` assessment.

## #39 — DATA-05 — Data availability after contract termination

**Assessment:** Requires organizational confirmation

**Confidence:** High

**Proposed HECVAT response:**

> EVA is deployed on customer-controlled infrastructure. Continuous recordings
> remain in the customer's Luxriot Evo system. EVA's sampled frames, embeddings,
> descriptions, probes, alerts, incidents, agent history, IAM, audit data, runtime
> state, configuration, and backups remain in the customer's PostgreSQL and local
> filesystems. EVA sends visual/text requests only to the inference endpoints
> configured for that deployment; no fixed Luxriot-controlled cloud storage or
> telemetry backend was identified in the product. EVA provides configurable
> retention, customer-side database backup, and selected incident/false-positive
> report exports. Contract-termination availability, assistance, complete data
> export format, deletion obligation, and timelines
> require contractual confirmation by Riot Engineering Ltd and/or the applicable
> contracting entity's Legal/Privacy owners, together with the customer.

**Evidence:**

- `docs/architecture/data_retention_privacy.md:8-23` — maps sampled frames,
  embeddings, descriptions, alerts/bookmarks, probes, IAM, and audit to local
  PostgreSQL/Evo stores and states that EVA does not keep a continuous video copy.
- `docs/architecture/system_architecture.md:76-83` — identifies PostgreSQL storage
  for archive vectors/thumbnails, probes, summaries, semantic history, prompt
  settings, desired sessions, IAM, sessions, and audit.
- `migrations/versions/20260612_0005_archive_runtime.py:31-180` — defines local
  PostgreSQL archive detections (including thumbnails and vector bytes), probes,
  runtime state, and FORCE RLS tenant isolation.
- `migrations/versions/20260726_0008_attention_storage.py:20-28,35-140` — defines
  local attention embedding/probe telemetry and its lineage tables.
- `migrations/versions/20260801_0011_incidents.py:25-132` and
  `migrations/versions/20260805_0012_incident_temporal_memory.py:114-349` — define
  local incident, observation, episode, relation, and transition records.
- `migrations/versions/20260609_0001_secure_foundation.py:93-145,238-405` — defines
  local IAM, sessions, agent conversations/tool runs/action plans, and audit data.
- `config.py:407-449,1181-1196` and
  `docs/architecture/data_retention_privacy.md:25-50` — archive, thumbnail,
  summary, and rollup retention are configurable; scheduled pruning and targeted
  database deletion are technically possible.
- `docs/admin/backup_recovery.md:1-38` — PostgreSQL is the primary durable asset;
  configuration, optional inference spool, TLS material, and service units are
  on-host items for customer backup.
- `oldapp.py:3958-3993,4437-4472` — false-positive and incident reports are
  generated from durable records and returned as Markdown/XML downloads; the
  report response is not a separate server-side long-term store.
- `scripts/install_port_appliance.py:2217-2306` — runtime paths and all Evo/model
  endpoints are site-configured; the standard database is local PostgreSQL.

**Responsibility boundary:**

The customer controls the on-premises hosts, EVA databases, filesystems, backups,
and configured inference hosts. EVA AI implements retention and export/read
surfaces for EVA AI data. Continuous recordings and bookmarks belong to the
external Luxriot Evo integration; this repository cannot establish their
retention/export behavior. Riot Engineering Ltd and/or the applicable contracting
entity's Legal/Privacy owners own termination commitments and support obligations;
the customer owns its own preservation or erasure duties.

**Gaps or caveats:**

- No contract-termination SOP, guaranteed access period, export SLA, media-return
  process, or certified-deletion procedure exists in this repository.
- The product has selected report exports and database backup/restore, but no
  complete tenant-data portability/export contract is documented.
- Targeted erasure is described as possible DB tooling, but the documentation
  explicitly calls for a formal erasure SOP and legal timelines.
- Inference endpoints receive sampled customer imagery and prompts. Their owner,
  retention behavior, and transport must be verified for each deployment; the
  code cannot prove that every configured endpoint is customer-controlled.
- EVA AI sends bookmark requests to the separately supplied Luxriot Evo system,
  while continuous recording also remains there. Whether and how Evo persists,
  retains, exports, or deletes those records is outside this EVA AI repository.

## #42 — DATA-08 — Physical protection of archival media

**Assessment:** N/A

**Confidence:** High

**Proposed HECVAT response:**

> EVA does not define, ship, or physically control archival media. Durable EVA
> data is stored in PostgreSQL and on-host configuration/backup paths in the
> customer's on-premises environment; continuous recordings remain in the
> customer's Luxriot Evo infrastructure. Physical protection, transport,
> destruction, and chain of custody for servers, disks, removable media, and
> backups are customer infrastructure controls unless a separate managed-service
> contract states otherwise.

**Evidence:**

- `docs/architecture/system_architecture.md:76-83` — durable application data is
  in PostgreSQL rather than an EVA-controlled physical archive.
- `docs/admin/backup_recovery.md:7-16,25-38` — backups are customer-created
  PostgreSQL dumps plus separately secured local configuration and optional
  runtime files.
- `docs/architecture/data_retention_privacy.md:20-23` — continuous video stays in
  Luxriot Evo; EVA stores sampled evidence rather than a parallel recording.
- `scripts/install_port_appliance.py:2217-2252,2289-2305` — application/data/config
  roots and PostgreSQL are installed on the target customer host.

**Responsibility boundary:**

Customer/on-prem infrastructure owns the physical host and backup media. Luxriot
Evo is a separate VMS/NVR dependency whose recording/media controls are not
implemented or evidenced here. Riot Engineering Ltd or another contracting
organization would share responsibility only under a separate service that takes
custody of media; this repository provides no evidence of such a service.

**Gaps or caveats:**

The repository cannot establish customer data-center controls, disk encryption,
backup-vault security, removable-media procedures, or physical destruction.

## #46 — FIDP-02 — Firewall rule change process

**Assessment:** Requires organizational confirmation

**Confidence:** High

**Proposed HECVAT response:**

> EVA documents and configures its service bindings but does not implement a
> formal firewall-rule approval/change-management process. The standard appliance
> exposes Nginx on TCP 80 (redirect only) and TCP 443, while EVA/Gunicorn TCP 5000,
> local PostgreSQL TCP 5432, local VLM TCP 1234, and optional deep-review TCP 1236
> are bound to loopback. EVA connects outbound/on-site to the configured Luxriot
> Evo endpoint (commonly TCP 8080 or HTTPS) and to any configured remote inference
> or database endpoint. The bundle includes the `ufw` package but does not apply a
> firewall policy. Host/network firewall rules and their approval process are a
> customer IT/Security responsibility; Luxriot must separately confirm any
> organizational change-control procedure used during deployment support.

**Evidence:**

- `scripts/install_port_appliance.py:3035-3111` — generated Nginx configuration
  listens on TCP 80/443 and proxies to loopback TCP 5000.
- `scripts/install_port_appliance.py:2289-2305,2623-2630,2671-2690,2790-2802` —
  standard PostgreSQL, local VLM, and optional deep-review endpoints are loopback
  TCP 5432, 1234, and 1236.
- `scripts/install_port_appliance.py:522-545,1320-1324` — the installer accepts a
  customer Evo HTTP(S) endpoint; a bare address uses port 8080.
- `scripts/install_port_appliance.py:1005-1033` — the Spark container path uses
  Docker host networking, so host firewall policy remains authoritative.
- `deployment/port_4070s/apt-packages-ubuntu-24.04.txt:34-39` and
  `deployment/spark_gb10/apt-packages-ubuntu-24.04.txt:29-34` — `ufw` is bundled as
  a maintenance utility.
- `docs/install/production_settings.md:7-19,45-64` — documents the browser,
  Gunicorn, Evo, database, and multi-host inference endpoints.
- Repository-wide search found no `ufw`, nftables, iptables, or firewalld rule
  application in the installer/runtime and no firewall change-approval workflow.

**Responsibility boundary:**

EVA AI Engineering owns accurate port/binding documentation and secure local
defaults. Customer IT/Security owns host/network rules and production change
approval. Riot Engineering Ltd Services/IT owns any internal support change
process.

**Gaps or caveats:**

- Installing `ufw` is not the same as configuring or monitoring a firewall.
- Site-specific remote inference, Evo, PostgreSQL, DNS/NTP, OS update, and support
  routes must be added to a deployment-specific data-flow/port matrix.
- No formal approval, review, rollback, periodic recertification, or rule-owner
  process is evidenced by this repository.

## #47 — FIDP-03 — Network IDS

**Assessment:** No

**Confidence:** High

**Proposed HECVAT response:**

> EVA does not bundle, configure, or integrate with a network intrusion detection
> or prevention system. No Suricata, Snort, Zeek, NIDS/IPS, or SIEM/syslog-forwarding
> integration is present in the repository. EVA application audit and health logs
> are not network IDS. A customer may deploy NIDS/IPS and log forwarding around
> the EVA host as part of its network-security architecture.

**Evidence:**

- `docs/admin/observability.md:6-11,63-66` — the documented observability surfaces
  are application health/readiness, the systemd journal, and EVA's database audit
  log; no NIDS surface is described.
- `docs/architecture/security_threat_model.md:24-39,68-84` — network protection is
  framed as a closed-network/customer boundary, and exposure to untrusted
  networks is out of pilot scope.
- `deployment/port_4070s/apt-packages-ubuntu-24.04.txt` and
  `deployment/spark_gb10/apt-packages-ubuntu-24.04.txt` — the shipped host package
  lists contain no network IDS/IPS product.
- Repository-wide search at the audited commit found no Suricata, Snort, Zeek,
  NIDS, intrusion-prevention, SIEM, or syslog-forwarding implementation.

**Responsibility boundary:**

Customer IT/Security owns NIDS/IPS for an on-premises deployment. EVA AI
Engineering owns application audit/health only. Riot Engineering Ltd
organizational IT/Security must confirm whether a separate managed-service control exists
outside this repository; no Luxriot Evo NIDS capability is asserted here.

**Gaps or caveats:**

There is no product NIDS integration or documented event-forwarding contract.
Ordinary application logging must not be cited as satisfying this control.

## #48 — FIDP-04 — Host-based IDS

**Assessment:** No

**Confidence:** High

**Proposed HECVAT response:**

> EVA does not install or integrate a host intrusion detection or endpoint
> detection product. No Wazuh, OSSEC, auditd rule set, Falco, EDR, or host-security
> agent is present. EVA's service watchdogs, readiness probes, bundle checksums,
> and application audit log provide availability/integrity evidence but do not
> perform host intrusion detection. Customer IT/Security must provide any HIDS or
> EDR required for the on-premises host.

**Evidence:**

- `docs/admin/observability.md:6-11,63-66` — documents health/readiness, systemd
  logs, and application audit, not host intrusion detection.
- `scripts/install_port_appliance.py:2780-2788` — the VLM watchdog only restarts a
  failed inference service; it is an availability recovery control, not HIDS.
- `scripts/eva_offline_deploy.py:127-248` — bundle hashes verify release payload
  integrity before deployment but do not monitor runtime host compromise.
- Shipped package lists under `deployment/port_4070s/` and
  `deployment/spark_gb10/` contain no HIDS/EDR product.
- Repository-wide search at the audited commit found no Wazuh, OSSEC, HIDS,
  auditd integration, EDR, Falco, CrowdStrike, or SentinelOne configuration.

**Responsibility boundary:**

Customer IT/Security owns host hardening, HIDS/EDR selection, monitoring, and
response. EVA AI Engineering owns application health and release-integrity checks.

**Gaps or caveats:**

No HIDS/EDR integration, alert-forwarding interface, or operational response
procedure is documented.

## #50 — PPPR-01 — Patch management process

**Assessment:** Yes

**Confidence:** Medium

**Proposed HECVAT response:**

> EVA has a controlled technical release/update mechanism for on-premises
> deployments. A checksummed offline bundle detects fresh install, interrupted
> install/resume, update, or report mode. Updates run a read-only preflight,
> preserve site configuration and inference routing, require a validated
> PostgreSQL backup, apply transactional Alembic migrations, verify database-row
> preservation, replace versioned application/UI assets, restart services, and
> run health/readiness and deployment acceptance checks. Backup paths and an exact
> rollback command are retained. Release notes and schema/version identifiers are
> maintained. The repository does not establish Luxriot's vulnerability triage,
> security-patch SLA, release cadence, emergency approval, or customer-notification
> policy; those process elements require organizational confirmation.

**Evidence:**

- `deployment/universal/START_HERE.md:1-44` — documents one checksummed offline
  entry point for fresh install, resume, update, and report, with a read-only
  preflight before mutation.
- `deployment/universal/START_HERE.md:46-87` — documents the acceptance report,
  backups, rollback boundary, preservation of external topology/site settings,
  stream resumption, and checksum binding for client update packs.
- `scripts/eva_offline_deploy.py:127-248,292-348,542-622` — verifies manifest and
  critical/dependency hashes, detects incomplete installations, runs update
  preflight, obtains a transient migration identity when needed, asks for final
  confirmation, applies, and produces post-update acceptance.
- `scripts/preflight_patch.sh:41-56,227-247,352-431` — read-only preflight checks
  bundle, dependencies, disk/backup capacity, PostgreSQL tooling/schema, and live
  health without stopping or editing the installation.
- `scripts/install_patch.sh:289-380,387-434,500-612` — creates mandatory database
  and runtime-state backups before migrations, verifies model payloads, applies
  `alembic upgrade head`, and invokes post-install verification.
- `scripts/database_preservation_guard.py:296-355,364-458` — fails closed when the
  migration role cannot see FORCE RLS tables, captures a pre-migration inventory,
  and verifies that pre-existing users, probes, runtime state, and archive rows
  remain after migration.
- `docs/install/offline_installer_083.md:125-177,196-225,248-278` — documents
  migration identity, mandatory dump, idempotence, automatic rollback boundary,
  serialization lock, and explicit manual recovery.
- `scripts/verify_patch.sh:180-248` — verifies service state, health/readiness, and
  React UI payload after update.
- `readiness/RELEASE_NOTES_0.8.7.md:1-10,46-85` and `CHANGELOG.md:1-68` — provide
  version/schema migration notes, release changes, verification totals, and
  deployment requirements.
- `.github/workflows/ci.yml:16-81` — migration, documentation drift, Python
  compilation/tests, React tests, and production UI build run on pushes/PRs.

**Responsibility boundary:**

EVA AI Engineering owns versioned code, migrations, bundle integrity, update
mechanics, release tests, and technical recovery. Riot Engineering Ltd
Product/Security/Ops owns prioritization, vulnerability triage, release
approval/cadence, customer
notification, and support SLAs. Customer administrators own maintenance windows,
backup custody, local approval, execution, and post-update operational checks.

**Gaps or caveats:**

- The repository shows a mature technical update path, not an approved corporate
  patch-management policy.
- No severity-to-remediation SLA, routine patch cadence, emergency patch process,
  accountable security owner, end-of-support policy, or notification timeline is
  evidenced.
- CI does not currently include automated vulnerability scanning; see #53.

## #51 — PPPR-02 — Compliance with customer privacy/security policies

**Assessment:** Requires organizational confirmation

**Confidence:** High

**Proposed HECVAT response:**

> EVA provides technical controls that can support customer privacy and security
> requirements: named users, role-based permissions, channel-scoped access,
> PostgreSQL row-level tenant isolation, audit logging, configurable retention,
> local/on-premises data residency, approval-gated sensitive actions, and secure
> deployment checks. These capabilities do not constitute a commitment to review
> or comply with every customer's institutional policy. Acceptance of and
> compliance with a specific customer policy must be confirmed contractually by
> Riot Engineering Ltd and/or the applicable contracting entity's
> Legal/Privacy/Security owners and mapped to the deployed configuration.

**Evidence:**

- `security/permissions.py:10-74,120-168` — defines admin/engineer/operator/viewer
  roles, explicit permissions, and server-side channel authorization.
- `docs/admin/admin_guide.md:9-44` — documents named users, least-channel grants,
  API/tool-gateway enforcement, and PostgreSQL RLS.
- `docs/architecture/security_threat_model.md:24-39,52-66` — documents trust
  boundaries, authentication, RBAC, per-channel grants, RLS, CSRF, audit, and
  human approval for sensitive agent actions.
- `docs/architecture/data_retention_privacy.md:25-50` — documents configurable
  retention, minimization, access controls, and the need for legal retention and
  erasure procedures.
- `docs/install/production_settings.md:21-43,88-99` — identifies required secure
  deployment configuration and client-facing blockers.

**Responsibility boundary:**

EVA AI Engineering provides technical capabilities. Riot Engineering Ltd and/or
the applicable contracting entity's Legal/Privacy/Security owners own policy
review and contractual commitment. Customer governance owners define
the applicable policy and accept the mapped deployment controls and residual
risks.

**Gaps or caveats:**

No repository document authorizes a general commitment to comply with arbitrary
customer policies, and no policy-intake, control-mapping, exception, attestation,
or periodic-review process is evidenced.

## #52 — PPPR-03 — Applicable jurisdiction / laws

**Assessment:** Requires organizational confirmation

**Confidence:** High

**Proposed HECVAT response:**

> The EVA repository does not establish Luxriot's country of incorporation,
> governing-law jurisdiction, contracting entity, or the laws applicable to a
> particular customer deployment. EVA includes technical privacy, retention,
> access-control, traceability, and human-oversight features, and its documentation
> discusses EU privacy/AI requirements as a draft technical mapping. The concrete
> jurisdiction and binding legal obligations must be supplied by Luxriot
> Legal/Management for the relevant contract and deployment location.

**Evidence:**

- `docs/architecture/data_retention_privacy.md:1-6,20-23` — explicitly separates
  technical facts from legal classification and marks jurisdictional conclusions
  for legal sign-off.
- `docs/architecture/data_retention_privacy.md:25-50` — documents technical
  retention/minimization and deletion capabilities while assigning binding policy
  and timelines to Legal.
- `docs/architecture/data_retention_privacy.md:52-72` — provides a draft EU AI Act
  technical mapping but explicitly requires legal review for classification,
  legal basis, DPIA, conformity, and notification duties.
- Repository-wide search at the audited commit found no authoritative governing
  law, company jurisdiction, contracting-entity, or country statement.

**Responsibility boundary:**

Riot Engineering Ltd and/or the applicable contracting entity's
Legal/Privacy/Management owners own the answer. EVA AI Engineering can provide
the technical-control mapping. Customer Legal determines customer-specific and
local deployment obligations.

**Gaps or caveats:**

A concrete country/jurisdiction cannot be inferred from source code, developer
locations, timezone defaults, pilot geography, or EU-oriented documentation.

## #53 — VULN-01 — Authenticated vulnerability scanning

**Assessment:** No

**Confidence:** High

**Proposed HECVAT response:**

> The audited repository does not demonstrate an established authenticated
> vulnerability-scanning process for a deployed EVA instance. CI runs database
> migrations, documentation checks, Python compilation and tests, React tests,
> and a production frontend build. No authenticated DAST, dependency scanner,
> SAST, CodeQL, container scanner, or scheduled vulnerability-scan gate is
> configured. Functional tests and bundle checksum verification are valuable
> quality/integrity controls but are not authenticated vulnerability scanning.

**Evidence:**

- `.github/workflows/ci.yml:16-81` — the only committed CI workflow installs test
  dependencies, migrates PostgreSQL, checks docs, compiles Python, runs pytest and
  React tests, and builds the UI; it contains no security-scanning job.
- `docs/frontend_rewrite/full_react_migration_checklist_ru.md:698-700` — mentions
  an `npm audit` as a planned/point-in-time compatibility check, not a CI control
  and not authenticated application scanning.
- `docs/architecture/security_threat_model.md:81-84` — formal penetration-test
  sign-off is explicitly out of pilot scope/future work.
- `scripts/eva_offline_deploy.py:127-248` — release checksums validate offline
  artifact integrity but do not assess vulnerabilities in the running application.
- Repository-wide search found no Dependabot configuration, Trivy, Grype, Snyk,
  pip-audit, Bandit, Semgrep, CodeQL, container scan, OWASP ZAP, Nessus, Qualys,
  Burp, OpenVAS, authenticated DAST, or equivalent release gate.

**Responsibility boundary:**

EVA AI Engineering owns engineering-side scan integration and remediation. Riot
Engineering Ltd organizational IT/Security owns any corporate scanning program,
credentials, scope, cadence, and acceptance gates. Customer Security may
independently validate a deployed instance only under an agreed authorization;
see #55.

**Gaps or caveats:**

- No authenticated scan, scan schedule, severity gate, evidence retention, or
  remediation SLA is established.
- Dependency, SAST, container, and secret scanning are also absent from committed
  CI; adding them would improve coverage but would still not alone satisfy the
  authenticated deployed-application requirement.
- EVA AI Engineering confirms that no authenticated vulnerability scan or
  penetration test has yet been performed. The trusted 0.8.7 pilot is not being
  represented as a 1.0 security release. Authenticated assessment and penetration
  testing are explicit pre-1.0 work in `docs/security/ROADMAP_TO_1_0.md`.

## #54 — VULN-02 — Providing vulnerability scan results to customers

**Assessment:** Requires organizational confirmation

**Confidence:** High

**Proposed HECVAT response:**

> The EVA repository contains no policy or commitment to provide vulnerability
> scan, penetration-test, or security-assessment results to customers. Riot
> Engineering Ltd and/or the applicable contracting entity's IT Security and
> Legal owners must confirm whether such reports exist, what may be shared,
> under what NDA, at what level of detail, and through which approved channel.

**Evidence:**

- `.github/workflows/ci.yml:16-81` — no vulnerability-scan report is generated by
  the committed CI workflow.
- `docs/architecture/security_threat_model.md:81-84` — formal penetration-test
  sign-off is listed as out of pilot scope/future work.
- `docs/legal/dependency_license_audit.md:1-10` — the existing assessment artifact
  is a dependency-license checklist, explicitly not legal advice; it is not a
  vulnerability or penetration-test report.
- Repository-wide search found no customer disclosure policy, NDA-controlled scan
  report procedure, vulnerability-report template, or security-assessment sharing
  commitment.

**Responsibility boundary:**

Riot Engineering Ltd and/or the applicable contracting entity's IT Security and
Legal/Management owners. EVA AI Engineering can produce technical artifacts but
cannot authorize disclosure.

**Gaps or caveats:**

Do not promise a report until its existence, owner, currency, customer scope,
redaction, NDA terms, and disclosure channel have been confirmed.

## #55 — VULN-03 — Customer-performed vulnerability scanning / penetration testing

**Assessment:** Requires organizational confirmation

**Confidence:** High

**Proposed HECVAT response:**

> The EVA repository neither authorizes nor prohibits customer vulnerability
> scanning or penetration testing. Authorization must be confirmed in writing by
> Riot Engineering Ltd and/or the applicable contracting entity's IT
> Security/Legal owners and coordinated with the customer. A future policy
> should define the target environment, scope and exclusions, test identities,
> data handling, source addresses, rate limits, maintenance window, incident
> contacts, stop conditions, and disclosure/remediation process. These are
> recommendations, not current repository-backed commitments.

**Evidence:**

- `docs/architecture/security_threat_model.md:81-84` — public internet exposure
  and formal penetration-test sign-off are explicitly out of pilot scope/future
  work.
- `deployment/universal/START_HERE.md:105-130` — operational diagnostics and
  failure handoff are documented, but no customer security-testing authorization
  or rules of engagement are defined.
- Repository-wide search found no policy allowing/prohibiting customer scans, no
  rules-of-engagement template, no safe-harbor language, and no coordinated
  penetration-testing procedure.

**Responsibility boundary:**

Riot Engineering Ltd and/or the applicable contracting entity's IT
Security/Legal owners own authorization and rules of engagement. Customer
Security owns its testers and adherence to the approved scope. EVA AI
Engineering supports test-environment preparation and remediation only after
authorization.

**Gaps or caveats:**

No customer is automatically authorized by the software license or repository to
scan production systems. The recommended conditions above are not an existing
policy.

## #56 — HIPA-01 — HIPAA/HITECH training

**Assessment:** N/A

**Confidence:** Medium

**Proposed HECVAT response:**

> EVA is documented as an on-premises public-space video decision-support product,
> and the current questionnaire scope states that it does not process PHI or
> HIPAA-regulated data. The repository makes no HIPAA/HITECH compliance or
> healthcare-specific claim and cannot establish employee training. HIPA-01 is
> therefore likely not applicable to this offering as scoped. Any statement about
> Riot Engineering Ltd staff HIPAA/HITECH training, or any future deployment
> involving PHI, requires confirmation by Riot Engineering Ltd and/or the
> applicable contracting entity's Legal/Privacy/HR owners and a renewed
> applicability assessment.

**Evidence:**

- `docs/architecture/data_retention_privacy.md:1-23` — describes a government
  public-space monitoring pilot and sampled public-space imagery, with no
  healthcare/PHI processing claim.
- `docs/architecture/security_threat_model.md:8-13` — identifies live/archived
  public-space imagery, derived alerts/vectors, credentials, IAM, audit, and agent
  history as assets; it does not identify PHI.
- Repository-wide search at the audited commit found no HIPAA, HITECH, PHI,
  protected-health-information, healthcare-control, or HIPAA-training statement.

**Responsibility boundary:**

Riot Engineering Ltd and/or the applicable contracting entity's Legal/Privacy/HR
owners own applicability and personnel-training claims. The customer owns
determining whether its selected use, cameras, integrations, or
operator inputs introduce PHI. EVA AI Engineering can map product controls after
that scope is defined.

**Gaps or caveats:**

Repository silence cannot prove that employees have or have not received
training. If EVA is deployed in healthcare or receives PHI, N/A must be revisited
and technical, contractual, and workforce controls reassessed.

## Items requiring human confirmation

### Confirmed EVA AI Engineering decisions

- The current local-password direction is a 12-character minimum with Argon2id,
  no forced composition/history/periodic expiry, and consistent enforcement at
  every supported and low-level provisioning path before 1.0.
- There is no separate audit-retention policy today; current code is the source
  of truth. Retention/export design is a 1.0 roadmap item and must not introduce
  silent deletion during an update.
- Plain HTTP on EVA-owned off-host links is not the desired final posture. The
  Luxriot Evo integration currently uses Evo's ordinary partner HTTP API; EVA AI
  will prefer HTTPS when the Evo team exposes it but does not own that server-side
  capability. Trusted-network/VPN handling and explicit warnings remain part of
  the 1.0 design.
- No authenticated scan or penetration test has been performed for 0.8.7. The
  product is a trusted pilot; automated security scanning, authenticated
  assessment, and penetration testing are tracked toward the 1.0 release.

These are EVA AI product/security decisions and must not be redirected to the
Luxriot Evo team. The living plan is `docs/security/ROADMAP_TO_1_0.md`. Roadmap
entries remain gaps rather than implemented controls until evidence exists.

### Outside EVA AI Engineering — organizational IT / Security

- Confirm the responding/contracting organization's InCommon/eduGAIN status if
  the workbook requires an organizational answer despite the on-premises N/A.
- Approve the production port/data-flow matrix and define firewall rule request,
  approval, rollback, review, and recertification procedures.
- Confirm required customer/host NIDS, HIDS/EDR, and SIEM/log-forwarding controls.
- Define vulnerability scan and penetration-test cadence, scope, severity gates,
  remediation SLAs, evidence retention, and security-patch escalation.
- Decide whether and under what NDA the responding/contracting organization
  provides scan/assessment results, and publish customer penetration-testing
  authorization/rules of engagement.

### Outside EVA AI Engineering — Legal / Privacy / Management

- Specify the contracting entity, governing law, applicable jurisdiction, and
  deployment-specific privacy/regulatory obligations.
- Define contract-termination data access, export assistance, retention/deletion,
  certified-erasure, and backup/media handling commitments and timelines.
- Define the process for receiving, mapping, accepting, or taking exceptions to a
  customer's privacy/security policies.
- Confirm contractual responsibility for customer-controlled physical media,
  backups, EVA databases, and third-party/customer inference hosts. Luxriot Evo
  recording behavior is explicitly outside EVA AI's engineering scope and must be
  answered by the owner of that VMS/NVR if the customer asks for it.
- Confirm HIPAA/HITECH/PHI applicability and any workforce-training statement;
  retain N/A only while PHI is explicitly outside the offering/deployment scope.

### Outside EVA AI Engineering — Luxriot Evo dependency

Do not answer the following from EVA AI code or attribute them to EVA AI:

- Luxriot Evo's password/authentication policy and security audit coverage;
- Evo server-side TLS support or enforcement;
- continuous-recording and bookmark persistence, retention, export, deletion,
  backup, availability, and physical-media controls;
- Evo server hardening, firewall management, NIDS/HIDS, vulnerability scanning,
  patching, and penetration-testing policy.

EVA AI Engineering can accurately describe only EVA's side of the integration:
configured partner API URL and credentials, HTTP Digest client behavior, media
consumption, bookmark requests, local evidence derived from sampled frames, and
the scheme/network validation EVA does or does not enforce. The Evo team owns the
API implementation, including whether an HTTPS endpoint is available.
