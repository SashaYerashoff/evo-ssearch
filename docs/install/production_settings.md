# Production Settings Checklist

Single-page checklist for the client-facing EVA AI control-plane. Detailed
variables live in [config_reference](../00_CANON/config_reference.md); this file
collects the deployment decisions that caused the office HTTP/HTTPS confusion.

## Browser URL vs. Internal App URL

| Surface | Expected setting |
|---|---|
| Operator browser URL | `https://<client-eva-host>/` or another site-approved HTTPS URL |
| Internal Gunicorn URL | `http://127.0.0.1:5000` by default |
| Reverse proxy | Nginx/site TLS boundary forwards HTTPS to internal HTTP |
| Health from server shell | `curl http://127.0.0.1:5000/health` or proxy URL if verifying TLS |
| Health from operator browser | HTTPS URL only for client-facing deployment |

`EVOSSEARCH_PORT=5000` is an internal HTTP port. Setting it to `5443` does not
make Gunicorn serve TLS. Use Nginx/site proxy for TLS unless a separate
TLS-enabled service unit is explicitly installed and verified.

## Required Secure-Pilot Env

```env
EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVA_DB_STRICT_RUNTIME_ROLES=true
EVOSSEARCH_ARCHIVE_STORE=postgres
EVOSSEARCH_EMBEDDER=clip
EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED=false
EVOSSEARCH_DINO_SEGMENTS_ENABLED=false
EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED=false
EVOSSEARCH_GUNICORN_WORKERS=1
EVOSSEARCH_HOST=127.0.0.1
EVOSSEARCH_PORT=5000
EVOSSEARCH_AUTH_COOKIE_SECURE=true
```

If the client temporarily runs HTTP-only in a closed lab, set
`EVOSSEARCH_AUTH_COOKIE_SECURE=false` or browser login cookies will not work.
That is a lab/demo exception, not the client-facing target state.

## Site-Specific Env To Fill

Keep real values in `/etc/eva-ai/eva-ai.env` only:

```env
EVOSSEARCH_AUTH_TENANT_ID=<uuid>
EVOSSEARCH_ARCHIVE_TENANT_ID=<same-or-site-uuid>
EVA_DATABASE_DSN=<secret>
EVA_AUDIT_DATABASE_DSN=<secret>
EVA_WORKER_DATABASE_DSN=<secret>
EVOSSEARCH_LUXRIOT_BASE_URL=http://<luxriot-evo-ip>:<port>
EVOSSEARCH_LUXRIOT_USERNAME=<secret>
EVOSSEARCH_LUXRIOT_PASSWORD=<secret>
EVOSSEARCH_LM_PROFILES=agent,vlm-a1,vlm-a0,vlm-b1,vlm-b0
EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=http://<agent-host>:1234/v1
EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL=http://<inference-a-ip>:8001/v1
EVOSSEARCH_LM_PROFILE_VLM_A0_BASE_URL=http://<inference-a-ip>:8002/v1
EVOSSEARCH_LM_PROFILE_VLM_B1_BASE_URL=http://<inference-b-ip>:8001/v1
EVOSSEARCH_LM_PROFILE_VLM_B0_BASE_URL=http://<inference-b-ip>:8002/v1
```

## Verification Commands

From the EVA AI server:

```bash
systemctl status eva-ai --no-pager -l
curl -sS http://127.0.0.1:5000/health | jq
curl -sS http://127.0.0.1:5000/ready | jq '.status, .checks.deployment_security, .checks.luxriot, .checks.lm_profiles'
```

If Nginx/TLS is configured:

```bash
curl -k -sS https://127.0.0.1/health | jq
curl -k -sS https://127.0.0.1/ready | jq '.status, .checks.deployment_security'
```

From an operator workstation, open the client-approved HTTPS URL and confirm
login works. If login fails only in the browser while `curl http://127.0.0.1:5000`
works on the server, check `EVOSSEARCH_AUTH_COOKIE_SECURE` versus the actual
browser-facing scheme.

## Acceptable Warnings

- Office/demo HTTP on `:5000` is acceptable only for internal testing.
- `inference_queue.status=disabled` is expected in the current pilot.

## Client-Facing Blockers

- Browser-facing URL is HTTP-only.
- `EVOSSEARCH_AUTH_COOKIE_SECURE=false` on a TLS deployment.
- `/ready.checks.deployment_security.issues` includes placeholder secrets.
- `EVOSSEARCH_GUNICORN_WORKERS` is not `1`.
- Luxriot or LM profile checks are not reachable.
