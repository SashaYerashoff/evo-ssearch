# Archive — API map (for React rewrite)

Scope: everything the **Archive** screen needs. Backend = existing Flask (`oldapp.py`), unchanged. React SPA consumes these endpoints.

## Auth (applies to all)
- Session is a cookie (`eva_session`) set by `POST /auth/login {username,password}`.
- Login response includes `csrfHeader: "X-CSRF-Token"` and sets cookie `eva_csrf`.
- **All POST/mutating calls must send header `X-CSRF-Token: <eva_csrf cookie value>`** + credentials (cookies). GET needs only the session cookie.
- `GET /auth/me` → `{ user: { tenantId, roles[], permissions[], allowedChannelIds[] } }`.

## Filter option sources
- **Streams (channels):** `GET /luxriot/channels` → `{ channels: [{ id:int, title:str, guid, server, ptzCapabilities }] }`. UI joins `detection.channel_id → title`.
- **Probes:** `GET /probes/list` → `{ probes: [...] }` (probe id + name for the "Probe" filter).
- **Source enum (client-side):** `"" (All evidence) | vlm_summary | vlm_alert | probe`.
- **Sort:** `similarity | time`. **Time range hours:** `1,6,24,72,168,0(=all)`.

## 1. Load Archive  — `GET /detections/list`
Query params (all optional): `probe_id, channel_id(int), source, since_ms(int), until_ms(int), hours(float, default 24; used only if since_ms absent; hours=0 → all time), limit(default 50, max 500), offset(default 0)`.
Response:
```json
{ "detections": [Detection...], "total": 6, "limit": 24, "offset": 0,
  "has_more": false, "filters": {probe_id,channel_id,source,since_ms,until_ms} }
```
Pagination = `offset += limit` while `has_more`. (UI Prev/Next.)

## 2. Text query — `POST /detections/search_text`  (JSON)
Body: `{ query(str, required), probe_id, channel_id, source, since_ms, until_ms, hours, limit(3..48), candidate_limit(20000), sort_by, embedder }`.
Response: `{ results: [Detection + score], coverage:{}, mode_requested, mode_used, filters, query }`.
Note: text search is always CLIP (DINO/fusion can't encode text). Empty query → 400.

## 3. Image query — `POST /detections/search_image`  (multipart/form-data)
Fields: `image` (file, required) + form fields `probe_id, channel_id, source, since_ms, until_ms, hours, limit, candidate_limit, sort_by, embedder`.
Response: `{ results:[Detection + score], coverage:{}, mode_used, filters }`.

## 4. Summary/counts — `GET /detections/summary`
Params: `source, channel_id, since_ms/hours, until_ms, limit(100)`.
Response: `{ summary:[{probe_id,hit_count,...}], count, filters }`. (Per-probe hit counts for the range.)

## 5. Images
- Full frame: `GET /detections/image?path=<detection.image_path>` (server-side path) or `GET /detections/thumbnail/<id>`.
- Each Detection also carries `thumbnail` = **base64 JPEG inline** (use directly as `data:image/jpeg;base64,<thumbnail>`), so the grid needs no extra image request.

## 6. Inspector actions
- **Describe frame:** `POST /describe_image` (JSON) — runs the VLM on the frame; returns a text description. (Inspector shows "No LLM description yet" until called.)
- **Find similar:** re-runs `POST /detections/search_image` using the selected frame. (CONFIRM exact payload — likely image_path or re-upload.)
- **Comments:** `GET/POST /comments`, `POST /commented_images`. NOTE: UI currently disables inline comments for archive-only results ("comments are unavailable for archive-only results").

## Detection object (verified live)
```jsonc
{
  "id": 28,
  "channel_id": 1590,                 // join to /luxriot/channels for title ("Stream")
  "probe_id": "probe-c62798334039",
  "probe_name": "Man on chair",       // "Name" in card
  "source": "probe",                  // probe | vlm_summary | vlm_alert
  "severity": "critical",
  "pos_score": 0.2707,                // "P" chip
  "neg_score": 0.2184,                // "N" chip
  "margin": 0.0523,                   // "M" chip
  "recorded_at_ms": 1782917454720,    // "Time"
  "thumbnail": "<base64 jpeg>",       // inline preview
  "image_path": "D:\\...\\....jpg",   // full-res via /detections/image?path=
  "has_clip": true, "has_dino": false,
  "bookmark_enabled": true, "bookmark_sent": true,
  "shard_key": "ch1590:20260701",
  "payload": {                        // rich detail for inspector
    "origin": "probe_daemon", "source": "probe", "hit_index": 0,
    "probe_window_sec": 300.0, "probe_fps": null,
    "hit": { "pos_score", "neg_score", "margin", "timestamp_ms", "channel_id" },
    "context": { "bookmark_gate": { "similarity", "reason", "sent", ... },
                 "roi_enabled": false, "roi_norm": null, "frames_indexed": 114 },
    "retention": { "decision", "kept", "record_persisted", ... }
  }
}
```
- **"Match %"** in the card = search-result `score`/similarity (search endpoints), or `payload.context.bookmark_gate.similarity` for list rows. Confirm which the UI uses per source.
- **"Origin"** = `payload.origin`. **"CLIP"** tag = `has_clip`.

## RESOLVED

### #1 Find similar (verified)
= `POST /detections/search_image` (multipart) with the selected frame's **image blob** as `image` + `limit`, `sort_by`, `embedder`, and current filters. Frontend gets the blob from the archive thumbnail/`/detections/image`, re-posts it. So "find similar" is just image-search seeded by the chosen frame.

### #2 describe_image (verified)
Request — two forms:
- Archive frame: **multipart** `image`(blob) + `prompt` + optional `model`.
- Folder/path: **JSON** `{ image_path, prompt, model, folder? }`.
Response:
```json
{ "summary": "<the VLM description text>",   // <-- description lives in `summary`
  "model": "...", "model_selection": "default_agent", "model_selector": "...",
  "profile_id": "default", "assigned_profile_id": "default",
  "filename": "frame.jpg", "thumbnail": "<base64>", "uploaded": "True" }
```

### #3 Match % + schema drift (IMPORTANT for React)
- There is **no `score` field**. Search rows carry **`similarity`** (0..1) → that's "Match %". List rows have no similarity; match derives from `payload.context.bookmark_gate.similarity`.
- **`/detections/list` and `/detections/search_*` return DIFFERENT object schemas — must be normalized in the client:**

| Concept        | `/detections/list`        | `/detections/search_*`                 |
|----------------|---------------------------|----------------------------------------|
| id             | `id`                      | `detection_id`                         |
| full image ref | `image_path`              | `path`                                 |
| timestamp      | `recorded_at_ms`          | `timestamp_ms`                         |
| match %        | (derive from payload)     | `similarity`                           |
| extra          | `bookmark_*`, `has_clip`  | `is_detection`, `source_label`, `search_mode`, `dino_fallback`, `filename`, `metadata`, `origin` |
| shared         | channel_id, probe_id, probe_name, source, severity, pos_score, neg_score, margin, thumbnail(base64), payload |

**Action for React:** define a single `Detection` view-model and a normalizer that maps both list-rows and search-rows into it (id ← id/detection_id, imageRef ← image_path/path, tsMs ← recorded_at_ms/timestamp_ms, matchPct ← similarity ?? payload…). This normalizer is the first util to build.
