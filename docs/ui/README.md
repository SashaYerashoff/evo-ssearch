# UI screenshot harness

Every UI picture in the operator and admin guides is generated, never hand-taken.
A scene in [`shots.json`](shots.json) declares which role, what to click and what
to frame; [`scripts/ui_shots.py`](../../scripts/ui_shots.py) drives a real browser
against a running EVA and writes the PNG plus a provenance sidecar into
[`assets/`](assets/).

The point is reproducibility: anyone — a person, Claude, or Codex — can regenerate
a guide picture after a UI change without guessing what the original screenshot
was showing.

## Scope: the React UI

The manifest documents the **React UI only** (`/?ui=react`), section by section
including tabs and modals:

| Area | Scenes |
|---|---|
| Shell | home, navigation drawer, appearance |
| Archive | filters, stream picker, period menu, date range, text query, image query, results, research review modal |
| Summaries | stream review, period, resolution, expanded description, preview drawer, stream settings, prompt modal (Stream / Alerts / rollups), incident modal |
| Probes | board, list view, selected-probe modal, probe editor, advanced sections, cast, channel groups |
| Agent | docked, panel controls, control strip, history / skills / streams menus, composer, full screen with the session rail, an answered transcript |
| Settings | Server, Users, Audit, Environment, Diagnostics |

The legacy template UI is deliberately **not** documented. It is carried for
compatibility; the guides describe the current UI. The harness still supports
`"ui": "legacy"` if that ever changes.

The scenes were derived from the React sources rather than from a live build, so
the first capture run is also their first real verification. A wrong selector
fails that one scene by name and leaves the rest of the run intact — read the
`FAILED` lines and fix the manifest.

## Privacy first

Evidence frames are operational camera data. They must never end up in a
committed PNG.

Every scene has a `media_policy`:

| Policy | What happens | When |
|---|---|---|
| `placeholder` (default) | The evidence endpoints (`/detections/image`, `/detections/thumbnail/*`, `/image/*`) are intercepted **in the browser**; a neutral graphic is served instead, so real pixels are never decoded. `video`/`canvas`, which the network layer cannot neutralize, are blurred. | Always, unless there is a specific reason not to |
| `blur` | Real media loads, then every selector in `redact` + `redact_always` is blurred before the shot. | Layout shots where a placeholder would be misleading |
| `pass` | No interception, no blur. Requires `--allow-real-media` on the command line, and the choice is recorded in the sidecar so review can see it. | Only against a deployment running **rights-cleared, non-personal** streams |

A scene can *add* selectors via `redact_extra`; it cannot drop the defaults.
Blurring is defense in depth — the real control is what the deployment is
streaming when you capture.

## Prerequisites

`validate` needs nothing but Python. Capturing needs the browser driver:

```bash
uv pip install --python .venv/bin/python -r requirements-docs.txt
.venv/bin/python -m playwright install chromium
```

This is workstation-only. Playwright is deliberately absent from
`requirements.txt`: the appliance runtime ships no browser driver and the offline
bundle does not carry ~120 MB of Chromium.

## Capturing

The React build must exist — the app falls back to the legacy UI and stamps
`X-EVA-UI-Fallback: react-dist-missing` when `react-ui/dist` is absent:

```bash
cd react-ui && npm ci && npm run build && cd ..
```

Then:

```bash
export EVA_SHOTS_BASE_URL=http://127.0.0.1:5000
export EVA_SHOTS_OPERATOR_USER=...  EVA_SHOTS_OPERATOR_PASSWORD=...
export EVA_SHOTS_ENGINEER_USER=...  EVA_SHOTS_ENGINEER_PASSWORD=...
export EVA_SHOTS_ADMIN_USER=...     EVA_SHOTS_ADMIN_PASSWORD=...

.venv/bin/python scripts/ui_shots.py list
.venv/bin/python scripts/ui_shots.py capture --scene react-summaries-review
.venv/bin/python scripts/ui_shots.py capture            # every enabled scene
```

Three roles are used on purpose: operator scenes must show what an operator
actually sees, engineer scenes cover prompt and probe editing, admin scenes cover
Settings. Capturing everything as admin would document a workspace no operator
has.

Credentials come from the environment only — never from the manifest, and never
into a sidecar. `EVA_SHOTS_USER` / `EVA_SHOTS_PASSWORD` are the fallback when a
role-specific pair is not set. Login uses the same `POST /auth/login` the UI uses,
so role-dependent screenshots really do show what that role sees.

Useful flags: `--out-dir` (write elsewhere, e.g. a smoke run that must not touch
the repo), `--headed` (watch it drive), `--check` (compare instead of write),
`--verify-tls` (off by default, for dev self-signed certificates).

## Putting a screenshot into a guide

Order matters — the CI gate fails on a doc that embeds a screenshot that does not
exist yet:

1. `capture` the scene.
2. `scripts/ui_shots.py snippet <scene-id>` prints the markdown with the correct
   relative path and the caption as alt text.
3. Paste it into the guide, and list that guide in the scene's `used_by`.
4. `bash scripts/check_docs_drift.sh`.

## What CI checks

`scripts/check_docs_drift.sh` runs `ui_shots.py validate` — stdlib only, no
browser, no running service.

**Fails the build:**

- a doc embeds a screenshot that does not exist, or that no scene produces;
- a PNG in `assets/` that no scene produces (a hand-taken screenshot);
- a malformed scene: bad id, unknown step verb, output outside `assets/`, missing
  caption, `used_by` pointing at a missing doc.

**Warns only** (use `validate --strict` to escalate):

- a selector naming a class or id that appears nowhere in `react-ui/src` — a
  spelling check that catches a rename before the next capture session, though a
  name that exists may still point at the wrong element;

- a screenshot whose sidecar fingerprint predates a change in the UI source that
  produced it — the signal that a picture needs re-shooting;
- a captured screenshot no doc uses yet;
- a screenshot captured with `media_policy: pass`.

Staleness is a warning rather than an error on purpose: a UI change should not
turn CI red on a machine that has no browser and no running service. It should
show up as a to-do for whoever next regenerates the guides.

## Scene format

```json
{
  "id": "react-summaries-review",
  "ui": "react",
  "auth": "operator",
  "title": "Summaries — stream review",
  "caption": "Used verbatim as the markdown alt text.",
  "path": "/?ui=react",
  "steps": [
    {"click": ".menu-ear"},
    {"click": ".menu-item:has-text('Summaries')"},
    {"wait_for": ".vid-cols"}
  ],
  "clip": null,
  "output": "docs/ui/assets/react-summaries-review.png",
  "used_by": ["docs/operator/operator_guide.md"],
  "requires": "free-text note about the data or permissions the scene needs"
}
```

`clip` is a selector for an element shot, or `null` for the whole viewport.
`enabled: false` parks a scene that is not ready without deleting it.

A scene that needs data should `wait_for` a content selector (`.card-grid .card`,
`.probe-card`, `.vid-sum-toggle`). Then a missing fixture fails the scene loudly
instead of quietly publishing a screenshot of an empty state.

Step verbs are a closed set — the manifest cannot execute arbitrary JavaScript:

| Verb | Payload |
|---|---|
| `click`, `wait_for`, `hide`, `scroll_to` | selector string |
| `wait_ms` | integer |
| `fill`, `set_text` | `{"selector": ..., "value": ...}` |
| `press` | `{"selector": ..., "key": ...}` |

`set_text` is how a volatile timestamp is pinned to a fixed value so the same
scene produces the same picture tomorrow. `hide` removes a transient banner.

## Determinism

Fixed viewport, fixed colour scheme and locale, animations and transitions
disabled, scrollbars hidden, and `local_storage` seeded (the React UI reads its
language from `eva.ui.language.v1`). `--check` allows a small pixel tolerance
(default 0.2%) so font antialiasing does not report a false change.

## Known rough edges

- **No URL routing** — `App.tsx` holds the active section in `useState`
  ([App.tsx:91](../../react-ui/src/App.tsx#L91)), so every scene clicks through the
  left-rail drawer. Reading a `?section=` parameter on mount would collapse three
  steps into a URL and would help support, too.
- **No `data-testid` anywhere**, so scenes match by class and by English label
  (`.menu-item:has-text('Archive')`). Class names are reasonably stable; labels
  break the moment a caption is reworded or the UI is captured in Latvian. A
  dozen test ids on the rail, the tab strips and the modal roots would remove
  that whole class of breakage.
- **Locale** is pinned to English by seeding `eva.ui.language.v1`. Latvian
  screenshots would need a second manifest with translated label selectors — one
  more reason to prefer test ids.
- **Data-dependent scenes** (archive results, probe board, expanded description,
  agent transcript) need a deployment with content. Their `requires` field says
  what.
- **`react-agent-transcript` is the one scene whose *content* is published**, not
  just its layout. Read the session before capturing it.

## Self-test

`tests/test_ui_shots.py` covers the manifest rules and the gate without a browser,
and runs an end-to-end capture against a local fixture page when Playwright is
installed — including the assertion that `placeholder` really does stop the
evidence request from reaching the server.
