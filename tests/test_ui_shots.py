"""Tests for the documentation screenshot harness (scripts/ui_shots.py).

The manifest/gate tests are deterministic and stdlib-only, so they run in CI next
to everything else. The browser test is skipped unless Playwright and its
Chromium build are present (requirements-docs.txt is a workstation-only install).
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "ui_shots.py"
SPEC = importlib.util.spec_from_file_location("ui_shots", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ui_shots = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ui_shots
SPEC.loader.exec_module(ui_shots)


MINIMAL_SCENE = {
    "id": "legacy-example",
    "ui": "legacy",
    "auth": "none",
    "title": "Example",
    "caption": "An example scene.",
    "path": "/?ui=legacy",
    "steps": [{"wait_for": "#root"}],
    "clip": None,
    "output": "docs/ui/assets/legacy-example.png",
    "used_by": [],
}


def manifest_with(*scenes: dict) -> dict:
    return {
        "version": 1,
        "defaults": {"media_policy": "placeholder", "redact_always": ["video"]},
        "scenes": [copy.deepcopy(scene) for scene in scenes],
    }


class ValidateInTempRepoTest(unittest.TestCase):
    """validate() reads the repo around it, so each case gets its own fake repo."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        (self.repo / "docs" / "ui" / "assets").mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)
        patcher_root = patch.object(ui_shots, "REPO_ROOT", self.repo)
        patcher_docs = patch.object(ui_shots, "DOC_ROOT", self.repo / "docs")
        patcher_root.start()
        patcher_docs.start()
        self.addCleanup(patcher_root.stop)
        self.addCleanup(patcher_docs.stop)

    def run_validate(self, manifest: dict, strict: bool = False) -> tuple:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = ui_shots.validate(manifest, strict=strict)
        return code, buffer.getvalue()

    def write_doc(self, relative: str, text: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_png(self, relative: str, payload: bytes = b"\x89PNG\r\n\x1a\n") -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_minimal_manifest_is_valid(self) -> None:
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE))
        self.assertEqual(code, 0, output)
        self.assertIn("not captured yet", output)

    def test_unknown_step_verb_is_rejected(self) -> None:
        scene = copy.deepcopy(MINIMAL_SCENE)
        scene["steps"] = [{"evaluate": "window.close()"}]
        code, output = self.run_validate(manifest_with(scene))
        self.assertEqual(code, 1)
        self.assertIn("unknown step verb", output)

    def test_step_needs_exactly_one_verb(self) -> None:
        scene = copy.deepcopy(MINIMAL_SCENE)
        scene["steps"] = [{"click": "#a", "wait_for": "#b"}]
        code, output = self.run_validate(manifest_with(scene))
        self.assertEqual(code, 1)
        self.assertIn("exactly one verb", output)

    def test_duplicate_scene_id_is_rejected(self) -> None:
        second = copy.deepcopy(MINIMAL_SCENE)
        second["output"] = "docs/ui/assets/other.png"
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE, second))
        self.assertEqual(code, 1)
        self.assertIn("duplicate id", output)

    def test_output_must_be_png_under_assets(self) -> None:
        scene = copy.deepcopy(MINIMAL_SCENE)
        scene["output"] = "docs/operator/sneaky.png"
        code, output = self.run_validate(manifest_with(scene))
        self.assertEqual(code, 1)
        self.assertIn("must be a .png under docs/ui/assets/", output)

    def test_caption_is_required(self) -> None:
        scene = copy.deepcopy(MINIMAL_SCENE)
        scene["caption"] = "  "
        code, output = self.run_validate(manifest_with(scene))
        self.assertEqual(code, 1)
        self.assertIn("'caption' is required", output)

    def test_doc_embedding_a_missing_screenshot_fails(self) -> None:
        self.write_doc("docs/operator/guide.md", "![Example](../ui/assets/legacy-example.png)\n")
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE))
        self.assertEqual(code, 1)
        self.assertIn("embeds missing screenshot", output)

    def test_doc_embedding_an_undeclared_screenshot_fails(self) -> None:
        self.write_png("docs/ui/assets/legacy-example.png")
        self.write_png("docs/ui/assets/hand-made.png")
        self.write_doc("docs/operator/guide.md", "![Hand made](../ui/assets/hand-made.png)\n")
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE))
        self.assertEqual(code, 1)
        self.assertIn("no scene in docs/ui/shots.json produces", output)

    def test_orphan_asset_fails(self) -> None:
        self.write_png("docs/ui/assets/left-over.png")
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE))
        self.assertEqual(code, 1)
        self.assertIn("orphan screenshot", output)

    def test_captured_but_unused_screenshot_warns_only(self) -> None:
        self.write_png("docs/ui/assets/legacy-example.png")
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE))
        self.assertEqual(code, 0, output)
        self.assertIn("no doc embeds it yet", output)

    def test_stale_fingerprint_warns_and_is_fatal_under_strict(self) -> None:
        self.write_png("docs/ui/assets/legacy-example.png")
        self.write_doc("docs/operator/guide.md", "![Example](../ui/assets/legacy-example.png)\n")
        self.write_doc("templates/index.html", "<html>new</html>")
        (self.repo / "docs/ui/assets/legacy-example.png.meta.json").write_text(
            json.dumps(
                {
                    "scene": "legacy-example",
                    "ui": "legacy",
                    "media_policy": "placeholder",
                    "source_fingerprint": {"templates/index.html": "stale-hash"},
                }
            ),
            encoding="utf-8",
        )
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE))
        self.assertEqual(code, 0, output)
        self.assertIn("predates changes in templates/index.html", output)

        strict_code, _ = self.run_validate(manifest_with(MINIMAL_SCENE), strict=True)
        self.assertEqual(strict_code, 1)

    def test_real_media_capture_is_flagged_for_review(self) -> None:
        self.write_png("docs/ui/assets/legacy-example.png")
        self.write_doc("docs/operator/guide.md", "![Example](../ui/assets/legacy-example.png)\n")
        (self.repo / "docs/ui/assets/legacy-example.png.meta.json").write_text(
            json.dumps({"scene": "legacy-example", "ui": "legacy", "media_policy": "pass",
                        "source_fingerprint": {}}),
            encoding="utf-8",
        )
        code, output = self.run_validate(manifest_with(MINIMAL_SCENE))
        self.assertEqual(code, 0, output)
        self.assertIn("media_policy=pass", output)


class RepoManifestTest(unittest.TestCase):
    def test_committed_manifest_passes_validation(self) -> None:
        manifest = ui_shots.load_manifest()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = ui_shots.validate(manifest)
        self.assertEqual(code, 0, buffer.getvalue())

    def test_every_scene_declares_a_privacy_policy_we_understand(self) -> None:
        manifest = ui_shots.load_manifest()
        defaults = ui_shots.scene_defaults(manifest)
        for scene in manifest["scenes"]:
            policy = ui_shots.resolve(scene, defaults, "media_policy", "placeholder")
            self.assertIn(policy, ui_shots.VALID_MEDIA_POLICIES, scene["id"])

    def test_redaction_css_cannot_be_emptied_by_a_scene(self) -> None:
        """A scene may add selectors; it must not be able to drop the defaults."""
        manifest = ui_shots.load_manifest()
        defaults = ui_shots.scene_defaults(manifest)
        scene = {"id": "x", "redact_extra": ["#custom"]}
        css = ui_shots.redaction_css(scene, defaults, "placeholder")
        for selector in defaults["redact_always"]:
            self.assertIn(selector, css)
        self.assertIn("#custom", css)
        self.assertIn("blur(", css)

    def test_pass_policy_disables_redaction_only_deliberately(self) -> None:
        manifest = ui_shots.load_manifest()
        defaults = ui_shots.scene_defaults(manifest)
        self.assertEqual(ui_shots.redaction_css({"id": "x"}, defaults, "pass"), "")


class SelectorLintTest(unittest.TestCase):
    def test_tokens_ignore_has_text_literals(self) -> None:
        tokens = ui_shots.selector_tokens(".menu-item:has-text('Archive · #12')")
        self.assertEqual(tokens, ["menu-item"])

    def test_tokens_cover_ids_and_nested_selectors(self) -> None:
        tokens = ui_shots.selector_tokens("#eva-main-menu .menu-item.on")
        self.assertEqual(tokens, ["eva-main-menu", "menu-item", "on"])

    def test_unknown_class_is_warned(self) -> None:
        manifest = manifest_with({
            **MINIMAL_SCENE,
            "ui": "react",
            "steps": [{"click": ".menu-ear"}, {"wait_for": ".menu-eear"}],
        })
        problems = ui_shots.Problems()
        with patch.object(ui_shots, "ui_source_text", return_value="className=\"menu-ear\""):
            ui_shots.lint_selectors(manifest, problems)
        self.assertEqual(len(problems.warnings), 1)
        self.assertIn("menu-eear", problems.warnings[0])
        self.assertNotIn("menu-ear,", problems.warnings[0])

    def test_known_classes_are_silent(self) -> None:
        manifest = manifest_with({**MINIMAL_SCENE, "ui": "react", "steps": [{"click": ".menu-ear"}]})
        problems = ui_shots.Problems()
        with patch.object(ui_shots, "ui_source_text", return_value="className=\"menu-ear\""):
            ui_shots.lint_selectors(manifest, problems)
        self.assertEqual(problems.warnings, [])

    def test_committed_manifest_selectors_exist_in_the_react_sources(self) -> None:
        """Renaming a class in the React UI must show up here, not on capture day."""
        manifest = ui_shots.load_manifest()
        problems = ui_shots.Problems()
        ui_shots.lint_selectors(manifest, problems)
        self.assertEqual(problems.warnings, [])


class ComparePngTest(unittest.TestCase):
    """--check must survive antialiasing noise but still notice a real UI change."""

    def setUp(self) -> None:
        try:
            from PIL import Image  # noqa: F401
        except ImportError:  # pragma: no cover - pillow is a runtime dependency
            self.skipTest("pillow not installed")
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name: str, mutate=None):
        from PIL import Image

        image = Image.new("RGB", (100, 100), (20, 24, 30))
        if mutate is not None:
            mutate(image)
        path = self.tmp / name
        image.save(path)
        return path

    def test_identical_images_report_no_change(self) -> None:
        first = self.write("a.png")
        second = self.write("b.png")
        fraction, summary = ui_shots.compare_png(first, second)
        self.assertEqual(fraction, 0.0, summary)

    def test_single_pixel_noise_stays_under_default_tolerance(self) -> None:
        first = self.write("a.png")
        second = self.write("b.png", lambda img: img.putpixel((10, 10), (255, 255, 255)))
        fraction, _ = ui_shots.compare_png(first, second)
        self.assertGreater(fraction, 0.0)
        self.assertLess(fraction, 0.002)

    def test_large_change_is_reported(self) -> None:
        from PIL import ImageDraw

        first = self.write("a.png")
        second = self.write("b.png", lambda img: ImageDraw.Draw(img).rectangle([0, 0, 99, 49], fill=(200, 30, 30)))
        fraction, summary = ui_shots.compare_png(first, second)
        self.assertGreater(fraction, 0.4, summary)

    def test_size_change_is_a_full_difference(self) -> None:
        from PIL import Image

        first = self.write("a.png")
        second_path = self.tmp / "c.png"
        Image.new("RGB", (120, 100), (20, 24, 30)).save(second_path)
        fraction, summary = ui_shots.compare_png(first, second_path)
        self.assertEqual(fraction, 1.0)
        self.assertIn("size", summary)


class CaptureGuardTest(unittest.TestCase):
    def test_pass_policy_requires_explicit_flag(self) -> None:
        scene = copy.deepcopy(MINIMAL_SCENE)
        scene["media_policy"] = "pass"
        manifest = manifest_with(scene)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = ui_shots.capture(
                manifest, scene_ids=[], base_url="http://127.0.0.1:1", out_dir=None,
                check=False, allow_real_media=False, verify_tls=False, headed=False, timeout_ms=1000,
            )
        self.assertEqual(code, 2)


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return False
    try:
        from playwright.sync_api import sync_playwright as sp

        with sp() as playwright:
            browser = playwright.chromium.launch()
            browser.close()
        return True
    except Exception:  # noqa: BLE001 - no browser build, no sandbox, headless shell missing
        return False


FIXTURE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>fixture</title>
<style>
 body { margin:0; background:#12151b; color:#dfe4ec; font-family:sans-serif; }
 #panel { width:520px; padding:24px; }
 #hidden-tab { display:none; }
 #hidden-tab.on { display:block; }
</style></head>
<body>
  <div id="panel">
    <h1 id="heading">EVA fixture</h1>
    <button id="open-tab" onclick="document.getElementById('hidden-tab').classList.add('on')">Open</button>
    <div id="hidden-tab"><p id="tab-body">TAB CONTENT</p></div>
    <span id="stamp">2026-01-01 00:00:00</span>
    <img id="evidence" src="/detections/thumbnail/1" width="200" height="120" alt="evidence">
    <video id="clip" width="200" height="120"></video>
    <div id="seeded"></div>
  </div>
  <script>
    try {
      if (window.localStorage.getItem('eva.test.seed') === 'ok') {
        document.getElementById('seeded').innerHTML = '<span id="seed-ok">seeded</span>';
      }
    } catch (e) { /* seeding failure must surface as a missing #seed-ok */ }
  </script>
</body></html>
"""

# A red square: if evidence interception ever breaks, the real bytes are what the
# screenshot would contain, and the request shows up in the server's access log.
REAL_EVIDENCE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
    "7753de0000000c4944415408d763f8cfc000000301010018dd8db00000000049454e44ae426082"
)


class _FixtureServer:
    """Serves the fixture page over http and records every request path."""

    def __init__(self, directory: Path) -> None:
        import http.server
        import threading

        requests_seen: List[str] = []
        self.requests = requests_seen

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(directory), **kwargs)

            def log_message(self, *args):  # keep test output quiet
                pass

            def do_GET(self):  # noqa: N802 - stdlib naming
                requests_seen.append(self.path)
                if self.path.startswith("/detections/"):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(REAL_EVIDENCE_PNG)))
                    self.end_headers()
                    self.wfile.write(REAL_EVIDENCE_PNG)
                    return
                super().do_GET()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@unittest.skipUnless(_browser_available(), "playwright chromium not installed (requirements-docs.txt)")
class BrowserCaptureTest(unittest.TestCase):
    """End-to-end: steps run, evidence never loads, and --check detects a UI change."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.page_path = self.tmp / "fixture.html"
        self.page_path.write_text(FIXTURE_PAGE, encoding="utf-8")
        self.out_dir = self.tmp / "out"
        self.out_dir.mkdir()
        self.server = _FixtureServer(self.tmp)
        self.addCleanup(self.server.stop)

        self.scene = {
            "id": "fixture-scene",
            "ui": "legacy",
            "auth": "none",
            "title": "Fixture",
            "caption": "Fixture page used by the harness self-test.",
            "path": "/fixture.html",
            "steps": [
                {"wait_for": "#open-tab"},
                {"click": "#open-tab"},
                {"wait_for": "#tab-body"},
                {"wait_for": "#seed-ok"},
                {"set_text": {"selector": "#stamp", "value": "2026-08-11 12:00:00"}},
                {"wait_ms": 50},
            ],
            "clip": "#panel",
            "output": "docs/ui/assets/fixture-scene.png",
            "used_by": [],
        }
        self.manifest = manifest_with(self.scene)
        self.manifest["defaults"]["viewport"] = {"width": 900, "height": 600}
        self.manifest["defaults"]["settle_ms"] = 50
        self.manifest["defaults"]["local_storage"] = {"eva.test.seed": "ok"}

    def run_capture(self, check: bool = False) -> int:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = ui_shots.capture(
                self.manifest, scene_ids=[], base_url=self.server.base_url, out_dir=self.out_dir,
                check=check, allow_real_media=False, verify_tls=False, headed=False, timeout_ms=10000,
            )
        self.last_output = buffer.getvalue()
        return code

    def test_capture_then_check_detects_a_ui_change(self) -> None:
        self.assertEqual(self.run_capture(), 0, getattr(self, "last_output", ""))
        shot = self.out_dir / "fixture-scene.png"
        self.assertTrue(shot.is_file())
        self.assertTrue(shot.stat().st_size > 500)
        self.assertEqual(shot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        # Unchanged page -> --check stays green.
        self.assertEqual(self.run_capture(check=True), 0, getattr(self, "last_output", ""))

        # Changed page -> --check reports the screenshot as stale instead of overwriting it.
        self.page_path.write_text(
            FIXTURE_PAGE.replace("EVA fixture", "EVA fixture (renamed heading)"), encoding="utf-8"
        )
        self.assertEqual(self.run_capture(check=True), 1)
        self.assertIn("STALE", self.last_output)

    def test_evidence_images_never_reach_the_browser(self) -> None:
        """media_policy=placeholder must intercept before the request is made."""
        self.assertEqual(self.run_capture(), 0, getattr(self, "last_output", ""))
        self.assertIn("/fixture.html", self.server.requests)
        evidence = [path for path in self.server.requests if path.startswith("/detections/")]
        self.assertEqual(evidence, [], f"evidence endpoint was fetched: {evidence}")

    def test_pass_policy_does_fetch_media_so_the_guard_matters(self) -> None:
        """Counterpart to the test above: without interception the bytes really do load."""
        self.manifest["defaults"]["media_policy"] = "pass"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = ui_shots.capture(
                self.manifest, scene_ids=[], base_url=self.server.base_url, out_dir=self.out_dir,
                check=False, allow_real_media=True, verify_tls=False, headed=False, timeout_ms=10000,
            )
        self.assertEqual(code, 0, buffer.getvalue())
        evidence = [path for path in self.server.requests if path.startswith("/detections/")]
        self.assertTrue(evidence, "fixture never requested the evidence endpoint at all")

    def test_capture_writes_no_sidecar_for_out_dir_runs(self) -> None:
        self.assertEqual(self.run_capture(), 0, getattr(self, "last_output", ""))
        self.assertFalse((self.out_dir / "fixture-scene.png.meta.json").exists())


if __name__ == "__main__":
    unittest.main()
