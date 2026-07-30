from pathlib import Path
from unittest.mock import patch

from oldapp import app, config


def test_react_ui_can_be_piloted_without_changing_default(tmp_path: Path):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><script type="module" src="/ui-assets/assets/app.js"></script>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("window.__evaReact = true", encoding="utf-8")

    with patch("oldapp._REACT_UI_DIST", dist), app.test_client() as client:
        index = client.get("/?ui=react")
        asset = client.get("/ui-assets/assets/app.js")

    assert index.status_code == 200
    assert index.headers["X-EVA-UI"] == "react"
    assert b"/ui-assets/assets/app.js" in index.data
    assert asset.status_code == 200
    assert asset.headers["Cache-Control"] == "public, max-age=31536000, immutable"
    assert asset.data == b"window.__evaReact = true"


def test_missing_react_build_fails_back_to_legacy(tmp_path: Path):
    with patch("oldapp._REACT_UI_DIST", tmp_path / "missing"), app.test_client() as client:
        response = client.get("/?ui=react")

    assert response.status_code == 200
    assert response.headers["X-EVA-UI"] == "legacy"
    assert response.headers["X-EVA-UI-Fallback"] == "react-dist-missing"


def test_invalid_ui_override_uses_configured_mode(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>React</title>", encoding="utf-8")

    with (
        patch("oldapp._REACT_UI_DIST", dist),
        patch.object(config, "UI_MODE", "react"),
        app.test_client() as client,
    ):
        response = client.get("/?ui=not-a-shell")

    assert response.status_code == 200
    assert response.headers["X-EVA-UI"] == "react"
