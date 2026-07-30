#!/usr/bin/env python3
"""Wait until an OpenAI-compatible endpoint exposes the configured model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def wait(base_url: str, model: str, timeout_sec: int) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_ids = {
                str(item.get("id") or "")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            }
            if model in model_ids:
                return True, model
            last_error = f"configured model is absent; available={sorted(model_ids)!r}"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(3)
    return False, last_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-prefix",
        default="EVOSSEARCH_LM_PROFILE_VLM",
        help="Environment prefix containing BASE_URL and MODEL.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    base_url = str(os.getenv(f"{args.env_prefix}_BASE_URL") or "").strip()
    model = str(os.getenv(f"{args.env_prefix}_MODEL") or "").strip()
    if not base_url or not model:
        print(
            f"{args.env_prefix}_BASE_URL and {args.env_prefix}_MODEL are required.",
            file=sys.stderr,
        )
        return 2
    ok, detail = wait(base_url, model, max(1, args.timeout))
    if not ok:
        print(f"Inference endpoint is not ready: {detail}", file=sys.stderr)
        return 1
    print(f"Inference endpoint ready: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
