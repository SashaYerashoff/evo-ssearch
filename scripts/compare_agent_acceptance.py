#!/usr/bin/env python3
"""Compare supervised EVA agent acceptance reports across inference models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print model-level and scenario-level deltas from live agent acceptance JSON reports."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    return parser


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _model_name(report: Mapping[str, Any], path: Path) -> str:
    runtime = _object(report.get("runtime_before"))
    raw = _object(runtime.get("agent_config"))
    candidates = [
        raw,
        _object(raw.get("config")),
        _object(raw.get("agent")),
        _object(_object(raw.get("config")).get("agent")),
    ]
    for row in candidates:
        for key in (
            "resolved_model",
            "default_resolved_model",
            "lm_model",
            "model",
            "default_model",
        ):
            value = str(row.get(key) or "").strip()
            if value:
                return value
    return path.stem


def _cell(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    reports: list[tuple[Path, Mapping[str, Any]]] = []
    for path in _parser().parse_args().reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise SystemExit(f"{path}: expected a JSON object")
        reports.append((path, payload))

    print("# EVA agent model comparison\n")
    print("| Model / report | Pass | p50 s | p95 s | Quality | Efficiency | Tools | LM calls | Max queue |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for path, report in reports:
        summary = _object(report.get("summary"))
        latency = _object(summary.get("latency_seconds"))
        passed = f"{summary.get('passed', 0)}/{summary.get('executed', 0)}"
        print(
            "| {model} (`{file}`) | {passed} | {p50} | {p95} | {quality} | {efficiency} | {tools} | {lm} | {queue} |".format(
                model=_model_name(report, path),
                file=path.name,
                passed=passed,
                p50=_cell(latency.get("median"), digits=3),
                p95=_cell(latency.get("p95"), digits=3),
                quality=_cell(summary.get("generation_quality_average")),
                efficiency=_cell(summary.get("tool_efficiency_average")),
                tools=_cell(summary.get("tool_calls_total")),
                lm=_cell(summary.get("agent_lm_admissions_total")),
                queue=_cell(summary.get("max_sampled_lm_queue")),
            )
        )

    scenario_names = sorted({
        str(row.get("name") or "")
        for _path, report in reports
        for row in (report.get("scenarios") or [])
        if isinstance(row, Mapping) and row.get("status") != "skipped"
    })
    print("\n## Per scenario\n")
    print("| Scenario | " + " | ".join(_model_name(report, path) for path, report in reports) + " |")
    print("|---|" + "---:|" * len(reports))
    for scenario_name in scenario_names:
        cells: list[str] = []
        for _path, report in reports:
            row = next(
                (
                    candidate for candidate in (report.get("scenarios") or [])
                    if isinstance(candidate, Mapping) and candidate.get("name") == scenario_name
                ),
                None,
            )
            if not isinstance(row, Mapping) or row.get("status") == "skipped":
                cells.append("—")
                continue
            quality = _object(row.get("generation_quality")).get("score")
            efficiency = _object(row.get("tool_efficiency")).get("score")
            cells.append(
                f"{row.get('status')} · {_cell(row.get('elapsed_seconds'), digits=2)}s · Q{_cell(quality)} / E{_cell(efficiency)}"
            )
        print(f"| `{scenario_name}` | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
