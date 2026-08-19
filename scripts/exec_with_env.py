#!/usr/bin/env python3
"""Safely exec a command with a dotenv file as its canonical environment.

Unlike sourcing the file in a shell, parsing through python-dotenv cannot turn
an operator-edited setting into shell code.  Existing process values are
deliberately replaced so an ARM container restart observes the same file that
the Settings UI updates.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import dotenv_values


ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def configured_environment(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"configuration file is missing: {path}")
    values = dotenv_values(path)
    invalid = sorted(str(key) for key in values if not ENV_KEY.fullmatch(str(key)))
    if invalid:
        raise ValueError("configuration contains invalid variable names")
    environment = dict(os.environ)
    for key, value in values.items():
        environment[str(key)] = "" if value is None else str(value)
    environment["EVOSSEARCH_CONFIG_ENV_FILE"] = str(path)
    return environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    try:
        environment = configured_environment(args.env_file)
    except (OSError, ValueError) as exc:
        print(f"EVA configuration load failed: {exc}", file=sys.stderr)
        return 2
    os.execvpe(command[0], command, environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
