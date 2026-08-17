from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "database_preservation_guard_test",
    ROOT / "scripts" / "database_preservation_guard.py",
)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


class _Cursor:
    """Answer only the catalogue probes the visibility gate issues."""

    def __init__(self, *, can_read_unfiltered: bool, force_rls: list[tuple[str, str]]) -> None:
        self._can_read_unfiltered = can_read_unfiltered
        self._force_rls = force_rls
        self._rows: list[tuple[object, ...]] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, statement: str, params: object = None) -> None:
        if not isinstance(statement, str):
            if not self._can_read_unfiltered:
                raise PermissionError("query would be affected by row-level security")
            self._rows = [(1,)]
        elif "relforcerowsecurity" in statement:
            self._rows = list(self._force_rls)
        elif "SET LOCAL row_security = off" in statement:
            self._rows = []
        else:  # pragma: no cover - the gate must run before anything else
            raise AssertionError(f"unexpected query before visibility gate: {statement}")

    def fetchone(self) -> tuple[object, ...]:
        return self._rows[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


class _Connection:
    def __init__(
        self,
        *,
        can_read_unfiltered: bool,
        force_rls: list[tuple[str, str]],
    ) -> None:
        self._can_read_unfiltered = can_read_unfiltered
        self._force_rls = force_rls

    def cursor(self, name: str | None = None) -> _Cursor:
        return _Cursor(
            can_read_unfiltered=self._can_read_unfiltered,
            force_rls=self._force_rls,
        )


PROTECTED = [("archive", "detections"), ("iam", "users"), ("archive", "probes")]


def test_guard_refuses_when_forced_rls_hides_every_protected_row():
    connection = _Connection(can_read_unfiltered=False, force_rls=PROTECTED)

    with pytest.raises(guard.PreservationError, match="FORCE ROW LEVEL SECURITY"):
        guard.assert_row_visibility(connection)


def test_capture_refuses_before_reading_anything_when_blind():
    connection = _Connection(can_read_unfiltered=False, force_rls=PROTECTED)

    # _Cursor raises AssertionError on any non-catalogue query, so this proves
    # the gate runs before a single count(*) is issued.
    with pytest.raises(guard.PreservationError):
        guard.capture(connection)


def test_guard_allows_an_identity_with_effective_unfiltered_access():
    connection = _Connection(can_read_unfiltered=True, force_rls=PROTECTED)

    guard.assert_row_visibility(connection)


def test_guard_allows_a_database_without_forced_row_security():
    connection = _Connection(can_read_unfiltered=False, force_rls=[])

    guard.assert_row_visibility(connection)


def test_all_zero_manifests_compare_clean_which_is_why_the_gate_exists():
    """An RLS-blind capture is self-consistent, so comparison cannot catch it.

    Both sides read as empty, ``compare_manifests`` reports no errors and the
    caller prints a preservation guarantee nothing verified.  The visibility
    gate is the only place this can be detected.
    """

    blind = {
        "manifest_version": guard.MANIFEST_VERSION,
        "database": "eva",
        "tables": {
            "archive.detections": {"row_count": 0, "key_hashes": [], "columns": []},
            "iam.users": {"row_count": 0, "key_hashes": [], "columns": []},
        },
    }

    assert guard.compare_manifests(blind, blind) == []
