import unittest
from pathlib import Path

from agent import AgentTools
from agent_help_index import (
    HelpIndex,
    build_help_response,
    get_help_index,
    partition_by_permission,
)

REPO = Path(__file__).resolve().parent.parent
_ALLOWED_PREFIXES = (
    "docs/operator/",
    "docs/admin/",
    "docs/install/",
    "docs/00_CANON/",
)
_BACKUP_QUERY = "how to backup the database before an update"


def _make_tools() -> AgentTools:
    # _lookup_help only needs self._local + the help index; deps are unused.
    return AgentTools(
        detections_store=None,
        probes_store=None,
        luxriot_manager=None,
        embed_text_fn=None,
        embed_image_fn=None,
        call_lm_fn=None,
        encode_jpeg_fn=None,
        search_indexed_folder_fn=None,
        search_detections_fn=None,
    )


class HelpIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = HelpIndex(REPO)

    def test_operator_query_finds_operator_section(self) -> None:
        res = self.index.query(
            "how do I ask the agent what happened on a channel over a period", pool=10
        )
        self.assertTrue(res)
        docs = {r["doc"] for r in res}
        self.assertTrue(
            any(d.startswith("docs/operator/") or d.endswith("glossary.md") for d in docs),
            f"expected an operator doc in {docs}",
        )

    def test_admin_query_can_retrieve_admin_section(self) -> None:
        res = self.index.query(
            "create a user reset a password assign channel grants disable account", pool=20
        )
        admin = [r for r in res if r["doc"] == "docs/admin/admin_guide.md"]
        self.assertTrue(admin, "admin_guide should be retrievable from the index")
        self.assertEqual(admin[0]["required_permission"], "users:manage")

    def test_operator_cannot_see_admin_procedure(self) -> None:
        res = self.index.query(
            "create a user reset a password assign channel grants disable account", pool=20
        )
        results, restricted = partition_by_permission(res, granted=set(), top_k=5)
        self.assertFalse(
            any(r["doc"] == "docs/admin/admin_guide.md" for r in results),
            "operator must not get admin procedure passages",
        )
        self.assertTrue(
            any(rm.get("required_permission") == "users:manage" for rm in restricted),
            "admin match should appear as a restricted redirect",
        )
        for rm in restricted:  # redacted: section/heading/permission/score only, no procedure text
            self.assertNotIn("snippet", rm)
            self.assertIn("score", rm)

        admin_results, _ = partition_by_permission(res, granted={"users:manage"}, top_k=5)
        self.assertTrue(
            any(r["doc"] == "docs/admin/admin_guide.md" for r in admin_results),
            "a user with users:manage should get the admin passage",
        )

    def test_backup_best_match_restricted_for_operator(self) -> None:
        res = self.index.query(_BACKUP_QUERY, pool=20)
        resp = build_help_response(_BACKUP_QUERY, res, granted=set(), top_k=3)
        self.assertTrue(resp["best_match_restricted"], "backup help best match is admin-only")
        self.assertEqual(resp["best_required_permission"], "settings:manage")
        self.assertFalse(
            any(r["doc"] == "docs/admin/backup_recovery.md" for r in resp["results"]),
            "no backup procedure snippet may leak to an operator",
        )

    def test_backup_visible_with_settings_manage(self) -> None:
        res = self.index.query(_BACKUP_QUERY, pool=20)
        resp = build_help_response(_BACKUP_QUERY, res, granted={"settings:manage"}, top_k=3)
        self.assertFalse(resp["best_match_restricted"])
        self.assertTrue(
            any(r["doc"] == "docs/admin/backup_recovery.md" for r in resp["results"]),
            "settings:manage should get the backup procedure",
        )

    def test_normal_ui_help_not_restricted(self) -> None:
        res = self.index.query("how do I search the archive for a fight", pool=20)
        resp = build_help_response("how do I search the archive for a fight", res, granted=set(), top_k=3)
        self.assertFalse(resp["best_match_restricted"])
        self.assertTrue(resp["results"], "operator UI help should return allowed passages")

    def test_only_allowlisted_docs_indexed(self) -> None:
        for doc in self.index.indexed_docs:
            ok = doc == "docs/known_limitations.md" or any(
                doc.startswith(p) for p in _ALLOWED_PREFIXES
            )
            self.assertTrue(ok, f"unexpected indexed doc: {doc}")
            for forbidden in ("readiness/history", "docs/gtm", "docs/legal", "field_rollout", ".env"):
                self.assertNotIn(forbidden, doc)

    def test_deterministic_rebuild(self) -> None:
        a = HelpIndex(REPO)
        b = HelpIndex(REPO)
        self.assertEqual(a.indexed_docs, b.indexed_docs)
        q = "how do I run the demo and search the archive"
        self.assertEqual(
            [(r["doc"], r["heading"]) for r in a.query(q, pool=8)],
            [(r["doc"], r["heading"]) for r in b.query(q, pool=8)],
        )

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.index.query("", pool=5), [])

    def test_singleton_builds(self) -> None:
        self.assertTrue(get_help_index().indexed_docs)


class LookupHelpLegacyPathTest(unittest.TestCase):
    """The legacy/non-secure tool path must not trust model-supplied permissions."""

    def test_legacy_lookup_help_ignores_model_supplied_permissions(self) -> None:
        tools = _make_tools()
        out = tools._lookup_help(
            {
                "query": _BACKUP_QUERY,
                "_granted_permissions": ["settings:manage", "users:manage"],
            }
        )
        # No trusted context set -> operator-only regardless of args.
        self.assertTrue(out["best_match_restricted"])
        self.assertFalse(
            any(r["doc"].startswith("docs/admin/") for r in out["results"]),
            "model-supplied permissions must not unlock admin passages",
        )

    def test_legacy_lookup_help_respects_trusted_thread_local(self) -> None:
        tools = _make_tools()
        tools._set_trusted_permissions(["settings:manage"])
        try:
            out = tools._lookup_help({"query": _BACKUP_QUERY})
        finally:
            tools._clear_trusted_permissions()
        self.assertFalse(out["best_match_restricted"])
        self.assertTrue(
            any(r["doc"] == "docs/admin/backup_recovery.md" for r in out["results"]),
            "trusted settings:manage should unlock the backup procedure",
        )


if __name__ == "__main__":
    unittest.main()
