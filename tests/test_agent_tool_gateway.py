import json
import unittest
from datetime import datetime, timedelta, timezone

from agent_security import (
    ApprovalArgumentMismatchError,
    ApprovalConsumedError,
    ApprovalError,
    ApprovalExpiredError,
    ApprovalRequiredError,
    AuditUnavailableError,
    ChannelAccessDeniedError,
    ContextInjectionError,
    InvalidToolArgumentsError,
    PermissionDeniedError,
    ToolExecutionContext,
    ToolGateway,
    ToolPolicy,
    ToolRegistry,
    ToolRisk,
)
from agent_security.eva_adapter import EvaAgentToolAdapter
from agent_security.output import sanitize_output


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class ToolGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.audit_events = []
        self.calls = []
        self.registry = ToolRegistry()
        self.context = ToolExecutionContext(
            actor_id="operator-1",
            tenant_id="tenant-1",
            roles={"operator"},
            permissions={"detections.read", "probe.write"},
            allowed_channels={"channel-1", "channel-2"},
            session_id="session-1",
            request_id="request-1",
            client_metadata={"client_ip": "192.0.2.10"},
        )

        self.registry.register(
            "search",
            self._record_call,
            ToolPolicy(
                required_permission="detections.read",
                allowed_arguments=frozenset(
                    {
                        "channel_id",
                        "channel_ids",
                        "limit",
                        "start_time",
                        "end_time",
                    }
                ),
                channel_required=True,
                max_rows=50,
                default_rows=20,
                max_time_window=timedelta(hours=1),
                default_time_window=timedelta(minutes=30),
                max_output_bytes=512,
                max_output_string_chars=100,
            ),
        )
        self.registry.register(
            "update_probe",
            self._record_call,
            ToolPolicy(
                required_permission="probe.write",
                risk=ToolRisk.WRITE,
                approval_required=True,
                allowed_arguments=frozenset(
                    {"channel_id", "probe_id", "threshold", "preview"}
                ),
                required_arguments=frozenset(
                    {"channel_id", "probe_id", "threshold"}
                ),
                channel_required=True,
                approval_ttl_seconds=30,
                plan_ttl_seconds=60,
            ),
        )
        self.registry.register(
            "delete_probe",
            self._record_call,
            ToolPolicy(
                required_permission="probe.write",
                risk=ToolRisk.WRITE,
                approval_required=True,
                allowed_arguments=frozenset({"channel_id", "probe_id"}),
                required_arguments=frozenset({"channel_id", "probe_id"}),
                channel_required=True,
            ),
        )
        self.registry.register(
            "apply_setting",
            self._record_call,
            ToolPolicy(
                required_permission="probe.write",
                risk=ToolRisk.WRITE,
                approval_required=True,
                approval_required_when=lambda arguments: arguments.get(
                    "preview",
                    True,
                )
                is not True,
                allowed_arguments=frozenset({"channel_id", "value", "preview"}),
                required_arguments=frozenset({"channel_id", "value"}),
                channel_required=True,
            ),
        )
        self.gateway = ToolGateway(
            self.registry,
            audit_callback=self.audit_events.append,
            clock=self.clock,
        )

    def tearDown(self) -> None:
        self.gateway.close()

    def _record_call(self, context, arguments):
        call = {"context": context, "arguments": arguments}
        self.calls.append(call)
        return call

    def test_denies_missing_permission(self) -> None:
        context = ToolExecutionContext(
            actor_id="viewer-1",
            tenant_id="tenant-1",
            permissions=set(),
            allowed_channels={"channel-1"},
        )

        with self.assertRaises(PermissionDeniedError):
            self.gateway.execute(
                "search",
                {"channel_id": "channel-1"},
                context,
            )

        self.assertEqual(self.calls, [])

    def test_denies_forbidden_singular_channel(self) -> None:
        with self.assertRaises(ChannelAccessDeniedError):
            self.gateway.execute(
                "search",
                {"channel_id": "channel-99"},
                self.context,
            )

        self.assertEqual(self.calls, [])

    def test_denies_list_if_any_channel_is_forbidden(self) -> None:
        with self.assertRaises(ChannelAccessDeniedError):
            self.gateway.execute(
                "search",
                {"channel_ids": ["channel-1", "channel-99"]},
                self.context,
            )

        self.assertEqual(self.calls, [])

    def test_rejects_context_injection_even_when_nested(self) -> None:
        with self.assertRaises(ContextInjectionError):
            self.gateway.execute(
                "search",
                {
                    "channel_id": "channel-1",
                    "filters": {"actor_id": "admin"},
                },
                self.context,
            )

        self.assertEqual(self.calls, [])

    def test_rejects_unknown_arguments_under_closed_policy(self) -> None:
        with self.assertRaises(InvalidToolArgumentsError):
            self.gateway.execute(
                "search",
                {"channel_id": "channel-1", "unbounded": True},
                self.context,
            )

    def test_normalizes_row_and_time_window_limits(self) -> None:
        result = self.gateway.execute(
            "search",
            {
                "channel_id": "channel-1",
                "limit": "500",
                "start_time": "2026-06-09T00:00:00Z",
                "end_time": "2026-06-09T12:00:00Z",
            },
            self.context,
        )

        arguments = self.calls[-1]["arguments"]
        self.assertEqual(arguments["limit"], 50)
        self.assertEqual(arguments["start_time"], "2026-06-09T11:00:00.000000Z")
        self.assertEqual(arguments["end_time"], "2026-06-09T12:00:00.000000Z")
        self.assertEqual(result["arguments"]["limit"], 50)

    def test_redacts_and_bounds_tool_output(self) -> None:
        registry = ToolRegistry()
        registry.register(
            "secrets",
            lambda _context, _arguments: {
                "approval_id": "one-time-bearer-value",
                "password": "not-for-model",
                "text": "x" * 500,
            },
            ToolPolicy(
                required_permission="detections.read",
                allowed_arguments=frozenset(),
                max_output_bytes=120,
                max_output_string_chars=500,
            ),
        )
        gateway = ToolGateway(registry, clock=self.clock)
        self.addCleanup(gateway.close)

        result = gateway.execute("secrets", {}, self.context)
        encoded = json.dumps(result, separators=(",", ":")).encode("utf-8")

        self.assertLessEqual(len(encoded), 120)
        self.assertNotIn("not-for-model", encoded.decode("utf-8"))
        self.assertNotIn("one-time-bearer-value", encoded.decode("utf-8"))

    def test_preview_false_is_not_approval(self) -> None:
        with self.assertRaises(ApprovalRequiredError):
            self.gateway.execute(
                "update_probe",
                {
                    "channel_id": "channel-1",
                    "probe_id": "probe-1",
                    "threshold": 0.8,
                    "preview": False,
                },
                self.context,
            )

        self.assertEqual(self.calls, [])

    def test_conditional_approval_allows_preview_and_requires_apply(self) -> None:
        preview = self.gateway.execute(
            "apply_setting",
            {
                "channel_id": "channel-1",
                "value": "quiet",
                "preview": True,
            },
            self.context,
        )

        self.assertEqual(preview["arguments"]["preview"], True)
        with self.assertRaises(ApprovalRequiredError):
            self.gateway.execute(
                "apply_setting",
                {
                    "channel_id": "channel-1",
                    "value": "quiet",
                    "preview": False,
                },
                self.context,
            )
        with self.assertRaises(ApprovalError):
            self.gateway.create_plan(
                "apply_setting",
                {
                    "channel_id": "channel-1",
                    "value": "quiet",
                    "preview": True,
                },
                self.context,
            )
        plan = self.gateway.create_plan(
            "apply_setting",
            {
                "channel_id": "channel-1",
                "value": "quiet",
                "preview": False,
            },
            self.context,
        )
        approval = self.gateway.approve(plan.plan_id, self.context)
        result = self.gateway.execute(
            "apply_setting",
            None,
            self.context,
            approval_id=approval.approval_id,
        )

        self.assertEqual(result["arguments"]["preview"], False)

    def test_approval_is_one_time_and_executes_stored_arguments(self) -> None:
        arguments = {
            "channel_id": "channel-1",
            "probe_id": "probe-1",
            "threshold": 0.8,
            "preview": True,
        }
        plan = self.gateway.create_plan(
            "update_probe", arguments, self.context
        )
        approval = self.gateway.approve(plan.plan_id, self.context)

        result = self.gateway.execute(
            "update_probe",
            None,
            self.context,
            approval_id=approval.approval_id,
        )

        self.assertEqual(result["arguments"], arguments)
        with self.assertRaises(ApprovalConsumedError):
            self.gateway.execute(
                "update_probe",
                None,
                self.context,
                approval_id=approval.approval_id,
            )
        self.assertEqual(len(self.calls), 1)

    def test_changed_arguments_cannot_execute_approved_plan(self) -> None:
        plan = self.gateway.create_plan(
            "update_probe",
            {
                "channel_id": "channel-1",
                "probe_id": "probe-1",
                "threshold": 0.8,
                "preview": True,
            },
            self.context,
        )
        approval = self.gateway.approve(plan.plan_id, self.context)

        with self.assertRaises(ApprovalArgumentMismatchError):
            self.gateway.execute(
                "update_probe",
                {
                    "channel_id": "channel-1",
                    "probe_id": "probe-1",
                    "threshold": 0.1,
                    "preview": False,
                },
                self.context,
                approval_id=approval.approval_id,
            )

        self.assertEqual(self.calls, [])
        result = self.gateway.execute(
            "update_probe",
            None,
            self.context,
            approval_id=approval.approval_id,
        )
        self.assertEqual(result["arguments"]["threshold"], 0.8)

    def test_approval_is_actor_and_action_bound(self) -> None:
        plan = self.gateway.create_plan(
            "update_probe",
            {
                "channel_id": "channel-1",
                "probe_id": "probe-1",
                "threshold": 0.8,
            },
            self.context,
        )
        approval = self.gateway.approve(plan.plan_id, self.context)
        other_actor = ToolExecutionContext(
            actor_id="operator-2",
            tenant_id="tenant-1",
            permissions={"probe.write"},
            allowed_channels={"channel-1"},
        )

        with self.assertRaises(ApprovalError):
            self.gateway.execute(
                "delete_probe",
                None,
                self.context,
                approval_id=approval.approval_id,
            )
        with self.assertRaises(ApprovalError):
            self.gateway.execute(
                "update_probe",
                None,
                other_actor,
                approval_id=approval.approval_id,
            )

    def test_expired_approval_cannot_execute(self) -> None:
        plan = self.gateway.create_plan(
            "update_probe",
            {
                "channel_id": "channel-1",
                "probe_id": "probe-1",
                "threshold": 0.8,
            },
            self.context,
        )
        approval = self.gateway.approve(
            plan.plan_id, self.context, ttl_seconds=5
        )
        self.clock.advance(seconds=6)

        with self.assertRaises(ApprovalExpiredError):
            self.gateway.execute(
                "update_probe",
                None,
                self.context,
                approval_id=approval.approval_id,
            )

        self.assertEqual(self.calls, [])

    def test_denied_call_is_audited(self) -> None:
        with self.assertRaises(ChannelAccessDeniedError):
            self.gateway.execute(
                "search",
                {"channel_id": "channel-99"},
                self.context,
            )

        denied = self.audit_events[-1]
        self.assertEqual(denied.phase, "deny")
        self.assertEqual(denied.operation, "execute")
        self.assertEqual(denied.tool_name, "search")
        self.assertEqual(denied.actor_id, "operator-1")
        self.assertEqual(denied.code, "channel_access_denied")

    def test_success_and_handler_error_have_audit_events(self) -> None:
        self.gateway.execute(
            "search", {"channel_id": "channel-1"}, self.context
        )
        self.assertEqual(
            [
                event.phase
                for event in self.audit_events
                if event.operation == "execute"
            ],
            ["allow", "result"],
        )

        registry = ToolRegistry()

        def fail(_context, _arguments):
            raise RuntimeError("service failed")

        registry.register(
            "fail",
            fail,
            ToolPolicy(
                required_permission="detections.read",
                allowed_arguments=frozenset(),
            ),
        )
        events = []
        gateway = ToolGateway(
            registry, audit_callback=events.append, clock=self.clock
        )
        self.addCleanup(gateway.close)

        with self.assertRaises(RuntimeError):
            gateway.execute("fail", {}, self.context)

        self.assertEqual([event.phase for event in events], ["allow", "error"])

    def test_missing_external_sink_still_leaves_internal_audit_events(self) -> None:
        registry = ToolRegistry()
        registry.register(
            "read",
            self._record_call,
            ToolPolicy(
                required_permission="detections.read",
                allowed_arguments=frozenset(),
            ),
        )
        gateway = ToolGateway(registry, clock=self.clock)
        self.addCleanup(gateway.close)

        gateway.execute("read", {}, self.context)

        self.assertEqual(
            [event.phase for event in gateway.audit_events],
            ["allow", "result"],
        )

    def test_audit_sink_failure_blocks_handler(self) -> None:
        registry = ToolRegistry()
        registry.register(
            "read",
            self._record_call,
            ToolPolicy(
                required_permission="detections.read",
                allowed_arguments=frozenset(),
            ),
        )

        def unavailable(_event):
            raise RuntimeError("audit database unavailable")

        gateway = ToolGateway(
            registry,
            audit_callback=unavailable,
            clock=self.clock,
        )
        self.addCleanup(gateway.close)

        with self.assertRaises(AuditUnavailableError):
            gateway.execute("read", {}, self.context)

        self.assertEqual(self.calls, [])
        self.assertEqual(gateway.audit_events[0].phase, "allow")


def _uncompacted_search_archive_row(index: int, *, realistic_size: bool = False) -> dict:
    """Mirror a real, un-compacted vlm_summary archive row (as produced by
    agent._annotate_archive_row / oldapp._build_detection_search_result),
    including a full payload with nested per-frame arrays. Compaction for
    the model (_compact_search_result_for_model) trims this down, but the
    security sanitizer runs on the *raw* result before compaction.

    realistic_size=True pads summary/state text to match measured live
    repro rows (~10.5KB/row with real VLM-written prose), for tests that
    exercise the byte budget specifically rather than the item-count one.
    """
    # Real captured VLM summaries run several sentences per section
    # (scene description, activity description, "worth to remember");
    # a single repeated short phrase understates real row size by ~5-10x.
    summary_text = (
        "### Scene description A person is seated at a desk in a dimly lit "
        "room, facing a large monitor displaying lines of code. ### Activity "
        "description The person remains seated and focused on the monitor "
        "throughout the batch, with only minor shifts in posture. "
        * (30 if realistic_size else 1)
    )
    return {
        "path": None,
        "filename": f"probe · 2026-07-19 20:3{index}:00",
        "similarity": 0.15 + index * 0.001,
        "score": 0.15 + index * 0.001,
        "timestamp_ms": 1_784_493_000_000 + index * 60_000,
        "source": "vlm_summary",
        "source_label": "Video-description frame",
        "archive_item_type": "video_description_frame",
        "probe_id": None,
        "probe_name": "VLM summary frame",
        "channel_id": 112,
        "severity": None,
        "pos_score": 0.15,
        "neg_score": 0.02,
        "margin": 0.13,
        "origin": "vlm_summary",
        "shard_key": "112",
        "search_mode": "clip",
        "dino_fallback": True,
        "detection_id": 3000 + index,
        "id": 3000 + index,
        "image_url": f"/detections/thumbnail/{3000 + index}",
        "payload": {
            "run_id": f"run-{index}",
            "batch_start_ms": 1_784_493_000_000,
            "batch_end_ms": 1_784_493_026_000,
            "frame_index": index,
            "frame_timestamp_ms": 1_784_493_000_000 + index * 2_000,
            "anchor_role": "primary",
            "alert_total": 0,
            "alert_counts": {},
            "summary_excerpt": summary_text,
            "state_observations": [
                {"key": f"obs_{i}", "value": f"val_{i}", "frame": i}
                for i in range(12)
            ],
            "state_transition_events": [
                {"from": "idle", "to": "typing", "frame": i} for i in range(6)
            ],
            "state_transition_total": 6,
            "vector_signal": [0.01 * i for i in range(16)],
        },
    }


class SearchArchiveCoverageBudgetTests(unittest.TestCase):
    """Regression coverage for the search_archive coverage-honesty bug:
    agent_security.output.sanitize_output enforces a shared item budget
    across an entire tool result, walking dict keys in insertion order. A
    12-row uncompacted search_archive result (full vlm_summary payload per
    row) burns through the generic 500-item default before the sanitizer
    reaches a trailing `coverage` key, silently dropping it and setting a
    spurious `_truncated` flag on searches that were not actually
    truncated — breaking the coverage-honesty gate
    (docs/tuktuk/grammar_pin.md) for both the operator UI and the model.
    See docs/tuktuk/grammar_review_questions.md (Resolved,
    "search_archive coverage truncation")."""

    def _build_result(self, n_rows: int = 12, *, coverage_last: bool = False) -> dict:
        coverage = {
            "candidate_limit": 20000,
            "scanned_candidates": 1906,
            "total_candidates": 1906,
            "truncated": False,
            "result_limit": n_rows,
            "source": "vlm_summary",
            "channel_id": 112,
            "must_state_coverage": False,
            "note": "Search ranked the full candidate set for the requested filters.",
        }
        rows = [_uncompacted_search_archive_row(i) for i in range(n_rows)]
        head = {
            "scope": "detections",
            "source": "vlm_summary",
            "source_label": "Video-description frame",
            "count": n_rows,
        }
        # coverage_last=True reproduces agent.py's pre-fix key order
        # (results before coverage); the default reproduces the fix.
        return (
            {**head, "results": rows, "coverage": coverage}
            if coverage_last
            else {**head, "coverage": coverage, "results": rows}
        )

    def test_adapter_budget_is_raised_for_search_archive(self) -> None:
        self.assertEqual(EvaAgentToolAdapter._max_output_items("search_archive"), 50_000)

    def test_old_order_and_default_budget_dropped_coverage(self) -> None:
        # Proves the failure mode this fix addresses: with coverage placed
        # after results (the pre-fix order) at the old shared generic
        # budget, a full page of real-shaped rows exhausts the item budget
        # before the sanitizer reaches the trailing coverage key.
        policy = ToolPolicy(required_permission="detections.read", max_output_items=500)
        sanitized = sanitize_output(self._build_result(coverage_last=True), policy)
        self.assertNotIn("coverage", sanitized)
        self.assertTrue(sanitized.get("_truncated"))

    def test_raised_budget_preserves_coverage_honesty(self) -> None:
        policy = ToolPolicy(
            required_permission="detections.read",
            max_output_items=EvaAgentToolAdapter._max_output_items("search_archive"),
        )
        sanitized = sanitize_output(self._build_result(), policy)
        self.assertIn("coverage", sanitized)
        self.assertEqual(sanitized["coverage"]["total_candidates"], 1906)
        self.assertFalse(sanitized["coverage"]["truncated"])
        self.assertNotIn("_truncated", sanitized)
        self.assertEqual(len(sanitized["results"]), 12)

    def test_search_archive_orders_coverage_before_results(self) -> None:
        # Even at the raised budget, coverage must be ordered before the
        # bulky results list: sanitize_output stops on a shared item
        # counter, so if a future row grows large enough to exhaust even
        # the raised budget, trailing keys are still the ones silently
        # dropped. Exercises the real agent.py AgentTools._search_archive,
        # not a test fixture, so a future reordering trips this test.
        from agent import AgentTools

        def fake_search_detections_fn(*, include_coverage=False, **_kwargs):
            if include_coverage:
                return {
                    "results": [_uncompacted_search_archive_row(0)],
                    "coverage": {"truncated": False, "scanned_candidates": 1},
                }
            return [_uncompacted_search_archive_row(0)]

        tools = AgentTools(
            detections_store=None,
            probes_store=None,
            luxriot_manager=None,
            embed_text_fn=lambda _text: [0.0],
            embed_image_fn=lambda _img: [0.0],
            call_lm_fn=lambda _messages: "",
            encode_jpeg_fn=lambda *_a, **_k: "",
            search_indexed_folder_fn=lambda **_kwargs: [],
            search_detections_fn=fake_search_detections_fn,
        )
        result = tools._search_archive({"query": "person at desk", "scope": "detections"})
        keys = list(result.keys())
        self.assertIn("coverage", keys)
        self.assertLess(keys.index("coverage"), keys.index("results"))


class OutputByteBudgetTests(unittest.TestCase):
    """Regression coverage for the second half of the coverage-truncation
    bug: sanitize_output's byte cap (_bound_serialized) replaces the
    *entire* result with a useless {"_truncated": true, "preview": "..."}
    envelope once serialized size exceeds max_output_bytes — independent
    of, and not helped by, the item-count budget or key ordering. A real
    12-row search_archive page (measured against live repro data with
    real VLM-written summaries) serializes to ~141KB; a real 20-row
    get_detections page measures ~221KB. Both routinely exceeded the old
    flat 96,000-byte default used for every tool. See
    docs/tuktuk/grammar_review_questions.md (Resolved,
    "search_archive coverage truncation")."""

    def _search_archive_result(self, n_rows: int = 12) -> dict:
        return {
            "scope": "detections",
            "source": "vlm_summary",
            "source_label": "Video-description frame",
            "count": n_rows,
            "coverage": {"truncated": False, "scanned_candidates": 1906},
            "results": [
                _uncompacted_search_archive_row(i, realistic_size=True)
                for i in range(n_rows)
            ],
        }

    def test_default_byte_budget_would_have_wiped_a_realistic_page(self) -> None:
        # Reproduces the exact live combination: the item budget is
        # already raised (so it does not trim rows first and mask the
        # byte cap), but max_output_bytes is still the generic default —
        # matching the state search_archive was deployed in before this
        # fix, where the item-count fix alone was not enough.
        result = self._search_archive_result(12)
        raw_bytes = len(json.dumps(result, default=str).encode("utf-8"))
        self.assertGreater(raw_bytes, 96_000, "fixture should mirror the measured live payload size")
        policy = ToolPolicy(
            required_permission="detections.read",
            max_output_bytes=96_000,
            max_output_items=EvaAgentToolAdapter._max_output_items("search_archive"),
        )
        sanitized = sanitize_output(result, policy)
        self.assertEqual(sanitized.get("_truncated"), True)
        self.assertNotIn("results", sanitized)
        self.assertNotIn("coverage", sanitized)

    def test_raised_byte_budget_preserves_a_realistic_page(self) -> None:
        result = self._search_archive_result(12)
        policy = ToolPolicy(
            required_permission="detections.read",
            max_output_bytes=EvaAgentToolAdapter._max_output_bytes("search_archive"),
            max_output_items=EvaAgentToolAdapter._max_output_items("search_archive"),
        )
        sanitized = sanitize_output(result, policy)
        self.assertNotIn("_truncated", sanitized)
        self.assertIn("coverage", sanitized)
        self.assertEqual(sanitized["coverage"]["scanned_candidates"], 1906)
        self.assertEqual(len(sanitized["results"]), 12)

    def test_get_detections_gets_a_raised_byte_and_item_budget(self) -> None:
        self.assertEqual(EvaAgentToolAdapter._max_output_bytes("get_detections"), 2_000_000)
        self.assertEqual(EvaAgentToolAdapter._max_output_items("get_detections"), 50_000)

    def test_video_summaries_preserve_realistic_evidence_before_compaction(self) -> None:
        result = {
            "channel_id": 118,
            "depth": "L1",
            "coverage": {"status": "complete", "truncated": False},
            "count": 1,
            "entries": [
                {
                    "level": "L1",
                    "summary": "Night drift fixture remained visible.",
                }
            ],
            # Live vlm_summary/vlm_alert payloads measured about 20 KB each.
            # Eight evidence frames are the normal tool default, not an
            # exceptional max-page request.
            "evidence_frames": [
                {
                    "id": index,
                    "source": "vlm_summary",
                    "payload": {"summary": "x" * 20_000},
                }
                for index in range(8)
            ],
        }
        raw_bytes = len(json.dumps(result, default=str).encode("utf-8"))
        self.assertGreater(raw_bytes, 96_000)
        policy = ToolPolicy(
            required_permission="streams.view",
            max_output_bytes=EvaAgentToolAdapter._max_output_bytes(
                "get_video_summaries"
            ),
            max_output_items=EvaAgentToolAdapter._max_output_items(
                "get_video_summaries"
            ),
            max_output_string_chars=24_000,
        )

        sanitized = sanitize_output(result, policy)

        self.assertNotIn("_truncated", sanitized)
        self.assertEqual(sanitized["channel_id"], 118)
        self.assertEqual(sanitized["coverage"]["status"], "complete")
        self.assertEqual(len(sanitized["evidence_frames"]), 8)

    def test_unlisted_tools_keep_the_conservative_default(self) -> None:
        self.assertEqual(EvaAgentToolAdapter._max_output_bytes("some_other_tool"), 96_000)
        self.assertEqual(EvaAgentToolAdapter._max_output_items("some_other_tool"), 500)


if __name__ == "__main__":
    unittest.main()
