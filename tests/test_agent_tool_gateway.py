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


if __name__ == "__main__":
    unittest.main()
