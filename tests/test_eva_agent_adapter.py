import unittest

from agent import _TOOL_SCHEMAS
from agent_security import (
    ApprovalRequiredError,
    AuditUnavailableError,
    ChannelAccessDeniedError,
    InvalidToolArgumentsError,
    ToolExecutionContext,
)
from agent_security.eva_adapter import EvaAgentToolAdapter
from security import Permission


class _ProbeStore:
    def list_probes(self):
        return [
            {"id": "probe-7", "name": "Door", "channel_id": 7},
            {"id": "probe-8", "name": "Yard", "channel_id": 8},
        ]


class _DetectionStore:
    def fetch_detections_by_ids(self, ids, include_vectors=False):
        if ids == [80]:
            return [{"id": 80, "channel_id": 8}]
        return []


class _LegacyTools:
    def __init__(self):
        self._ps = _ProbeStore()
        self._ds = _DetectionStore()
        self.calls = []

    def execute(self, name, arguments, progress_cb=None):
        self.calls.append((name, arguments))
        if name == "list_channels":
            return {
                "count": 2,
                "channels": [{"id": 7}, {"id": 8}],
            }
        if name == "list_probes":
            return {
                "count": 2,
                "probes": self._ps.list_probes(),
            }
        return {"status": "preview", "arguments": arguments}

    def _resolve_channel_id(self, arguments, required=False):
        return {"door": 7, "yard": 8}.get(arguments.get("channel_ref"))


class EvaAgentToolAdapterTests(unittest.TestCase):
    def setUp(self):
        self.legacy = _LegacyTools()
        self.audit_events = []
        self.adapter = EvaAgentToolAdapter(
            self.legacy,
            _TOOL_SCHEMAS,
            audit_callback=self.audit_events.append,
        )
        self.addCleanup(self.adapter.close)
        self.context = ToolExecutionContext(
            actor_id="361fe45f-f277-42f8-ae35-eaa0fc81cf38",
            tenant_id="59da6ca3-51b7-4d91-9190-aae06b76d846",
            roles={"engineer"},
            permissions={permission.value for permission in Permission},
            allowed_channel_ids={"7"},
            request_id="request-1",
            client_ip="192.0.2.10",
        )

    def test_model_schemas_remove_unsafe_surfaces(self):
        schemas = {
            item["function"]["name"]: item
            for item in self.adapter.available_tool_schemas(self.context)
        }

        self.assertNotIn("create_bookmark", schemas)
        search = schemas["search_archive"]["function"]["parameters"]
        self.assertNotIn("folder", search["properties"])
        self.assertEqual(
            search["properties"]["scope"]["enum"],
            ["detections"],
        )
        describe = schemas["describe_frame"]["function"]["parameters"]
        self.assertNotIn("image_path", describe["properties"])
        preview = schemas["update_probe"]["function"]["parameters"][
            "properties"
        ]["preview"]
        self.assertEqual(preview["enum"], [True])

    def test_list_results_are_filtered_to_channel_grants(self):
        channels = self.adapter.execute("list_channels", {}, self.context)
        probes = self.adapter.execute("list_probes", {}, self.context)

        self.assertEqual(channels["channels"], [{"id": 7}])
        self.assertEqual(
            [item["id"] for item in probes["probes"]],
            ["probe-7"],
        )

    def test_unauthorized_channel_is_denied_and_audited(self):
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "get_detections",
                {"channel_id": 8},
                self.context,
            )

        self.assertEqual(self.legacy.calls, [])
        self.assertEqual(self.audit_events[-1].phase, "deny")
        self.assertEqual(
            self.audit_events[-1].code,
            "channel_access_denied",
        )

    def test_scoped_aggregate_requires_explicit_channel(self):
        multi_channel_context = ToolExecutionContext(
            actor_id=self.context.actor_id,
            tenant_id=self.context.tenant_id,
            permissions=self.context.permissions,
            allowed_channel_ids={"7", "8"},
        )

        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "get_detection_summary",
                {},
                multi_channel_context,
            )

        self.assertEqual(self.audit_events[-1].phase, "deny")

    def test_filesystem_inputs_are_denied_before_legacy_dispatch(self):
        with self.assertRaises(InvalidToolArgumentsError):
            self.adapter.execute(
                "search_archive",
                {
                    "query": "person",
                    "scope": "indexed_folder",
                    "folder": "/etc",
                },
                self.context,
            )
        with self.assertRaises(InvalidToolArgumentsError):
            self.adapter.execute(
                "describe_frame",
                {"image_path": "/etc/passwd"},
                self.context,
            )

        self.assertEqual(self.legacy.calls, [])
        self.assertEqual(
            [event.phase for event in self.audit_events],
            ["deny", "deny"],
        )

    def test_detection_ownership_is_resolved_before_dispatch(self):
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "describe_frame",
                {"detection_id": 80},
                self.context,
            )

        self.assertEqual(self.legacy.calls, [])

    def test_write_tools_are_preview_only(self):
        result = self.adapter.execute(
            "update_probe",
            {
                "probe_id": "probe-7",
                "changes": {"pos_floor": 0.4},
                "preview": True,
            },
            self.context,
        )
        self.assertEqual(result["status"], "preview")

        with self.assertRaises(ApprovalRequiredError):
            self.adapter.execute(
                "update_probe",
                {
                    "probe_id": "probe-7",
                    "changes": {"pos_floor": 0.4},
                    "preview": False,
                },
                self.context,
            )

        self.assertEqual(len(self.legacy.calls), 1)

    def test_write_apply_requires_plan_then_executes_stored_arguments(self):
        plan = self.adapter.create_plan(
            "update_probe",
            {
                "probe_id": "probe-7",
                "changes": {"pos_floor": 0.4},
                "preview": False,
            },
            self.context,
        )

        result = self.adapter.approve_and_execute(plan.plan_id, self.context)

        self.assertEqual(result["status"], "preview")
        self.assertEqual(result["arguments"]["preview"], False)
        self.assertEqual(len(self.legacy.calls), 1)

    def test_audit_failure_blocks_legacy_handler(self):
        adapter = EvaAgentToolAdapter(
            self.legacy,
            _TOOL_SCHEMAS,
            audit_callback=lambda _event: (_ for _ in ()).throw(
                RuntimeError("audit down")
            ),
        )
        self.addCleanup(adapter.close)

        with self.assertRaises(AuditUnavailableError):
            adapter.execute("list_channels", {}, self.context)

        self.assertEqual(self.legacy.calls, [])


if __name__ == "__main__":
    unittest.main()
