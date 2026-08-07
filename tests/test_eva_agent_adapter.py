import unittest

from agent import AgentTools, _TOOL_SCHEMAS
from agent_security import (
    ApprovalRequiredError,
    AuditUnavailableError,
    ChannelAccessDeniedError,
    InvalidToolArgumentsError,
    PermissionDeniedError,
    ToolExecutionContext,
)
from agent_security.eva_adapter import EvaAgentToolAdapter
from deployment_workflow import ProtocolDeploymentStore
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
        if ids == [70, 71]:
            return [
                {"id": 70, "channel_id": 7},
                {"id": 71, "channel_id": 7},
            ]
        if ids == [70, 80]:
            return [
                {"id": 70, "channel_id": 7},
                {"id": 80, "channel_id": 8},
            ]
        return []


class _IncidentCommands:
    def __init__(self):
        self.calls = []
        self.channel_ids = [7]
        self.stored_actor_ids = []
        self.reviewed = []

    def get(self, incident_id):
        self.calls.append(incident_id)
        return {
            "id": incident_id,
            "revision": 3,
            "state": "draft",
            "channel_ids": list(self.channel_ids),
        }

    def build_draft(self, *, channel_id, anchor_detection_id, since_ms, until_ms):
        return {
            "title": "Grounded incident",
            "channel_ids": [channel_id],
            "anchor": {"detection_id": anchor_detection_id},
            "time_bounds": {"observed_start_ms": 1_000, "observed_end_ms": 2_000},
            "timeline": [],
            "evidence": [],
            "coverage": {"status": "covered"},
        }

    def draft_digest(self, _draft):
        return "digest-v1"

    def store_draft(self, draft, *, actor_id):
        self.stored_actor_ids.append(actor_id)
        return {"id": "00000000-0000-0000-0000-000000000118", "revision": 1, **draft}

    def public_record(self, record):
        return {**record, "incident_id": record.get("id")}

    def temporal_context(self, incident):
        return {
            "supported": True,
            "incident_id": incident["id"],
            "episodes": [],
            "episode_total": 0,
            "series_links": [],
            "relation_total": 0,
            "correction_count": 0,
            "lifecycle_history": [],
            "transition_total": 0,
        }

    def review_incident(self, incident_id, *, actor_id, action, expected_revision):
        self.reviewed.append(
            {
                "incident_id": incident_id,
                "actor_id": actor_id,
                "action": action,
                "expected_revision": expected_revision,
            }
        )
        return {
            "id": incident_id,
            "revision": expected_revision + 1,
            "state": "confirmed",
            "case_state": "confirmed",
            "channel_ids": list(self.channel_ids),
        }


class _LegacyTools:
    def __init__(self):
        self._ps = _ProbeStore()
        self._ds = _DetectionStore()
        self.calls = []
        self._trusted = None
        self.seen_trusted = None
        self.fail_name = None
        self.results = {}
        self._deployment_store = ProtocolDeploymentStore()
        self._incident_commands = _IncidentCommands()

    def _set_trusted_permissions(self, permissions):
        self._trusted = frozenset(str(item) for item in (permissions or ()))

    def _clear_trusted_permissions(self):
        self._trusted = None

    def execute(self, name, arguments, progress_cb=None):
        self.calls.append((name, arguments))
        self.seen_trusted = self._trusted
        if name == self.fail_name:
            raise RuntimeError("boom")
        if name in self.results:
            return self.results[name]
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

    def test_lookup_help_uses_trusted_context_not_model_args(self):
        operator_context = ToolExecutionContext(
            actor_id="361fe45f-f277-42f8-ae35-eaa0fc81cf38",
            tenant_id="59da6ca3-51b7-4d91-9190-aae06b76d846",
            roles={"operator"},
            permissions={
                Permission.AGENT_USE.value,
                Permission.DETECTIONS_VIEW.value,
            },
            allowed_channel_ids={"7"},
            request_id="request-2",
            client_ip="192.0.2.10",
        )
        # The model attempts to grant itself users:manage via args.
        self.adapter.execute(
            "lookup_help",
            {
                "query": "how do I reset a user password",
                "_granted_permissions": ["users:manage"],
            },
            operator_context,
        )
        name, prepared = self.legacy.calls[-1]
        self.assertEqual(name, "lookup_help")
        # Model-supplied permissions are stripped from the tool arguments.
        self.assertNotIn("_granted_permissions", prepared)
        # Trusted permissions seen by the tool come from the context, not the model.
        self.assertEqual(
            self.legacy.seen_trusted,
            frozenset({Permission.AGENT_USE.value, Permission.DETECTIONS_VIEW.value}),
        )
        self.assertNotIn("users:manage", self.legacy.seen_trusted)

    def test_video_summary_security_boundary_preserves_entries_and_image_urls(self):
        self.legacy.results["get_video_summaries"] = {
            "channel_id": 7,
            "count": 25,
            "coverage": {
                "status": "truncated",
                "windows": [
                    {"index": index, "status": "covered", "metadata": {"source": "L0"}}
                    for index in range(300)
                ],
            },
            "entries": [
                {"time": f"10:{index:02d}", "summary": f"summary {index}"}
                for index in range(25)
            ],
            "evidence_frames": [
                {
                    "id": 501,
                    "detection_id": 501,
                    "channel_id": 7,
                    "image_url": "/detections/thumbnail/501",
                }
            ],
        }

        result = self.adapter.execute(
            "get_video_summaries",
            {"channel_id": 7, "limit": 25, "include_evidence_frames": True},
            self.context,
        )

        self.assertEqual(len(result["entries"]), 25)
        self.assertEqual(result["evidence_frames"][0]["image_url"], "/detections/thumbnail/501")
        self.assertNotIn("_truncated", result)

    def test_summary_restore_is_preview_only_and_scoped_to_authorized_channels(self):
        schemas = {
            item["function"]["name"]: item
            for item in self.adapter.available_tool_schemas(self.context)
        }
        preview = schemas["restore_video_summary_history"]["function"]["parameters"]["properties"]["preview"]
        self.assertEqual(preview["enum"], [True])

        result = self.adapter.execute(
            "restore_video_summary_history",
            {"relative_range": "last two weeks", "preview": True},
            self.context,
        )

        self.assertEqual(result["status"], "preview")
        self.assertEqual(result["arguments"]["channel_ids"], ["7"])
        self.assertTrue(result["arguments"]["preview"])

        self.assertNotIn("get_video_summary_restore_status", schemas)
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "get_video_summary_restore_status",
                {},
                self.context,
            )

    def test_trusted_permissions_are_cleared_after_tool_exception(self):
        self.legacy.fail_name = "lookup_help"

        with self.assertRaises(RuntimeError):
            self.adapter.execute(
                "lookup_help",
                {"query": "how do I reset a user password"},
                self.context,
            )

        self.assertIsNone(self.legacy._trusted)
        self.assertIsNotNone(self.legacy.seen_trusted)

    def test_incident_follow_preview_binds_server_channels_and_revision(self):
        schemas = {
            item["function"]["name"]: item
            for item in self.adapter.available_tool_schemas(self.context)
        }
        preview = schemas["follow_incident"]["function"]["parameters"]["properties"]["preview"]
        self.assertEqual(preview["enum"], [True])

        result = self.adapter.execute(
            "follow_incident",
            {
                "incident_id": "00000000-0000-0000-0000-000000000117",
                "mode": "critical",
                "ttl_seconds": 300,
                "preview": True,
                # Untrusted values are overwritten from durable state.
                "expected_revision": 999,
                "channel_ids": [999],
            },
            self.context,
        )

        self.assertEqual(result["status"], "preview")
        self.assertIn("approval", result)
        name, prepared = self.legacy.calls[-1]
        self.assertEqual(name, "follow_incident")
        self.assertEqual(prepared["channel_ids"], ["7"])
        self.assertEqual(prepared["expected_revision"], 3)

    def test_incident_lookup_does_not_leak_before_permission_check(self):
        no_reports = ToolExecutionContext(
            actor_id=self.context.actor_id,
            tenant_id=self.context.tenant_id,
            roles={"operator"},
            permissions={Permission.AGENT_USE.value},
            allowed_channel_ids={"7"},
        )
        with self.assertRaises(PermissionDeniedError):
            self.adapter.execute(
                "get_incident",
                {"incident_id": "00000000-0000-0000-0000-000000000117"},
                no_reports,
            )
        self.assertEqual(self.legacy._incident_commands.calls, [])

    def test_incident_channel_ownership_is_resolved_from_durable_record(self):
        self.legacy._incident_commands.channel_ids = [8]
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "get_incident",
                {"incident_id": "00000000-0000-0000-0000-000000000117"},
                self.context,
            )

    def test_incident_draft_rejects_anchor_channel_spoofing(self):
        all_channels = ToolExecutionContext(
            actor_id=self.context.actor_id,
            tenant_id=self.context.tenant_id,
            roles=self.context.roles,
            permissions=self.context.permissions,
            allowed_channel_ids={"*"},
        )
        with self.assertRaises(InvalidToolArgumentsError):
            self.adapter.execute(
                "draft_incident",
                {
                    "channel_id": 7,
                    "anchor_detection_id": 80,
                    "preview": True,
                },
                all_channels,
            )

        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "draft_incident",
                {"anchor_detection_id": 80, "preview": True},
                self.context,
            )

    def test_incident_draft_apply_uses_authenticated_actor_and_preview_digest(self):
        commands = _IncidentCommands()
        tools = AgentTools(
            detections_store=_DetectionStore(),
            probes_store=_ProbeStore(),
            luxriot_manager=None,
            embed_text_fn=None,
            embed_image_fn=None,
            call_lm_fn=None,
            encode_jpeg_fn=None,
            search_indexed_folder_fn=None,
            search_detections_fn=None,
            incident_command_service=commands,
        )
        adapter = EvaAgentToolAdapter(
            tools,
            _TOOL_SCHEMAS,
            audit_callback=self.audit_events.append,
        )
        self.addCleanup(adapter.close)
        context = ToolExecutionContext(
            actor_id=self.context.actor_id,
            tenant_id=self.context.tenant_id,
            roles={"operator"},
            permissions={permission.value for permission in Permission},
            allowed_channel_ids={"8"},
        )

        preview = adapter.execute(
            "draft_incident",
            {"anchor_detection_id": 80, "preview": True},
            context,
        )
        applied = adapter.approve_and_execute(
            preview["approval"]["plan_id"],
            context,
        )

        self.assertEqual(applied["status"], "applied")
        self.assertEqual(commands.stored_actor_ids, [self.context.actor_id])
        self.assertEqual(applied["incident"]["channel_ids"], [8])

    def test_incident_review_requires_approval_and_binds_revision_to_durable_state(self):
        commands = _IncidentCommands()
        tools = AgentTools(
            detections_store=_DetectionStore(),
            probes_store=_ProbeStore(),
            luxriot_manager=None,
            embed_text_fn=None,
            embed_image_fn=None,
            call_lm_fn=None,
            encode_jpeg_fn=None,
            search_indexed_folder_fn=None,
            search_detections_fn=None,
            incident_command_service=commands,
        )
        adapter = EvaAgentToolAdapter(
            tools,
            _TOOL_SCHEMAS,
            audit_callback=self.audit_events.append,
        )
        self.addCleanup(adapter.close)

        preview = adapter.execute(
            "review_incident",
            {
                "incident_id": "00000000-0000-0000-0000-000000000117",
                "action": "confirm",
                "preview": True,
                # Neither a model nor stale UI may choose the write revision.
                "expected_revision": 999,
            },
            self.context,
        )
        self.assertEqual(preview["status"], "preview")
        self.assertEqual(preview["proposed_review"]["action"], "confirm")

        applied = adapter.approve_and_execute(
            preview["approval"]["plan_id"],
            self.context,
        )
        self.assertEqual(applied["status"], "applied")
        self.assertEqual(
            commands.reviewed,
            [
                {
                    "incident_id": "00000000-0000-0000-0000-000000000117",
                    "actor_id": self.context.actor_id,
                    "action": "confirm",
                    "expected_revision": 3,
                }
            ],
        )

    def test_lookup_help_real_agent_tools_keeps_permissions_across_executor(self):
        tools = AgentTools(
            detections_store=_DetectionStore(),
            probes_store=_ProbeStore(),
            luxriot_manager=None,
            embed_text_fn=None,
            embed_image_fn=None,
            call_lm_fn=None,
            encode_jpeg_fn=None,
            search_indexed_folder_fn=None,
            search_detections_fn=None,
        )
        adapter = EvaAgentToolAdapter(
            tools,
            _TOOL_SCHEMAS,
            audit_callback=self.audit_events.append,
        )
        self.addCleanup(adapter.close)
        settings_context = ToolExecutionContext(
            actor_id="361fe45f-f277-42f8-ae35-eaa0fc81cf38",
            tenant_id="59da6ca3-51b7-4d91-9190-aae06b76d846",
            roles={"engineer"},
            permissions={
                Permission.AGENT_USE.value,
                Permission.SETTINGS_MANAGE.value,
            },
            allowed_channel_ids={"*"},
            request_id="request-3",
            client_ip="192.0.2.10",
        )

        result = adapter.execute(
            "lookup_help",
            {"query": "how to backup the database before an update"},
            settings_context,
        )

        self.assertFalse(result["best_match_restricted"])
        self.assertTrue(
            any(
                row.get("doc") == "docs/admin/backup_recovery.md"
                for row in result.get("results") or []
            ),
            "settings:manage should unlock backup help through the secure adapter",
        )

    def test_model_schemas_remove_unsafe_surfaces(self):
        schemas = {
            item["function"]["name"]: item
            for item in self.adapter.available_tool_schemas(self.context)
        }

        self.assertNotIn("create_bookmark", schemas)
        self.assertIn("normalize_time_window", schemas)
        self.assertIn("list_video_summary_channels", schemas)
        self.assertIn("count_video_summary_events", schemas)
        self.assertIn("track_visual_state_transitions", schemas)
        self.assertIn("calibrate_probe_from_archive", schemas)
        search = schemas["search_archive"]["function"]["parameters"]
        self.assertNotIn("folder", search["properties"])
        self.assertEqual(
            search["properties"]["scope"]["enum"],
            ["detections"],
        )
        self.assertEqual(
            search["properties"]["source"]["enum"],
            ["probe", "vlm_summary", "vlm_alert"],
        )
        describe = schemas["describe_frame"]["function"]["parameters"]
        self.assertNotIn("image_path", describe["properties"])
        self.assertIn("detection_ids", describe["properties"])
        self.assertEqual(describe["properties"]["detection_ids"]["maxItems"], 9)
        self.assertNotIn("channel_ids", describe["properties"])
        create_probe = schemas["create_probe"]["function"]
        self.assertIn("VLM-alert follow-up", create_probe["description"])
        self.assertIn(
            "Avoid personal names",
            create_probe["parameters"]["properties"]["positives"]["description"],
        )
        self.assertIn(
            "Do not write literal negation",
            create_probe["parameters"]["properties"]["negatives"]["description"],
        )
        calibrate = schemas["calibrate_probe_from_archive"]["function"]
        self.assertIn("Read-only CLIP P/N/M calibration", calibrate["description"])
        self.assertIn("channel_ids", calibrate["parameters"]["properties"])
        self.assertIn("max_channels_per_call", calibrate["parameters"]["properties"])
        preview = schemas["update_probe"]["function"]["parameters"][
            "properties"
        ]["preview"]
        self.assertEqual(preview["enum"], [True])
        deployment_apply = schemas["apply_deployment_plan"]["function"][
            "parameters"
        ]["properties"]["preview"]
        self.assertEqual(deployment_apply["enum"], [True])

    def test_deployment_scope_is_resolved_before_dispatch(self):
        result = self.adapter.execute(
            "start_deployment",
            {"target_channel_count": 8},
            self.context,
        )
        self.assertEqual(result["arguments"]["channel_ids"], ["7"])

        state = self.legacy._deployment_store.start(
            [{"id": 7, "title": "Door"}, {"id": 8, "title": "Yard"}],
            resume_latest=False,
        )
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "configure_deployment",
                {
                    "deployment_id": state["deployment_id"],
                    "channel_ids": [8],
                },
                self.context,
            )

        scoped = self.legacy._deployment_store.start(
            [{"id": 7, "title": "Door"}],
            resume_latest=False,
        )
        self.legacy._deployment_store.configure(
            scoped["deployment_id"],
            channel_ids=[7],
            groups=[{"name": "Door group", "channel_ids": [7]}],
        )
        self.adapter.execute(
            "configure_deployment",
            {
                "deployment_id": scoped["deployment_id"],
                "requirements": [
                    {"name": "Door routine", "channel_ids": [7]}
                ],
            },
            self.context,
        )
        _name, forwarded = self.legacy.calls[-1]
        self.assertNotIn("channel_ids", forwarded)
        self.assertNotIn("_eva_deployment_scope_guard_only", forwarded)
        self.assertEqual(
            forwarded["requirements"][0]["channel_ids"],
            [7],
        )

    def test_counted_metric_profile_is_channel_scoped(self):
        self.legacy._deployment_store.save_counted_profiles(
            [
                {
                    "id": "metric-yard",
                    "name": "Yard occupancy",
                    "channel_id": 8,
                }
            ]
        )
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "query_counted_state_metric",
                {"metric_id": "metric-yard"},
                self.context,
            )

    def test_deployment_apply_requires_all_composite_permissions(self):
        state = self.legacy._deployment_store.start(
            [{"id": 7, "title": "Door"}],
            resume_latest=False,
        )
        self.legacy._deployment_store.configure(
            state["deployment_id"],
            channel_ids=[7],
        )
        missing_capture = ToolExecutionContext(
            actor_id=self.context.actor_id,
            tenant_id=self.context.tenant_id,
            roles={"admin"},
            permissions={
                permission.value
                for permission in Permission
                if permission is not Permission.CAPTURE_MANAGE
            },
            allowed_channel_ids={"7"},
        )

        with self.assertRaises(PermissionDeniedError) as raised:
            self.adapter.execute(
                "apply_deployment_plan",
                {
                    "deployment_id": state["deployment_id"],
                    "preview": True,
                },
                missing_capture,
            )

        self.assertIn(
            Permission.CAPTURE_MANAGE.value,
            raised.exception.details["missing_permissions"],
        )

    def test_list_results_are_filtered_to_channel_grants(self):
        channels = self.adapter.execute("list_channels", {}, self.context)
        probes = self.adapter.execute("list_probes", {}, self.context)

        self.assertEqual(channels["channels"], [{"id": 7}])
        self.assertEqual(
            [item["id"] for item in probes["probes"]],
            ["probe-7"],
        )

    def test_list_channels_accepts_now_alias_without_leaking_unknown_arg(self):
        self.adapter.execute("list_channels", {"now": True}, self.context)

        self.assertEqual(
            self.legacy.calls[-1],
            ("list_channels", {"force": True}),
        )

    def test_probe_calibration_batch_items_are_filtered_to_channel_grants(self):
        result = self.adapter.execute(
            "prepare_probe_calibration_batch",
            {
                "items": [
                    {
                        "name": "allowed",
                        "event_query": "person lying on ground",
                        "contrast_query": "people walking normally",
                        "channel_id": 7,
                    },
                    {
                        "name": "blocked",
                        "event_query": "vehicle drifting",
                        "contrast_query": "normal traffic",
                        "channel_id": 8,
                    },
                    {
                        "name": "mixed",
                        "event_query": "smoke visible",
                        "contrast_query": "clear roadway",
                        "channel_ids": [7, 8],
                    },
                ],
            },
            self.context,
        )

        prepared = result["arguments"]
        self.assertEqual(prepared["channel_ids"], ["7"])
        self.assertEqual([item["name"] for item in prepared["items"]], ["allowed", "mixed"])
        self.assertEqual(prepared["items"][1]["channel_ids"], [7])

    def test_normalize_time_window_allows_operator_relative_range(self):
        result = self.adapter.execute(
            "normalize_time_window",
            {"relative_range": "last two hours"},
            self.context,
        )

        self.assertEqual(result["arguments"]["relative_range"], "last two hours")

    def test_normalize_time_window_does_not_require_iso_start_end(self):
        result = self.adapter.execute(
            "normalize_time_window",
            {"start_time": "01:30", "end_time": "08:30"},
            self.context,
        )

        self.assertEqual(result["arguments"]["start_time"], "01:30")
        self.assertEqual(result["arguments"]["end_time"], "08:30")

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

    def test_visual_window_signals_are_channel_scoped(self):
        result = self.adapter.execute(
            "get_visual_window_signals",
            {"positive_query": "dog without visible ear tag"},
            self.context,
        )

        self.assertEqual(result["arguments"]["channel_id"], "7")

        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "get_visual_window_signals",
                {"channel_id": 8, "positive_query": "dog without visible ear tag"},
                self.context,
            )

        self.assertEqual(self.legacy.calls[-1][0], "get_visual_window_signals")
        self.assertEqual(self.audit_events[-1].phase, "deny")

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

    def test_summary_channel_inventory_defaults_to_scoped_channels(self):
        result = self.adapter.execute(
            "list_video_summary_channels",
            {"from_ts": 100.0, "to_ts": 200.0},
            self.context,
        )

        self.assertEqual(result["arguments"]["channel_ids"], ["7"])

    def test_generate_report_defaults_to_scoped_channels(self):
        result = self.adapter.execute(
            "generate_report",
            {"report_type": "video_descriptions", "from_ts": 100.0, "to_ts": 200.0},
            self.context,
        )

        self.assertEqual(result["arguments"]["channel_ids"], ["7"])

    def test_generate_report_accepts_millisecond_window_aliases(self):
        result = self.adapter.execute(
            "generate_report",
            {
                "report_type": "video_descriptions",
                "since_ms": 100_000,
                "until_ms": 200_000,
            },
            self.context,
        )

        self.assertEqual(result["arguments"]["from_ts"], 100.0)
        self.assertEqual(result["arguments"]["to_ts"], 200.0)
        self.assertNotIn("since_ms", result["arguments"])
        self.assertNotIn("until_ms", result["arguments"])
        self.assertEqual(result["arguments"]["channel_ids"], ["7"])

    def test_visual_state_transitions_defaults_to_scoped_channel(self):
        result = self.adapter.execute(
            "track_visual_state_transitions",
            {
                "positive_state_query": "cat on top of computer tower",
                "negative_state_query": "empty computer tower",
                "from_ts": 100.0,
                "to_ts": 200.0,
            },
            self.context,
        )

        self.assertEqual(result["arguments"]["channel_id"], "7")

    def test_calibrate_probe_defaults_to_scoped_channels(self):
        result = self.adapter.execute(
            "calibrate_probe_from_archive",
            {
                "event_query": "two people fighting",
                "contrast_query": "people walking normally",
                "from_ts": 100.0,
                "to_ts": 200.0,
            },
            self.context,
        )

        self.assertEqual(result["arguments"]["channel_ids"], ["7"])

    def test_detection_ownership_is_resolved_before_dispatch(self):
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "describe_frame",
                {"detection_id": 80},
                self.context,
            )

        self.assertEqual(self.legacy.calls, [])

    def test_detection_ownership_overrides_caller_supplied_channel(self):
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "describe_frame",
                {"detection_id": 80, "channel_id": 7},
                self.context,
            )

        self.assertEqual(self.legacy.calls, [])

    def test_detection_batch_resolves_hidden_channel_ownership(self):
        result = self.adapter.execute(
            "describe_frame",
            {"detection_ids": [70, 71], "prompt": "sphynx cat"},
            self.context,
        )

        self.assertEqual(result["arguments"]["detection_ids"], [70, 71])
        self.assertEqual(result["arguments"]["channel_ids"], ["7"])
        self.assertEqual(result["arguments"]["channel_id"], "7")

    def test_detection_batch_rejects_one_unauthorized_candidate(self):
        with self.assertRaises(ChannelAccessDeniedError):
            self.adapter.execute(
                "describe_frame",
                {"detection_ids": [70, 80], "prompt": "sphynx cat"},
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
        self.assertIn("approval", result)
        self.assertIn("plan_id", result["approval"])
        self.assertNotIn("approval_id", result["approval"])

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

    def test_probe_write_rejects_negated_negative_prompts_through_adapter(self):
        tools = AgentTools(
            detections_store=_DetectionStore(),
            probes_store=_ProbeStore(),
            luxriot_manager=object(),
            embed_text_fn=lambda _text: None,
            embed_image_fn=lambda _image: None,
            call_lm_fn=lambda *_args, **_kwargs: "",
            encode_jpeg_fn=lambda *_args, **_kwargs: "",
            search_indexed_folder_fn=lambda **_kwargs: [],
            search_detections_fn=lambda **_kwargs: [],
        )
        adapter = EvaAgentToolAdapter(
            tools,
            _TOOL_SCHEMAS,
            audit_callback=self.audit_events.append,
        )
        self.addCleanup(adapter.close)

        with self.assertRaisesRegex(Exception, "literal negation"):
            adapter.execute(
                "create_probe",
                {
                    "name": "unsafe vehicle probe",
                    "channel_id": 7,
                    "positives": ["vehicle doing burnout"],
                    "negatives": ["no vehicle"],
                    "preview": True,
                },
                self.context,
            )

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
        self.assertEqual(result["action_receipt"]["status"], "applied")
        self.assertEqual(result["action_receipt"]["tool"], "update_probe")
        self.assertEqual(result["action_receipt"]["result_status"], "preview")
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
