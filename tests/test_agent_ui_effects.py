from agent_ui_effects import derive_agent_ui_effects


def test_archive_result_projects_closed_filter_effect():
    effects = derive_agent_ui_effects(
        "search_archive",
        {
            "query": "person near gate",
            "channel_id": 12,
            "source": "vlm_alert",
            "since_ms": 1000,
            "until_ms": 2000,
        },
        {"results": [{"id": 1}, {"id": 2}]},
        seed="call-1",
    )

    assert effects == [
        {
            "version": 1,
            "effect_id": effects[0]["effect_id"],
            "target": "archive",
            "action": "show_results",
            "source": {"tool": "search_archive", "committed": False},
            "payload": {
                "channel_id": 12,
                "source": "vlm_alert",
                "since_ms": 1000,
                "until_ms": 2000,
                "query": "person near gate",
                "result_count": 2,
            },
        }
    ]


def test_preview_write_never_claims_committed_ui_state():
    [effect] = derive_agent_ui_effects(
        "update_probe",
        {"probe_id": "p1", "channel_id": 7, "preview": True},
        {"status": "preview", "approval": {"plan_id": "plan-1"}},
    )

    assert effect["target"] == "probes"
    assert effect["action"] == "show_preview"
    assert effect["source"]["committed"] is False


def test_apply_receipt_projects_refresh():
    [effect] = derive_agent_ui_effects(
        "update_probe",
        {},
        {
            "status": "saved",
            "probe": {"id": "p1", "channel_id": 7},
            "action_receipt": {
                "status": "applied",
                "plan_id": "plan-1",
                "tool": "update_probe",
            },
        },
        committed=True,
        seed="plan-1",
    )

    assert effect["action"] == "refresh"
    assert effect["source"]["committed"] is True
    assert effect["payload"]["id"] == "p1"


def test_failed_or_unmapped_tools_do_not_drive_console():
    assert derive_agent_ui_effects("search_archive", {}, {"error": "denied"}) == []
    assert derive_agent_ui_effects("lookup_help", {}, {"results": []}) == []


def test_unscoped_video_read_does_not_override_live_depth():
    [effect] = derive_agent_ui_effects(
        "get_video_summaries",
        {"channel_id": 112, "depth": "L1", "limit": 20},
        {"channel_id": 112, "time_window": {"from_ts": 100.0, "to_ts": 200.0}},
    )

    assert effect["action"] == "show_period"
    assert effect["payload"]["channel_id"] == 112
    assert "depth" not in effect["payload"]
    assert "since_ms" not in effect["payload"]


def test_explicit_video_period_projects_depth_with_resolved_bounds():
    [effect] = derive_agent_ui_effects(
        "get_video_summaries",
        {"channel_id": 112, "depth": "L1", "relative_range": "last hour"},
        {"channel_id": 112, "time_window": {"from_ts": 100.0, "to_ts": 200.0}},
    )

    assert effect["payload"]["depth"] == "L1"
    assert effect["payload"]["since_ms"] == 100_000
    assert effect["payload"]["until_ms"] == 200_000
