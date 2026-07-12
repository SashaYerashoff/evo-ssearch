from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static/css/app.css").read_text(encoding="utf-8")
JS = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
TEMPLATE = (ROOT / "templates/index.html").read_text(encoding="utf-8")


def _cef_safe_section() -> str:
    marker = "CEF-safe rendering contract"
    assert marker in CSS
    return CSS.split(marker, 1)[1]


def test_cef_safe_section_keeps_repeated_cards_static_on_hover():
    section = _cef_safe_section()
    assert ".result-item:hover" in section
    assert ".probe-mini-card:hover" in section
    assert ".result-item.selected" in section
    assert ".probe-mini-card.active" in section
    assert "transform: none;" in section
    assert "transition:" in section
    assert "transition: all" not in section


def test_cef_safe_section_removes_card_backdrop_filters():
    section = _cef_safe_section()
    assert ".expand-collapse-icon" in section
    assert ".action-icon" in section
    assert ".probe-action-btn" in section
    assert "backdrop-filter: none;" in section


def test_studio_scroll_contract_has_stable_gutters():
    section = _cef_safe_section()
    for selector in (
        ".results-grid",
        "#probeCards",
        ".archive-inspector-body",
        ".luxriot-summaries",
        ".agent-context",
    ):
        assert selector in section
    assert "scrollbar-gutter: stable;" in section


def test_probe_board_has_one_scroll_owner():
    section = _cef_safe_section()
    scroll_group = section.split("scrollbar-width: thin;", 1)[0]
    hidden_group = section.split(".archive-inspector-panel,", 1)[1].split("{", 1)[0]
    assert "#probeCards" in scroll_group
    assert ".probe-shell" not in scroll_group
    assert ".probe-shell" in hidden_group


def test_probe_cards_have_stable_action_zones():
    assert "probe-mini-card-head" in JS
    assert "probe-mini-content" in JS
    assert "probe-mini-card-foot" in JS
    assert "probe-mini-overlay" not in JS
    assert "probe-mini-primary-actions" in JS
    assert "probe-mini-danger-actions" in JS
    assert 'data-action="${toggleAction}"' in JS
    assert 'data-action="expand"' in JS
    assert 'data-action="delete"' in JS
    assert 'data-action="run" data-id="${p.id}"' not in JS
    assert "probeActionIcon(toggleAction)" in JS

    assert "Probe board card flow layout" in CSS
    section = CSS.split("Probe board card flow layout", 1)[1].split("UI layout consistency contract", 1)[0]
    assert ".probe-mini-card-head" in section
    assert ".probe-mini-content" in section
    assert ".probe-mini-card-foot" in section
    assert ".probe-mini-primary-actions" in section
    assert ".probe-mini-danger-actions" in section
    assert ".probe-mini-overlay" not in section
    assert "grid-template-rows: auto auto auto;" in section
    assert "grid-template-columns: minmax(0, 1fr) auto;" in section
    assert "grid-auto-rows: max-content;" in section
    assert "contain: none;" in section
    assert "overflow: hidden;" in section
    assert "position: absolute;" not in section
    assert "position: static;" in section
    assert "right: auto;" in section
    assert "bottom: auto;" in section
    assert "flex-wrap: nowrap;" in section
    assert "max-width: 100%;" in section
    assert "height: clamp(" in section
    assert "aspect-ratio: auto;" in section
    assert "-webkit-line-clamp: 2;" in section
    assert "overflow-wrap: anywhere;" in section


def test_repeated_grid_consistency_contract_prevents_card_overlap():
    assert "UI layout consistency contract" in CSS
    section = CSS.split("UI layout consistency contract", 1)[1]
    for selector in (
        "#probeCards",
        ".results-grid",
        ".agent-det-grid",
        ".agent-search-results-grid",
    ):
        assert selector in section
    assert "grid-auto-rows: max-content;" in section
    assert "overflow-x: hidden;" in section
    assert "overflow-y: auto;" in section
    assert "repeat(auto-fit, minmax(min(100%, 120px), 1fr))" in section
    assert "repeat(auto-fit, minmax(min(100%, 88px), 1fr))" in section
    assert ".archive-inspector-body .result-item" in section
    assert "contain: none;" in section


def test_modal_consistency_contract_has_bounded_scroll_bodies():
    section = CSS.split("UI layout consistency contract", 1)[1]
    assert ".settings-modal {" in section
    assert "overflow: auto;" in section
    assert ".settings-modal-content," in section
    assert ".archive-review-shell" in section
    assert "max-block-size: calc(100dvh - 32px);" in section
    assert ".prompt-modal-scrollarea" in section
    assert "grid-template-rows: auto minmax(0, 1fr) auto auto auto minmax(0, 0.55fr);" in section
    assert "position: relative;" in section
    assert "inset: auto;" in section
    assert "width: min(96vw, 1120px);" in section
    assert "height: min(calc(100dvh - 32px), 960px);" in section
    assert "border-radius: 22px;" in section


def test_archive_review_modal_uses_shared_surface_not_hud_chrome():
    section = CSS.split("Unified modal behavior", 1)[1]
    shell_section = section.split(".archive-review-head", 1)[0]
    assert ".archive-review-shell" in shell_section
    assert "position: relative;" in shell_section
    assert "inset: auto;" in shell_section
    assert "border: 1px solid rgba(37, 51, 66, 0.92);" in shell_section
    assert "border-radius: 22px;" in shell_section
    assert "border: 1px dashed" not in shell_section
    assert "background-size:" not in section


def test_unified_scrollbar_contract_covers_modals_and_panels():
    assert "Unified scrollbar contract" in CSS
    section = CSS.split("Unified scrollbar contract", 1)[1]
    for selector in (
        ".settings-modal",
        ".archive-review-summary",
        ".probe-editor-modal-body",
        ".probe-cast-body",
        ".prompt-modal-body",
        ".prompt-modal-scrollarea",
        ".settings-sheet-scroll",
        "#probeCards",
        ".agent-messages",
    ):
        assert selector in section
    assert "scrollbar-width: thin;" in section
    assert "scrollbar-color: var(--eva-scrollbar-thumb) transparent;" in section
    assert "::-webkit-scrollbar" in section
    assert "::-webkit-scrollbar-thumb" in section
    assert "background-clip: content-box;" in section
    assert "var(--eva-scrollbar-thumb-hover)" in section


def test_modal_markup_has_no_obvious_duplicate_wrappers_or_options():
    assert TEMPLATE.count('class="probe-editor-modal-body"') == 1
    assert TEMPLATE.count('<option value="skip" selected>Skip matching</option>') == 1


def test_video_workspace_keeps_preview_and_feed_controls_in_their_compact_panes():
    channel = TEMPLATE.index('id="luxriotChannelSelect"')
    preview = TEMPLATE.index('id="luxriotViewport"')
    cadence = TEMPLATE.index('id="luxriotBatchSize"')
    feed = TEMPLATE.index('class="luxriot-card luxriot-summaries-card')
    feed_filters = TEMPLATE.index('id="luxriotSummaryChannelSelect"')
    runtime = TEMPLATE.index('class="luxriot-selected-runtime"')
    streams = TEMPLATE.index('id="luxriotStreams"')

    assert channel < preview < cadence < feed < feed_filters < runtime < streams
    assert TEMPLATE.count('id="luxriotViewport"') == 1
    assert TEMPLATE.count('id="luxriotSummaryChannelSelect"') == 1
    assert 'data-road-scene-grounding' not in TEMPLATE
    assert '.luxriot-sidebar-preview .luxriot-viewport' in CSS
    assert '.luxriot-summaries-card .video-feed-head' in CSS
    assert '.luxriot-selected-runtime-facts' in CSS


def test_video_workspace_compacts_controls_without_hiding_mobile_panes():
    assert 'class="luxriot-row video-channel-row"' in TEMPLATE
    assert 'class="luxriot-row video-model-row"' in TEMPLATE
    assert TEMPLATE.count('class="runtime-fact-secondary"') == 2
    assert '.video-command-panel #luxriotBatchInfo' in CSS
    assert '.luxriot-stream-card > .studio-panel-head' in CSS
    assert 'Narrow Video workspace: stack complete panes' in CSS

    narrow = CSS.split('Narrow Video workspace: stack complete panes', 1)[1]
    assert '.search-panel,' in narrow
    assert '.video-box,' in narrow
    assert '.video-rail,' in narrow
    assert '.video-workspace' in narrow
    assert 'overflow: visible;' in narrow
    assert 'grid-auto-rows: max-content;' in narrow


def test_remaining_studio_tabs_keep_inspectors_until_tablet_and_stack_on_mobile():
    assert 'Compact density for the non-video workspaces' in CSS
    compact = CSS.split('Compact density for the non-video workspaces', 1)[1]
    for selector in (
        '.archive-search-shell .archive-section',
        '.monitor-selection-panel',
        '.agent-chat-topbar',
        '.agent-messages',
    ):
        assert selector in compact

    tablet = CSS.split('@media (max-width: 1440px)', 1)[1].split(
        '@media (max-width: 1260px)', 1
    )[0]
    assert '.monitor-box {' in tablet
    assert '.agent-box {' in tablet
    assert '.monitor-inspector' not in tablet
    assert '.agent-inspector' not in tablet

    assert 'Narrow Archive/Monitoring/Agent workspaces' in CSS
    narrow = CSS.split('Narrow Archive/Monitoring/Agent workspaces', 1)[1]
    assert '.archive-search-shell' in narrow
    assert '.monitor-board-panel' in narrow
    assert '.agent-chat-area' in narrow
    assert 'grid-auto-rows: max-content;' in narrow
    assert '.results-grid,' in narrow
    assert '#probeCards,' in narrow
    assert '.agent-messages' in narrow


def test_escape_closes_all_primary_modals():
    for token in (
        "setProbeSnapModalVisibility(false)",
        "setProbeCastModalVisibility(false)",
        "setProbeEditorModalVisibility(false)",
        "closeLuxriotPromptModal()",
        "agentSkillModal.style.display = 'none'",
    ):
        assert token in JS


def test_ui_lite_mode_is_available_for_embedded_webview_bisect():
    assert "uiLiteMode" in JS
    assert "document.documentElement.classList.add('ui-lite')" in JS
    assert "document.body.classList.add('ui-lite')" in JS
    section = _cef_safe_section()
    assert "html.ui-lite *" in section
    assert "transition: none !important" in section
    assert "backdrop-filter: none !important" in section


def test_archive_review_vlm_feed_jump_is_time_addressed():
    assert "archiveResultCanOpenVlmFeed" in JS
    assert "archiveResultSummaryWindow" in JS
    assert "scrollLuxriotSummaryToTimestamp" in JS
    assert "data-summary-created-ms" in JS
    assert "data-summary-batch-start-ms" in JS
    assert "data-summary-batch-end-ms" in JS
    assert "Opening VLM feed around" in JS


def test_agent_probe_approval_is_standalone_not_research_trace():
    assert "function isStandaloneProbeApprovalResult" in JS
    assert "function buildAgentProbeApprovalCard" in JS
    assert "function promoteStandaloneAgentApprovalCards" in JS
    assert "function isLegacyProbeApprovalCard" in JS
    assert "card.dataset.agentStandaloneApproval = 'true'" in JS
    assert "card.dataset && card.dataset.agentStandaloneApproval === 'true'" in JS
    assert "const card = isStandaloneProbeApprovalResult(name, result)" in JS
    assert "bubble.bodyEl.insertBefore(card, before)" in JS
    assert "bubble.actionsEl.appendChild(card)" in JS
    assert "promoteStandaloneAgentApprovalCards(bubble)" in JS
    assert "isProbeMutationTool(toolName)" in JS
    for tool_name in ("create_probe", "update_probe", "delete_probes"):
        assert tool_name in JS

    trace_route = JS.split("function appendActionCard", 1)[1].split("function appendProgressNote", 1)[0]
    assert "standaloneApproval" in trace_route
    assert "bubble.actionCount = (bubble.actionCount || 0) + 1;" in trace_route
    assert "if (bubble.traceEl) bubble.traceEl.hidden = false;" in trace_route
    assert "bubble.traceEl.hidden = !hasActions;" in JS

    assert ".agent-approval-card" in CSS
    assert ".agent-approval-card-head" in CSS
    assert ".agent-approval-card-body" in CSS
    assert ".agent-approval-fields" in CSS
    assert ".agent-approval-card + .agent-tool-trace" in CSS
    assert ".agent-approval-card-legacy" in CSS


def test_vlm_machine_json_label_is_semantic_not_generic_only():
    assert "function summarizeMachineJson" in JS
    assert "System message" in JS
    assert "Memory/homeostasis" in JS
    assert "kind: 'alert'" in JS
    assert "kind: 'memory'" in JS
    assert "kind: 'system'" in JS
    assert "summary-json-${escapeHtml(summary.kind)}" in JS
    assert "renderSummaryMachineJson(summaryJson, 'Machine JSON', summaryParts.marker)" in JS
    assert ".summary-json-alert" in CSS
    assert ".summary-json-memory" in CSS
    assert ".summary-json-system" in CSS


def test_monitor_probe_filmstrip_is_not_rendered_in_inspector():
    assert "Latest CLIP Hits" not in TEMPLATE
    assert "monitor-detections-panel" not in TEMPLATE
    assert 'id="probeResults"' not in TEMPLATE


def test_agent_thumbnail_grid_has_missing_image_fallback():
    assert "function _makeThumb" in JS
    assert "function agentImageUrlForItem" in JS
    assert "showMissingImage" in JS
    assert "img.addEventListener('error', showMissingImage, { once: true })" in JS
    assert "agent-thumb-missing-image" in JS
    assert ".agent-thumb-missing-image" in CSS


def test_shared_channel_surfaces_abort_and_reject_stale_responses():
    for token in (
        "luxriotPromptFormChannelId",
        "luxriotPromptRequestGeneration",
        "luxriotSummaryRequestGeneration",
        "luxriotSummaryActiveRequest",
        "archiveEvidenceRequestGeneration",
        "archiveFilterRequestGeneration",
        "archiveReviewRequestGeneration",
        "new AbortController()",
        "signal: requestContext.controller.signal",
    ):
        assert token in JS
    assert "The selected channel changed. Its prompt settings were reloaded" in JS
    assert "getSelectedSummaryChannel() === requestContext.channelId" in JS
    assert "archiveReviewContext === requestContext.context" in JS
    assert "function invalidateArchiveResultContext" in JS
    summary_poll = JS.split("function startLuxriotSummaryPoll()", 1)[1].split(
        "async function startLuxriotCapture", 1
    )[0]
    assert "stopLuxriotSummaryPoll()" not in summary_poll


def test_agent_cards_surface_coverage_and_incomplete_scope():
    assert "function appendAgentCompleteness" in JS
    assert "toolName === 'list_video_summary_channels'" in JS
    for label in (
        "Coverage: not reported by the backend",
        "Backend truncated:",
        "Result truncation:",
        "Unchecked:",
        "Deferred:",
        "Errors:",
    ):
        assert label in JS
    search_card = JS.split("if (toolName === 'search_archive')", 1)[1].split("toolName === 'get_detections'", 1)[0]
    assert "alwaysCoverage: true" in search_card
    assert "alwaysTruncation: true" in search_card


def test_agent_sidebar_polls_analytics_runtime_and_lm_admission_only_when_active():
    assert "Analytics Streams" in JS
    assert "not the full camera inventory" in JS
    assert "agentLoadAnalyticsStreams" in JS
    assert "agentSetContextActive" in JS
    assert "window._agentSetActive = agentSetContextActive" in JS
    assert "fetch(`/luxriot/streams?t=${Date.now()}`" in JS
    assert "fetch(`/lm/admission?t=${Date.now()}`" in JS
    assert "oldest_queue_age_sec" in JS
    assert "currentMode !== 'agent'" in JS


def test_operator_media_uses_same_origin_broker_and_explicit_player_states():
    assert "fetch(normalizedUrl" in JS
    assert "normalizedUrl.startsWith('/luxriot/media/')" in JS
    assert "normalizedUrl.startsWith('/luxriot/attention_stream/')" in JS
    assert "`/luxriot/archive_snapshot/${encodeURIComponent(String(channelId))}" in JS
    assert "method: 'HEAD'" in JS
    assert "Media broker URL must be same-origin" in JS
    for state in ("'loading'", "'playing'", "'degraded'", "'error'"):
        assert state in JS
    assert "Static frame fallback — not video" in JS
    assert "this is not video or a snapshot slideshow" in JS
    assert "function startLuxriotLegacySnapshotPreview" not in JS
    assert "{luxriot_base_url_json}" not in JS
    assert "luxriotDefaults.baseUrl" not in JS.split("function luxriotMediaBrokerUrl", 1)[1].split(
        "function setRoadSceneGroundingConfidence", 1
    )[0]


def test_operator_media_rejects_late_channel_and_archive_player_responses():
    assert "requestSeq === luxriotPreviewRequestSeq" in JS
    assert "generation === probePreviewGeneration" in JS
    assert "requestContext.generation === archiveMediaRequestGeneration" in JS
    assert "archiveReviewFrameIdentity(requestContext.result) === requestContext.identity" in JS
    assert "abortUiRequest(archiveMediaAbortController)" in JS
    assert "abortUiRequest(probePreviewAbortController)" in JS


def test_bounded_live_media_is_renewed_before_freeze_and_stalls_have_watchdogs():
    assert "numericHeader('X-EVA-Media-Renew-After-Ms')" in JS
    for token in (
        "function scheduleLuxriotPreviewRenewal",
        "function armLuxriotPreviewStallWatchdog",
        "function scheduleProbePreviewRenewal",
        "function armProbePreviewStallWatchdog",
        "startProbePreview(channelId, true)",
        "video.onprogress = clearLuxriotPreviewStallWatchdog",
        "video.ontimeupdate = clearLuxriotPreviewStallWatchdog",
        "video.onprogress = clearProbePreviewStallWatchdog",
        "video.ontimeupdate = clearProbePreviewStallWatchdog",
    ):
        assert token in JS

    live_preview = JS.split("function startLuxriotPreview(", 1)[1].split(
        "function setRoadSceneGroundingConfidence", 1
    )[0]
    assert "negotiated.renewAfterMs" in live_preview
    assert "Renewing the bounded MJPEG connection" in live_preview
    assert "Renewing the bounded video connection" in live_preview
    assert "options.reuseNegotiation" in live_preview
    assert "Promise.resolve(cachedNegotiation)" in live_preview
    assert "startLuxriotPreview({ reuseNegotiation: true })" in JS

    stop_preview = JS.split("function stopLuxriotPreview", 1)[1].split(
        "function setLuxriotPreviewSignalLost", 1
    )[0]
    assert "clearTimeout(luxriotPreviewRenewTimer)" in stop_preview
    assert "clearTimeout(luxriotPreviewStallTimer)" in stop_preview

    probe_preview = JS.split("function startProbePreview", 1)[1].split(
        "function syncProbePreview", 1
    )[0]
    assert "!force" in probe_preview
    assert "negotiated.renewAfterMs" in probe_preview
    assert "probePreviewNegotiation" in probe_preview
    assert "Promise.resolve(cachedNegotiation)" in probe_preview


def test_running_analytics_uses_shared_attention_preview_unless_operator_requests_full_live():
    for token in (
        "/luxriot/attention_stream/",
        "X-EVA-Attention-Preview",
        "attentionPreview",
        "useAttentionPreview",
        "luxriotPreferFullOperatorMedia",
        "maybeSwitchLuxriotPreviewToAttention",
        "Full live",
        "Model view",
        "no second recorder stream competes with analytics",
        "replaceLuxriotPreviewImageElement",
        "replaceProbePreviewImageElement",
    ):
        assert token in JS
    assert "Boolean(videoStream?.running) && !luxriotPreferFullOperatorMedia" in JS
    assert "Boolean(sharedVideoStream?.running)" in JS
    assert "currentSource.includes('/luxriot/attention_stream/')" in JS


def test_live_runtime_surfaces_completed_and_inflight_attention_throughput():
    for token in (
        "last_live_segment_target_seconds",
        "last_live_segment_summary_target_seconds",
        "last_live_segment_represented_seconds",
        "live_segment_inflight_target_seconds",
        "live_segment_inflight_raw_frame_budget",
        "live_segment_inflight_frames",
        "live_segment_inflight_represented_seconds",
        "attentionRealtimeRatio",
        "attentionUnderfilled",
        "attentionBehindRealtime",
        "label: 'apex-lag'",
        "dense frames",
        "descriptions every",
        "Dense capture progress",
        "x realtime",
    ):
        assert token in JS
    assert "activeCaptureSource !== 'live_segment'" in JS
    assert "currentSnapshotSlow = source !== 'live_segment'" in JS


def test_expected_summary_backpressure_is_not_rendered_as_a_capture_failure():
    for token in (
        "function classifyLuxriotStreamIssue",
        "summary queue overflow",
        "label: 'backpressure'",
        "label: 'aggregating'",
        "Aggregation backpressure: capture continues",
        "aggregation backpressure",
        "Boolean(streamIssue.hardError)",
        "detailParts.push('backpressure')",
    ):
        assert token in JS
    health = JS.split("function getLuxriotStreamHealth", 1)[1].split(
        "function renderLuxriotHealthBadge", 1
    )[0]
    assert "if (issue.hardError)" in health
    assert "if (issue.backpressure || droppedBatches > 0)" in health
    assert health.index("if (issue.hardError)") < health.index("if (issue.backpressure || droppedBatches > 0)")


def test_archive_media_loops_the_complete_description_batch():
    assert "numericHeader('X-Stream-Last-Sample-Timestamp')" in JS
    assert "numericHeader('X-EVA-Archive-Resolved-Time-Ms')" in JS
    assert "numericHeader('X-EVA-Archive-Duration-Seconds')" in JS
    assert "X-EVA-Archive-Frame-Alignment" in JS
    assert "X-EVA-HTML5-Compatible" in JS
    assert "function archivePlaybackWindow" in JS
    assert "Math.ceil(batchSpanMs / 1000) + 1" in JS
    assert "Math.min(15, Math.ceil(batchSpanMs / 1000) + 1)" in JS
    assert "params.set('duration_sec'" in JS
    assert "durationSec * 3000 + 15000" in JS
    assert "function fetchLuxriotMediaBlob" in JS
    assert "URL.createObjectURL(negotiated.blob)" in JS
    assert "URL.revokeObjectURL(archiveMediaObjectUrl)" in JS
    assert ".archive-review-frame .feature-btn[hidden]" in CSS
    assert "video.loop = true" in JS
    assert "video.currentTime = 0" in JS
    assert "isCurrentArchiveMediaRequest(requestContext)" in JS
    assert "Loading the next recorded archive segment" not in JS


def test_burst_attention_is_visible_in_summary_and_archive_ui():
    for token in (
        "function summaryBurstAttention",
        "function renderSummaryBurstAttentionChip",
        "capture_attention",
        "⚡ burst ×",
        "Motion far above this channel's measured norm; snapshot numbers:",
        "function archiveFrameRoleLabel",
        "burst apex",
        "sharper companion (burst)",
        "archive-review-strip-attention",
        "Burst attention frame",
    ):
        assert token in JS
    assert ".summary-attention-chip" in CSS
    assert ".result-badge.attention" in CSS
    assert ".archive-review-strip-frame .archive-review-strip-attention" in CSS


def test_rollup_aggregation_progress_is_targeted_generation_safe_and_cleared():
    rollups = JS.split("async function refreshLuxriotRollups", 1)[1].split(
        "async function refreshLuxriotSummaryView", 1
    )[0]
    assert "params.set('target_level', targetLevel)" in rollups
    assert "`Aggregating ${targetLevel}…`" in rollups
    assert "renderAggregationProgress" in rollups
    assert "isCurrentLuxriotSummaryRequest(requestContext)" in rollups
    assert "shared LM queue" in rollups
    assert "clearInterval(progressTimer)" in rollups


def test_prompt_and_environment_setting_sources_are_operator_visible():
    assert "luxriotPromptSettingSources" in JS
    assert "persisted runtime default" in JS
    assert "persistence error:" in JS
    assert "saved revision" in JS
    assert "different_process_and_file_keys" in JS
    assert "declared_file_matches_project" in JS
    assert ".settings-status.warning" in CSS


def test_prompt_apply_only_posts_fields_changed_from_loaded_channel_settings():
    collect = JS.split("function collectLuxriotPromptSettings", 1)[1].split(
        "function applyLuxriotPromptSettingsFromPayload", 1
    )[0]
    assert "luxriotPromptLoadedSettings" in collect
    assert "const payload = {}" in collect
    assert "const changedRollups = {}" in collect
    assert "payload.rollup_prompts = changedRollups" in collect
    assert "current.bookmark_enabled" in collect
    assert "baseline.bookmark_enabled" in collect


def test_prompt_modal_can_explicitly_reset_channel_overrides_to_inherited_defaults():
    assert 'id="luxriotPromptResetBtn"' in TEMPLATE
    assert "function getClearableLuxriotPromptOverrideFields" in JS
    assert "function resetLuxriotPromptOverrides" in JS
    assert "clear_override_fields: clearOverrideFields" in JS
    assert "use inherited defaults" in JS.lower()
