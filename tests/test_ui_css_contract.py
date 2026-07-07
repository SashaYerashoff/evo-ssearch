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
    assert "grid-template-rows: auto minmax(0, 1fr) auto auto minmax(0, 0.55fr);" in section
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
