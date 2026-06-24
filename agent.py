"""
agent.py — AI agent backend for evo-ssearch.

Provides:
  AgentTools   — tool dispatcher (one method per tool)
  AgentRunner  — tool-calling loop + SSE streaming generator
  _AgentLMClient — LM Studio client (tools + streaming), separate from _call_lm_chat

Instantiated once in oldapp.py after all globals are ready, injected with
callables so this module never imports from oldapp.py.
"""

from __future__ import annotations

import base64
import collections
import copy
import json
import queue
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Mapping, Optional, Sequence, Tuple, cast
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

from agent_security import ToolExecutionContext, ToolGatewayError
from agent_security.audit import ToolAuditEvent
from agent_security.eva_adapter import EvaAgentToolAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_MAX_HISTORY_MESSAGES = 20       # last N messages kept as context
AGENT_HEARTBEAT_INTERVAL   = 15       # seconds between SSE heartbeats
AGENT_SESSION_TTL_DAYS     = 30       # sessions older than this are GC'd
AGENT_MAX_SESSIONS         = 100      # sessions kept per store (GC oldest)
AGENT_MAX_MESSAGES_PER_SESSION = 200  # messages kept per session (prune oldest)
AGENT_MAX_RUNTIME_SKILLS_CHARS = 3_500
AGENT_MAX_ACTIVE_SKILL_CHARS   = 6_000
AGENT_MAX_PROBES_IN_PROMPT     = 12
AGENT_MAX_TOOL_CALLS_PER_TURN  = 64
AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN = 8
AGENT_VIDEO_SUMMARY_DEFAULT_LEVEL_LIMIT = 500
AGENT_VIDEO_SUMMARY_MAX_LEVEL_LIMIT = 2_000
ARCHIVE_SOURCE_LABELS = {
    "probe": "Probe hit",
    "vlm_summary": "Video-description frame",
    "vlm_alert": "VLM alert frame",
}
ARCHIVE_SOURCE_ITEM_TYPES = {
    "probe": "probe_detection",
    "vlm_summary": "video_description_frame",
    "vlm_alert": "video_description_alert",
}
ARCHIVE_SOURCE_ALIASES = {
    "detection": "probe",
    "detections": "probe",
    "probe_hit": "probe",
    "probe_hits": "probe",
    "probes_run": "probe",
    "probes_query": "probe",
    "probe_daemon": "probe",
    "summary": "vlm_summary",
    "video_summary": "vlm_summary",
    "video_summaries": "vlm_summary",
    "video_description": "vlm_summary",
    "video_descriptions": "vlm_summary",
    "alert": "vlm_alert",
    "alerts": "vlm_alert",
    "vlm_summary_frame": "vlm_summary",
    "vlm_alert_frame": "vlm_alert",
}


def _strip_image_data_url_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("data:image/") and "," in text:
        return text.split(",", 1)[1].strip()
    return text


def _image_data_url(value: Any, default_mime: str = "image/jpeg") -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.lower().startswith("data:image/"):
        return text
    return f"data:{default_mime};base64,{text}"


_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_archive",
            "description": (
                "Semantic text search over either an indexed folder (CLIP/FAISS) or "
                "the frame archive (vector search over probe hits and video-description frames). "
                "Returns ranked results with image URLs/previews, similarity scores, source labels, and metadata. "
                "When source is vlm_summary or vlm_alert, the result is from video descriptions, not a probe detection. "
                "TODO v1.1: add image_path/detection_id params for image-similarity search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query, e.g. 'person in red jacket near door'.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["indexed_folder", "detections"],
                        "description": (
                            "indexed_folder: searches a FAISS-indexed image directory. "
                            "detections: searches the frame archive via stored CLIP vectors."
                        ),
                    },
                    "folder": {
                        "type": "string",
                        "description": "Required when scope=indexed_folder. Absolute path to an indexed folder.",
                    },
                    "probe_id": {
                        "type": "string",
                        "description": "Optional. When scope=detections, restrict to one archive item/probe id.",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["probe", "vlm_summary", "vlm_alert"],
                        "description": (
                            "Optional archive source filter. probe = probe hits/detections; "
                            "vlm_summary = regular frames saved from video-description batches; "
                            "vlm_alert = frames anchored to video-description alerts."
                        ),
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict archive search to this channel.",
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Only include results from the past N hours. Default: 24.",
                    },
                    "since_ms": {
                        "type": "integer",
                        "description": "Optional absolute lower timestamp bound in Unix milliseconds.",
                    },
                    "until_ms": {
                        "type": "integer",
                        "description": "Optional absolute upper timestamp bound in Unix milliseconds.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["similarity", "time"],
                        "description": "similarity: semantic relevance. time: newest first.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["clip", "dino", "fusion"],
                        "description": "Frame archive search only. clip: fast/default. dino/fusion: richer visual matching when available.",
                    },
                    "candidate_limit": {
                        "type": "integer",
                        "description": "Frame archive search only. Candidate pool before final ranking. Default: 20000.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return. Default: 12, max: 48.",
                    },
                },
                "required": ["query", "scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_visual_window_signals",
            "description": (
                "Compare a positive and optional negative semantic phrase against CLIP-indexed "
                "video-description archive frames for one channel/time window. Returns P/N/M "
                "(positive score, negative score, margin) as an attention signal only, not proof. "
                "Use this to choose candidate windows/frames before calling describe_frame."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Luxriot channel ID to score.",
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title.",
                    },
                    "positive_query": {
                        "type": "string",
                        "description": "Visible-event phrase to look for, e.g. 'dog without visible ear tag'.",
                    },
                    "negative_query": {
                        "type": "string",
                        "description": "Optional contrast phrase, e.g. 'empty street with no dogs'.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["vlm_summary", "vlm_alert", "probe"]},
                        "description": (
                            "Archive sources to compare. Default: ['vlm_alert','vlm_summary']; "
                            "probe should be used only when the operator explicitly asks for probe corroboration."
                        ),
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Score frames from the past N hours. Default: 6.",
                    },
                    "since_ms": {
                        "type": "integer",
                        "description": "Optional absolute lower timestamp bound in Unix milliseconds.",
                    },
                    "until_ms": {
                        "type": "integer",
                        "description": "Optional absolute upper timestamp bound in Unix milliseconds.",
                    },
                    "limit_per_source": {
                        "type": "integer",
                        "description": "Top results to inspect per query/source. Default: 8, max: 24.",
                    },
                    "candidate_limit": {
                        "type": "integer",
                        "description": "Candidate pool before CLIP ranking. Default: 20000.",
                    },
                },
                "required": ["positive_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detections",
            "description": (
                "Retrieve recent archive frame records. By default this includes probe hits, "
                "video-description frames, and VLM alert frames. Use source=probe when you need "
                "actual probe detections only. Returns timestamps, source labels, scores where present, "
                "and preview image URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {
                        "type": "string",
                        "description": "Archive item/probe ID to filter by. If omitted, returns records across all archive items.",
                    },
                    "probe_name": {
                        "type": "string",
                        "description": "Human-readable probe name. Resolved to probe_id internally.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict to archive records from this channel.",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["probe", "vlm_summary", "vlm_alert"],
                        "description": (
                            "Optional archive source filter. Use source=probe for real probe detections, "
                            "source=vlm_summary for regular video-description frames, and source=vlm_alert for VLM alert frames."
                        ),
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Fetch detections from the past N hours. Default: 24.",
                    },
                    "until_hours": {
                        "type": "number",
                        "description": "Optional upper bound: detections older than N hours ago.",
                    },
                    "since_ms": {
                        "type": "integer",
                        "description": "Optional absolute lower timestamp bound in Unix milliseconds.",
                    },
                    "until_ms": {
                        "type": "integer",
                        "description": "Optional absolute upper timestamp bound in Unix milliseconds.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset within the selected time window.",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["newest", "oldest"],
                        "description": "Order detections by event time. Default: newest.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max detections to return. Default: 20, max: 100.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detection_summary",
            "description": (
                "Aggregate archive frame statistics grouped by archive item/probe. Returns counts, "
                "source labels, and latest activity timestamps. Use source=probe for actual probe detection summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since_hours": {
                        "type": "number",
                        "description": "Summarize detections from the past N hours. Default: 24.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict summary to one channel.",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["probe", "vlm_summary", "vlm_alert"],
                        "description": "Optional archive source filter: probe, vlm_summary, or vlm_alert.",
                    },
                    "since_ms": {
                        "type": "integer",
                        "description": "Optional absolute lower timestamp bound in Unix milliseconds.",
                    },
                    "until_ms": {
                        "type": "integer",
                        "description": "Optional absolute upper timestamp bound in Unix milliseconds.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_channels",
            "description": "List available Luxriot channels with IDs and titles. Use before deployment, survey, or prompt tuning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "If true, refresh channel list from Luxriot instead of relying on cache.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_time_window",
            "description": (
                "Convert an operator-facing time range into exact Unix seconds/milliseconds. "
                "Use before archive or video-summary tools when the user gives relative or local times "
                "such as 'last day', 'last night from 1:30am to 8:30am', or 'past 90 minutes'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Optional local date in YYYY-MM-DD. If omitted, day_hint is used.",
                    },
                    "day_hint": {
                        "type": "string",
                        "enum": ["today", "yesterday", "last_night", "explicit"],
                        "description": "How to choose the date when date is omitted. Default: today.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Local start time, e.g. '01:30', '1:30am', or '23:00'.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Local end time, e.g. '08:30', '8:30am', or '02:00'.",
                    },
                    "relative_range": {
                        "type": "string",
                        "description": "Optional relative range such as 'last day' (rolling 24h), 'last two hours', 'past 90 minutes', or 'last hour'. Prefer this for phrases like 'during the last two hours'.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone. Default: Europe/Riga.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_video_summary_channels",
            "description": (
                "Inventory which Luxriot channels have VLM video-summary coverage in a time window. "
                "Use this before broad video-description review when the operator did not name channels. "
                "Returns metadata and candidate channels, not full summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional channel IDs to inspect. Omit to inspect all available authorized channels.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["live", "L1", "L2", "L3"],
                        "description": "Summary level to count. Default: L1.",
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Return coverage from the past N hours. Default: 6.",
                    },
                    "from_ts": {
                        "type": "number",
                        "description": "Optional absolute lower timestamp bound in Unix seconds. Milliseconds are accepted and normalized.",
                    },
                    "to_ts": {
                        "type": "number",
                        "description": "Optional absolute upper timestamp bound in Unix seconds. Milliseconds are accepted and normalized.",
                    },
                    "run": {
                        "type": "string",
                        "description": "Optional run selector: latest, running, all, or a concrete run id.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum candidate channels to return. Default: 16, max: 100.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_probes",
            "description": "List configured probes with their IDs, channels, thresholds, and recent hit counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "since_hours": {
                        "type": "number",
                        "description": "Recent window used for hit counts. Default: 24.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "survey_channels",
            "description": (
                "Capture a short batch of snapshots from one or more channels over 10-15 seconds "
                "and summarize what each camera is looking at."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional explicit list of channel IDs. Omit to survey all available channels.",
                    },
                    "fast_mode": {
                        "type": "boolean",
                        "description": "If true, use a shorter demo survey with fewer samples and less wait time.",
                    },
                    "duration_sec": {
                        "type": "number",
                        "description": "Approximate capture duration per channel. Default: 12 seconds.",
                    },
                    "sample_count": {
                        "type": "integer",
                        "description": "How many snapshots to collect per channel. Default: 4.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional instruction for the VLM summarizer.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_research_batch",
            "description": (
                "Assemble a representative batch of detections for agent research across multiple time windows "
                "and confidence bands. Use before probe tuning, archive investigations, or prompt refinement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {
                        "type": "string",
                        "description": "Optional probe ID to restrict the batch.",
                    },
                    "probe_name": {
                        "type": "string",
                        "description": "Optional human-readable probe name, resolved to probe_id internally.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional channel to restrict the batch.",
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Default window lower bound if periods are not provided. Default: 24.",
                    },
                    "until_hours": {
                        "type": "number",
                        "description": "Optional relative upper bound if periods are not provided.",
                    },
                    "since_ms": {
                        "type": "integer",
                        "description": "Optional absolute lower timestamp bound if periods are not provided.",
                    },
                    "until_ms": {
                        "type": "integer",
                        "description": "Optional absolute upper timestamp bound if periods are not provided.",
                    },
                    "periods": {
                        "type": "array",
                        "description": "Optional named time slices to sample independently.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "since_hours": {"type": "number"},
                                "until_hours": {"type": "number"},
                                "since_ms": {"type": "integer"},
                                "until_ms": {"type": "integer"}
                            },
                            "additionalProperties": False
                        },
                    },
                    "bands": {
                        "type": "array",
                        "description": "Optional confidence bands for stratified sampling.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "score_field": {
                                    "type": "string",
                                    "enum": ["pos_score", "margin", "neg_score"]
                                },
                                "min": {"type": "number"},
                                "max": {"type": "number"}
                            },
                            "additionalProperties": False
                        },
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["newest", "oldest", "highest_pos", "lowest_pos", "highest_margin", "lowest_margin"],
                        "description": "How to rank candidates before sampling within each period/band.",
                    },
                    "per_period_limit": {
                        "type": "integer",
                        "description": "Max detections to keep per period before the final merge. Default: 24.",
                    },
                    "per_band_limit": {
                        "type": "integer",
                        "description": "Max detections to keep per band inside each period. Default: 6.",
                    },
                    "max_candidates": {
                        "type": "integer",
                        "description": "Max raw detections to scan per period before sampling. Default: 1000.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_probe",
            "description": (
                "Create a new probe with text pairs and thresholds. "
                "IMPORTANT: call with preview=true first unless the operator explicitly authorized direct deployment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "channel_id": {"type": "integer"},
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "positives": {"type": "array", "items": {"type": "string"}},
                    "negatives": {"type": "array", "items": {"type": "string"}},
                    "pos_floor": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "margin_thr": {"type": "number", "minimum": 0.0},
                    "top_k": {"type": "integer", "minimum": 1},
                    "window_sec": {"type": "number", "minimum": 0.0},
                    "severity": {
                        "type": "string",
                        "enum": ["info", "low", "normal", "high", "critical"],
                    },
                    "bookmark_enabled": {"type": "boolean"},
                    "bookmark_cooldown_sec": {"type": "number", "minimum": 0.0},
                    "bookmark_dedupe_window_sec": {"type": "number", "minimum": 0.5},
                    "enabled": {"type": "boolean"},
                    "update_existing": {
                        "type": "boolean",
                        "description": "If true, reuse an existing probe with the same name on the same channel instead of creating a duplicate. Default: true.",
                    },
                    "preview": {"type": "boolean"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deploy_summary",
            "description": (
                "Record the final outcome of a deployment or survey-only pass as a structured summary card. "
                "Use at the end of Protocol Deploy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["standard", "magic", "survey_only"],
                    },
                    "wipe": {"type": "boolean"},
                    "elapsed_sec": {"type": "number"},
                    "overview": {"type": "string"},
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "probes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "prompt_targets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "notes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_probes",
            "description": (
                "Delete one or more probes, or all probes. "
                "IMPORTANT: call with preview=true first unless the operator explicitly authorized destructive deployment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit probe IDs to delete.",
                    },
                    "delete_all": {
                        "type": "boolean",
                        "description": "If true, delete all configured probes.",
                    },
                    "preview": {
                        "type": "boolean",
                        "description": "If true, return a deletion plan without applying. Default: true.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_probe",
            "description": (
                "Modify a probe's configuration: text pairs (positives/negatives), "
                "detection thresholds, or enabled state. "
                "IMPORTANT: always call with preview=true first to show the user a diff. "
                "Only call with preview=false after the user has explicitly confirmed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_name": {
                        "type": "string",
                        "description": "Human-readable name of the probe to modify.",
                    },
                    "probe_id": {
                        "type": "string",
                        "description": "Probe ID. Use if probe_name is ambiguous.",
                    },
                    "changes": {
                        "type": "object",
                        "description": "Partial update — only the fields to change. All other probe settings are preserved.",
                        "properties": {
                            "positives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "New list of positive text descriptions (replaces current list).",
                            },
                            "negatives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "New list of negative/exclusion text descriptions.",
                            },
                            "pos_floor": {
                                "type": "number",
                                "description": "Minimum positive similarity score to trigger (0.0–1.0).",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "margin_thr": {
                                "type": "number",
                                "description": "Minimum margin (pos_score - neg_score) to trigger.",
                                "minimum": 0.0,
                            },
                            "top_k": {"type": "integer", "minimum": 1},
                            "window_sec": {"type": "number", "minimum": 0.0},
                            "enabled": {"type": "boolean"},
                            "severity": {
                                "type": "string",
                                "enum": ["info", "low", "normal", "high", "critical"],
                            },
                            "bookmark_enabled": {"type": "boolean"},
                            "bookmark_cooldown_sec": {"type": "number", "minimum": 0.0},
                            "bookmark_dedupe_window_sec": {"type": "number", "minimum": 0.5},
                        },
                        "additionalProperties": False,
                    },
                    "preview": {
                        "type": "boolean",
                        "description": "If true, return a diff without applying. If false, apply the changes. Default: true.",
                    },
                },
                "required": ["changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_frame",
            "description": (
                "Send an image frame to the vision language model for a detailed description. "
                "Accepts a live camera snapshot (channel_id), a detection record (detection_id), "
                "or a filesystem path (image_path). "
                "Use to understand what is happening on camera right now, or to analyze a past detection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": (
                            "Luxriot channel ID. Captures a live snapshot from the camera right now "
                            "and describes it. Use when the operator asks about the current scene."
                        ),
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Absolute filesystem path to an image file.",
                    },
                    "detection_id": {
                        "type": "integer",
                        "description": "Detection record ID. The stored thumbnail will be used.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional instruction to the VLM. Defaults to a general scene description.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prompt_settings",
            "description": (
                "Read the effective Luxriot VLM prompt settings. "
                "Without channel_id returns global defaults. With channel_id returns the effective per-channel view."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional Luxriot channel ID. Omit to read global defaults.",
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_prompt_settings",
            "description": (
                "Modify Luxriot VLM prompt settings for either global defaults or a single channel. "
                "IMPORTANT: always call with preview=true first and apply only after explicit confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional Luxriot channel ID. Omit to update global defaults.",
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "changes": {
                        "type": "object",
                        "description": (
                            "Partial prompt/settings update. "
                            "L0/live descriptions use stream_system_prompt. "
                            "L1/L2/L3 summaries use rollup_prompts. "
                            "Behavioral bookmark instructions belong inside the L0/live prompt. "
                            "json_alert_prompt is only the structured alert-output template."
                        ),
                        "properties": {
                            "stream_system_prompt": {"type": "string"},
                            "l0_prompt": {"type": "string"},
                            "live_prompt": {"type": "string"},
                            "json_alert_prompt": {"type": "string"},
                            "bookmark_rule_prompt": {
                                "type": "string",
                                "description": "A bookmark/alert instruction line to add to the L0/live stream prompt.",
                            },
                            "bookmark_enabled": {"type": "boolean"},
                            "bookmark_cooldown_sec": {"type": "number", "minimum": 0.0},
                            "l1_prompt": {"type": "string"},
                            "l2_prompt": {"type": "string"},
                            "l3_prompt": {"type": "string"},
                            "rollup_prompts": {
                                "type": "object",
                                "description": "Partial L1/L2/L3 prompt overrides.",
                                "properties": {
                                    "L1": {"type": "string"},
                                    "L2": {"type": "string"},
                                    "L3": {"type": "string"}
                                },
                                "additionalProperties": False
                            },
                        },
                        "additionalProperties": False,
                    },
                    "preview": {
                        "type": "boolean",
                        "description": "If true, return a diff without applying. If false, apply the changes. Default: true.",
                    },
                },
                "required": ["changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_bookmark",
            "description": (
                "Create a bookmark in Luxriot EVA on a camera channel, visible in the Luxriot client. "
                "Use to flag significant events for operator review. "
                "Only call when there is a concrete reason — do not create bookmarks speculatively."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Luxriot channel (camera) ID to attach the bookmark to.",
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short bookmark title (max 80 characters).",
                        "maxLength": 80,
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed event description (max 240 characters).",
                        "maxLength": 240,
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["info", "low", "normal", "high", "critical"],
                        "description": "Event severity level. Default: normal.",
                    },
                    "timestamp_ms": {
                        "type": "integer",
                        "description": "Optional event timestamp in milliseconds. Defaults to now.",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_summaries",
            "description": (
                "Retrieve VLM-generated video summaries for a Luxriot channel. "
                "Returns narrative text summaries at different depths: "
                "live (L0, per-batch captions), L1 (minute-level), L2 (hour-level), L3 (day-level). "
                "Use to answer 'what happened on camera X in the last hour?' without re-analyzing frames."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Luxriot channel ID to query.",
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["live", "L1", "L2", "L3"],
                        "description": (
                            "live: raw per-batch captions (most detail, most entries). "
                            "L1: minute-window rollups. L2: hour rollups. L3: day rollups. "
                            "Default: L1."
                        ),
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Return summaries from the past N hours. Default: 6.",
                    },
                    "from_ts": {
                        "type": "number",
                        "description": "Optional absolute lower timestamp bound in Unix seconds.",
                    },
                    "to_ts": {
                        "type": "number",
                        "description": "Optional absolute upper timestamp bound in Unix seconds.",
                    },
                    "run": {
                        "type": "string",
                        "description": "Optional run selector: latest, running, or a concrete run id.",
                    },
                    "level_limit": {
                        "type": "integer",
                        "description": (
                            "Max nodes per rollup level to scan before slicing the requested depth. "
                            "Defaults to a high bounded scan independent of display limit."
                        ),
                    },
                    "include_evidence_frames": {
                        "type": "boolean",
                        "description": (
                            "If true, also return representative archive frame records from source=vlm_summary "
                            "and source=vlm_alert for the requested channel/time window. Use when the operator "
                            "asks to confirm video-summary findings with images/snaps."
                        ),
                    },
                    "evidence_frame_limit": {
                        "type": "integer",
                        "description": "Max evidence frames to include when include_evidence_frames=true. Default: 8, max: 24.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max summary entries to return. Default: 20.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_video_summary_events",
            "description": (
                "Count appearance/disappearance style events from VLM video-summary text for one channel/time window. "
                "Use for questions like 'how many times did X appear/disappear', 'count entries/exits', or "
                "'how often did an object leave/return'. The result is a structured count from summaries, with "
                "coverage and transition evidence; it is not frame-level ground truth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Luxriot channel ID to query.",
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "entity_query": {
                        "type": "string",
                        "description": "Entity/object to track, e.g. 'person', 'white BMW', 'delivery courier', or 'animal'.",
                    },
                    "anchor_query": {
                        "type": "string",
                        "description": "Optional place/object relation, e.g. 'front door', 'loading bay', 'gate', or 'central square'.",
                    },
                    "event_kind": {
                        "type": "string",
                        "enum": ["presence_transitions"],
                        "description": "Currently supports presence/absence transitions. Default: presence_transitions.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["live", "L1", "L2", "L3"],
                        "description": "Summary level to scan. Use L1 for day-scale counts, live/L0 only for short exact windows.",
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Return summaries from the past N hours. Default: 6.",
                    },
                    "from_ts": {
                        "type": "number",
                        "description": "Optional absolute lower timestamp bound in Unix seconds. Milliseconds are accepted.",
                    },
                    "to_ts": {
                        "type": "number",
                        "description": "Optional absolute upper timestamp bound in Unix seconds. Milliseconds are accepted.",
                    },
                    "run": {
                        "type": "string",
                        "description": "Optional run selector: latest, running, all, or a concrete run id.",
                    },
                    "level_limit": {
                        "type": "integer",
                        "description": "Max nodes per rollup level to scan. Default: high bounded scan.",
                    },
                    "timeline_limit": {
                        "type": "integer",
                        "description": "Max classified timeline rows to return. Default: 40, max: 120.",
                    },
                    "event_limit": {
                        "type": "integer",
                        "description": "Max transition events to return. Default: 40, max: 120.",
                    },
                },
                "required": ["entity_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_visual_state_transitions",
            "description": (
                "Track visual state changes over archived video-description frames using CLIP P/N/M signals. "
                "Use for unexpected/count questions such as 'how many times did X appear/disappear', "
                "'when did a door open/close', or 'did an object leave/return'. Builds a timeline of stable "
                "visual state segments and returns boundary frame evidence. This is stronger than summary-text "
                "counts but still returns candidates, not legal/medical/intent conclusions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {"type": "integer", "description": "Luxriot channel ID to query."},
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "subject_query": {
                        "type": "string",
                        "description": "Optional subject label for reporting, e.g. 'delivery courier', 'white van', or 'animal'.",
                    },
                    "positive_state_query": {
                        "type": "string",
                        "description": "Visual state to count as positive, e.g. 'vehicle parked at gate' or 'person standing near doorway'.",
                    },
                    "negative_state_query": {
                        "type": "string",
                        "description": "Visible contrast state for CLIP scoring, e.g. 'empty gate' or 'empty doorway'. Avoid phrases like 'no person'/'without vehicle' when possible; literal negation is unreliable for CLIP.",
                    },
                    "alternate_state_query": {
                        "type": "string",
                        "description": "Optional third state, e.g. 'vehicle visible elsewhere, not at gate'.",
                    },
                    "positive_label": {
                        "type": "string",
                        "description": "State label for positive_state_query. Default: positive.",
                    },
                    "negative_label": {
                        "type": "string",
                        "description": "State label for negative_state_query. Default: negative.",
                    },
                    "alternate_label": {
                        "type": "string",
                        "description": "State label for alternate_state_query. Default: alternate.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["vlm_summary", "vlm_alert", "probe"]},
                        "description": "Archive frame sources to scan. Default: ['vlm_summary','vlm_alert'].",
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Scan frames from the past N hours. Default: 6.",
                    },
                    "from_ts": {
                        "type": "number",
                        "description": "Optional absolute lower timestamp bound in Unix seconds. Milliseconds are accepted.",
                    },
                    "to_ts": {
                        "type": "number",
                        "description": "Optional absolute upper timestamp bound in Unix seconds. Milliseconds are accepted.",
                    },
                    "candidate_limit": {
                        "type": "integer",
                        "description": "Max archive frames to scan across sources. Default: 20000, max: 100000.",
                    },
                    "positive_floor": {
                        "type": "number",
                        "description": "Minimum positive CLIP score for the positive state. Default: 0.18.",
                    },
                    "negative_floor": {
                        "type": "number",
                        "description": "Minimum negative CLIP score for the negative state. Default: 0.18.",
                    },
                    "margin_threshold": {
                        "type": "number",
                        "description": "Minimum CLIP margin between winning state and runner-up. Default: 0.03.",
                    },
                    "min_state_samples": {
                        "type": "integer",
                        "description": "Minimum samples for a stable segment. Default: 2.",
                    },
                    "min_state_duration_sec": {
                        "type": "number",
                        "description": "Minimum duration for a stable segment when sample count is low. Default: 2.",
                    },
                    "merge_gap_sec": {
                        "type": "number",
                        "description": "Merge short unknown gaps between same states. Default: 3.",
                    },
                    "transition_limit": {
                        "type": "integer",
                        "description": "Max transitions to return. Default: 40, max: 120.",
                    },
                    "segment_limit": {
                        "type": "integer",
                        "description": "Max segments to return. Default: 80, max: 200.",
                    },
                    "evidence_limit": {
                        "type": "integer",
                        "description": "Max boundary evidence frames to return. Default: 24, max: 48.",
                    },
                },
                "required": ["positive_state_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Compile a structured detection report for a time period. "
                "Aggregates per-probe statistics, highlights peak activity hours, "
                "and lists the most significant detection events. "
                "Use for daily/weekly summaries or to answer 'give me a report on this week'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since_hours": {
                        "type": "number",
                        "description": "Report covers detections from the past N hours. Default: 24.",
                    },
                    "until_hours": {
                        "type": "number",
                        "description": "Optional upper bound. Omit for up-to-now.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict report to one channel.",
                    },
                    "channel_ref": {
                        "type": "string",
                        "description": "Optional channel reference such as '#115', '115', or a title like 'stream'.",
                    },
                    "include_probes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Probe names to include. If omitted, includes all active probes.",
                    },
                    "top_events": {
                        "type": "integer",
                        "description": "Include the N highest-margin events per probe. Default: 5.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# LM client
# ---------------------------------------------------------------------------

@dataclass
class _ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class _LMResponse:
    content: Optional[str]
    tool_calls: List[_ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class _AgentLMClient:
    """
    Minimal OpenAI-compatible client for the tool-calling loop.
    Completely separate from _call_lm_chat in oldapp.py.
    """

    def __init__(self, base_url: str, model: str, api_key: str, timeout: int) -> None:
        if not base_url:
            raise ValueError("LM base URL is not configured (EVOSSEARCH_LM_BASE_URL).")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model    = model
        self.timeout  = timeout
        self.connect_timeout = min(15, max(5, int(timeout or 120)))
        self.read_timeout = max(int(timeout or 120), 900)
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> _LMResponse:
        """Blocking non-streaming call with tools. Returns parsed response."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": _TOOL_SCHEMAS if tools is None else tools,
            "tool_choice": "auto",
            "stream": False,
        }
        resp = requests.post(
            self.endpoint,
            json=payload,
            headers=self.headers,
            timeout=(self.connect_timeout, self.read_timeout),
        )
        resp.raise_for_status()
        data   = resp.json()
        choice = data["choices"][0]
        msg    = choice.get("message", {}) or {}
        finish = choice.get("finish_reason", "stop")

        # Parse tool_calls
        tool_calls: List[_ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError, TypeError):
                args = {}
            tool_calls.append(_ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                name=tc["function"]["name"],
                args=args,
            ))

        # content may be None on pure tool-call turns
        raw_content = msg.get("content")
        if isinstance(raw_content, list):
            raw_content = " ".join(
                p.get("text", "") for p in raw_content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip() or None

        return _LMResponse(
            content=raw_content,
            tool_calls=tool_calls,
            finish_reason=finish,
        )

    def stream_text(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        """Streaming call without tools. Yields text delta chunks."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        with requests.post(
            self.endpoint,
            json=payload,
            headers=self.headers,
            timeout=(self.connect_timeout, self.read_timeout),
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk  = json.loads(line)
                    delta  = chunk["choices"][0]["delta"]
                    text   = delta.get("content")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Raised by AgentTools to signal a user-facing error to the model."""


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class AgentTools:
    """
    Executes agent tools. All dependencies are injected at construction.
    Never imports from oldapp.py.
    """

    def __init__(
        self,
        *,
        detections_store: Any,
        probes_store: Any,
        luxriot_manager: Any,
        embed_text_fn: Callable[[str], np.ndarray],
        embed_image_fn: Callable[[Image.Image], np.ndarray],
        call_lm_fn: Callable[..., str],
        encode_jpeg_fn: Callable[..., str],
        search_indexed_folder_fn: Callable[..., List[Dict[str, Any]]],
        search_detections_fn: Callable[..., List[Dict[str, Any]]],
    ) -> None:
        self._ds   = detections_store
        self._ps   = probes_store
        self._lxm  = luxriot_manager
        self._emb_text  = embed_text_fn
        self._emb_image = embed_image_fn
        self._lm   = call_lm_fn
        self._jpeg = encode_jpeg_fn
        self._search_folder    = search_indexed_folder_fn
        self._search_det       = search_detections_fn
        self._local = threading.local()

    def execute(
        self,
        name: str,
        args: Dict[str, Any],
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Dispatch to the named tool. Returns a dict always."""
        dispatch = {
            "search_archive":       self._search_archive,
            "get_visual_window_signals": self._get_visual_window_signals,
            "get_detections":       self._get_detections,
            "get_detection_summary": self._get_detection_summary,
            "list_channels":        self._list_channels,
            "normalize_time_window": self._normalize_time_window,
            "list_video_summary_channels": self._list_video_summary_channels,
            "list_probes":          self._list_probes,
            "survey_channels":      self._survey_channels,
            "build_research_batch": self._build_research_batch,
            "create_probe":         self._create_probe,
            "deploy_summary":       self._deploy_summary,
            "delete_probes":        self._delete_probes,
            "update_probe":         self._update_probe,
            "describe_frame":       self._describe_frame,
            "get_prompt_settings":  self._get_prompt_settings,
            "update_prompt_settings": self._update_prompt_settings,
            "get_video_summaries":  self._get_video_summaries,
            "count_video_summary_events": self._count_video_summary_events,
            "track_visual_state_transitions": self._track_visual_state_transitions,
            "create_bookmark":      self._create_bookmark,
            "generate_report":      self._generate_report,
        }
        fn = dispatch.get(name)
        if fn is None:
            raise ToolError(f"Unknown tool: {name!r}")
        self._local.progress_cb = progress_cb
        try:
            return fn(args)
        finally:
            self._local.progress_cb = None

    # ── search_archive ──────────────────────────────────────────────────────

    def _search_archive(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query  = str(args.get("query") or "").strip()
        scope  = str(args.get("scope") or "detections")
        limit  = max(1, min(48, int(args.get("limit") or 12)))
        sort_by = str(args.get("sort_by") or "similarity").strip().lower()

        if not query:
            raise ToolError("'query' is required.")

        if scope == "indexed_folder":
            folder = str(args.get("folder") or "").strip()
            if not folder:
                raise ToolError("'folder' is required when scope='indexed_folder'.")
            if sort_by not in {"similarity", "time"}:
                sort_by = "similarity"
            results = self._search_folder(query=query, folder=folder, limit=limit, sort_by=sort_by)
            return {"scope": scope, "count": len(results), "results": _strip_thumbnails(results, folder=folder)}

        # scope == "detections"
        since_ms, until_ms = self._resolve_time_window(args, default_since_hours=24.0)
        probe_id    = str(args.get("probe_id") or "").strip() or None
        channel_id  = _opt_int(args.get("channel_id"))
        source = _normalize_archive_source(args.get("source"))
        mode = str(args.get("mode") or "clip").strip().lower()
        if mode not in {"clip", "dino", "fusion"}:
            mode = "clip"
        if sort_by not in {"similarity", "time"}:
            sort_by = "similarity"
        candidate_limit = max(limit, min(100_000, int(args.get("candidate_limit") or 20_000)))

        results = self._search_det(
            query=query,
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            sort_by=sort_by,
            candidate_limit=candidate_limit,
            mode=mode,
            include_coverage=True,
        )
        coverage = None
        if isinstance(results, Mapping):
            coverage_raw = results.get("coverage")
            coverage = dict(coverage_raw) if isinstance(coverage_raw, Mapping) else None
            results = results.get("results") if isinstance(results.get("results"), list) else []
        return {
            "scope": scope,
            "source": source,
            "source_label": _archive_source_label(source),
            "count": len(results),
            "results": _strip_thumbnails([_annotate_archive_row(result) for result in results]),
            "coverage": coverage,
        }

    # ── get_visual_window_signals ─────────────────────────────────────────────

    def _get_visual_window_signals(self, args: Dict[str, Any]) -> Dict[str, Any]:
        positive_query = str(args.get("positive_query") or args.get("query") or "").strip()
        negative_query = str(args.get("negative_query") or "").strip()
        if not positive_query:
            raise ToolError("'positive_query' is required.")
        channel_id = self._resolve_channel_id(args, required=True)
        since_ms, until_ms = self._resolve_time_window(args, default_since_hours=6.0)
        limit_per_source = max(1, min(24, int(args.get("limit_per_source") or 8)))
        candidate_limit = max(limit_per_source, min(100_000, int(args.get("candidate_limit") or 20_000)))

        raw_sources = args.get("sources")
        sources: List[str] = []
        if isinstance(raw_sources, Sequence) and not isinstance(raw_sources, (str, bytes, bytearray)):
            for raw_source in raw_sources:
                normalized = _normalize_archive_source(raw_source)
                if normalized and normalized not in sources:
                    sources.append(normalized)
        else:
            raw_single_source = args.get("source")
            if raw_single_source:
                normalized = _normalize_archive_source(raw_single_source)
                if normalized:
                    sources.append(normalized)
        if not sources:
            sources = ["vlm_alert", "vlm_summary"]

        by_source: List[Dict[str, Any]] = []
        frame_slots: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        search_errors: List[Dict[str, Any]] = []

        def run_query(source: str, query: str, polarity: str) -> List[Dict[str, Any]]:
            try:
                rows = self._search_det(
                    query=query,
                    probe_id=None,
                    channel_id=channel_id,
                    source=source,
                    since_ms=since_ms,
                    until_ms=until_ms,
                    limit=limit_per_source,
                    sort_by="similarity",
                    candidate_limit=candidate_limit,
                    mode="clip",
                )
            except Exception as exc:
                search_errors.append({
                    "source": source,
                    "polarity": polarity,
                    "error": str(exc),
                })
                return []
            prepared: List[Dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                prepared.append(_safe_detection(_annotate_archive_row(row)))
            return prepared

        for source in sources:
            positive_hits = run_query(source, positive_query, "positive")
            negative_hits = run_query(source, negative_query, "negative") if negative_query else []
            source_p = _best_search_score(positive_hits)
            source_n = _best_search_score(negative_hits) if negative_query else None
            source_margin = (
                float(source_p) - float(source_n)
                if source_p is not None and source_n is not None
                else None
            )

            for polarity, hits in (("positive", positive_hits), ("negative", negative_hits)):
                for hit in hits:
                    key = _visual_signal_row_key(hit)
                    slot = frame_slots.setdefault(
                        key,
                        {
                            "detection": hit,
                            "source": hit.get("source") or source,
                            "timestamp_ms": _detection_timestamp_ms(hit),
                            "positive_score": None,
                            "negative_score": None,
                        },
                    )
                    score = _search_result_score(hit)
                    if score is None:
                        continue
                    score_key = "positive_score" if polarity == "positive" else "negative_score"
                    current = slot.get(score_key)
                    if current is None or float(score) > float(current):
                        slot[score_key] = float(score)

            by_source.append(
                {
                    "source": source,
                    "source_label": _archive_source_label(source),
                    "positive_count": len(positive_hits),
                    "negative_count": len(negative_hits),
                    "pnm": _visual_signal_pnm(source_p, source_n),
                    "positive_hits": [_compact_visual_signal_hit(row, "positive") for row in positive_hits[:5]],
                    "negative_hits": [_compact_visual_signal_hit(row, "negative") for row in negative_hits[:5]],
                    "margin": source_margin,
                }
            )

        candidate_frames: List[Dict[str, Any]] = []
        for slot in frame_slots.values():
            pos_score = slot.get("positive_score")
            neg_score = slot.get("negative_score")
            margin = (
                float(pos_score) - float(neg_score)
                if pos_score is not None and neg_score is not None
                else None
            )
            detection = slot.get("detection") if isinstance(slot.get("detection"), dict) else {}
            candidate = _compact_detection_for_model(cast(Dict[str, Any], detection))
            candidate.update(
                {
                    "positive_score": pos_score,
                    "negative_score": neg_score,
                    "margin": margin,
                    "pnm_state": _visual_signal_state(pos_score, neg_score),
                    "needs_describe_frame": bool(candidate.get("image_url")),
                }
            )
            candidate_frames.append(candidate)

        candidate_frames.sort(
            key=lambda row: (
                row.get("margin") is not None,
                float(row.get("margin") if row.get("margin") is not None else -999.0),
                float(row.get("positive_score") if row.get("positive_score") is not None else -999.0),
                -int(row.get("timestamp_ms") or 0),
            ),
            reverse=True,
        )

        overall_p = _best_search_score([
            hit
            for source_row in by_source
            for hit in source_row.get("positive_hits", [])
            if isinstance(hit, dict)
        ])
        overall_n = _best_search_score([
            hit
            for source_row in by_source
            for hit in source_row.get("negative_hits", [])
            if isinstance(hit, dict)
        ]) if negative_query else None

        return {
            "channel_id": channel_id,
            "positive_query": positive_query,
            "negative_query": negative_query or None,
            "sources": sources,
            "since_ms": since_ms,
            "until_ms": until_ms,
            "limit_per_source": limit_per_source,
            "candidate_limit": candidate_limit,
            "score_semantics": "clip_retrieval_signal_not_proof",
            "pnm": _visual_signal_pnm(overall_p, overall_n),
            "by_source": by_source,
            "candidate_frames": candidate_frames[: max(1, min(12, limit_per_source))],
            "search_errors": search_errors,
            "operator_note": (
                "P/N/M is a CLIP attention signal, not visual proof. Use returned frame image_url "
                "with describe_frame before saying an event was visually confirmed."
            ),
        }

    # ── get_detections ──────────────────────────────────────────────────────

    def _get_detections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        probe_id = str(args.get("probe_id") or "").strip() or None

        # Resolve probe_name → probe_id if needed
        probe_name_raw = str(args.get("probe_name") or "").strip()
        if not probe_id and probe_name_raw:
            probe_id = self._resolve_probe_id_by_name(probe_name_raw)

        channel_id  = _opt_int(args.get("channel_id"))
        source = _normalize_archive_source(args.get("source"))
        since_ms, until_ms = self._resolve_time_window(args, default_since_hours=24.0)
        limit       = max(1, min(100, int(args.get("limit") or 20)))
        offset      = max(0, int(args.get("offset") or 0))
        sort_by     = str(args.get("sort_by") or "newest").strip().lower()
        if sort_by not in {"newest", "oldest"}:
            sort_by = "newest"

        rows, total = self._list_detection_window(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
        )

        return {
            "probe_id": probe_id,
            "channel_id": channel_id,
            "source": source,
            "source_label": _archive_source_label(source),
            "since_ms": since_ms,
            "until_ms": until_ms,
            "total_in_window": total,
            "returned": len(rows),
            "offset": offset,
            "sort_by": sort_by,
            "detections": [_safe_detection(_annotate_archive_row(r)) for r in rows],
        }

    # ── get_detection_summary ───────────────────────────────────────────────

    def _get_detection_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id  = _opt_int(args.get("channel_id"))
        source = _normalize_archive_source(args.get("source"))
        since_ms, until_ms = self._resolve_time_window(args, default_since_hours=24.0)

        if until_ms is None:
            rows = self._ds.summarize_by_probe(since_ms=since_ms, channel_id=channel_id, source=source)
        else:
            grouped: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
            offset = 0
            while True:
                batch, _total = self._ds.list_detections(
                    probe_id=None,
                    channel_id=channel_id,
                    source=source,
                    since_ms=since_ms,
                    until_ms=until_ms,
                    limit=500,
                    offset=offset,
                )
                if not batch:
                    break
                for row in batch:
                    row_source = str(row.get("source") or "").strip().lower()
                    key = (str(row.get("probe_id") or ""), int(row.get("channel_id") or 0), row_source)
                    slot = grouped.setdefault(key, {
                        "probe_id": row.get("probe_id"),
                        "probe_name": row.get("probe_name"),
                        "channel_id": row.get("channel_id"),
                        "source": row_source,
                        "source_label": _archive_source_label(row_source),
                        "archive_item_type": _archive_item_type(row_source),
                        "hit_count": 0,
                        "latest_timestamp_ms": 0,
                    })
                    slot["hit_count"] += 1
                    slot["latest_timestamp_ms"] = max(
                        int(slot["latest_timestamp_ms"] or 0),
                        _detection_timestamp_ms(row),
                    )
                offset += len(batch)
                if len(batch) < 500:
                    break
            rows = sorted(
                grouped.values(),
                key=lambda row: int(row.get("latest_timestamp_ms") or 0),
                reverse=True,
            )
        rows = [_annotate_archive_row(row) for row in rows]
        return {
            "since_ms": since_ms,
            "until_ms": until_ms,
            "source": source,
            "source_label": _archive_source_label(source),
            "probe_count": len(rows),
            "total_detections": sum(r.get("hit_count", 0) for r in rows),
            "total_archive_items": sum(r.get("hit_count", 0) for r in rows),
            "by_probe": rows,
        }

    # ── list_channels ──────────────────────────────────────────────────────

    def _list_channels(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self._lxm, "get_channels"):
            raise ToolError("Luxriot manager is not available or not configured.")
        force = bool(args.get("force", False))
        try:
            channels = self._lxm.get_channels(force=force)
        except Exception as exc:
            raise ToolError(f"Could not fetch channels: {exc}") from exc
        clean_channels: List[Dict[str, Any]] = []
        for channel in channels if isinstance(channels, list) else []:
            if not isinstance(channel, dict):
                continue
            clean_channels.append({
                "id": channel.get("id"),
                "title": channel.get("title") or channel.get("name") or channel.get("label") or f"channel-{channel.get('id')}",
                "enabled": channel.get("enabled"),
                "status": channel.get("status"),
            })
        return {
            "count": len(clean_channels),
            "channels": clean_channels,
        }

    # ── normalize_time_window ───────────────────────────────────────────────

    def _normalize_time_window(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime, time as time_cls, timedelta
        from zoneinfo import ZoneInfo

        tz_name = str(args.get("timezone") or "Europe/Riga").strip() or "Europe/Riga"
        try:
            tz = ZoneInfo(tz_name)
        except Exception as exc:
            raise ToolError(f"Unknown timezone: {tz_name}") from exc

        now_local = datetime.now(tz)
        relative_candidates = [
            args.get("relative_range"),
            args.get("start_time"),
            args.get("end_time"),
            args.get("date"),
        ]
        for candidate in relative_candidates:
            parsed_relative = _parse_relative_window_seconds(candidate)
            if parsed_relative is None:
                continue
            duration_sec, normalized_relative = parsed_relative
            end_local = now_local
            start_local = end_local - timedelta(seconds=duration_sec)
            from_ts = int(start_local.timestamp())
            to_ts = int(end_local.timestamp())
            return {
                "timezone": tz_name,
                "day_hint": "relative",
                "relative_range": normalized_relative,
                "from_local": start_local.isoformat(),
                "to_local": end_local.isoformat(),
                "from_ts": from_ts,
                "to_ts": to_ts,
                "since_ms": from_ts * 1000,
                "until_ms": to_ts * 1000,
                "duration_sec": max(0, to_ts - from_ts),
            }

        date_raw = str(args.get("date") or "").strip()
        day_hint = str(args.get("day_hint") or "today").strip().lower()
        if date_raw:
            try:
                base_date = datetime.strptime(date_raw, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ToolError("'date' must be YYYY-MM-DD.") from exc
        elif day_hint == "yesterday":
            base_date = (now_local - timedelta(days=1)).date()
        elif day_hint == "last_night":
            # Before noon, "last night" usually means the current local date after midnight.
            # Later in the day, choose the previous overnight period unless the user supplies date.
            base_date = now_local.date() if now_local.hour < 12 else (now_local - timedelta(days=1)).date()
        else:
            base_date = now_local.date()

        start_raw = str(args.get("start_time") or "").strip()
        end_raw = str(args.get("end_time") or "").strip()
        day_hint_raw = str(args.get("day_hint") or "").strip()
        if not start_raw and not end_raw and not date_raw and not day_hint_raw:
            return {
                "timezone": tz_name,
                "status": "not_specified",
                "has_time_window": False,
                "message": "No explicit time window was provided by the operator.",
            }
        if not start_raw and not end_raw and (date_raw or day_hint_raw):
            if day_hint == "last_night":
                start_local = datetime.combine(base_date, time_cls(22, 0), tzinfo=tz)
                end_local = datetime.combine(base_date + timedelta(days=1), time_cls(8, 0), tzinfo=tz)
            else:
                start_local = datetime.combine(base_date, time_cls(0, 0), tzinfo=tz)
                full_day_end = start_local + timedelta(days=1)
                end_local = min(now_local, full_day_end) if base_date == now_local.date() else full_day_end
            from_ts = int(start_local.timestamp())
            to_ts = int(end_local.timestamp())
            return {
                "timezone": tz_name,
                "day_hint": day_hint,
                "from_local": start_local.isoformat(),
                "to_local": end_local.isoformat(),
                "from_ts": from_ts,
                "to_ts": to_ts,
                "since_ms": from_ts * 1000,
                "until_ms": to_ts * 1000,
                "duration_sec": max(0, to_ts - from_ts),
            }
        if not start_raw or not end_raw:
            raise ToolError(
                "Provide either relative_range (for example 'last two hours') or both start_time and end_time."
            )

        def _parse_iso_local(value: str) -> Optional[datetime]:
            text = str(value or "").strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=tz)
            return parsed.astimezone(tz)

        start_iso = _parse_iso_local(start_raw)
        end_iso = _parse_iso_local(end_raw)
        if start_iso is not None and end_iso is not None:
            start_local = start_iso
            end_local = end_iso
            if end_local <= start_local:
                raise ToolError("'start_time' must be before 'end_time' for ISO timestamps.")
            from_ts = int(start_local.timestamp())
            to_ts = int(end_local.timestamp())
            return {
                "timezone": tz_name,
                "day_hint": day_hint,
                "from_local": start_local.isoformat(),
                "to_local": end_local.isoformat(),
                "from_ts": from_ts,
                "to_ts": to_ts,
                "since_ms": from_ts * 1000,
                "until_ms": to_ts * 1000,
                "duration_sec": max(0, to_ts - from_ts),
            }

        start_clock = _parse_operator_clock(start_raw)
        end_clock = _parse_operator_clock(end_raw)
        if start_clock is None or end_clock is None:
            raise ToolError(
                "'start_time' and 'end_time' must be parseable local times, or use relative_range like 'last two hours'."
            )

        start_local = datetime.combine(base_date, start_clock, tzinfo=tz)
        end_local = datetime.combine(base_date, end_clock, tzinfo=tz)
        if end_local <= start_local:
            end_local += timedelta(days=1)

        from_ts = int(start_local.timestamp())
        to_ts = int(end_local.timestamp())
        return {
            "timezone": tz_name,
            "day_hint": day_hint,
            "from_local": start_local.isoformat(),
            "to_local": end_local.isoformat(),
            "from_ts": from_ts,
            "to_ts": to_ts,
            "since_ms": from_ts * 1000,
            "until_ms": to_ts * 1000,
            "duration_sec": max(0, to_ts - from_ts),
        }

    # ── list_video_summary_channels ─────────────────────────────────────────

    def _list_video_summary_channels(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self._lxm, "get_channels") or not hasattr(self._lxm, "session_status"):
            raise ToolError("Luxriot manager is not available or not configured.")
        depth = _normalize_summary_depth(args.get("depth"))
        limit = max(1, min(100, int(args.get("limit") or 16)))
        run_selector = str(args.get("run") or "all").strip() or "all"
        from_ts, to_ts, time_meta = self._resolve_summary_time_window(args, default_since_hours=6.0)
        requested_ids = args.get("channel_ids") if isinstance(args.get("channel_ids"), list) else None
        requested_ids_set = {
            int(item) for item in (requested_ids or [])
            if _opt_int(item) is not None and int(item) > 0
        }
        try:
            channels = self._lxm.get_channels(force=False)
        except Exception as exc:
            raise ToolError(f"Could not fetch channels: {exc}") from exc

        valid_channel_ids: set[int] = set()
        for channel in channels if isinstance(channels, list) else []:
            if not isinstance(channel, dict):
                continue
            parsed_channel_id = _opt_int(channel.get("id"))
            if parsed_channel_id is not None and parsed_channel_id > 0:
                valid_channel_ids.add(int(parsed_channel_id))
        requested_count = len(requested_ids_set) if requested_ids_set else len(valid_channel_ids)
        checked_channel_ids: set[int] = set()
        channel_rows: List[Dict[str, Any]] = []
        inactive_count = 0
        errors: List[Dict[str, Any]] = []
        for channel in channels if isinstance(channels, list) else []:
            if not isinstance(channel, dict):
                continue
            channel_id = _opt_int(channel.get("id"))
            if channel_id is None or channel_id <= 0:
                continue
            if requested_ids_set and channel_id not in requested_ids_set:
                continue
            title = str(channel.get("title") or channel.get("name") or f"channel-{channel_id}")
            checked_channel_ids.add(channel_id)
            try:
                status = self._lxm.session_status(
                    channel_id=channel_id,
                    run_selector=run_selector,
                    start_ts=from_ts,
                    end_ts=to_ts,
                    limit=None,
                )
            except Exception as exc:
                errors.append({"channel_id": channel_id, "title": title, "error": str(exc)[:200]})
                continue
            logs = status.get("logs") if isinstance(status, dict) else []
            logs = logs if isinstance(logs, list) else []
            if not logs:
                inactive_count += 1
                continue
            starts: List[float] = []
            ends: List[float] = []
            frame_count = 0
            alert_counts: Dict[str, int] = {}
            for log in logs:
                if not isinstance(log, dict):
                    continue
                created = _coerce_epoch_seconds(log.get("created_at"))
                batch_start_ms = _opt_int(log.get("batch_start_ms"))
                batch_end_ms = _opt_int(log.get("batch_end_ms"))
                if batch_start_ms is not None or batch_end_ms is not None:
                    if batch_start_ms is None:
                        batch_start_ms = batch_end_ms
                    if batch_end_ms is None:
                        batch_end_ms = batch_start_ms
                    if batch_start_ms is not None and batch_end_ms is not None:
                        if batch_end_ms < batch_start_ms:
                            batch_start_ms, batch_end_ms = batch_end_ms, batch_start_ms
                        starts.append(float(batch_start_ms) / 1000.0)
                        ends.append(float(batch_end_ms) / 1000.0)
                elif created is not None:
                    starts.append(created)
                    ends.append(created)
                frame_count += int(_opt_int(log.get("frame_count")) or 0)
                raw_counts = log.get("alert_counts")
                if isinstance(raw_counts, dict):
                    for key, value in raw_counts.items():
                        severity = str(key or "normal").strip().lower() or "normal"
                        alert_counts[severity] = alert_counts.get(severity, 0) + int(_opt_int(value) or 0)
                else:
                    total = _opt_int(log.get("alert_total")) or 0
                    if total > 0:
                        severity = str(log.get("severity") or "normal").strip().lower() or "normal"
                        alert_counts[severity] = alert_counts.get(severity, 0) + total
            latest_ts = max(ends) if ends else None
            first_ts = min(starts) if starts else None
            alert_total = int(sum(alert_counts.values()))
            channel_rows.append(
                {
                    "channel_id": channel_id,
                    "title": title,
                    "summary_depth_recommended": depth,
                    "summary_count": len(logs),
                    "first_ts": first_ts,
                    "latest_ts": latest_ts,
                    "first_time": _format_epoch_minute(first_ts),
                    "latest_time": _format_epoch_minute(latest_ts),
                    "frame_count": frame_count,
                    "alert_total": alert_total,
                    "alert_counts": alert_counts,
                    "running": bool(status.get("running")) if isinstance(status, dict) else False,
                    "selected_run": status.get("selected_run") if isinstance(status, dict) else None,
                }
            )
        channel_rows.sort(
            key=lambda row: (
                int(row.get("alert_total") or 0),
                float(row.get("latest_ts") or 0.0),
                int(row.get("summary_count") or 0),
            ),
            reverse=True,
        )
        active_count = len(channel_rows)
        per_turn_limit = AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN
        effective_limit = min(limit, per_turn_limit)
        candidate_channels = channel_rows[:effective_limit]
        deferred_channel_ids = [
            int(row["channel_id"])
            for row in channel_rows[effective_limit:]
            if _opt_int(row.get("channel_id")) is not None
        ]
        unchecked_channel_ids = (
            sorted(requested_ids_set.difference(checked_channel_ids))
            if requested_ids_set
            else []
        )
        unchecked_count = len(unchecked_channel_ids)
        return {
            "depth": depth,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "time_window": time_meta,
            "requested_count": requested_count,
            "unchecked_count": unchecked_count,
            "unchecked_channel_ids": unchecked_channel_ids,
            "total_channels_checked": active_count + inactive_count + len(errors),
            "active_count": active_count,
            "inactive_count": inactive_count,
            "error_count": len(errors),
            "returned": len(candidate_channels),
            "deferred_count": len(deferred_channel_ids),
            "deferred_channel_ids": deferred_channel_ids,
            "per_turn_channel_limit": per_turn_limit,
            "requires_confirmation": active_count > per_turn_limit,
            "full_research_note": (
                f"{active_count} active channel(s) have summaries in this window. "
                f"Reviewing more than {per_turn_limit} channels should be confirmed and chunked."
            ),
            "candidate_channels": candidate_channels,
            "errors": errors[:8],
        }

    # ── list_probes ────────────────────────────────────────────────────────

    def _list_probes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        since_hours = float(args.get("since_hours") or 24)
        since_ms = int(time.time() * 1000 - since_hours * 3_600_000)
        probes = self._ps.list_probes()
        summary_rows = self._ds.summarize_by_probe(since_ms=since_ms, source="probe")
        summary_by_probe = {str(row.get("probe_id") or ""): row for row in summary_rows}
        items: List[Dict[str, Any]] = []
        for probe in probes:
            probe_id = str(probe.get("id") or "")
            summary = summary_by_probe.get(probe_id, {})
            items.append({
                "id": probe.get("id"),
                "name": probe.get("name"),
                "channel_id": probe.get("channel_id"),
                "enabled": probe.get("enabled", True),
                "severity": probe.get("severity"),
                "bookmark": probe.get("bookmark"),
                "pos_floor": probe.get("pos_floor"),
                "margin": probe.get("margin"),
                "hit_count_24h": int(summary.get("hit_count") or 0),
                "latest_timestamp_ms": int(summary.get("latest_timestamp_ms") or 0),
            })
        return {
            "count": len(items),
            "since_hours": since_hours,
            "probes": items,
        }

    # ── survey_channels ────────────────────────────────────────────────────

    def _survey_channels(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self._lxm, "get_channels") or not hasattr(self._lxm, "get_snapshot_base64"):
            raise ToolError("Luxriot manager is not available or not configured.")
        requested_ids = args.get("channel_ids") if isinstance(args.get("channel_ids"), list) else None
        requested_ids_set = {
            int(item) for item in (requested_ids or [])
            if _opt_int(item) is not None
        }
        fast_mode = bool(args.get("fast_mode", False))
        default_duration = 4.0 if fast_mode else 12.0
        default_samples = 2 if fast_mode else 4
        duration_sec = max(1.0, min(20.0, float(args.get("duration_sec") or default_duration)))
        sample_count = max(2, min(6, int(args.get("sample_count") or default_samples)))
        prompt = str(args.get("prompt") or "").strip() or (
            "You are surveying CCTV channels during deployment. "
            "Summarize what this camera is pointed at, what usually occupies the scene, "
            "whether it is indoor or outdoor, and 2-4 plausible monitoring scenarios."
        )
        try:
            channels = self._lxm.get_channels(force=False)
        except Exception as exc:
            raise ToolError(f"Could not fetch channels for survey: {exc}") from exc

        survey_items: List[Dict[str, Any]] = []
        target_channels = []
        for channel in channels if isinstance(channels, list) else []:
            if not isinstance(channel, dict):
                continue
            channel_id = _opt_int(channel.get("id"))
            if channel_id is None:
                continue
            if requested_ids_set and channel_id not in requested_ids_set:
                continue
            target_channels.append(channel)

        interval_sec = duration_sec / max(1, sample_count - 1)
        self._report_progress({
            "tool": "survey_channels",
            "stage": "start",
            "message": f"Surveying {len(target_channels)} channel(s){' in fast mode' if fast_mode else ''}",
            "channel_count": len(target_channels),
            "fast_mode": fast_mode,
        })
        for channel in target_channels:
            channel_id = int(channel.get("id"))
            channel_title = str(channel.get("title") or channel.get("name") or f"channel-{channel_id}")
            self._report_progress({
                "tool": "survey_channels",
                "stage": "capture",
                "channel_id": channel_id,
                "title": channel_title,
                "message": f"Capturing samples from CH {channel_id} ({channel_title})",
            })
            snapshots: List[str] = []
            capture_errors: List[str] = []
            for idx in range(sample_count):
                try:
                    encoded, _meta = self._lxm.get_snapshot_base64(channel_id)
                    snapshots.append(encoded)
                    self._report_progress({
                        "tool": "survey_channels",
                        "stage": "capture_sample",
                        "channel_id": channel_id,
                        "title": channel_title,
                        "sample": idx + 1,
                        "sample_count": sample_count,
                        "message": f"Captured sample {idx + 1}/{sample_count} for CH {channel_id}",
                    })
                except Exception as exc:
                    capture_errors.append(str(exc))
                if idx < sample_count - 1:
                    time.sleep(interval_sec)
            if not snapshots:
                self._report_progress({
                    "tool": "survey_channels",
                    "stage": "capture_failed",
                    "channel_id": channel_id,
                    "title": channel_title,
                    "message": f"Could not capture samples from CH {channel_id}",
                })
                survey_items.append({
                    "channel_id": channel_id,
                    "title": channel_title,
                    "sample_count": 0,
                    "survey": "",
                    "error": capture_errors[-1] if capture_errors else "No snapshots captured.",
                })
                continue

            self._report_progress({
                "tool": "survey_channels",
                "stage": "analyze",
                "channel_id": channel_id,
                "title": channel_title,
                "message": f"Analyzing CH {channel_id} ({channel_title})",
            })
            user_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"Channel {channel_id} ({channel_title}).\nTask: {prompt}"}
            ]
            for snap in snapshots:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{snap}",
                        "detail": "low",
                    },
                })
            messages = [
                {"role": "system", "content": "You are an expert CCTV deployment analyst. Be concise and operational."},
                {"role": "user", "content": user_content},
            ]
            try:
                survey = self._lm(messages)
            except Exception as exc:
                raise ToolError(f"Could not analyze channel {channel_id}: {exc}") from exc
            self._report_progress({
                "tool": "survey_channels",
                "stage": "done_channel",
                "channel_id": channel_id,
                "title": channel_title,
                "message": f"Survey complete for CH {channel_id} ({channel_title})",
            })
            survey_items.append({
                "channel_id": channel_id,
                "title": channel_title,
                "sample_count": len(snapshots),
                "duration_sec": duration_sec,
                "survey": survey,
                "errors": capture_errors,
            })

        return {
            "duration_sec": duration_sec,
            "sample_count": sample_count,
            "fast_mode": fast_mode,
            "channels": survey_items,
        }

    # ── build_research_batch ───────────────────────────────────────────────

    def _build_research_batch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        probe_id = str(args.get("probe_id") or "").strip() or None
        probe_name_raw = str(args.get("probe_name") or "").strip()
        if not probe_id and probe_name_raw:
            probe_id = self._resolve_probe_id_by_name(probe_name_raw)

        channel_id = _opt_int(args.get("channel_id"))
        sort_by = str(args.get("sort_by") or "highest_margin").strip().lower()
        if sort_by not in {
            "newest", "oldest", "highest_pos", "lowest_pos", "highest_margin", "lowest_margin"
        }:
            sort_by = "highest_margin"
        per_period_limit = max(1, min(100, int(args.get("per_period_limit") or 24)))
        per_band_limit = max(1, min(per_period_limit, int(args.get("per_band_limit") or 6)))
        max_candidates = max(per_period_limit, min(5_000, int(args.get("max_candidates") or 1_000)))

        raw_periods = args.get("periods") if isinstance(args.get("periods"), list) else None
        periods: List[Dict[str, Any]] = []
        if raw_periods:
            for idx, raw_period in enumerate(raw_periods, start=1):
                if not isinstance(raw_period, dict):
                    continue
                since_ms, until_ms = self._resolve_time_window(raw_period, default_since_hours=24.0)
                periods.append({
                    "label": str(raw_period.get("label") or f"period_{idx}").strip() or f"period_{idx}",
                    "since_ms": since_ms,
                    "until_ms": until_ms,
                })
        if not periods:
            since_ms, until_ms = self._resolve_time_window(args, default_since_hours=24.0)
            periods = [{"label": "primary_window", "since_ms": since_ms, "until_ms": until_ms}]

        raw_bands = args.get("bands") if isinstance(args.get("bands"), list) else None
        bands: List[Dict[str, Any]] = []
        if raw_bands:
            for idx, raw_band in enumerate(raw_bands, start=1):
                if not isinstance(raw_band, dict):
                    continue
                score_field = str(raw_band.get("score_field") or "margin").strip().lower()
                if score_field not in {"pos_score", "margin", "neg_score"}:
                    score_field = "margin"
                bands.append({
                    "label": str(raw_band.get("label") or f"band_{idx}").strip() or f"band_{idx}",
                    "score_field": score_field,
                    "min": _opt_float(raw_band.get("min")),
                    "max": _opt_float(raw_band.get("max")),
                })

        selected: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()
        period_reports: List[Dict[str, Any]] = []
        band_reports: List[Dict[str, Any]] = []

        for period in periods:
            window_rows, total = self._list_detection_window(
                probe_id=probe_id,
                channel_id=channel_id,
                source="probe",
                since_ms=period["since_ms"],
                until_ms=period["until_ms"],
                limit=max_candidates,
                offset=0,
                sort_by=sort_by,
                max_scan=max_candidates,
            )
            kept_for_period = 0
            if bands:
                for band in bands:
                    band_rows = self._filter_detection_band(window_rows, band)
                    band_take = 0
                    for row in band_rows:
                        det_id = int(row.get("id") or 0)
                        if det_id <= 0 or det_id in seen_ids:
                            continue
                        selected.append(row)
                        seen_ids.add(det_id)
                        kept_for_period += 1
                        band_take += 1
                        if band_take >= per_band_limit or kept_for_period >= per_period_limit:
                            break
                    band_reports.append({
                        "period": period["label"],
                        "label": band["label"],
                        "score_field": band["score_field"],
                        "min": band["min"],
                        "max": band["max"],
                        "matched": len(band_rows),
                        "selected": band_take,
                    })
                    if kept_for_period >= per_period_limit:
                        break
            else:
                for row in window_rows:
                    det_id = int(row.get("id") or 0)
                    if det_id <= 0 or det_id in seen_ids:
                        continue
                    selected.append(row)
                    seen_ids.add(det_id)
                    kept_for_period += 1
                    if kept_for_period >= per_period_limit:
                        break

            period_reports.append({
                "label": period["label"],
                "since_ms": period["since_ms"],
                "until_ms": period["until_ms"],
                "total_candidates": total,
                "scanned": len(window_rows),
                "selected": kept_for_period,
            })

        return {
            "probe_id": probe_id,
            "channel_id": channel_id,
            "source": "probe",
            "source_label": _archive_source_label("probe"),
            "sort_by": sort_by,
            "periods": period_reports,
            "bands": band_reports,
            "batch_size": len(selected),
            "detections": [_safe_detection(_annotate_archive_row(row)) for row in selected],
        }

    # ── create_probe ───────────────────────────────────────────────────────

    def _create_probe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = str(args.get("name") or "").strip()
        channel_id = self._resolve_channel_id(args, required=True)
        preview = bool(args.get("preview", True))
        update_existing = bool(args.get("update_existing", True))
        if not name:
            raise ToolError("'name' is required.")
        if channel_id is None:
            raise ToolError("'channel_id' or 'channel_ref' is required.")

        positives = [str(item).strip() for item in (args.get("positives") or []) if str(item).strip()]
        negatives = [str(item).strip() for item in (args.get("negatives") or []) if str(item).strip()]
        probe = {
            "name": name,
            "channel_id": channel_id,
            "positives": positives,
            "negatives": negatives,
            "pos_floor": _opt_float(args.get("pos_floor")) if args.get("pos_floor") is not None else 0.2,
            "margin": _opt_float(args.get("margin_thr")) if args.get("margin_thr") is not None else 0.05,
            "bookmark_cooldown_sec": _opt_float(args.get("bookmark_cooldown_sec")) if args.get("bookmark_cooldown_sec") is not None else 8.0,
            "bookmark_dedupe_window_sec": _opt_float(args.get("bookmark_dedupe_window_sec")) if args.get("bookmark_dedupe_window_sec") is not None else 20.0,
            "top_k": max(1, int(args.get("top_k") or 6)),
            "window_sec": max(0.0, float(args.get("window_sec") or 300.0)),
            "severity": str(args.get("severity") or "critical").strip().lower(),
            "bookmark": bool(args.get("bookmark_enabled", True)),
            "enabled": bool(args.get("enabled", True)),
            "image_probe": {"enabled": False, "data": None, "name": None, "pos_floor": 0.7},
            "roi_enabled": False,
            "roi_norm": None,
            "pairs": _probe_pairs_from_lists(positives, negatives),
            "last_hit": None,
            "recent_hits": [],
            "bookmark_gate": None,
            "bookmark_gate_updated_at_ms": None,
        }
        errors = _validate_probe(probe)
        if errors:
            raise ToolError("Validation failed: " + "; ".join(errors))
        existing = [
            p for p in self._ps.list_probes()
            if str(p.get("name") or "").strip().lower() == name.lower() and _opt_int(p.get("channel_id")) == channel_id
        ]
        if len(existing) > 1:
            raise ToolError(
                "Multiple existing probes have the same name on this channel. "
                "Clean up duplicates first or choose a unique probe name."
            )
        existing_probe = existing[0] if existing else None
        action = "create_new"
        if existing_probe and update_existing:
            probe = _merge_probe(existing_probe, probe)
            probe["id"] = existing_probe.get("id")
            action = "update_existing"
        if preview:
            return {
                "status": "preview",
                "exists": bool(existing_probe),
                "action": action,
                "conflicts": [_probe_summary(p) for p in existing],
                "proposed": _probe_summary(probe),
            }
        saved = self._ps.upsert_probe(probe)
        return {
            "status": "applied",
            "action": action,
            "exists": bool(existing_probe),
            "probe_id": saved.get("id"),
            "probe_name": saved.get("name"),
            "probe": _probe_summary(saved),
        }

    # ── deploy_summary ─────────────────────────────────────────────────────

    def _deploy_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(args.get("mode") or "standard").strip().lower()
        if mode not in {"standard", "magic", "survey_only"}:
            mode = "standard"
        channels = [str(item).strip() for item in (args.get("channels") or []) if str(item).strip()]
        probes = [str(item).strip() for item in (args.get("probes") or []) if str(item).strip()]
        prompt_targets = [str(item).strip() for item in (args.get("prompt_targets") or []) if str(item).strip()]
        notes = [str(item).strip() for item in (args.get("notes") or []) if str(item).strip()]
        return {
            "mode": mode,
            "wipe": bool(args.get("wipe", False)),
            "elapsed_sec": _opt_float(args.get("elapsed_sec")),
            "overview": str(args.get("overview") or "").strip(),
            "channels": channels,
            "probes": probes,
            "prompt_targets": prompt_targets,
            "notes": notes,
        }

    # ── delete_probes ──────────────────────────────────────────────────────

    def _delete_probes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        preview = bool(args.get("preview", True))
        delete_all = bool(args.get("delete_all", False))
        probe_ids = [
            str(item).strip() for item in (args.get("probe_ids") or [])
            if str(item).strip()
        ]
        probes = self._ps.list_probes()
        if delete_all:
            targets = probes
        else:
            wanted = set(probe_ids)
            targets = [probe for probe in probes if str(probe.get("id") or "") in wanted]
        if not targets:
            raise ToolError("No probes selected for deletion.")
        summary = [
            {
                "id": probe.get("id"),
                "name": probe.get("name"),
                "channel_id": probe.get("channel_id"),
            }
            for probe in targets
        ]
        if preview:
            return {
                "status": "preview",
                "delete_all": delete_all,
                "count": len(summary),
                "targets": summary,
            }
        deleted = 0
        for probe in targets:
            if self._ps.delete_probe(str(probe.get("id") or "")):
                deleted += 1
        return {
            "status": "applied",
            "delete_all": delete_all,
            "deleted": deleted,
            "targets": summary,
        }

    # ── update_probe ────────────────────────────────────────────────────────

    def _update_probe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        changes  = args.get("changes") or {}
        preview  = bool(args.get("preview", True))
        probe_id = str(args.get("probe_id") or "").strip() or None
        probe_name_raw = str(args.get("probe_name") or "").strip()

        if not changes:
            raise ToolError("'changes' must contain at least one field to modify.")

        # Resolve probe
        if not probe_id and probe_name_raw:
            probe_id = self._resolve_probe_id_by_name(probe_name_raw)
        if not probe_id:
            raise ToolError("Provide 'probe_id' or 'probe_name'.")

        current = self._find_probe(probe_id)

        # Deep merge — only touch what changes specifies
        merged = _merge_probe(current, changes)

        # Validate merged object
        errors = _validate_probe(merged)
        if errors:
            raise ToolError("Validation failed: " + "; ".join(errors))

        # Build human-readable diff
        diff = _probe_diff(current, merged)

        if preview:
            return {
                "status": "preview",
                "probe_id": probe_id,
                "probe_name": current.get("name"),
                "diff": diff,
                "current": _probe_summary(current),
                "proposed": _probe_summary(merged),
            }

        # Apply
        self._ps.upsert_probe(merged)
        return {
            "status": "applied",
            "probe_id": probe_id,
            "probe_name": merged.get("name"),
            "diff": diff,
        }

    # ── describe_frame ──────────────────────────────────────────────────────

    def _describe_frame(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id   = self._resolve_channel_id(args, required=False)
        image_path   = str(args.get("image_path") or "").strip() or None
        detection_id = _opt_int(args.get("detection_id"))
        prompt = str(args.get("prompt") or "").strip() or (
            "Describe what is happening in this image in detail. "
            "Note any people, vehicles, objects, or unusual activity."
        )

        # Prefer explicit archive/file evidence over a live snapshot. Users often
        # ask about "channel X" while also passing detection_id from archive search.
        if image_path is None and detection_id is None and channel_id is not None:
            if not hasattr(self._lxm, "get_snapshot_base64"):
                raise ToolError("Luxriot manager is not available or not configured.")
            try:
                encoded, meta = self._lxm.get_snapshot_base64(channel_id)
            except Exception as exc:
                raise ToolError(f"Could not capture snapshot from channel {channel_id}: {exc}") from exc
            messages = [
                {"role": "system", "content": "You are an expert visual analyst. Be concise and factual."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Live feed from channel {channel_id}.\n\nTask: {prompt}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}",
                        "detail": "high",
                    }},
                ]},
            ]
            description = self._lm(messages)
            return {
                "description": description,
                "source": "live_snapshot",
                "channel_id": channel_id,
                "image_path": None,
                "snapshot_b64": encoded,
            }

        if image_path is None and detection_id is None:
            raise ToolError("Provide 'channel_id'/'channel_ref', 'image_path', or 'detection_id'.")

        # Resolve image from detection record if needed
        resolved_path: Optional[str] = image_path
        if detection_id is not None and image_path is None:
            records = self._ds.fetch_detections_by_ids([detection_id], include_vectors=False)
            if not records:
                raise ToolError(f"Detection ID {detection_id} not found.")
            rec = records[0]
            resolved_path = str(rec.get("image_path") or "").strip() or None
            if not resolved_path:
                # Fall back to thumbnail
                thumb = rec.get("thumbnail")
                if not thumb:
                    raise ToolError(f"Detection {detection_id} has no image_path or thumbnail.")
                return self._describe_from_thumb_b64(thumb, prompt, detection_id=detection_id)

        if resolved_path and resolved_path.lower().startswith("data:image/"):
            return self._describe_from_thumb_b64(resolved_path, prompt)

        if not resolved_path or not Path(resolved_path).exists():
            raise ToolError(f"Image file not found: {resolved_path!r}")

        # Build messages inline (mirrors _build_image_messages in oldapp.py)
        try:
            encoded = self._jpeg(Image.open(resolved_path), max_edge=960, quality=88)
        except Exception as exc:
            raise ToolError(f"Could not open image: {exc}") from exc

        messages = [
            {"role": "system", "content": "You are an expert visual analyst. Be concise and factual."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Image: {Path(resolved_path).name}\n\nTask: {prompt}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                }},
            ]},
        ]
        description = self._lm(messages)
        return {
            "description": description,
            "source": "image_path",
            "image_path": resolved_path,
            "image_url": f"/detections/image?image_path={quote(resolved_path, safe='')}",
            "detection_id": detection_id,
            "snapshot_b64": encoded,
        }

    def _describe_from_thumb_b64(
        self,
        thumb_b64: str,
        prompt: str,
        *,
        detection_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized_b64 = _strip_image_data_url_prefix(thumb_b64)
        image_url = _image_data_url(normalized_b64)
        if not normalized_b64 or not image_url:
            raise ToolError("Detection thumbnail is empty.")
        messages = [
            {"role": "system", "content": "You are an expert visual analyst. Be concise and factual."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Task: {prompt}"},
                {"type": "image_url", "image_url": {
                    "url": image_url,
                    "detail": "high",
                }},
            ]},
        ]
        description = self._lm(messages)
        return {
            "description": description,
            "source": "thumbnail",
            "image_path": None,
            "image_url": f"/detections/thumbnail/{int(detection_id)}" if detection_id is not None else None,
            "detection_id": detection_id,
            "snapshot_b64": normalized_b64,
            "note": "low-res thumbnail used",
        }

    # ── prompt settings ────────────────────────────────────────────────────

    def _get_prompt_settings(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self._lxm, "get_prompt_settings"):
            raise ToolError("Luxriot manager prompt settings are not available.")
        channel_id = self._resolve_channel_id(args, required=False)
        try:
            return self._lxm.get_prompt_settings(channel_id=channel_id)
        except Exception as exc:
            raise ToolError(f"Could not fetch prompt settings: {exc}") from exc

    def _update_prompt_settings(self, args: Dict[str, Any]) -> Dict[str, Any]:
        changes = _normalize_prompt_setting_changes(args.get("changes") or {})
        preview = bool(args.get("preview", True))
        channel_id = self._resolve_channel_id(args, required=False)

        if not changes:
            raise ToolError("'changes' must contain at least one field to modify.")
        if not hasattr(self._lxm, "get_prompt_settings") or not hasattr(self._lxm, "update_prompt_settings"):
            raise ToolError("Luxriot manager prompt settings are not available.")

        try:
            current = self._lxm.get_prompt_settings(channel_id=channel_id)
        except Exception as exc:
            raise ToolError(f"Could not fetch current prompt settings: {exc}") from exc

        proposed = _merge_prompt_settings_snapshot(current, changes)
        diff = _prompt_settings_diff(current, proposed)
        if not diff:
            return {
                "status": "noop",
                "channel_id": channel_id,
                "current": current,
                "proposed": proposed,
                "diff": {},
            }

        if preview:
            return {
                "status": "preview",
                "channel_id": channel_id,
                "diff": diff,
                "current": current,
                "proposed": proposed,
            }

        try:
            applied = self._lxm.update_prompt_settings(
                channel_id=channel_id,
                stream_system_prompt=changes.get("stream_system_prompt"),
                rollup_prompts=changes.get("rollup_prompts"),
                json_alert_prompt=changes.get("json_alert_prompt"),
                bookmark_enabled=changes.get("bookmark_enabled"),
                bookmark_cooldown_sec=changes.get("bookmark_cooldown_sec"),
            )
        except Exception as exc:
            raise ToolError(f"Could not update prompt settings: {exc}") from exc

        return {
            "status": "applied",
            "channel_id": channel_id,
            "diff": diff,
            "effective": applied,
        }

    # ── get_video_summaries ─────────────────────────────────────────────────

    def _get_video_summaries(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id  = self._resolve_channel_id(args, required=True)
        if channel_id is None:
            raise ToolError("'channel_id' or 'channel_ref' is required.")
        depth       = _normalize_summary_depth(args.get("depth"))
        limit       = max(1, min(100, int(args.get("limit") or 20)))
        requested_level_limit = _opt_int(args.get("level_limit"))
        if requested_level_limit is None:
            level_limit = max(limit, AGENT_VIDEO_SUMMARY_DEFAULT_LEVEL_LIMIT)
        else:
            level_limit = max(
                limit,
                min(AGENT_VIDEO_SUMMARY_MAX_LEVEL_LIMIT, int(requested_level_limit)),
            )
        run_selector = str(args.get("run") or "all").strip() or "all"
        include_evidence_frames = bool(args.get("include_evidence_frames", False))
        evidence_frame_limit = max(1, min(24, int(args.get("evidence_frame_limit") or 8)))

        if not hasattr(self._lxm, "summary_rollups"):
            raise ToolError("Luxriot manager is not available or not configured.")

        from_ts, to_ts, time_meta = self._resolve_summary_time_window(args, default_since_hours=6.0)

        try:
            rollups = self._summary_rollups_readonly(
                channel_id=channel_id,
                run_selector=run_selector,
                start_ts=from_ts,
                end_ts=to_ts,
                level_limit=level_limit,
            )
        except Exception as exc:
            raise ToolError(f"Could not fetch summaries: {exc}") from exc

        levels = rollups.get("levels") if isinstance(rollups.get("levels"), dict) else {}
        source_counts_raw = rollups.get("source_counts") if isinstance(rollups.get("source_counts"), dict) else {}
        source_counts: Dict[str, int] = {}
        for level_name in ("L0", "L1", "L2", "L3"):
            raw_count = source_counts_raw.get(level_name)
            parsed_count = _opt_int(raw_count)
            if parsed_count is None:
                level_nodes = levels.get(level_name) if isinstance(levels, dict) else []
                parsed_count = len(level_nodes) if isinstance(level_nodes, list) else 0
            source_counts[level_name] = int(parsed_count)
        level_limit_applied = _opt_int(rollups.get("level_limit")) or level_limit
        backend_truncated = bool(
            level_limit_applied > 0
            and any(count >= int(level_limit_applied) for count in source_counts.values())
        )
        nodes = levels.get(depth) or []

        # Flatten to a clean list for the LLM
        entries = []
        filtered_nodes = [
            node for node in nodes
            if isinstance(node, dict) and _summary_node_overlaps(node, from_ts, to_ts)
        ]
        filtered_nodes.sort(
            key=lambda node: (
                _summary_node_bounds(node)[0] if _summary_node_bounds(node)[0] is not None else 0.0,
                _summary_node_bounds(node)[1] if _summary_node_bounds(node)[1] is not None else 0.0,
            )
        )
        returned_nodes, selection_strategy = _select_summary_nodes_for_period(filtered_nodes, limit)
        for node in returned_nodes:
            entry: Dict[str, Any] = {}
            start, end = _summary_node_bounds(node)
            if start is not None:
                entry["time"] = _format_epoch_minute(start)
                entry["window_start"] = start
            if end is not None:
                entry["window_end"] = end
                entry["window_end_time"] = _format_epoch_minute(end)
            for key in ("level", "frame_count", "item_count", "alert_total", "alert_counts", "alert_severities"):
                if key in node:
                    entry[key] = node.get(key)
            text = str(node.get("summary") or "").strip()
            if text:
                entry["summary"] = text[:800]
            if entry.get("summary"):
                entries.append(entry)

        truncated = len(filtered_nodes) > len(returned_nodes)
        coverage = _video_summary_coverage_contract(
            available_nodes=filtered_nodes,
            returned_nodes=returned_nodes,
            from_ts=from_ts,
            to_ts=to_ts,
            truncated=truncated,
            selection_strategy=selection_strategy,
        )
        evidence_sources = ("vlm_alert", "vlm_summary")
        evidence_frame_query = {
            "tool": "get_detections",
            "channel_id": channel_id,
            "sources": list(evidence_sources),
            "since_ms": int(float(from_ts) * 1000.0),
            "until_ms": int(float(to_ts) * 1000.0),
            "sort_by": "oldest",
            "limit": min(24, evidence_frame_limit),
        }
        evidence_frame_queries: List[Dict[str, Any]] = []
        evidence_attempted_sources: List[str] = []
        evidence_frames: List[Dict[str, Any]] = []
        evidence_totals: Dict[str, int] = {}
        if include_evidence_frames:
            source_rows: Dict[str, List[Dict[str, Any]]] = {}
            for source in evidence_sources:
                source_query = dict(evidence_frame_query)
                source_query.pop("sources", None)
                source_query["source"] = source
                evidence_frame_queries.append(source_query)
                evidence_attempted_sources.append(source)
                rows, total = self._list_detection_window(
                    probe_id=None,
                    channel_id=channel_id,
                    source=source,
                    since_ms=evidence_frame_query["since_ms"],
                    until_ms=evidence_frame_query["until_ms"],
                    limit=evidence_frame_limit,
                    offset=0,
                    sort_by="oldest",
                    max_scan=1000,
                )
                evidence_totals[source] = total
                source_rows[source] = [
                    _safe_detection(_annotate_archive_row(row))
                    for row in rows
                ]
            evidence_frames = _select_evidence_frame_rows(source_rows, evidence_frame_limit)

        return {
            "channel_id": channel_id,
            "depth": depth,
            "display_limit": limit,
            "level_limit_applied": level_limit_applied,
            "backend_truncated": backend_truncated,
            "source_counts": source_counts,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "time_window": time_meta,
            "coverage": coverage,
            "run": run_selector,
            "count": len(entries),
            "total_in_window": len(filtered_nodes),
            "truncated": truncated,
            "selection_strategy": selection_strategy,
            "selected_run": rollups.get("selected_run"),
            "run_filter_id": rollups.get("run_filter_id"),
            "running": bool(rollups.get("running")),
            "evidence_frame_query": evidence_frame_query,
            "evidence_frame_queries": evidence_frame_queries,
            "evidence_frame_attempted_sources": evidence_attempted_sources,
            "attempted_sources": evidence_attempted_sources,
            "evidence_frame_totals": evidence_totals,
            "totals": evidence_totals,
            "evidence_frames": evidence_frames,
            "entries": entries,
        }

    # ── count_video_summary_events ──────────────────────────────────────────

    def _count_video_summary_events(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id = self._resolve_channel_id(args, required=True)
        if channel_id is None:
            raise ToolError("'channel_id' or 'channel_ref' is required.")
        entity_query = str(args.get("entity_query") or "").strip()
        if not entity_query:
            raise ToolError("'entity_query' is required.")
        anchor_query = str(args.get("anchor_query") or "").strip()
        event_kind = str(args.get("event_kind") or "presence_transitions").strip().lower()
        if event_kind != "presence_transitions":
            raise ToolError("Only event_kind='presence_transitions' is supported.")
        depth = _normalize_summary_depth(args.get("depth") or "L1")
        requested_level_limit = _opt_int(args.get("level_limit"))
        if requested_level_limit is None:
            level_limit = AGENT_VIDEO_SUMMARY_MAX_LEVEL_LIMIT
        else:
            level_limit = max(1, min(AGENT_VIDEO_SUMMARY_MAX_LEVEL_LIMIT, int(requested_level_limit)))
        timeline_limit = max(1, min(120, int(args.get("timeline_limit") or 40)))
        event_limit = max(1, min(120, int(args.get("event_limit") or 40)))
        run_selector = str(args.get("run") or "all").strip() or "all"

        if not hasattr(self._lxm, "summary_rollups"):
            raise ToolError("Luxriot manager is not available or not configured.")

        from_ts, to_ts, time_meta = self._resolve_summary_time_window(args, default_since_hours=6.0)
        try:
            rollups = self._summary_rollups_readonly(
                channel_id=channel_id,
                run_selector=run_selector,
                start_ts=from_ts,
                end_ts=to_ts,
                level_limit=level_limit,
            )
        except Exception as exc:
            raise ToolError(f"Could not fetch summaries: {exc}") from exc

        levels = rollups.get("levels") if isinstance(rollups.get("levels"), dict) else {}
        source_counts_raw = rollups.get("source_counts") if isinstance(rollups.get("source_counts"), dict) else {}
        source_counts: Dict[str, int] = {}
        for level_name in ("L0", "L1", "L2", "L3"):
            raw_count = source_counts_raw.get(level_name)
            parsed_count = _opt_int(raw_count)
            if parsed_count is None:
                level_nodes = levels.get(level_name) if isinstance(levels, dict) else []
                parsed_count = len(level_nodes) if isinstance(level_nodes, list) else 0
            source_counts[level_name] = int(parsed_count)

        level_limit_applied = _opt_int(rollups.get("level_limit")) or level_limit
        backend_truncated = bool(
            level_limit_applied > 0
            and any(count >= int(level_limit_applied) for count in source_counts.values())
        )
        nodes_raw = levels.get(depth) or []
        filtered_nodes = [
            node for node in nodes_raw
            if isinstance(node, dict) and _summary_node_overlaps(node, from_ts, to_ts)
        ]
        filtered_nodes.sort(
            key=lambda node: (
                _summary_node_bounds(node)[0] if _summary_node_bounds(node)[0] is not None else 0.0,
                _summary_node_bounds(node)[1] if _summary_node_bounds(node)[1] is not None else 0.0,
            )
        )
        coverage = _video_summary_coverage_contract(
            available_nodes=filtered_nodes,
            returned_nodes=filtered_nodes,
            from_ts=from_ts,
            to_ts=to_ts,
            truncated=backend_truncated,
            selection_strategy="full_scan",
        )
        count_result = _count_summary_presence_transitions(
            filtered_nodes,
            entity_query=entity_query,
            anchor_query=anchor_query,
            timeline_limit=timeline_limit,
            event_limit=event_limit,
        )
        notes = [
            "Counts are derived from VLM summary text, not exhaustive frame-level reanalysis.",
            "Explicit events come from summary wording; inferred events come from adjacent summary state changes.",
            "If coverage is partial/no_data/truncated, counts apply only to the covered/scanned summaries.",
        ]
        return {
            "channel_id": channel_id,
            "depth": depth,
            "event_kind": event_kind,
            "entity_query": entity_query,
            "anchor_query": anchor_query or None,
            "level_limit_applied": level_limit_applied,
            "backend_truncated": backend_truncated,
            "source_counts": source_counts,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "time_window": time_meta,
            "coverage": coverage,
            "run": run_selector,
            "selected_run": rollups.get("selected_run"),
            "run_filter_id": rollups.get("run_filter_id"),
            "running": bool(rollups.get("running")),
            "total_in_window": len(filtered_nodes),
            "scan_strategy": "sequential_summary_state_count",
            "score_semantics": "summary_text_count_not_frame_ground_truth",
            "notes": notes,
            **count_result,
        }

    # ── track_visual_state_transitions ──────────────────────────────────────

    def _track_visual_state_transitions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id = self._resolve_channel_id(args, required=True)
        if channel_id is None:
            raise ToolError("'channel_id' or 'channel_ref' is required.")
        positive_query = str(args.get("positive_state_query") or "").strip()
        if not positive_query:
            raise ToolError("'positive_state_query' is required.")
        negative_query = str(args.get("negative_state_query") or "").strip()
        alternate_query = str(args.get("alternate_state_query") or "").strip()
        subject_query = str(args.get("subject_query") or "").strip() or None
        negative_query_effective, negative_query_warnings = _clip_effective_negative_state_query(
            negative_query,
            subject_query=subject_query,
        )
        positive_label = str(args.get("positive_label") or "positive").strip() or "positive"
        negative_label = str(args.get("negative_label") or "negative").strip() or "negative"
        alternate_label = str(args.get("alternate_label") or "alternate").strip() or "alternate"
        positive_floor = float(_opt_float(args.get("positive_floor")) or 0.18)
        negative_floor = float(_opt_float(args.get("negative_floor")) or 0.18)
        margin_threshold = float(_opt_float(args.get("margin_threshold")) or 0.03)
        min_state_samples = max(1, min(20, int(args.get("min_state_samples") or 2)))
        min_state_duration_sec = max(0.0, min(120.0, float(_opt_float(args.get("min_state_duration_sec")) or 2.0)))
        merge_gap_sec = max(0.0, min(300.0, float(_opt_float(args.get("merge_gap_sec")) or 3.0)))
        candidate_limit = max(1, min(100_000, int(args.get("candidate_limit") or 20_000)))
        transition_limit = max(1, min(120, int(args.get("transition_limit") or 40)))
        segment_limit = max(1, min(200, int(args.get("segment_limit") or 80)))
        evidence_limit = max(1, min(48, int(args.get("evidence_limit") or 24)))

        raw_sources = args.get("sources")
        sources: List[str] = []
        if isinstance(raw_sources, Sequence) and not isinstance(raw_sources, (str, bytes, bytearray)):
            for raw_source in raw_sources:
                normalized = _normalize_archive_source(raw_source)
                if normalized and normalized not in sources:
                    sources.append(normalized)
        else:
            normalized = _normalize_archive_source(args.get("source"))
            if normalized:
                sources.append(normalized)
        if not sources:
            sources = ["vlm_summary", "vlm_alert"]

        from_ts, to_ts, time_meta = self._resolve_summary_time_window(args, default_since_hours=6.0)
        since_ms = int(from_ts * 1000.0)
        until_ms = int(to_ts * 1000.0)

        positive_vec = _agent_normalized_vec(self._emb_text(positive_query))
        if positive_vec is None:
            raise ToolError("CLIP text embedder did not return a positive query vector.")
        negative_vec = _agent_normalized_vec(self._emb_text(negative_query_effective)) if negative_query_effective else None
        alternate_vec = _agent_normalized_vec(self._emb_text(alternate_query)) if alternate_query else None

        rows, source_totals, source_returned, fetch_warnings = self._list_vector_frame_window(
            channel_id=channel_id,
            sources=sources,
            since_ms=since_ms,
            until_ms=until_ms,
            candidate_limit=candidate_limit,
        )
        samples: List[Dict[str, Any]] = []
        for row in rows:
            clip_vec = row.get("clip_vec")
            positive_score = _agent_dot_score(positive_vec, clip_vec)
            if positive_score is None:
                continue
            negative_score = _agent_dot_score(negative_vec, clip_vec) if negative_vec is not None else None
            alternate_score = _agent_dot_score(alternate_vec, clip_vec) if alternate_vec is not None else None
            samples.append(
                _state_sample_from_scores(
                    row,
                    positive_score=positive_score,
                    negative_score=negative_score,
                    alternate_score=alternate_score,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    alternate_label=alternate_label,
                    positive_floor=positive_floor,
                    negative_floor=negative_floor,
                    margin_threshold=margin_threshold,
                )
            )
        samples.sort(key=lambda item: (int(item.get("timestamp_ms") or 0), int(item.get("detection_id") or 0)))

        frame_nodes = [
            {
                "window_start": float(sample["timestamp_ms"]) / 1000.0,
                "window_end": float(sample["timestamp_ms"]) / 1000.0,
            }
            for sample in samples
            if int(sample.get("timestamp_ms") or 0) > 0
        ]
        truncated = any(int(source_totals.get(source) or 0) > int(source_returned.get(source) or 0) for source in sources)
        coverage = _video_summary_coverage_contract(
            available_nodes=frame_nodes,
            returned_nodes=frame_nodes,
            from_ts=from_ts,
            to_ts=to_ts,
            truncated=truncated,
            selection_strategy="full_frame_state_scan",
        )

        segments_all = _build_state_segments_from_samples(
            samples,
            min_state_samples=min_state_samples,
            min_state_duration_sec=min_state_duration_sec,
            merge_gap_sec=merge_gap_sec,
        )
        transitions, boundary_frames, transition_counts = _build_state_transitions(
            segments_all,
            positive_label=positive_label,
            negative_label=negative_label,
            transition_limit=transition_limit,
            evidence_limit=evidence_limit,
        )
        candidate_frames = _select_state_candidate_frames(samples, evidence_limit=evidence_limit)

        state_counts: Dict[str, int] = {}
        confirmed_state_counts: Dict[str, int] = {}
        for segment in segments_all:
            state = str(segment.get("state") or "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1
            if segment.get("stability") == "confirmed":
                confirmed_state_counts[state] = confirmed_state_counts.get(state, 0) + 1
        warnings = [*fetch_warnings, *negative_query_warnings]
        if negative_vec is None:
            warnings.append("negative_state_query was not provided; unknown/positive separation is weaker.")
        if truncated:
            warnings.append("Frame candidate scan was truncated by candidate_limit; counts apply to scanned frames only.")
        if not samples:
            warnings.append("No archived frames with CLIP vectors were available for this channel/time/source filter.")

        return {
            "channel_id": channel_id,
            "subject_query": subject_query,
            "positive_state_query": positive_query,
            "negative_state_query": negative_query or None,
            "negative_state_query_effective": negative_query_effective or None,
            "alternate_state_query": alternate_query or None,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "alternate_label": alternate_label if alternate_query else None,
            "sources": sources,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "time_window": time_meta,
            "since_ms": since_ms,
            "until_ms": until_ms,
            "candidate_limit": candidate_limit,
            "source_totals": source_totals,
            "source_returned": source_returned,
            "frame_count": len(samples),
            "coverage": coverage,
            "score_semantics": "clip_pnm_state_machine_not_ground_truth",
            "thresholds": {
                "positive_floor": positive_floor,
                "negative_floor": negative_floor,
                "margin_threshold": margin_threshold,
                "min_state_samples": min_state_samples,
                "min_state_duration_sec": min_state_duration_sec,
                "merge_gap_sec": merge_gap_sec,
            },
            "counts": {
                **transition_counts,
                "segment_count": len(segments_all),
                "confirmed_segment_count": sum(1 for segment in segments_all if segment.get("stability") == "confirmed"),
                "state_counts": state_counts,
                "confirmed_state_counts": confirmed_state_counts,
            },
            "segments": [_public_state_segment(segment) for segment in segments_all[:segment_limit]],
            "segments_total": len(segments_all),
            "transitions": transitions,
            "transitions_total": transition_counts.get("transition_count", 0),
            "boundary_frames": boundary_frames,
            "candidate_frames": candidate_frames,
            "warnings": warnings,
            "operator_note": (
                "This tool tracks visual state transitions from archived CLIP-scored frames. "
                "Use boundary frame thumbnails and describe_frame for final visual confirmation."
            ),
        }

    # ── create_bookmark ─────────────────────────────────────────────────────

    def _create_bookmark(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id = self._resolve_channel_id(args, required=True)
        if channel_id is None:
            raise ToolError("'channel_id' or 'channel_ref' is required.")
        title = str(args.get("title") or "").strip()[:80]
        if not title:
            raise ToolError("'title' is required.")
        description = str(args.get("description") or "").strip()[:240]
        severity    = str(args.get("severity") or "normal").lower()
        timestamp_ms = _opt_int(args.get("timestamp_ms")) or int(time.time() * 1000)

        if not hasattr(self._lxm, "send_bookmark_event"):
            raise ToolError("Luxriot manager is not available or not configured.")

        try:
            self._lxm.send_bookmark_event(
                channel_id=channel_id,
                title=title,
                description=description,
                severity=severity,
                state="new",
                timestamp_ms=timestamp_ms,
            )
        except Exception as exc:
            raise ToolError(f"Bookmark creation failed: {exc}") from exc

        return {
            "status": "created",
            "channel_id": channel_id,
            "title": title,
            "severity": severity,
        }

    # ── generate_report ─────────────────────────────────────────────────────

    def _generate_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        since_hours = float(args.get("since_hours") or 24)
        until_hours = _opt_float(args.get("until_hours"))
        channel_id  = self._resolve_channel_id(args, required=False)
        include_probes: Optional[List[str]] = args.get("include_probes") or None
        top_events  = max(1, min(20, int(args.get("top_events") or 5)))

        now_ms   = int(time.time() * 1000)
        since_ms = int(now_ms - since_hours * 3_600_000)
        until_ms = int(now_ms - until_hours * 3_600_000) if until_hours is not None else None

        summary_rows = self._ds.summarize_by_probe(since_ms=since_ms, channel_id=channel_id, source="probe")

        # Filter probes if requested
        if include_probes:
            names_lower = {n.lower() for n in include_probes}
            summary_rows = [
                r for r in summary_rows
                if str(r.get("probe_name") or r.get("probe_id") or "").lower() in names_lower
            ]

        probes_data = []
        activity_by_hour: Dict[str, int] = collections.defaultdict(int)
        total_detections = 0

        for row in summary_rows:
            pid = row["probe_id"]
            total_detections += row.get("hit_count", 0)

            # Fetch top events for this probe
            top_rows, _ = self._ds.list_detections(
                probe_id=pid,
                channel_id=channel_id,
                source="probe",
                since_ms=since_ms,
                until_ms=until_ms,
                limit=top_events,
                offset=0,
            )

            # Accumulate hourly buckets
            all_rows, _ = self._ds.list_detections(
                probe_id=pid,
                channel_id=channel_id,
                source="probe",
                since_ms=since_ms,
                until_ms=until_ms,
                limit=500,
                offset=0,
            )
            for det in all_rows:
                ts = _detection_timestamp_ms(det)
                hour_key = time.strftime("%Y-%m-%d %H:00", time.localtime(ts / 1000))
                activity_by_hour[hour_key] += 1

            probes_data.append({
                "probe_id": pid,
                "probe_name": row.get("probe_name"),
                "channel_id": row.get("channel_id"),
                "hit_count": row.get("hit_count", 0),
                "latest_ts": row.get("latest_timestamp_ms"),
                "top_events": [_safe_detection(_annotate_archive_row(r)) for r in top_rows],
            })

        return {
            "period": {
                "since_ms": since_ms,
                "until_ms": until_ms,
                "since_hours": since_hours,
            },
            "total_detections": total_detections,
            "probe_count": len(probes_data),
            "probes": probes_data,
            "activity_by_hour": dict(sorted(activity_by_hour.items())),
        }

    # ── helpers ─────────────────────────────────────────────────────────────

    def _summary_rollups_readonly(
        self,
        *,
        channel_id: int,
        run_selector: str,
        start_ts: float,
        end_ts: float,
        level_limit: int,
    ) -> Dict[str, Any]:
        kwargs = {
            "channel_id": channel_id,
            "run_selector": run_selector,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "level_limit": level_limit,
        }
        try:
            return self._lxm.summary_rollups(**kwargs, synthesize=False)
        except TypeError as exc:
            if "synthesize" not in str(exc):
                raise
            return self._lxm.summary_rollups(**kwargs)

    def _resolve_summary_time_window(
        self,
        args: Dict[str, Any],
        *,
        default_since_hours: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        raw_from = args.get("from_ts")
        raw_to = args.get("to_ts")
        from_ts = _coerce_epoch_seconds(raw_from)
        to_ts = _coerce_epoch_seconds(raw_to)
        normalized_units = {
            "from_ts": _epoch_input_unit(raw_from),
            "to_ts": _epoch_input_unit(raw_to),
        }
        if from_ts is None and to_ts is None:
            since_hours = float(args.get("since_hours") or default_since_hours)
            to_ts = time.time()
            from_ts = to_ts - since_hours * 3600.0
        else:
            if to_ts is None:
                to_ts = time.time()
            if from_ts is None:
                from_ts = max(0.0, float(to_ts) - default_since_hours * 3600.0)
        if from_ts > to_ts:
            from_ts, to_ts = to_ts, from_ts
        return (
            float(from_ts),
            float(to_ts),
            {
                "from_ts": float(from_ts),
                "to_ts": float(to_ts),
                "since_ms": int(float(from_ts) * 1000.0),
                "until_ms": int(float(to_ts) * 1000.0),
                "from_time": _format_epoch_minute(from_ts),
                "to_time": _format_epoch_minute(to_ts),
                "normalized_input_units": normalized_units,
            },
        )

    def _resolve_time_window(
        self,
        args: Dict[str, Any],
        *,
        default_since_hours: float,
    ) -> Tuple[Optional[int], Optional[int]]:
        since_ms = _opt_int(args.get("since_ms"))
        until_ms = _opt_int(args.get("until_ms"))
        if since_ms is None:
            since_hours = _opt_float(args.get("since_hours"))
            if since_hours is None:
                since_hours = default_since_hours
            since_ms = int(time.time() * 1000 - since_hours * 3_600_000)
        if until_ms is None:
            until_hours = _opt_float(args.get("until_hours"))
            if until_hours is not None:
                until_ms = int(time.time() * 1000 - until_hours * 3_600_000)
        if since_ms is not None and until_ms is not None and since_ms > until_ms:
            since_ms, until_ms = until_ms, since_ms
        return since_ms, until_ms

    def _list_vector_frame_window(
        self,
        *,
        channel_id: int,
        sources: Sequence[str],
        since_ms: Optional[int],
        until_ms: Optional[int],
        candidate_limit: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int], List[str]]:
        per_source_limit = max(1, int(candidate_limit // max(1, len(sources))))
        rows_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        source_totals: Dict[str, int] = {}
        source_returned: Dict[str, int] = {}
        warnings: List[str] = []

        for source in sources:
            source_rows: List[Dict[str, Any]] = []
            total: Optional[int] = None
            ids_clean: List[int] = []
            if hasattr(self._ds, "count_vector_candidates"):
                try:
                    total = int(self._ds.count_vector_candidates(
                        channel_id=channel_id,
                        source=source,
                        since_ms=since_ms,
                        until_ms=until_ms,
                        only_with_clip=True,
                    ))
                except Exception as exc:
                    warnings.append(f"count_vector_candidates failed for {source}: {str(exc)[:160]}")
            if hasattr(self._ds, "list_vector_candidates"):
                try:
                    source_rows = list(self._ds.list_vector_candidates(
                        channel_id=channel_id,
                        source=source,
                        since_ms=since_ms,
                        until_ms=until_ms,
                        limit=per_source_limit,
                        only_with_clip=True,
                        include_vectors=True,
                        include_thumbnail=False,
                    ))
                except TypeError:
                    try:
                        source_rows = list(self._ds.list_vector_candidates(
                            channel_id=channel_id,
                            source=source,
                            since_ms=since_ms,
                            until_ms=until_ms,
                            limit=per_source_limit,
                        ))
                    except Exception as exc:
                        warnings.append(f"list_vector_candidates failed for {source}: {str(exc)[:160]}")
                except Exception as exc:
                    warnings.append(f"list_vector_candidates failed for {source}: {str(exc)[:160]}")

            if not source_rows:
                try:
                    plain_rows, plain_total = self._list_detection_window(
                        probe_id=None,
                        channel_id=channel_id,
                        source=source,
                        since_ms=since_ms,
                        until_ms=until_ms,
                        limit=per_source_limit,
                        offset=0,
                        sort_by="oldest",
                        max_scan=per_source_limit,
                    )
                    total = int(plain_total) if total is None else total
                    ids = [
                        _opt_int(row.get("id") or row.get("detection_id"))
                        for row in plain_rows
                    ]
                    ids_clean = [int(item) for item in ids if item is not None]
                    if ids_clean and hasattr(self._ds, "fetch_detections_by_ids"):
                        try:
                            fetched = self._ds.fetch_detections_by_ids(
                                ids_clean,
                                include_vectors=True,
                                include_thumbnail=False,
                            )
                        except TypeError:
                            fetched = self._ds.fetch_detections_by_ids(
                                ids_clean,
                                include_vectors=True,
                            )
                        source_rows = [dict(row) for row in fetched if isinstance(row, dict)]
                    else:
                        source_rows = plain_rows
                except Exception as exc:
                    warnings.append(f"vector fallback failed for {source}: {str(exc)[:160]}")

            prepared_rows = []
            for row in source_rows:
                if not isinstance(row, dict):
                    continue
                clip_vec = row.get("clip_vec")
                if _agent_normalized_vec(clip_vec) is None:
                    continue
                annotated = _annotate_archive_row(dict(row))
                if not annotated.get("source"):
                    annotated["source"] = source
                prepared_rows.append(annotated)
                key = _visual_signal_row_key(annotated)
                existing = rows_by_key.get(key)
                if existing is None or _detection_timestamp_ms(annotated) < _detection_timestamp_ms(existing):
                    rows_by_key[key] = annotated
            source_returned[source] = len(prepared_rows)
            source_totals[source] = int(total) if total is not None else len(prepared_rows)

        rows = list(rows_by_key.values())
        rows.sort(key=lambda row: (_detection_timestamp_ms(row), _opt_int(row.get("id") or row.get("detection_id")) or 0))
        return rows[:candidate_limit], source_totals, source_returned, warnings

    def _list_detection_window(
        self,
        *,
        probe_id: Optional[str],
        channel_id: Optional[int],
        source: Optional[str],
        since_ms: Optional[int],
        until_ms: Optional[int],
        limit: int,
        offset: int,
        sort_by: str,
        max_scan: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        max_scan = max(limit + offset, max_scan or (limit + offset))
        max_scan = max(limit + offset, min(5_000, int(max_scan)))
        batch_size = min(500, max_scan)
        scanned: List[Dict[str, Any]] = []
        total = 0
        next_offset = 0
        while next_offset < max_scan:
            rows, total = self._ds.list_detections(
                probe_id=probe_id,
                channel_id=channel_id,
                source=source,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=min(batch_size, max_scan - next_offset),
                offset=next_offset,
            )
            if not rows:
                break
            scanned.extend(rows)
            next_offset += len(rows)
            if len(rows) < batch_size or next_offset >= total:
                break
        ordered = _sort_detection_rows(scanned, sort_by)
        return ordered[offset: offset + limit], total

    def _filter_detection_band(
        self,
        rows: List[Dict[str, Any]],
        band: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        score_field = str(band.get("score_field") or "margin")
        lower = band.get("min")
        upper = band.get("max")
        out: List[Dict[str, Any]] = []
        for row in rows:
            value = _opt_float(row.get(score_field))
            if value is None:
                continue
            if lower is not None and value < float(lower):
                continue
            if upper is not None and value > float(upper):
                continue
            out.append(row)
        return out

    def _resolve_probe_id_by_name(self, name: str) -> str:
        name_lower = name.lower()
        matches = [
            p for p in self._ps.list_probes()
            if str(p.get("name") or "").lower() == name_lower
        ]
        if not matches:
            raise ToolError(f"No probe found with name {name!r}.")
        if len(matches) > 1:
            ids = [p.get("id") for p in matches]
            raise ToolError(
                f"Multiple probes named {name!r}: {ids}. Use probe_id to be specific."
            )
        return str(matches[0]["id"])

    def _find_probe_mentions_in_text(self, text: str) -> List[Dict[str, Any]]:
        normalized_text = _normalize_probe_match_text(text)
        if not normalized_text:
            return []
        hits: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for probe in self._ps.list_probes():
            probe_name = str(probe.get("name") or "").strip()
            probe_id = str(probe.get("id") or "").strip()
            if not probe_name or not probe_id:
                continue
            normalized_name = _normalize_probe_match_text(probe_name)
            if not normalized_name:
                continue
            if normalized_name in normalized_text and probe_id not in seen_ids:
                hits.append(probe)
                seen_ids.add(probe_id)
        return hits

    @staticmethod
    def _normalize_channel_ref(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw.startswith("#"):
            raw = raw[1:]
        return re.sub(r"[^a-z0-9]+", " ", raw).strip()

    def _resolve_channel_id(self, args: Dict[str, Any], *, required: bool = False) -> Optional[int]:
        channel_id = _opt_int(args.get("channel_id"))
        if channel_id is not None:
            return channel_id

        raw_ref = None
        for field_name in ("channel_ref", "channel", "channel_title", "channel_name"):
            value = args.get(field_name)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                raw_ref = value_str
                break

        if raw_ref is None:
            if required:
                raise ToolError("Provide 'channel_id' or 'channel_ref'.")
            return None

        numeric_ref = _opt_int(str(raw_ref).lstrip("#"))
        if numeric_ref is not None:
            return numeric_ref

        raw_channels = getattr(self._lxm, "channels", None) or []
        channels = [
            ch for ch in raw_channels
            if isinstance(ch, dict) and _opt_int(ch.get("id")) is not None
        ]
        if not channels:
            raise ToolError(
                f"Could not resolve channel {raw_ref!r}: Luxriot did not report any channels. "
                "Call list_channels first or verify the connection."
            )

        ref_norm = self._normalize_channel_ref(raw_ref)
        matches: List[Tuple[int, str]] = []
        for channel in channels:
            cid = _opt_int(channel.get("id"))
            if cid is None:
                continue
            title = str(channel.get("title") or channel.get("name") or channel.get("label") or f"channel-{cid}")
            title_norm = self._normalize_channel_ref(title)
            if not title_norm:
                continue
            if ref_norm == title_norm or ref_norm in title_norm or title_norm in ref_norm:
                matches.append((cid, title))

        if not matches:
            known = ", ".join(
                f"#{_opt_int(ch.get('id'))} {str(ch.get('title') or ch.get('name') or ch.get('label') or 'unknown')}"
                for ch in channels[:8]
                if _opt_int(ch.get("id")) is not None
            ) or "none"
            raise ToolError(f"No Luxriot channel matches {raw_ref!r}. Known channels: {known}")
        if len(matches) > 1:
            raise ToolError(
                f"Channel reference {raw_ref!r} is ambiguous. Matches: "
                + ", ".join(f"#{cid} {title}" for cid, title in matches[:8])
            )
        return matches[0][0]

    def _find_probe(self, probe_id: str) -> Dict[str, Any]:
        for p in self._ps.list_probes():
            if str(p.get("id")) == probe_id:
                return copy.deepcopy(p)
        raise ToolError(f"Probe not found: {probe_id!r}")

    def _report_progress(self, payload: Dict[str, Any]) -> None:
        cb = getattr(self._local, "progress_cb", None)
        if callable(cb):
            try:
                cb(dict(payload))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Probe merge / validate helpers
# ---------------------------------------------------------------------------

_SCALAR_PROBE_FIELDS = (
    "pos_floor", "margin", "top_k", "window_sec",
    "enabled", "severity", "bookmark",
    "bookmark_cooldown_sec", "bookmark_dedupe_window_sec",
    "name", "channel_id",
)

def _merge_probe(current: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge agent changes onto the full probe object.
    Only the fields present in 'changes' are updated.
    ROI and image_probe are never touched — they are preserved as-is.
    """
    merged = copy.deepcopy(current)

    # Map agent field name 'margin_thr' → stored field 'margin'
    if "margin_thr" in changes:
        changes = dict(changes)
        changes["margin"] = changes.pop("margin_thr")
    if "bookmark_enabled" in changes:
        changes = dict(changes)
        changes["bookmark"] = changes.pop("bookmark_enabled")

    # Scalar fields
    for f in _SCALAR_PROBE_FIELDS:
        if f in changes:
            merged[f] = changes[f]

    # List fields: full replacement, cleaned
    for f in ("positives", "negatives"):
        if f in changes:
            raw = changes[f]
            if not isinstance(raw, list):
                raise ToolError(f"'{f}' must be a list of strings.")
            merged[f] = [str(s).strip() for s in raw if str(s).strip()]

    if "positives" in changes or "negatives" in changes:
        merged["pairs"] = _probe_pairs_from_lists(
            merged.get("positives") or [],
            merged.get("negatives") or [],
        )

    # ROI and image_probe: intentionally not in the schema — preserved unchanged.

    return merged


def _validate_probe(probe: Dict[str, Any]) -> List[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: List[str] = []
    positives  = probe.get("positives") or []
    image_data = (probe.get("image_probe") or {}).get("data")
    image_en   = (probe.get("image_probe") or {}).get("enabled", True)

    if not positives and not (image_data and image_en is not False):
        errors.append("Probe must have at least one positive text or an enabled image probe.")

    pos_floor = probe.get("pos_floor")
    if pos_floor is not None:
        try:
            v = float(pos_floor)
            if not (0.0 <= v <= 1.0):
                errors.append(f"pos_floor must be between 0.0 and 1.0, got {v}.")
        except (TypeError, ValueError):
            errors.append(f"pos_floor must be a number, got {pos_floor!r}.")

    margin = probe.get("margin")
    if margin is not None:
        try:
            v = float(margin)
            if v < 0.0:
                errors.append(f"margin must be >= 0.0, got {v}.")
        except (TypeError, ValueError):
            errors.append(f"margin must be a number, got {margin!r}.")

    bookmark_cooldown_sec = probe.get("bookmark_cooldown_sec")
    if bookmark_cooldown_sec is not None:
        try:
            v = float(bookmark_cooldown_sec)
            if v < 0.0:
                errors.append(f"bookmark_cooldown_sec must be >= 0.0, got {v}.")
        except (TypeError, ValueError):
            errors.append(f"bookmark_cooldown_sec must be a number, got {bookmark_cooldown_sec!r}.")

    bookmark_dedupe_window_sec = probe.get("bookmark_dedupe_window_sec")
    if bookmark_dedupe_window_sec is not None:
        try:
            v = float(bookmark_dedupe_window_sec)
            if v < 0.5:
                errors.append(f"bookmark_dedupe_window_sec must be >= 0.5, got {v}.")
        except (TypeError, ValueError):
            errors.append(f"bookmark_dedupe_window_sec must be a number, got {bookmark_dedupe_window_sec!r}.")

    return errors


def _probe_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Return {field: {before, after}} for changed fields."""
    diff: Dict[str, Any] = {}
    all_keys = set(before) | set(after)
    skip = {"bookmark_gate", "bookmark_gate_updated_at_ms", "last_hit", "recent_hits", "image_probe", "roi_norm"}
    for k in sorted(all_keys - skip):
        if before.get(k) != after.get(k):
            diff[k] = {"before": before.get(k), "after": after.get(k)}
    return diff


def _probe_summary(probe: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":          probe.get("id"),
        "name":        probe.get("name"),
        "channel_id":  probe.get("channel_id"),
        "enabled":     probe.get("enabled"),
        "positives":   probe.get("positives"),
        "negatives":   probe.get("negatives"),
        "pos_floor":   probe.get("pos_floor"),
        "margin":      probe.get("margin"),
        "top_k":       probe.get("top_k"),
        "window_sec":  probe.get("window_sec"),
        "severity":    probe.get("severity"),
        "bookmark":    probe.get("bookmark"),
        "bookmark_cooldown_sec": probe.get("bookmark_cooldown_sec"),
        "bookmark_dedupe_window_sec": probe.get("bookmark_dedupe_window_sec"),
    }


def _probe_pairs_from_lists(positives: Sequence[str], negatives: Sequence[str]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    max_len = max(len(positives), len(negatives))
    for idx in range(max_len):
        pos = str(positives[idx]).strip() if idx < len(positives) else ""
        neg = str(negatives[idx]).strip() if idx < len(negatives) else ""
        if pos or neg:
            pairs.append({"positive": pos, "negative": neg})
    return pairs


def _merge_prompt_settings_snapshot(current: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(current)
    for field_name in ("stream_system_prompt", "json_alert_prompt", "bookmark_enabled", "bookmark_cooldown_sec"):
        if field_name in changes:
            merged[field_name] = changes[field_name]
    bookmark_rule_prompt = str(changes.get("bookmark_rule_prompt") or "").strip()
    if bookmark_rule_prompt:
        current_stream_prompt = str(merged.get("stream_system_prompt") or "").strip()
        if bookmark_rule_prompt not in current_stream_prompt:
            merged["stream_system_prompt"] = (
                f"{current_stream_prompt}\n- {bookmark_rule_prompt}"
                if current_stream_prompt else bookmark_rule_prompt
            )
    if isinstance(changes.get("rollup_prompts"), dict):
        rollups = dict(merged.get("rollup_prompts") or {})
        for level, prompt in changes["rollup_prompts"].items():
            level_key = str(level).strip().upper()
            if level_key in {"L1", "L2", "L3"}:
                rollups[level_key] = str(prompt)
        merged["rollup_prompts"] = rollups
    return merged


def _normalize_prompt_setting_changes(changes: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(changes, dict):
        return {}
    normalized = copy.deepcopy(changes)

    if "stream_system_prompt" not in normalized:
        for alias in ("l0_prompt", "live_prompt"):
            if alias in normalized:
                normalized["stream_system_prompt"] = normalized[alias]
                break

    rollups = dict(normalized.get("rollup_prompts") or {}) if isinstance(normalized.get("rollup_prompts"), dict) else {}
    for alias, level in (("l1_prompt", "L1"), ("l2_prompt", "L2"), ("l3_prompt", "L3")):
        if alias in normalized:
            rollups[level] = normalized[alias]
    if rollups:
        normalized["rollup_prompts"] = rollups

    for alias in ("l0_prompt", "live_prompt", "bookmark_rule_prompt", "l1_prompt", "l2_prompt", "l3_prompt"):
        normalized.pop(alias, None)

    return normalized


def _prompt_settings_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for field_name in (
        "stream_system_prompt",
        "json_alert_prompt",
        "bookmark_enabled",
        "bookmark_cooldown_sec",
        "rollup_prompts",
    ):
        if before.get(field_name) != after.get(field_name):
            diff[field_name] = {
                "before": before.get(field_name),
                "after": after.get(field_name),
            }
    return diff


def _load_runtime_skill_docs() -> List[Dict[str, str]]:
    skills_root = Path(__file__).resolve().parent / "skills"
    docs: List[Dict[str, str]] = []
    if not skills_root.exists():
        return docs
    for skill_file in sorted(skills_root.rglob("SKILL.md")):
        try:
            text = skill_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not text:
            continue
        docs.append({
            "slug": skill_file.parent.name,
            "name": skill_file.parent.name.replace("_", " "),
            "content": text,
        })
    return docs


def _skill_summary_line(content: str) -> str:
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return line[:220]
    return "No summary provided."


def _skill_trigger_phrases(content: str) -> List[str]:
    phrases: List[str] = []
    lines = str(content or "").splitlines()
    in_section = False
    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()
        if not in_section:
            if lower == "trigger phrases:":
                in_section = True
            continue
        if not line:
            if phrases:
                break
            continue
        if line.startswith("#"):
            break
        if re.match(r"^[A-Za-z][A-Za-z0-9 _-]*:$", line):
            break
        if line.startswith("- "):
            phrase = line[2:].strip().strip("`").strip()
            if phrase:
                phrases.append(phrase)
            continue
        if phrases:
            break
    return phrases


def _format_runtime_skill_index_for_prompt() -> str:
    docs = _load_runtime_skill_docs()
    if not docs:
        return ""
    remaining = AGENT_MAX_RUNTIME_SKILLS_CHARS
    parts: List[str] = []
    for doc in docs:
        if remaining <= 0:
            break
        line = f"- {doc['slug']}: {_skill_summary_line(doc['content'])}"
        if len(line) > remaining:
            break
        parts.append(line)
        remaining -= len(line) + 1
    if not parts:
        return ""
    return "\n\nRepository Playbooks Index:\n" + "\n".join(parts)


def _extract_requested_skill_slugs(message: Any) -> List[str]:
    text = ""
    if isinstance(message, str):
        text = message
    elif isinstance(message, list):
        chunks: List[str] = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text") or ""))
        text = "\n".join(chunks)
    raw = str(text or "")
    if not raw:
        return []
    docs = _load_runtime_skill_docs()
    by_slug = {str(doc.get("slug") or "").strip(): doc for doc in docs}
    hits: List[str] = []
    lower = raw.lower()
    for slug in by_slug:
        if not slug:
            continue
        if f'use playbook "{slug.lower()}"' in lower or f"use playbook '{slug.lower()}'" in lower:
            hits.append(slug)
        elif f"skill:{slug.lower()}" in lower or f"playbook:{slug.lower()}" in lower:
            hits.append(slug)
            continue
        doc = by_slug.get(slug) or {}
        for phrase in _skill_trigger_phrases(str(doc.get("content") or "")):
            if phrase and phrase.lower() in lower:
                hits.append(slug)
                break
    return list(dict.fromkeys(hits))


def _extract_text_from_message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        chunks: List[str] = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text") or ""))
        return "\n".join(chunks)
    return str(message or "")


def _normalize_probe_match_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _normalize_archive_source(value: Any) -> Optional[str]:
    source = str(value or "").strip().lower()
    if not source:
        return None
    source = ARCHIVE_SOURCE_ALIASES.get(source, source)
    if source not in ARCHIVE_SOURCE_LABELS:
        raise ToolError("source must be one of: probe, vlm_summary, vlm_alert")
    return source


def _archive_source_label(source: Any) -> str:
    normalized = str(source or "").strip().lower()
    normalized = ARCHIVE_SOURCE_ALIASES.get(normalized, normalized)
    return ARCHIVE_SOURCE_LABELS.get(normalized, "Archive frame")


def _archive_item_type(source: Any) -> str:
    normalized = str(source or "").strip().lower()
    normalized = ARCHIVE_SOURCE_ALIASES.get(normalized, normalized)
    return ARCHIVE_SOURCE_ITEM_TYPES.get(normalized, "archive_frame")


def _annotate_archive_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    source = str(out.get("source") or "").strip().lower()
    if source:
        out["source"] = source
    logical_source = ARCHIVE_SOURCE_ALIASES.get(source, source)
    if logical_source and logical_source != source:
        out["logical_source"] = logical_source
    out["source_label"] = _archive_source_label(source)
    out["archive_item_type"] = _archive_item_type(source)
    out["is_probe_detection"] = logical_source == "probe"
    out["is_video_description_frame"] = logical_source in {"vlm_summary", "vlm_alert"}
    return out


def _format_active_skill_docs_for_prompt(skill_slugs: Sequence[str]) -> str:
    if not skill_slugs:
        return ""
    docs = _load_runtime_skill_docs()
    by_slug = {str(doc.get("slug") or "").strip(): doc for doc in docs}
    remaining = AGENT_MAX_ACTIVE_SKILL_CHARS
    parts: List[str] = []
    for slug in skill_slugs:
        doc = by_slug.get(str(slug or "").strip())
        if not doc or remaining <= 0:
            continue
        header = f"### Active Playbook: {doc['slug']}\n"
        budget = remaining - len(header)
        if budget <= 0:
            break
        content = str(doc.get("content") or "")
        if len(content) > budget:
            content = content[: max(0, budget - 16)].rstrip() + "\n[truncated]"
        block = header + content
        parts.append(block)
        remaining -= len(block) + 2
    if not parts:
        return ""
    return "\n\nActivated Playbooks:\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    probes_store: Any,
    detections_store: Any,
    luxriot_manager: Any,
    active_skill_slugs: Optional[Sequence[str]] = None,
    allowed_channel_ids: Optional[Sequence[str]] = None,
    secure_tool_mode: bool = False,
) -> str:
    now_str = time.strftime("%Y-%m-%d %H:%M")

    # Active probes
    try:
        probes = probes_store.list_probes()
    except Exception:
        probes = []
    allowed_channels = (
        None
        if allowed_channel_ids is None or "*" in allowed_channel_ids
        else {str(channel_id) for channel_id in allowed_channel_ids}
    )
    if allowed_channels is not None:
        probes = [
            probe
            for probe in probes
            if str(probe.get("channel_id")) in allowed_channels
        ]

    # Recent detection counts (last 24h)
    since_ms = int(time.time() * 1000 - 86_400_000)
    try:
        probe_summary = detections_store.summarize_by_probe(since_ms=since_ms, source="probe")
        hit_by_probe  = {r["probe_id"]: r["hit_count"] for r in probe_summary}
    except Exception:
        hit_by_probe  = {}

    # Available channels
    try:
        raw_channels = getattr(luxriot_manager, "channels", None) or []
        raw_channel_list = (
            raw_channels if isinstance(raw_channels, list) else []
        )
        visible_channels = (
            raw_channel_list
            if allowed_channels is None
            else [
                channel
                for channel in raw_channel_list
                if str(channel.get("id")) in allowed_channels
            ]
        )
        channels_str = ", ".join(
            f"{c.get('id')} ({c.get('title', 'unknown')})"
            for c in visible_channels
        ) or "unknown (Luxriot not connected)"
    except Exception:
        channels_str = "unknown"

    probe_lines = []
    for idx, p in enumerate(probes):
        if idx >= AGENT_MAX_PROBES_IN_PROMPT:
            break
        pid    = p.get("id", "?")
        pname  = p.get("name", pid)
        ch     = p.get("channel_id", "?")
        floor  = p.get("pos_floor", "?")
        margin = p.get("margin", "?")
        en     = "enabled" if p.get("enabled", True) else "disabled"
        hits   = hit_by_probe.get(pid, 0)
        probe_lines.append(
            f"  - \"{pname}\" [id={pid}, ch={ch}]  "
            f"pos_floor={floor}, margin={margin}, {en}, {hits} hits/24h"
        )
    remaining_probe_count = max(0, len(probes) - len(probe_lines))
    if remaining_probe_count > 0:
        probe_lines.append(f"  - ... {remaining_probe_count} more probe(s) not expanded here")

    probe_block = "\n".join(probe_lines) if probe_lines else "  (no probes configured)"
    skills_block = _format_runtime_skill_index_for_prompt()
    active_skills_block = _format_active_skill_docs_for_prompt(active_skill_slugs or [])
    secure_rules = (
        "\n- Tool-driven probe and prompt changes are preview-only in this "
        "deployment. Never request preview=false or claim that a preview was applied."
        "\n- Bookmark creation is unavailable until the stored approval workflow "
        "is enabled."
        if secure_tool_mode
        else ""
    )

    return (
        f"You are the AI operations assistant for Luxriot EVA AI — a CCTV intelligent "
        f"monitoring platform. You have tools to search the archive, inspect detections, "
        f"assemble research batches, tune probes, adjust prompt settings, describe frames, "
        f"create bookmarks, and compile reports.\n"
        f"Be concise and operator-focused. Never fabricate detection data.\n\n"
        f"Current time: {now_str}\n\n"
        f"Active probes ({len(probes)} total):\n{probe_block}\n\n"
        f"Available channels: {channels_str}\n\n"
        f"Rules:\n"
        f"- For probe modifications: always call update_probe with preview=true first, "
        f"show the user the diff, and only apply after explicit confirmation.\n"
        f"- For prompt-setting modifications: always call update_prompt_settings with preview=true first, "
        f"show the user the diff, and only apply after explicit confirmation.\n"
        f"- Prompt field mapping: L0/live feed prompt = stream_system_prompt. L1/L2/L3 rollups = rollup_prompts.L1/L2/L3. Behavioral bookmark instructions belong inside the L0/live prompt. json_alert_prompt is only the structured alert-output template.\n"
        f"- There is no separate bookmark-rule registry. A bookmark rule only exists after update_prompt_settings applies the underlying L0/live prompt change.\n"
        f"- Do not rewrite json_alert_prompt unless the operator explicitly asks to change the structured alert/parsing template.\n"
        f"- Never claim that a prompt change, bookmark rule, or channel-specific setting was applied unless the corresponding tool returned status=applied in this turn.\n"
        f"- Never claim that Luxriot is disconnected or that a channel does not exist unless list_channels or another Luxriot tool in this turn confirmed that failure.\n"
        f"- Probe-threshold semantics are strict: raising pos_floor or raising margin makes a probe stricter; lowering pos_floor or lowering margin makes it more permissive. Never describe lowering margin as tightening, filtering more, or reducing noise.\n"
        f"- Detection hit counts over 24h are historical archive summaries. After a probe threshold change, do not claim that the 24h hit count already improved, dropped, or 'took effect' unless you explicitly measured a fresh post-change window.\n"
        f"- If the operator asks for probe status immediately after an update, report the saved settings and explain that effect on live volume still requires post-change observation unless a fresh post-change query was run.\n"
        f"- Do not claim support for PDF export, CSV export, emails, file links, async report queues, or background jobs unless a tool explicitly returns that artifact.\n"
        f"- If an operator asks for an unsupported export, say so plainly and offer the closest available format, such as a structured chat report.\n"
        f"- Prefer absolute time windows (since_ms/until_ms or from_ts/to_ts) when the operator asks about a specific date or period.\n"
        f"- For video-description or video-summary review over a period, first call normalize_time_window unless the user already provided exact Unix timestamps. Use from_ts/to_ts for video summaries and since_ms/until_ms for detection archive tools.\n"
        f"- For relative period phrases such as 'last day', 'last 24 hours', 'last two hours', or 'past 90 minutes', call normalize_time_window with relative_range set to the phrase; do not invent local start_time/end_time strings. Interpret 'last day' as rolling 24 hours; use 'yesterday' only for the previous calendar day.\n"
        f"- If the operator asks for video-summary review without naming channel(s), call list_video_summary_channels for the normalized period before reading full summaries. If active_count exceeds the per_turn_channel_limit, present candidate channels and ask the operator to choose channels or confirm full multi-turn research.\n"
        f"- Do not review more than {AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN} channels of video summaries in one turn unless the operator explicitly confirmed broad research. For broad research, work in chunks and report unchecked channels.\n"
        f"- For video event investigations over a non-trivial period, use rollups as a map before detail: L2 for broad context, L1 for candidate windows, and live/L0 only to verify exact events and evidence. Do not treat L2/L1 as visual proof.\n"
        f"- For count/state-change questions that can be checked visually, such as 'how many times did X appear/disappear', 'when did the door open/close', or 'did the object leave/return', prefer track_visual_state_transitions after normalize_time_window. Provide positive_state_query and a visible-background negative_state_query; avoid literal negation like 'no X'/'without X' because CLIP does not reliably understand negation. Use L2/L1 summaries as a map and use count_video_summary_events only as summary-text fallback when archived CLIP frames are unavailable. Report that CLIP P/N/M state transitions are candidates and cite boundary frame evidence before strong conclusions.\n"
        f"- For count questions over video summaries, such as counting mentions in summaries, use count_video_summary_events after normalize_time_window. If the operator did not name a channel, call list_video_summary_channels first and then call count_video_summary_events separately for each returned candidate channel, up to the per-turn channel limit. Never call count_video_summary_events without channel_id/channel_ref. Do not call get_detections with probe_name unless the operator named an actual configured probe. Report counts with coverage and distinguish explicit summary mentions from inferred adjacent-window state changes.\n"
        f"- Archive source semantics: source=probe rows are real probe hits/detections; source=vlm_summary rows are sampled frames saved from video-description batches; source=vlm_alert rows are frames anchored to VLM alerts from video descriptions.\n"
        f"- Do not call vlm_summary or vlm_alert rows probe detections. When answering from archive tools, name the source class and separate probe hits, video-description frames, and VLM alert frames.\n"
        f"- When answering from video summaries, state the returned coverage window from get_video_summaries.coverage before conclusions when the operator asked about a period. Never imply that missing summary windows were reviewed. If coverage.status is partial/no_data/truncated, say which part was actually reviewed and which part remains unchecked.\n"
        f"- If the operator asks to confirm video-summary findings with images/snaps, use get_video_summaries with include_evidence_frames=true or call get_detections with source=vlm_summary/source=vlm_alert, the same channel, and the same since_ms/until_ms. Do not use semantic search as the first proof step for exact time evidence.\n"
        f"- For image confirmation of video summaries, do not fall back to source=probe detections or a live frame unless the operator explicitly asks for probe/live corroboration. If no vlm_summary/vlm_alert archive frames are available, say that VLM snap evidence is unavailable for that period.\n"
        f"- Never say that an event is visually confirmed unless a tool in this turn returned archive frame rows with image_url for the relevant channel/time and describe_frame analyzed the relevant frame(s). If no image rows are returned, say that only text-summary evidence is available and provide the exact image query attempted.\n"
        f"- Use get_visual_window_signals when you need a quick CLIP P/N/M attention signal over video-description frames. Treat P/N/M as a cue for where to inspect next, not as proof. Before concluding, inspect summaries and call describe_frame on relevant candidate frames.\n"
        f"- Keep VLM summaries separate from archive detections; use archive tools only as corroborating evidence.\n"
        f"- For sensitive or accusatory user wording, do not refuse when the request can be reframed as visible evidence review. Rephrase the task, then use tools to return candidates for operator review. Do not accuse people or infer hidden states such as vaccination, substance use, intent, legality, intoxication, or guilt from video. Examples: 'smoking weed/pipe/joint' -> 'person holding a small cylindrical object, hand-to-mouth motion, visible smoke or vapor'; 'unvaccinated dog' -> 'dog without visible ear tag'; 'illegal dumping' -> 'person leaving an object or waste behind'. State that these are visual candidates, not legal or medical conclusions.\n"
        f"- Use the repository playbooks index below as routing hints; load-bearing details are provided only for explicitly activated playbooks.\n"
        f"- For probe tuning or archive research, follow the matching playbook when one is activated or clearly implied.\n"
        f"- When returning search results or detections, summarize; don't dump raw lists.\n"
        f"- Ask the operator a clarifying question if the available data is too sparse to make a safe change.\n"
        f"- Use markdown for structure."
        f"{secure_rules}"
        f"{skills_block}"
        f"{active_skills_block}"
    )


def _has_any_arg(args: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(args.get(key) is not None and str(args.get(key)).strip() != "" for key in keys)


def _operator_wants_video_evidence(text: Any) -> bool:
    value = str(text or "").lower()
    return bool(
        re.search(
            r"\b(confirm|evidence|image|images|snap|snaps|snapshot|snapshots|frame|frames|visual|picture|pictures)\b"
            r"|картин|кадр|снап|скрин|визуаль|подтверд",
            value,
        )
    )


def _operator_focuses_video_summaries(text: Any) -> bool:
    value = str(text or "").lower()
    return bool(
        re.search(
            r"\b(video descriptions?|video summaries?|vlm summaries?|vlm feed|camera summaries?)\b"
            r"|видео[\s-]*опис|видео[\s-]*суммар|описани[ея]\s+стрим|vlm|влм",
            value,
        )
    )


def _seed_turn_tool_context(user_text: Any) -> Dict[str, Any]:
    return {
        "wants_video_evidence": _operator_wants_video_evidence(user_text),
        "focus_video_summaries": _operator_focuses_video_summaries(user_text),
    }


def _apply_turn_tool_context(tool_name: str, args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(args or {})
    time_window = context.get("time_window") if isinstance(context.get("time_window"), dict) else {}
    if time_window:
        if tool_name in {"get_video_summaries", "count_video_summary_events", "track_visual_state_transitions", "list_video_summary_channels"} and not _has_any_arg(
            prepared,
            ("from_ts", "to_ts", "since_hours"),
        ):
            if time_window.get("from_ts") is not None:
                prepared["from_ts"] = time_window.get("from_ts")
            if time_window.get("to_ts") is not None:
                prepared["to_ts"] = time_window.get("to_ts")
        if tool_name in {"get_detections", "get_detection_summary", "search_archive"} and not _has_any_arg(
            prepared,
            ("since_ms", "until_ms", "since_hours", "until_hours"),
        ):
            if time_window.get("since_ms") is not None:
                prepared["since_ms"] = time_window.get("since_ms")
            if time_window.get("until_ms") is not None:
                prepared["until_ms"] = time_window.get("until_ms")

    channel_id = context.get("channel_id")
    should_default_channel = not (
        tool_name == "describe_frame"
        and _has_any_arg(prepared, ("detection_id", "image_path"))
    )
    if should_default_channel and channel_id is not None and tool_name in {
        "get_video_summaries",
        "get_detections",
        "get_detection_summary",
        "search_archive",
        "describe_frame",
        "count_video_summary_events",
        "track_visual_state_transitions",
    } and not _has_any_arg(prepared, ("channel_id", "channel_ref", "channel", "channel_title", "channel_name")):
        prepared["channel_id"] = channel_id

    if tool_name == "get_video_summaries" and context.get("wants_video_evidence"):
        prepared.setdefault("include_evidence_frames", True)
        prepared.setdefault("evidence_frame_limit", 8)

    if (
        tool_name in {"get_detections", "get_detection_summary", "search_archive"}
        and context.get("focus_video_summaries")
        and context.get("wants_video_evidence")
        and not prepared.get("source")
    ):
        prepared["source"] = "vlm_summary"

    return prepared


def _remember_turn_tool_result(tool_name: str, result: Any, context: Dict[str, Any]) -> None:
    if not isinstance(result, Mapping):
        return
    if result.get("error"):
        return

    if tool_name == "normalize_time_window":
        if result.get("from_ts") is not None and result.get("to_ts") is not None:
            context["time_window"] = {
                "from_ts": result.get("from_ts"),
                "to_ts": result.get("to_ts"),
                "since_ms": result.get("since_ms"),
                "until_ms": result.get("until_ms"),
                "from_local": result.get("from_local"),
                "to_local": result.get("to_local"),
            }
        return

    if tool_name == "list_video_summary_channels":
        time_window = result.get("time_window")
        if isinstance(time_window, Mapping) and time_window.get("from_ts") is not None and time_window.get("to_ts") is not None:
            context["time_window"] = {
                "from_ts": time_window.get("from_ts"),
                "to_ts": time_window.get("to_ts"),
                "since_ms": time_window.get("since_ms"),
                "until_ms": time_window.get("until_ms"),
            }
        candidates = result.get("candidate_channels")
        if isinstance(candidates, list) and len(candidates) == 1 and isinstance(candidates[0], Mapping):
            channel_id = _opt_int(candidates[0].get("channel_id"))
            if channel_id is not None:
                context["channel_id"] = channel_id
        return

    if tool_name in {"get_video_summaries", "count_video_summary_events", "track_visual_state_transitions"}:
        channel_id = _opt_int(result.get("channel_id"))
        if channel_id is not None:
            context["channel_id"] = channel_id
        time_window = result.get("time_window")
        if isinstance(time_window, Mapping) and time_window.get("from_ts") is not None and time_window.get("to_ts") is not None:
            context["time_window"] = {
                "from_ts": time_window.get("from_ts"),
                "to_ts": time_window.get("to_ts"),
                "since_ms": time_window.get("since_ms"),
                "until_ms": time_window.get("until_ms"),
            }
        return


# ---------------------------------------------------------------------------
# Agent runner (tool loop + SSE)
# ---------------------------------------------------------------------------

class AgentRunner:
    """
    Drives the tool-calling loop and yields SSE events as strings.
    Instantiated once in oldapp.py.
    """

    def __init__(
        self,
        *,
        embed_text_fn: Callable[[str], np.ndarray],
        embed_image_fn: Callable[[Image.Image], np.ndarray],
        call_lm_fn: Callable[..., str],
        encode_jpeg_fn: Callable[..., str],
        probes_store: Any,
        detections_store: Any,
        luxriot_manager: Any,
        search_indexed_folder_fn: Callable[..., List[Dict[str, Any]]],
        search_detections_fn: Callable[..., List[Dict[str, Any]]],
        lm_base_url: str,
        lm_model: str,
        lm_api_key: str,
        lm_timeout: int,
        store: Any | None = None,
        tool_audit_callback: Optional[
            Callable[[ToolAuditEvent], None]
        ] = None,
        tool_plan_store: Any | None = None,
        tool_approval_store: Any | None = None,
    ) -> None:
        self._ps  = probes_store
        self._ds  = detections_store
        self._lxm = luxriot_manager
        if store is None:
            raise RuntimeError("PostgreSQL-backed agent store is required")
        self.store = store
        self._lm_client = _AgentLMClient(
            base_url=lm_base_url,
            model=lm_model,
            api_key=lm_api_key,
            timeout=lm_timeout,
        )
        self._tools = AgentTools(
            detections_store=detections_store,
            probes_store=probes_store,
            luxriot_manager=luxriot_manager,
            embed_text_fn=embed_text_fn,
            embed_image_fn=embed_image_fn,
            call_lm_fn=call_lm_fn,
            encode_jpeg_fn=encode_jpeg_fn,
            search_indexed_folder_fn=search_indexed_folder_fn,
            search_detections_fn=search_detections_fn,
        )
        self._secure_tools = (
            EvaAgentToolAdapter(
                self._tools,
                _TOOL_SCHEMAS,
                audit_callback=tool_audit_callback,
                plan_store=tool_plan_store,
                approval_store=tool_approval_store,
            )
            if tool_audit_callback is not None
            else None
        )

    def approve_action_plan(
        self,
        plan_id: str,
        tool_context: ToolExecutionContext,
    ) -> Any:
        if self._secure_tools is None:
            raise ToolGatewayError("Authorized agent tools are unavailable.")
        return self._secure_tools.approve_and_execute(plan_id, tool_context)

    def stream_chat(
        self,
        session_id: Optional[str],
        message: str,
        image_b64: Optional[str] = None,
        tool_context: Optional[ToolExecutionContext] = None,
    ) -> Generator[str, None, None]:
        """
        Main entry point. Yields SSE-formatted strings.
        Each event is:  'data: <json>\\n\\n'
        """
        # ── session setup ──────────────────────────────────────────────────
        if tool_context is None:
            store_owner: Dict[str, str] = {}
            session_exists = bool(
                session_id and self.store.session_exists(session_id)
            )
        else:
            store_owner = {
                "tenant_id": tool_context.tenant_id,
                "actor_id": tool_context.actor_id,
            }
            session_exists = bool(
                session_id
                and self.store.session_exists(
                    session_id,
                    **store_owner,
                )
            )
        if not session_exists:
            if tool_context is None:
                session_id = self.store.create_session()
            else:
                session_id = self.store.create_session(**store_owner)
        if tool_context is not None:
            tool_context = ToolExecutionContext(
                actor_id=tool_context.actor_id,
                tenant_id=tool_context.tenant_id,
                roles=tool_context.roles,
                permissions=tool_context.permissions,
                allowed_channel_ids=tool_context.allowed_channel_ids,
                agent_session_id=session_id,
                request_id=tool_context.request_id,
                client_metadata=tool_context.client_metadata,
            )
            if self._secure_tools is None:
                yield _sse(
                    {
                        "type": "error",
                        "message": "Authorized agent tools are unavailable.",
                    }
                )
                yield _sse({"type": "done", "session_id": session_id})
                return

        title = message[:60].strip() or "Chat"
        self.store.touch_session(session_id, title=title, **store_owner)

        yield _sse({"type": "session", "session_id": session_id})

        # ── persist user message ───────────────────────────────────────────
        user_content: Any
        if image_b64:
            user_content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                }},
            ]
        else:
            user_content = message

        self.store.add_message(session_id, role="user", content=message, **store_owner)

        # ── build messages for LM ──────────────────────────────────────────
        requested_skill_slugs = _extract_requested_skill_slugs(user_content)
        user_text = _extract_text_from_message_content(user_content)
        if "probe_tuning" in requested_skill_slugs:
            normalized_user_text = _normalize_probe_match_text(user_text)
            wants_all_probes = any(
                token in normalized_user_text
                for token in ("all probes", "all probe", "every probe", "probe audit", "all probes audit")
            )
            mentioned_probes = self._tools._find_probe_mentions_in_text(user_text)
            visible_probes = (
                self._secure_tools.visible_probes(tool_context)
                if tool_context is not None and self._secure_tools is not None
                else self._ps.list_probes()
            )
            visible_probe_ids = {
                str(probe.get("id") or "") for probe in visible_probes
            }
            mentioned_probes = [
                probe
                for probe in mentioned_probes
                if str(probe.get("id") or "") in visible_probe_ids
            ]
            if len(mentioned_probes) != 1 and not wants_all_probes:
                probe_names = [
                    str(probe.get("name") or "").strip()
                    for probe in visible_probes
                    if str(probe.get("name") or "").strip()
                ]
                prompt_suffix = ""
                if probe_names:
                    prompt_suffix = " Available probes: " + ", ".join(probe_names[:8])
                    if len(probe_names) > 8:
                        prompt_suffix += f", and {len(probe_names) - 8} more."
                    else:
                        prompt_suffix += "."
                clarification = (
                    "Which probe should I tune?"
                    " Name the probe explicitly, or say that you want an all-probes audit."
                    + prompt_suffix
                )
                self.store.add_message(
                    session_id,
                    role="assistant",
                    content=clarification,
                    **store_owner,
                )
                yield _sse({"type": "text", "content": clarification})
                yield _sse({"type": "done", "session_id": session_id})
                return
        system_prompt = build_system_prompt(
            self._ps,
            self._ds,
            self._lxm,
            active_skill_slugs=requested_skill_slugs,
            allowed_channel_ids=(
                sorted(tool_context.allowed_channel_ids)
                if tool_context is not None
                else None
            ),
            secure_tool_mode=tool_context is not None,
        )
        history = self.store.load_history(session_id, **store_owner)

        # Replace the stored user content with the full (possibly image-bearing) one
        in_flight: List[Dict[str, Any]] = (
            [{"role": "system", "content": system_prompt}]
            + history[:-1]                          # all but the just-added user msg
            + [{"role": "user", "content": user_content}]
        )

        # Accumulated messages from this turn (to persist after streaming)
        new_assistant_messages: List[Dict[str, Any]] = []
        available_tool_schemas = (
            self._secure_tools.available_tool_schemas(tool_context)
            if tool_context is not None and self._secure_tools is not None
            else _TOOL_SCHEMAS
        )
        turn_tool_context = _seed_turn_tool_context(user_text)
        try:
            mentioned_channel_id = self._tools._resolve_channel_id(
                {"channel_ref": user_text},
                required=False,
            )
        except Exception:
            mentioned_channel_id = None
        if mentioned_channel_id is not None:
            turn_tool_context["channel_id"] = mentioned_channel_id

        # ── tool loop ──────────────────────────────────────────────────────
        tool_calls_used = 0
        while True:
            if tool_calls_used >= AGENT_MAX_TOOL_CALLS_PER_TURN:
                in_flight.append(
                    {
                        "role": "system",
                        "content": (
                            f"Tool budget exhausted after {tool_calls_used} tool call(s). "
                            "Give the operator a concise partial answer, list what remains unchecked, "
                            "and ask for confirmation before continuing in another turn."
                        ),
                    }
                )
                yield _sse(
                    {
                        "type": "tool_budget",
                        "message": (
                            f"Stopped tool use after {tool_calls_used} call(s); "
                            "preparing a partial answer."
                        ),
                        "max_tool_calls": AGENT_MAX_TOOL_CALLS_PER_TURN,
                    }
                )
                break
            # Run the blocking LM call in a thread so we can emit heartbeats
            lm_response: _LMResponse
            try:
                lm_response = yield from _run_with_heartbeats(
                    fn=lambda: self._lm_client.call_with_tools(
                        in_flight,
                        tools=available_tool_schemas,
                    ),
                    heartbeat_interval=AGENT_HEARTBEAT_INTERVAL,
                )
            except Exception as exc:
                yield _sse({"type": "error", "message": f"LM error: {exc}"})
                yield _sse({"type": "done", "session_id": session_id})
                return

            if lm_response.finish_reason != "tool_calls" or not lm_response.tool_calls:
                # Model wants to respond with text — break out to streaming phase
                break

            for tool_call in lm_response.tool_calls:
                tool_call.args = _apply_turn_tool_context(
                    tool_call.name,
                    tool_call.args,
                    turn_tool_context,
                )

            # Append assistant turn with tool_calls to in-flight history
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": lm_response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in lm_response.tool_calls
                ],
            }
            in_flight.append(assistant_msg)
            new_assistant_messages.append(assistant_msg)

            # Execute each tool call
            for tc in lm_response.tool_calls:
                yield _sse({"type": "tool_call", "name": tc.name, "args": tc.args})
                progress_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
                if tool_calls_used >= AGENT_MAX_TOOL_CALLS_PER_TURN:
                    error_payload = {
                        "error": (
                            "Tool budget exhausted. Ask the operator to continue "
                            "or narrow the channel/time scope."
                        )
                    }
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(error_payload)}
                    yield _sse({"type": "tool_result", "name": tc.name,
                                "result": error_payload, "error": error_payload["error"]})
                    in_flight.append(result_msg)
                    new_assistant_messages.append(result_msg)
                    continue
                tool_calls_used += 1

                try:
                    result = yield from _run_with_heartbeats(
                        fn=lambda tc=tc, progress_queue=progress_queue: (
                            self._secure_tools.execute(
                                tc.name,
                                tc.args,
                                tool_context,
                                progress_cb=lambda event: progress_queue.put(event),
                            )
                            if tool_context is not None
                            and self._secure_tools is not None
                            else self._tools.execute(
                                tc.name,
                                tc.args,
                                progress_cb=lambda event: progress_queue.put(event),
                            )
                        ),
                        heartbeat_interval=AGENT_HEARTBEAT_INTERVAL,
                        progress_queue=progress_queue,
                    )
                    result_for_model = _compact_tool_result_for_model(tc.name, result)
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(result_for_model, default=str)}
                    _remember_turn_tool_result(tc.name, result, turn_tool_context)
                    yield _sse({
                        "type": "tool_result",
                        "name": tc.name,
                        "result": _tool_result_for_ui(tc.name, result),
                    })
                except (ToolError, ToolGatewayError) as exc:
                    error_payload = {"error": str(exc)}
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(error_payload)}
                    yield _sse({"type": "tool_result", "name": tc.name,
                                "result": error_payload, "error": str(exc)})
                except Exception as exc:
                    error_payload = {"error": f"Internal tool error: {exc}"}
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(error_payload)}
                    yield _sse({"type": "tool_result", "name": tc.name,
                                "result": error_payload, "error": str(exc)})

                in_flight.append(result_msg)
                new_assistant_messages.append(result_msg)

        # ── final streaming text response ──────────────────────────────────
        full_text_parts: List[str] = []
        try:
            for chunk in self._lm_client.stream_text(in_flight):
                full_text_parts.append(chunk)
                yield _sse({"type": "text", "content": chunk})
        except Exception as exc:
            yield _sse({"type": "error", "message": f"Streaming error: {exc}"})
            yield _sse({"type": "done", "session_id": session_id})
            return

        final_text = "".join(full_text_parts)

        # ── persist assistant turn ─────────────────────────────────────────
        # Persist intermediate tool-call/result pairs
        for msg in new_assistant_messages:
            role = msg.get("role", "assistant")
            if role == "assistant":
                tcs = msg.get("tool_calls")
                self.store.add_message(
                    session_id,
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=tcs,
                    **store_owner,
                )
            elif role == "tool":
                # Strip large binary content before persisting
                raw_content = msg.get("content", "{}")
                try:
                    parsed = json.loads(raw_content)
                    parsed = _strip_thumbnails_deep(parsed)
                    content_to_store = json.dumps(parsed, default=str)
                except Exception:
                    content_to_store = raw_content
                self.store.add_message(
                    session_id,
                    role="tool",
                    tool_call_id=msg.get("tool_call_id"),
                    tool_name=msg.get("name"),
                    tool_result=content_to_store,
                    **store_owner,
                )

        # Persist final assistant text
        if final_text:
            self.store.add_message(
                session_id,
                role="assistant",
                content=final_text,
                **store_owner,
            )

        yield _sse({"type": "done", "session_id": session_id})


# ---------------------------------------------------------------------------
# Heartbeat helper
# ---------------------------------------------------------------------------

def _run_with_heartbeats(
    fn: Callable[[], Any],
    heartbeat_interval: float = AGENT_HEARTBEAT_INTERVAL,
    progress_queue: Optional["queue.Queue[Dict[str, Any]]"] = None,
) -> Generator[str, None, Any]:
    """
    Run fn() in a thread. Yield SSE heartbeat events every heartbeat_interval
    seconds while waiting. Return the result when done.
    Usage:  result = yield from _run_with_heartbeats(fn)
    """
    result_holder: Dict[str, Any] = {}
    exc_holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            result_holder["v"] = fn()
        except Exception as exc:
            exc_holder["v"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    last_heartbeat = time.time()
    while t.is_alive():
        t.join(timeout=0.25)
        if progress_queue is not None:
            while True:
                try:
                    event = progress_queue.get_nowait()
                except queue.Empty:
                    break
                yield _sse({"type": "tool_progress", **event})
        now = time.time()
        if t.is_alive() and now - last_heartbeat >= heartbeat_interval:
            yield _sse({"type": "heartbeat"})
            last_heartbeat = now

    if progress_queue is not None:
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break
            yield _sse({"type": "tool_progress", **event})

    if "v" in exc_holder:
        raise exc_holder["v"]
    return result_holder.get("v")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _epoch_input_unit(v: Any) -> Optional[str]:
    value = _opt_float(v)
    if value is None:
        return None
    return "milliseconds" if abs(value) >= 10_000_000_000 else "seconds"


def _coerce_epoch_seconds(v: Any) -> Optional[float]:
    value = _opt_float(v)
    if value is None:
        return None
    if abs(value) >= 10_000_000_000:
        value = value / 1000.0
    # Guard against obviously invalid epoch values while still allowing old archives.
    if value < 0:
        return None
    return float(value)


def _format_epoch_minute(v: Any) -> Optional[str]:
    value = _coerce_epoch_seconds(v)
    if value is None:
        return None
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


def _parse_operator_clock(value: str) -> Optional[Any]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = text.replace(".", ":").replace(" ", "")
    ampm: Optional[str] = None
    if text.endswith("am") or text.endswith("pm"):
        ampm = text[-2:]
        text = text[:-2]
    if ":" in text:
        hour_text, minute_text = text.split(":", 1)
    else:
        hour_text, minute_text = text, "0"
    try:
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        return None
    if minute < 0 or minute > 59:
        return None
    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    elif hour < 0 or hour > 23:
        return None
    from datetime import time as _time
    return _time(hour=hour, minute=minute)


_RELATIVE_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "couple": 2,
    "few": 3,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "пара": 2,
}


def _parse_relative_window_seconds(value: Any) -> Optional[Tuple[int, str]]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    normalized = re.sub(r"[,\.;]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    number_pattern = (
        r"\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|couple|few|один|одна|одно|два|две|три|четыре|пять|"
        r"шесть|семь|восемь|девять|десять|пара"
    )
    unit_pattern = (
        r"hours?|hrs?|h|час(?:а|ов)?|час|"
        r"minutes?|mins?|min|m|минут(?:ы|у)?|м(?:ин)?|"
        r"days?|d|д(?:ень|ня|ней)?|сут(?:ки|ок|ка)?"
    )
    match = re.search(
        rf"(?:last|past|previous|prior|recent|за последние|за последн\w+|последн\w+)\s+"
        rf"(?P<num>{number_pattern})\s*(?P<unit>{unit_pattern})\b",
        normalized,
    )
    if not match:
        match = re.search(
            rf"\b(?P<num>{number_pattern})\s*(?P<unit>{unit_pattern})\s+"
            rf"(?:ago|back|назад)\b",
            normalized,
        )
    if not match:
        implicit = re.search(
            r"(?:last|past|previous|prior|recent|последн\w+)\s+"
            r"(?P<unit>hour|час|minute|минута|day|день|сутки)\b",
            normalized,
        )
        if implicit:
            raw_number = "1"
            unit = implicit.group("unit")
        else:
            return None
    else:
        raw_number = match.group("num")
        unit = match.group("unit")

    if raw_number in _RELATIVE_NUMBER_WORDS:
        amount = float(_RELATIVE_NUMBER_WORDS[raw_number])
    else:
        try:
            amount = float(raw_number)
        except ValueError:
            return None
    if not amount or amount <= 0:
        return None
    unit = str(unit or "").lower()
    if unit.startswith(("d", "day", "д", "сут")):
        seconds = int(amount * 86400)
    elif unit.startswith(("h", "hr", "hour", "час")):
        seconds = int(amount * 3600)
    elif unit.startswith(("m", "min", "minute", "мин", "м")):
        seconds = int(amount * 60)
    else:
        return None
    if seconds <= 0:
        return None
    return seconds, normalized


def _normalize_summary_depth(value: Any) -> str:
    depth = str(value or "L1").strip().upper()
    if depth == "LIVE":
        return "L0"
    if depth in {"L0", "L1", "L2", "L3"}:
        return depth
    return "L1"


def _summary_node_bounds(node: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    start = _coerce_epoch_seconds(node.get("window_start"))
    if start is None:
        start = _coerce_epoch_seconds(node.get("created_at"))
    end = _coerce_epoch_seconds(node.get("window_end"))
    if end is None:
        end = start
    return start, end


def _summary_node_overlaps(node: Mapping[str, Any], from_ts: float, to_ts: float) -> bool:
    start, end = _summary_node_bounds(node)
    if start is None and end is None:
        return False
    if start is None:
        start = end
    if end is None:
        end = start
    if end is None or start is None:
        return False
    start_f = float(start)
    end_f = float(end)
    from_f = float(from_ts)
    to_f = float(to_ts)
    if start_f == end_f:
        return from_f <= start_f <= to_f
    return end_f > from_f and start_f < to_f


def _summary_coverage_from_nodes(
    nodes: Sequence[Mapping[str, Any]],
    from_ts: float,
    to_ts: float,
    *,
    label: str,
) -> Dict[str, Any]:
    requested_start = float(from_ts)
    requested_end = float(to_ts)
    if requested_start > requested_end:
        requested_start, requested_end = requested_end, requested_start
    requested_span = max(0.0, requested_end - requested_start)

    intervals: List[Tuple[float, float]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        start, end = _summary_node_bounds(node)
        if start is None and end is None:
            continue
        if start is None:
            start = end
        if end is None:
            end = start
        if start is None or end is None:
            continue
        start_f = max(requested_start, float(start))
        end_f = min(requested_end, float(end))
        if end_f < start_f:
            start_f, end_f = end_f, start_f
        intervals.append((start_f, end_f))

    base: Dict[str, Any] = {
        "label": label,
        "entry_count": len(intervals),
        "requested_from_ts": requested_start,
        "requested_to_ts": requested_end,
        "requested_from_time": _format_epoch_minute(requested_start),
        "requested_to_time": _format_epoch_minute(requested_end),
        "requested_span_sec": requested_span,
    }
    if not intervals:
        return {
            **base,
            "status": "no_data",
            "first_ts": None,
            "last_ts": None,
            "first_time": None,
            "last_time": None,
            "observed_span_sec": 0.0,
            "coverage_ratio": 0.0,
            "leading_gap_sec": requested_span,
            "trailing_gap_sec": requested_span,
            "internal_gap_count": 0,
            "large_internal_gaps": [],
            "note": "No video-summary entries were available inside the requested window.",
        }

    intervals.sort(key=lambda item: (item[0], item[1]))
    first_ts = intervals[0][0]
    last_ts = max(end for _start, end in intervals)
    observed_span = max(0.0, last_ts - first_ts)
    leading_gap = max(0.0, first_ts - requested_start)
    trailing_gap = max(0.0, requested_end - last_ts)

    merged: List[List[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    large_gaps: List[Dict[str, Any]] = []
    for prev, nxt in zip(merged, merged[1:]):
        gap = max(0.0, nxt[0] - prev[1])
        if gap < 120.0:
            continue
        large_gaps.append(
            {
                "from_ts": prev[1],
                "to_ts": nxt[0],
                "from_time": _format_epoch_minute(prev[1]),
                "to_time": _format_epoch_minute(nxt[0]),
                "gap_sec": gap,
            }
        )

    tolerance = 120.0
    if leading_gap > tolerance or trailing_gap > tolerance or large_gaps:
        status = "partial"
    else:
        status = "covered"
    coverage_ratio = 1.0 if requested_span <= 0 else max(0.0, min(1.0, observed_span / requested_span))
    note = (
        f"{label} covers {_format_epoch_minute(first_ts)} to {_format_epoch_minute(last_ts)} "
        f"inside requested {_format_epoch_minute(requested_start)} to {_format_epoch_minute(requested_end)}."
    )
    if status != "covered":
        note += " Treat this as partial coverage; do not imply the whole requested period was reviewed."

    return {
        **base,
        "status": status,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "first_time": _format_epoch_minute(first_ts),
        "last_time": _format_epoch_minute(last_ts),
        "observed_span_sec": observed_span,
        "coverage_ratio": coverage_ratio,
        "leading_gap_sec": leading_gap,
        "trailing_gap_sec": trailing_gap,
        "internal_gap_count": len(large_gaps),
        "large_internal_gaps": large_gaps[:6],
        "note": note,
    }


def _video_summary_coverage_contract(
    *,
    available_nodes: Sequence[Mapping[str, Any]],
    returned_nodes: Sequence[Mapping[str, Any]],
    from_ts: float,
    to_ts: float,
    truncated: bool,
    selection_strategy: str = "all",
) -> Dict[str, Any]:
    available = _summary_coverage_from_nodes(available_nodes, from_ts, to_ts, label="available_entries")
    returned = _summary_coverage_from_nodes(returned_nodes, from_ts, to_ts, label="returned_entries")
    status = "covered"
    if available.get("status") == "no_data":
        status = "no_data"
    elif truncated:
        status = "truncated"
    elif available.get("status") != "covered" or returned.get("status") != "covered":
        status = "partial"
    note = str(returned.get("note") or available.get("note") or "").strip()
    if truncated:
        sampled_note = (
            " Returned entries were selected across the requested period with alert/deviation priority."
            if selection_strategy == "period_sample_alert_priority"
            else ""
        )
        note = (
            "The summary tool found more entries than it returned. The assistant only reviewed "
            "the returned subset unless it requests another page/window."
            + sampled_note
            + " "
            + note
        ).strip()
    return {
        "status": status,
        "must_state_coverage": True,
        "truncated": bool(truncated),
        "selection_strategy": selection_strategy,
        "available": available,
        "returned": returned,
        "note": note,
    }


def _summary_node_alert_score(node: Mapping[str, Any]) -> int:
    score = 0
    raw_total = _opt_int(node.get("alert_total"))
    if raw_total is not None and raw_total > 0:
        score += int(raw_total)
    raw_counts = node.get("alert_counts")
    if isinstance(raw_counts, Mapping):
        for value in raw_counts.values():
            parsed = _opt_int(value)
            if parsed is not None and parsed > 0:
                score += int(parsed)
    text = str(node.get("summary") or "").lower()
    for marker in (
        "alert",
        "deviation",
        "incident",
        "violence",
        "fire",
        "smoke",
        "drift",
        "crash",
        "fall",
        "collapsed",
        "weapon",
        "forced entry",
        "дра",
        "пожар",
        "дым",
        "дрифт",
        "авари",
        "упал",
        "потерял созн",
    ):
        if marker in text:
            score += 1
    return score


def _evenly_spaced_indices(total: int, limit: int) -> List[int]:
    if total <= 0 or limit <= 0:
        return []
    if total <= limit:
        return list(range(total))
    if limit == 1:
        return [0]
    return sorted({
        int(round((total - 1) * (index / float(limit - 1))))
        for index in range(limit)
    })


def _select_summary_nodes_for_period(
    nodes: Sequence[Mapping[str, Any]],
    limit: int,
) -> Tuple[List[Mapping[str, Any]], str]:
    if len(nodes) <= limit:
        return list(nodes), "all"
    if limit <= 0:
        return [], "none"

    total = len(nodes)
    selected = set(_evenly_spaced_indices(total, limit))
    alert_indices = [
        index
        for index, node in enumerate(nodes)
        if isinstance(node, Mapping) and _summary_node_alert_score(node) > 0
    ]
    alert_indices.sort(
        key=lambda index: (
            -_summary_node_alert_score(cast(Mapping[str, Any], nodes[index])),
            index,
        )
    )
    protected = {0, total - 1}
    for alert_index in alert_indices:
        if alert_index in selected:
            continue
        if len(selected) < limit:
            selected.add(alert_index)
            continue
        replaceable = [
            index for index in selected
            if index not in protected and index not in alert_indices
        ]
        if not replaceable:
            continue
        victim = min(replaceable, key=lambda index: abs(index - alert_index))
        selected.remove(victim)
        selected.add(alert_index)

    ordered = sorted(selected)
    return [nodes[index] for index in ordered], "period_sample_alert_priority"


_COUNT_EVENT_APPEARANCE_PATTERNS = (
    r"\b(?:reappears?|reappeared|returns?|returned|arrives?|arrived|enters?|entered)\b",
    r"\b(?:jumps?|jumped|climbs?|climbed|gets?)\s+(?:back\s+)?(?:on|onto|atop)\b",
)
_COUNT_EVENT_DISAPPEARANCE_PATTERNS = (
    r"\b(?:exits?|exited|leaves?|left|disappears?|disappeared)\b",
    r"\b(?:no\s+longer\s+(?:visible|present|on|atop))\b",
    r"\b(?:jumps?|jumped|climbs?|climbed|gets?)\s+(?:down\s+)?(?:from|off)\b",
    r"\b(?:moves?|moved)\s+(?:off|out\s+of|away\s+from)\b",
)
_COUNT_EVENT_PRESENT_PATTERNS = (
    r"\b(?:visible|present|stationary|perched|resting|sitting|seated|atop|on\s+top\s+of)\b",
    r"\b(?:remains?|remained|is|was)\s+(?:visible|present|stationary|perched|resting|on|atop)\b",
)
_COUNT_EVENT_ABSENT_PATTERNS = (
    r"\b(?:absent|not\s+visible|not\s+present|out\s+of\s+view|off\s+screen)\b",
    r"\bno\s+(?:cats?|animals?|pets?|persons?|people|vehicles?|objects?)\s+(?:are\s+|were\s+|is\s+|was\s+)?(?:visible|present|in\s+view|in\s+frame)\b",
    r"\b(?:empty\s+of\s+(?:cats?|animals?|pets?|people|persons?))\b",
)


def _summary_count_terms(value: Any) -> List[str]:
    text = str(value or "").strip().lower()
    terms = re.findall(r"[a-zа-яё0-9]+", text, flags=re.IGNORECASE)
    prepared: List[str] = []
    for term in terms:
        cleaned = term.strip().lower()
        if len(cleaned) < 3:
            continue
        if cleaned not in prepared:
            prepared.append(cleaned)
    if "computer" in prepared and "pc" not in prepared:
        prepared.append("pc")
    return prepared[:12]


def _summary_count_has_term(text: str, terms: Sequence[str]) -> bool:
    if not terms:
        return True
    for term in terms:
        escaped = re.escape(str(term).lower())
        if re.search(rf"\b{escaped}\w*\b", text, flags=re.IGNORECASE):
            return True
    return False


def _summary_count_has_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _summary_count_excerpt(text: str, max_len: int = 360) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max(0, max_len - 3)].rstrip() + "..."


def _classify_summary_presence_event(
    node: Mapping[str, Any],
    *,
    entity_terms: Sequence[str],
    anchor_terms: Sequence[str],
) -> Dict[str, Any]:
    text_raw = str(node.get("summary") or "")
    text = text_raw.lower()
    entity_seen = _summary_count_has_term(text, entity_terms)
    anchor_seen = _summary_count_has_term(text, anchor_terms)
    relevant = bool(entity_seen and anchor_seen)
    explicit_in = bool(relevant and _summary_count_has_pattern(text, _COUNT_EVENT_APPEARANCE_PATTERNS))
    explicit_out = bool(relevant and _summary_count_has_pattern(text, _COUNT_EVENT_DISAPPEARANCE_PATTERNS))
    present_signal = bool(relevant and _summary_count_has_pattern(text, _COUNT_EVENT_PRESENT_PATTERNS))
    absent_signal = bool(entity_seen and _summary_count_has_pattern(text, _COUNT_EVENT_ABSENT_PATTERNS))
    if anchor_terms and absent_signal:
        absent_signal = absent_signal or anchor_seen
    if absent_signal and present_signal and not explicit_in:
        strong_present_signal = bool(
            relevant
            and re.search(
                r"\b(?:stationary|perched|resting|sitting|seated|atop|on\s+top\s+of|on\s+(?:the\s+)?(?:pc|computer|tower))\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        if not strong_present_signal:
            present_signal = False

    if explicit_in and explicit_out:
        state = "mixed"
    elif explicit_out:
        state = "absent"
    elif explicit_in:
        state = "present"
    elif present_signal and absent_signal:
        state = "mixed"
    elif present_signal:
        state = "present"
    elif absent_signal:
        state = "absent"
    else:
        state = "unknown"

    start, end = _summary_node_bounds(node)
    return {
        "state": state,
        "entity_seen": entity_seen,
        "anchor_seen": anchor_seen,
        "explicit_appearance": explicit_in,
        "explicit_disappearance": explicit_out,
        "present_signal": present_signal,
        "absent_signal": absent_signal,
        "window_start": start,
        "window_end": end,
        "time": _format_epoch_minute(start) if start is not None else None,
        "window_end_time": _format_epoch_minute(end) if end is not None else None,
        "summary": _summary_count_excerpt(text_raw),
    }


def _count_summary_presence_transitions(
    nodes: Sequence[Mapping[str, Any]],
    *,
    entity_query: str,
    anchor_query: str,
    timeline_limit: int,
    event_limit: int,
) -> Dict[str, Any]:
    entity_terms = _summary_count_terms(entity_query)
    anchor_terms = _summary_count_terms(anchor_query)
    timeline: List[Dict[str, Any]] = []
    transition_events: List[Dict[str, Any]] = []
    previous_state: Optional[str] = None
    previous_row: Optional[Dict[str, Any]] = None
    counts = {
        "appearance_count": 0,
        "disappearance_count": 0,
        "explicit_appearance_count": 0,
        "explicit_disappearance_count": 0,
        "inferred_appearance_count": 0,
        "inferred_disappearance_count": 0,
        "present_windows": 0,
        "absent_windows": 0,
        "mixed_windows": 0,
        "unknown_windows": 0,
    }

    def add_event(kind: str, row: Dict[str, Any], basis: str, previous: Optional[Dict[str, Any]] = None) -> None:
        if kind == "appearance":
            counts["appearance_count"] += 1
            if basis == "explicit_summary_mention":
                counts["explicit_appearance_count"] += 1
            else:
                counts["inferred_appearance_count"] += 1
        elif kind == "disappearance":
            counts["disappearance_count"] += 1
            if basis == "explicit_summary_mention":
                counts["explicit_disappearance_count"] += 1
            else:
                counts["inferred_disappearance_count"] += 1
        if len(transition_events) >= event_limit:
            return
        event = {
            "type": kind,
            "basis": basis,
            "time": row.get("time"),
            "window_start": row.get("window_start"),
            "window_end": row.get("window_end"),
            "window_end_time": row.get("window_end_time"),
            "summary": row.get("summary"),
        }
        if previous:
            event["previous_time"] = previous.get("time")
            event["previous_state"] = previous.get("state")
        transition_events.append(event)

    for node in nodes:
        row = _classify_summary_presence_event(
            node,
            entity_terms=entity_terms,
            anchor_terms=anchor_terms,
        )
        state = str(row.get("state") or "unknown")
        if state == "present":
            counts["present_windows"] += 1
        elif state == "absent":
            counts["absent_windows"] += 1
        elif state == "mixed":
            counts["mixed_windows"] += 1
        else:
            counts["unknown_windows"] += 1

        if row.get("explicit_appearance"):
            add_event("appearance", row, "explicit_summary_mention")
        if row.get("explicit_disappearance"):
            add_event("disappearance", row, "explicit_summary_mention")

        if state in {"present", "absent"}:
            if previous_state == "absent" and state == "present" and not row.get("explicit_appearance"):
                add_event("appearance", row, "inferred_adjacent_summary_state_change", previous_row)
            elif previous_state == "present" and state == "absent" and not row.get("explicit_disappearance"):
                add_event("disappearance", row, "inferred_adjacent_summary_state_change", previous_row)
            previous_state = state
            previous_row = row

        if len(timeline) < timeline_limit:
            timeline.append(row)

    counts["transition_count"] = counts["appearance_count"] + counts["disappearance_count"]
    return {
        "entity_terms": entity_terms,
        "anchor_terms": anchor_terms,
        "counts": counts,
        "transition_events": transition_events,
        "timeline": timeline,
        "timeline_returned": len(timeline),
        "timeline_total": len(nodes),
        "event_returned": len(transition_events),
        "event_total": counts["transition_count"],
    }


def _detection_timestamp_ms(row: Mapping[str, Any]) -> int:
    for key in ("timestamp_ms", "event_timestamp_ms", "recorded_at_ms"):
        parsed = _opt_int(row.get(key))
        if parsed is not None:
            return int(parsed)
    return 0


def _archive_score_semantics(source: Any) -> str:
    source_text = str(source or "").strip().lower()
    logical_source = ARCHIVE_SOURCE_ALIASES.get(source_text, source_text)
    if logical_source in {"vlm_summary", "vlm_alert"}:
        return "not_applicable"
    return "probe_threshold_scores"


def _search_result_score(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("score", "similarity", "clip_similarity"):
        parsed = _opt_float(row.get(key))
        if parsed is not None:
            return float(parsed)
    fusion = row.get("fusion")
    if isinstance(fusion, Mapping):
        parsed = _opt_float(fusion.get("clip_similarity"))
        if parsed is not None:
            return float(parsed)
    return None


def _best_search_score(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    scores = [
        float(score)
        for row in rows
        for score in [_search_result_score(row)]
        if score is not None
    ]
    return max(scores) if scores else None


def _visual_signal_state(
    positive_score: Optional[Any],
    negative_score: Optional[Any],
) -> str:
    pos = _opt_float(positive_score)
    neg = _opt_float(negative_score)
    if pos is None:
        return "no_positive_candidates"
    if neg is None:
        return "positive_only"
    margin = float(pos) - float(neg)
    if margin >= 0.06:
        return "positive_separation"
    if margin >= 0.02:
        return "weak_positive_separation"
    if margin > -0.02:
        return "ambiguous"
    return "negative_dominant"


def _visual_signal_pnm(
    positive_score: Optional[Any],
    negative_score: Optional[Any],
) -> Dict[str, Any]:
    pos = _opt_float(positive_score)
    neg = _opt_float(negative_score)
    margin = float(pos) - float(neg) if pos is not None and neg is not None else None
    state = _visual_signal_state(pos, neg)
    notes = {
        "no_positive_candidates": "No positive CLIP candidates were found in the selected window.",
        "positive_only": "Only positive retrieval was scored; margin is unavailable without a negative phrase.",
        "positive_separation": "Positive phrase separates from the negative phrase enough to prioritize review.",
        "weak_positive_separation": "Positive phrase is only weakly separated; treat as a tentative cue.",
        "ambiguous": "Positive and negative phrases are close; review frames before concluding.",
        "negative_dominant": "Negative phrase scores higher than the positive phrase; event support is weak.",
    }
    return {
        "p": pos,
        "n": neg,
        "m": margin,
        "state": state,
        "score_semantics": "clip_retrieval_signal_not_proof",
        "note": notes.get(state, "Use this as attention signal only."),
    }


def _agent_normalized_vec(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float32, copy=False)


def _agent_dot_score(query_vec: Optional[np.ndarray], frame_vec: Any) -> Optional[float]:
    query = _agent_normalized_vec(query_vec)
    frame = _agent_normalized_vec(frame_vec)
    if query is None or frame is None or query.shape != frame.shape:
        return None
    return float(np.dot(query, frame))


_CLIP_NEGATION_TERMS = {
    "no",
    "not",
    "without",
    "absent",
    "missing",
    "visible",
    "present",
    "with",
}


def _clip_query_terms(value: Any) -> List[str]:
    terms: List[str] = []
    for raw in re.findall(r"[a-zа-яё0-9]+", str(value or "").lower(), flags=re.IGNORECASE):
        cleaned = raw.strip().lower()
        if len(cleaned) < 3:
            continue
        if cleaned not in terms:
            terms.append(cleaned)
    return terms


def _clip_effective_negative_state_query(
    negative_query: str,
    *,
    subject_query: Optional[str] = None,
) -> Tuple[str, List[str]]:
    original = str(negative_query or "").strip()
    if not original:
        return "", []
    lowered = original.lower()
    if not re.search(r"\b(?:no|not|without|absent|missing)\b", lowered, flags=re.IGNORECASE):
        return original, []

    remove_terms: set[str] = set(_clip_query_terms(subject_query))
    for match in re.finditer(
        r"\b(?:no|without|not(?:\s+(?:visible|present))?|absent|missing)\b\s+([^,.;]+)",
        lowered,
        flags=re.IGNORECASE,
    ):
        target = match.group(1)
        target = re.split(
            r"\b(?:near|at|on|onto|in|inside|outside|by|beside|around|from|to|of|under|over|behind|ahead)\b",
            target,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        remove_terms.update(_clip_query_terms(target))

    tokens = _clip_query_terms(original)
    kept = [
        token for token in tokens
        if token not in _CLIP_NEGATION_TERMS and token not in remove_terms
    ]
    if not kept:
        return original, [
            "negative_state_query contains negation; CLIP may not understand it reliably."
        ]

    effective = " ".join(kept)
    if remove_terms and not re.search(r"\b(?:empty|clear|unoccupied|background|vacant)\b", effective, flags=re.IGNORECASE):
        effective = f"empty {effective}".strip()
    if effective == original:
        return original, []
    return effective, [
        (
            "negative_state_query contained negated target terms; CLIP contrast was scored against "
            f"{effective!r} instead of the literal negative phrase."
        )
    ]


def _transition_label(previous_state: str, current_state: str, positive_label: str, negative_label: str) -> str:
    if previous_state == negative_label and current_state == positive_label:
        return "appearance"
    if previous_state == positive_label and current_state == negative_label:
        return "disappearance"
    return f"{previous_state}_to_{current_state}"


def _state_sample_from_scores(
    row: Dict[str, Any],
    *,
    positive_score: Optional[float],
    negative_score: Optional[float],
    alternate_score: Optional[float],
    positive_label: str,
    negative_label: str,
    alternate_label: str,
    positive_floor: float,
    negative_floor: float,
    margin_threshold: float,
) -> Dict[str, Any]:
    candidates: List[Tuple[str, Optional[float], float]] = [
        (positive_label, positive_score, positive_floor),
    ]
    if negative_score is not None:
        candidates.append((negative_label, negative_score, negative_floor))
    if alternate_score is not None:
        candidates.append((alternate_label, alternate_score, positive_floor))
    valid_scores = [(label, float(score), floor) for label, score, floor in candidates if score is not None]
    valid_scores.sort(key=lambda item: item[1], reverse=True)
    state = "unknown"
    confidence = "low"
    margin = None
    winning_score = None
    runner_score = None
    if valid_scores:
        winner_label, winner_score, winner_floor = valid_scores[0]
        winning_score = winner_score
        runner_score = valid_scores[1][1] if len(valid_scores) > 1 else None
        margin = winner_score - runner_score if runner_score is not None else None
        margin_ok = margin is None or margin >= margin_threshold
        if winner_score >= winner_floor and margin_ok:
            state = winner_label
            confidence = "high" if margin is None or margin >= margin_threshold * 2 else "medium"
        elif winner_score >= winner_floor * 0.85:
            confidence = "low"
    ts_ms = _detection_timestamp_ms(row)
    return {
        "state": state,
        "confidence": confidence,
        "timestamp_ms": ts_ms,
        "time": _format_epoch_minute(float(ts_ms) / 1000.0) if ts_ms else None,
        "detection_id": row.get("detection_id") or row.get("id"),
        "source": row.get("source"),
        "source_label": _archive_source_label(row.get("source")),
        "positive_score": positive_score,
        "negative_score": negative_score,
        "alternate_score": alternate_score,
        "winning_score": winning_score,
        "runner_score": runner_score,
        "margin": margin,
        "image_url": _detection_image_url(row),
        "raw": row,
    }


def _frame_sample_public(sample: Mapping[str, Any]) -> Dict[str, Any]:
    row = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    public = _compact_detection_for_model(cast(Dict[str, Any], row)) if row else {}
    for key in (
        "state",
        "confidence",
        "positive_score",
        "negative_score",
        "alternate_score",
        "winning_score",
        "runner_score",
        "margin",
    ):
        if key in sample:
            public[key] = sample.get(key)
    public["timestamp_ms"] = sample.get("timestamp_ms")
    public["time"] = sample.get("time")
    public["image_url"] = sample.get("image_url") or public.get("image_url")
    public["detection_id"] = sample.get("detection_id") or public.get("detection_id")
    public["source"] = sample.get("source") or public.get("source")
    public["source_label"] = sample.get("source_label") or public.get("source_label")
    return public


def _build_state_segments_from_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    min_state_samples: int,
    min_state_duration_sec: float,
    merge_gap_sec: float,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for sample in samples:
        state = str(sample.get("state") or "unknown")
        ts_ms = int(sample.get("timestamp_ms") or 0)
        if ts_ms <= 0:
            continue
        if segments and segments[-1]["state"] == state:
            segment = segments[-1]
            segment["end_ms"] = ts_ms
            segment["sample_count"] += 1
            segment["last_sample"] = sample
            continue
        segments.append(
            {
                "state": state,
                "start_ms": ts_ms,
                "end_ms": ts_ms,
                "sample_count": 1,
                "first_sample": sample,
                "last_sample": sample,
            }
        )

    changed = True
    while changed:
        changed = False
        merged: List[Dict[str, Any]] = []
        index = 0
        while index < len(segments):
            if (
                index + 2 < len(segments)
                and segments[index]["state"] == segments[index + 2]["state"]
                and segments[index + 1]["state"] == "unknown"
                and (float(segments[index + 1]["end_ms"]) - float(segments[index + 1]["start_ms"])) / 1000.0 <= merge_gap_sec
            ):
                combined = dict(segments[index])
                combined["end_ms"] = segments[index + 2]["end_ms"]
                combined["sample_count"] = (
                    int(segments[index]["sample_count"])
                    + int(segments[index + 1]["sample_count"])
                    + int(segments[index + 2]["sample_count"])
                )
                combined["last_sample"] = segments[index + 2]["last_sample"]
                merged.append(combined)
                index += 3
                changed = True
            else:
                merged.append(segments[index])
                index += 1
        segments = merged

    for segment in segments:
        duration = max(0.0, (float(segment["end_ms"]) - float(segment["start_ms"])) / 1000.0)
        segment["duration_sec"] = duration
        segment["start_time"] = _format_epoch_minute(float(segment["start_ms"]) / 1000.0)
        segment["end_time"] = _format_epoch_minute(float(segment["end_ms"]) / 1000.0)
        if segment["state"] == "unknown":
            segment["stability"] = "unknown"
        elif int(segment["sample_count"]) >= min_state_samples or duration >= min_state_duration_sec:
            segment["stability"] = "confirmed"
        else:
            segment["stability"] = "possible"
    return segments


def _public_state_segment(segment: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "state": segment.get("state"),
        "stability": segment.get("stability"),
        "start_ms": segment.get("start_ms"),
        "end_ms": segment.get("end_ms"),
        "start_time": segment.get("start_time"),
        "end_time": segment.get("end_time"),
        "duration_sec": segment.get("duration_sec"),
        "sample_count": segment.get("sample_count"),
        "first_frame": _frame_sample_public(cast(Mapping[str, Any], segment.get("first_sample") or {})),
        "last_frame": _frame_sample_public(cast(Mapping[str, Any], segment.get("last_sample") or {})),
    }


def _build_state_transitions(
    segments: Sequence[Mapping[str, Any]],
    *,
    positive_label: str,
    negative_label: str,
    transition_limit: int,
    evidence_limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    stable = [
        segment for segment in segments
        if segment.get("state") != "unknown" and segment.get("stability") == "confirmed"
    ]
    transitions: List[Dict[str, Any]] = []
    boundary_frames: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "transition_count": 0,
        "appearance_count": 0,
        "disappearance_count": 0,
    }
    seen_frame_ids: set[Any] = set()

    def add_boundary(sample: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(sample, Mapping):
            return None
        public = _frame_sample_public(sample)
        key = public.get("detection_id") or (public.get("timestamp_ms"), public.get("source"))
        if len(boundary_frames) < evidence_limit and key not in seen_frame_ids:
            seen_frame_ids.add(key)
            boundary_frames.append(public)
        return public

    previous: Optional[Mapping[str, Any]] = None
    for current in stable:
        if previous is None:
            previous = current
            continue
        prev_state = str(previous.get("state") or "")
        current_state = str(current.get("state") or "")
        if not prev_state or not current_state or prev_state == current_state:
            previous = current
            continue
        kind = _transition_label(prev_state, current_state, positive_label, negative_label)
        counts["transition_count"] += 1
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "appearance":
            counts["appearance_count"] += 1
        elif kind == "disappearance":
            counts["disappearance_count"] += 1
        before_frame = add_boundary(previous.get("last_sample"))
        after_frame = add_boundary(current.get("first_sample"))
        if len(transitions) < transition_limit:
            transitions.append(
                {
                    "type": kind,
                    "from_state": prev_state,
                    "to_state": current_state,
                    "from_time": previous.get("end_time"),
                    "to_time": current.get("start_time"),
                    "from_ms": previous.get("end_ms"),
                    "to_ms": current.get("start_ms"),
                    "gap_sec": max(0.0, (float(current.get("start_ms") or 0) - float(previous.get("end_ms") or 0)) / 1000.0),
                    "before_frame": before_frame,
                    "after_frame": after_frame,
                }
            )
        previous = current
    return transitions, boundary_frames, counts


def _select_state_candidate_frames(
    samples: Sequence[Mapping[str, Any]],
    *,
    evidence_limit: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        samples,
        key=lambda sample: (
            float(sample.get("positive_score") or -1.0),
            float(sample.get("margin") or -999.0),
            int(sample.get("timestamp_ms") or 0),
        ),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    seen: set[Any] = set()
    for sample in ranked:
        public = _frame_sample_public(sample)
        key = public.get("detection_id") or (public.get("timestamp_ms"), public.get("source"))
        if key in seen:
            continue
        seen.add(key)
        public["role"] = "positive_candidate"
        out.append(public)
        if len(out) >= evidence_limit:
            break
    return out


def _visual_signal_row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    row_id = row.get("detection_id") or row.get("id")
    if row_id is not None:
        return ("id", row_id)
    return (
        "frame",
        row.get("source"),
        row.get("channel_id"),
        _detection_timestamp_ms(row),
        row.get("image_path") or row.get("image_url") or row.get("path"),
    )


def _compact_visual_signal_hit(row: Dict[str, Any], polarity: str) -> Dict[str, Any]:
    compact = _compact_detection_for_model(row)
    score = _search_result_score(row)
    compact["score"] = score
    compact["polarity"] = polarity
    compact["score_semantics"] = "clip_retrieval_signal_not_proof"
    return compact


def _evidence_row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    row_id = row.get("id") or row.get("detection_id")
    if row_id is not None:
        return ("id", row_id)
    return (
        "frame",
        row.get("source"),
        row.get("channel_id"),
        _detection_timestamp_ms(row),
        row.get("image_path") or row.get("image_url"),
    )


def _select_evidence_frame_rows(
    source_rows: Mapping[str, Sequence[Dict[str, Any]]],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    source_priority = {"vlm_alert": 0, "vlm_summary": 1}
    selected: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    for source in ("vlm_alert", "vlm_summary"):
        rows = source_rows.get(source) or []
        if not rows or len(selected) >= limit:
            continue
        first = dict(rows[0])
        key = _evidence_row_key(first)
        selected.append(first)
        seen.add(key)

    remaining: List[Dict[str, Any]] = []
    for rows in source_rows.values():
        for row in rows:
            key = _evidence_row_key(row)
            if key in seen:
                continue
            remaining.append(dict(row))

    remaining.sort(
        key=lambda row: (
            _detection_timestamp_ms(row),
            source_priority.get(str(row.get("source") or ""), 9),
            _opt_int(row.get("id") or row.get("detection_id")) or 0,
        )
    )
    for row in remaining:
        if len(selected) >= limit:
            break
        selected.append(row)

    selected.sort(
        key=lambda row: (
            _detection_timestamp_ms(row),
            source_priority.get(str(row.get("source") or ""), 9),
            _opt_int(row.get("id") or row.get("detection_id")) or 0,
        )
    )
    return selected[:limit]


def _sort_detection_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    sort_key: Callable[[Dict[str, Any]], Any]
    reverse = True
    if sort_by == "oldest":
        sort_key = lambda r: (_detection_timestamp_ms(r), _opt_int(r.get("id")) or 0)
        reverse = False
    elif sort_by == "highest_pos":
        sort_key = lambda r: (float(r.get("pos_score") or 0.0), _detection_timestamp_ms(r))
    elif sort_by == "lowest_pos":
        sort_key = lambda r: (float(r.get("pos_score") or 0.0), _detection_timestamp_ms(r))
        reverse = False
    elif sort_by == "highest_margin":
        sort_key = lambda r: (float(r.get("margin") or 0.0), _detection_timestamp_ms(r))
    elif sort_by == "lowest_margin":
        sort_key = lambda r: (float(r.get("margin") or 0.0), _detection_timestamp_ms(r))
        reverse = False
    else:
        sort_key = lambda r: (_detection_timestamp_ms(r), _opt_int(r.get("id")) or 0)
    return sorted(rows, key=sort_key, reverse=reverse)


def _detection_image_url(r: Dict[str, Any]) -> Optional[str]:
    """Return a URL the frontend can use to load the detection's image."""
    ip = str(r.get("image_path") or r.get("path") or "").strip()
    if ip:
        from urllib.parse import quote
        return f"/detections/image?image_path={quote(ip, safe='')}"
    detection_id = _opt_int(r.get("detection_id") or r.get("id"))
    if detection_id is not None:
        return f"/detections/thumbnail/{detection_id}"
    if r.get("thumbnail"):
        # Last-resort inline data URI for legacy rows without an ID.
        return _image_data_url(r.get("thumbnail"))
    return None


def _archive_result_image_url(r: Dict[str, Any], folder: Optional[str] = None) -> Optional[str]:
    """Return a URL the frontend can use to load an archive search result image."""
    fp = str(r.get("filepath") or r.get("path") or "").strip()
    if fp:
        from urllib.parse import quote
        if folder:
            return f"/image?folder={quote(folder, safe='')}&image_path={quote(fp, safe='')}"
        return f"/image/{quote(fp, safe='/')}"
    return None


def _safe_detection(r: Dict[str, Any]) -> Dict[str, Any]:
    """Return detection dict with thumbnail stripped, plus a serveable image_url."""
    out = {k: v for k, v in r.items() if k not in ("thumbnail", "clip_vec", "dino_vec")}
    source = str(out.get("source") or "").strip().lower()
    if source:
        out["source"] = source
    logical_source = ARCHIVE_SOURCE_ALIASES.get(source, source)
    if logical_source and logical_source != source:
        out["logical_source"] = logical_source
    out["source_label"] = _archive_source_label(source)
    out["archive_item_type"] = _archive_item_type(source)
    out["is_probe_detection"] = logical_source == "probe"
    out["is_video_description_frame"] = logical_source in {"vlm_summary", "vlm_alert"}
    if out.get("detection_id") is None and out.get("id") is not None:
        out["detection_id"] = out.get("id")
    out["score_semantics"] = _archive_score_semantics(source)
    out["has_thumbnail"] = bool(r.get("thumbnail"))
    url = _detection_image_url(r)
    if url:
        out["image_url"] = url
    return out


def _strip_thumbnails(results: List[Dict[str, Any]], folder: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("thumbnail", "thumbnail_b64", "clip_vec", "dino_vec")}
        if row.get("source") is not None:
            row = _annotate_archive_row(row)
        is_detection = bool(r.get("is_detection")) or bool(r.get("detection_id")) or bool(r.get("image_path"))
        url = _detection_image_url(r) if is_detection else _archive_result_image_url(r, folder=folder)
        if url:
            row["image_url"] = url
        out.append(row)
    return out


def _strip_thumbnails_deep(obj: Any) -> Any:
    """Recursively remove thumbnail/vector fields from nested dicts/lists."""
    if isinstance(obj, dict):
        return {
            k: _strip_thumbnails_deep(v)
            for k, v in obj.items()
            if k not in ("thumbnail", "thumbnail_b64", "clip_vec", "dino_vec")
        }
    if isinstance(obj, list):
        return [_strip_thumbnails_deep(item) for item in obj]
    return obj


def _tool_result_for_ui(tool_name: str, result: Any) -> Any:
    """Return a frontend-safe tool result without large inline images when a URL exists."""
    if tool_name != "describe_frame" or not isinstance(result, dict):
        return result
    out = dict(result)
    if out.get("image_url"):
        out.pop("snapshot_b64", None)
    return out


def _compact_detection_for_model(r: Dict[str, Any]) -> Dict[str, Any]:
    source = str(r.get("source") or "").strip().lower()
    detection_id = r.get("detection_id")
    if detection_id is None:
        detection_id = r.get("id")
    row = {
        "id": r.get("id"),
        "detection_id": detection_id,
        "timestamp_ms": _detection_timestamp_ms(r),
        "source": r.get("source"),
        "source_label": r.get("source_label") or _archive_source_label(r.get("source")),
        "archive_item_type": r.get("archive_item_type") or _archive_item_type(r.get("source")),
        "is_probe_detection": bool(
            r.get(
                "is_probe_detection",
                ARCHIVE_SOURCE_ALIASES.get(str(r.get("source") or "").lower(), str(r.get("source") or "").lower()) == "probe",
            )
        ),
        "probe_id": r.get("probe_id"),
        "probe_name": r.get("probe_name"),
        "channel_id": r.get("channel_id"),
        "severity": r.get("severity"),
        "bookmark_sent": r.get("bookmark_sent"),
        "pos_score": r.get("pos_score"),
        "neg_score": r.get("neg_score"),
        "margin": r.get("margin"),
        "score_semantics": _archive_score_semantics(source),
        "image_url": r.get("image_url") or _detection_image_url(r),
    }
    payload = r.get("payload")
    if isinstance(payload, dict):
        compact_payload: Dict[str, Any] = {}
        for key in (
            "run_id",
            "batch_start_ms",
            "batch_end_ms",
            "frame_timestamp_ms",
            "frame_index",
            "anchor_role",
            "severity",
            "alert_total",
            "alert_counts",
        ):
            if key in payload:
                compact_payload[key] = payload.get(key)
        summary = str(payload.get("summary") or "").strip()
        if summary:
            compact_payload["summary_excerpt"] = summary[:300]
        if compact_payload:
            row["payload"] = compact_payload
    return row


def _compact_search_result_for_model(r: Dict[str, Any]) -> Dict[str, Any]:
    detection_id = r.get("detection_id")
    if detection_id is None:
        detection_id = r.get("id")
    row: Dict[str, Any] = {
        "path": r.get("filepath") or r.get("path") or r.get("image_path"),
        "score": r.get("score"),
        "timestamp_ms": _detection_timestamp_ms(r),
        "source": r.get("source"),
        "source_label": r.get("source_label") or _archive_source_label(r.get("source")),
        "archive_item_type": r.get("archive_item_type") or _archive_item_type(r.get("source")),
        "probe_name": r.get("probe_name"),
        "channel_id": r.get("channel_id"),
        "image_url": r.get("image_url"),
    }
    if detection_id is not None:
        row["detection_id"] = detection_id
    return row


def _compact_prompt_settings_for_model(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"value": result}
    current = result.get("current") if isinstance(result.get("current"), dict) else result
    rollups = current.get("rollup_prompts") if isinstance(current.get("rollup_prompts"), dict) else {}
    stream_prompt = str(current.get("stream_system_prompt") or "")
    json_prompt = str(current.get("json_alert_prompt") or "")
    return {
        "scope": current.get("scope") or result.get("scope"),
        "channel_id": current.get("channel_id") or result.get("channel_id"),
        "stream_system_prompt": stream_prompt[:1000],
        "L0_live_prompt": stream_prompt[:1000],
        "json_alert_prompt": json_prompt[:800],
        "rollup_prompts": {
            level: str(prompt or "")[:600]
            for level, prompt in rollups.items()
            if str(level).strip().upper() in {"L1", "L2", "L3"}
        },
        "L1_prompt": str(rollups.get("L1") or "")[:600],
        "L2_prompt": str(rollups.get("L2") or "")[:600],
        "L3_prompt": str(rollups.get("L3") or "")[:600],
        "bookmark_enabled": current.get("bookmark_enabled"),
        "bookmark_cooldown_sec": current.get("bookmark_cooldown_sec"),
    }


def _compact_tool_result_for_model(tool_name: str, result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    if tool_name == "get_visual_window_signals":
        source_rows = result.get("by_source") if isinstance(result.get("by_source"), list) else []
        frame_rows = result.get("candidate_frames") if isinstance(result.get("candidate_frames"), list) else []
        errors = result.get("search_errors") if isinstance(result.get("search_errors"), list) else []
        return {
            "channel_id": result.get("channel_id"),
            "positive_query": result.get("positive_query"),
            "negative_query": result.get("negative_query"),
            "sources": result.get("sources"),
            "since_ms": result.get("since_ms"),
            "until_ms": result.get("until_ms"),
            "score_semantics": result.get("score_semantics"),
            "pnm": result.get("pnm"),
            "operator_note": result.get("operator_note"),
            "search_errors": errors[:4],
            "by_source": [
                {
                    "source": row.get("source"),
                    "source_label": row.get("source_label"),
                    "positive_count": row.get("positive_count"),
                    "negative_count": row.get("negative_count"),
                    "pnm": row.get("pnm"),
                    "margin": row.get("margin"),
                }
                for row in source_rows[:6]
                if isinstance(row, dict)
            ],
            "candidate_frames": [
                {
                    "detection_id": row.get("detection_id"),
                    "timestamp_ms": row.get("timestamp_ms"),
                    "source": row.get("source"),
                    "source_label": row.get("source_label"),
                    "positive_score": row.get("positive_score"),
                    "negative_score": row.get("negative_score"),
                    "margin": row.get("margin"),
                    "pnm_state": row.get("pnm_state"),
                    "image_url": row.get("image_url"),
                    "needs_describe_frame": row.get("needs_describe_frame"),
                }
                for row in frame_rows[:8]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "get_detections":
        detections = result.get("detections") if isinstance(result.get("detections"), list) else []
        return {
            "probe_id": result.get("probe_id"),
            "channel_id": result.get("channel_id"),
            "source": result.get("source"),
            "source_label": result.get("source_label") or _archive_source_label(result.get("source")),
            "since_ms": result.get("since_ms"),
            "until_ms": result.get("until_ms"),
            "total_in_window": result.get("total_in_window"),
            "returned": result.get("returned"),
            "offset": result.get("offset"),
            "sort_by": result.get("sort_by"),
            "detections": [_compact_detection_for_model(r) for r in detections[:8] if isinstance(r, dict)],
        }

    if tool_name == "search_archive":
        rows = result.get("results") if isinstance(result.get("results"), list) else []
        return {
            "scope": result.get("scope"),
            "source": result.get("source"),
            "source_label": result.get("source_label") or _archive_source_label(result.get("source")),
            "count": result.get("count"),
            "results": [_compact_search_result_for_model(r) for r in rows[:8] if isinstance(r, dict)],
        }

    if tool_name == "build_research_batch":
        rows = result.get("detections") if isinstance(result.get("detections"), list) else []
        return {
            "probe_id": result.get("probe_id"),
            "channel_id": result.get("channel_id"),
            "source": result.get("source"),
            "source_label": result.get("source_label") or _archive_source_label(result.get("source")),
            "sort_by": result.get("sort_by"),
            "batch_size": result.get("batch_size"),
            "periods": result.get("periods"),
            "bands": result.get("bands"),
            "detections": [_compact_detection_for_model(r) for r in rows[:8] if isinstance(r, dict)],
        }

    if tool_name == "get_detection_summary":
        rows = result.get("by_probe") if isinstance(result.get("by_probe"), list) else []
        compact_rows = []
        for row in rows[:12]:
            if not isinstance(row, dict):
                continue
            compact_rows.append({
                "probe_id": row.get("probe_id"),
                "probe_name": row.get("probe_name"),
                "channel_id": row.get("channel_id"),
                "source": row.get("source"),
                "source_label": row.get("source_label") or _archive_source_label(row.get("source")),
                "archive_item_type": row.get("archive_item_type") or _archive_item_type(row.get("source")),
                "hit_count": row.get("hit_count"),
                "latest_timestamp_ms": row.get("latest_timestamp_ms"),
            })
        return {
            "since_ms": result.get("since_ms"),
            "until_ms": result.get("until_ms"),
            "source": result.get("source"),
            "source_label": result.get("source_label") or _archive_source_label(result.get("source")),
            "probe_count": result.get("probe_count"),
            "total_detections": result.get("total_detections"),
            "total_archive_items": result.get("total_archive_items"),
            "by_probe": compact_rows,
        }

    if tool_name == "list_channels":
        rows = result.get("channels") if isinstance(result.get("channels"), list) else []
        return {
            "count": result.get("count"),
            "channels": [
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "enabled": row.get("enabled"),
                    "status": row.get("status"),
                }
                for row in rows[:12]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "normalize_time_window":
        return {
            "timezone": result.get("timezone"),
            "relative_range": result.get("relative_range"),
            "from_local": result.get("from_local"),
            "to_local": result.get("to_local"),
            "from_ts": result.get("from_ts"),
            "to_ts": result.get("to_ts"),
            "since_ms": result.get("since_ms"),
            "until_ms": result.get("until_ms"),
            "duration_sec": result.get("duration_sec"),
        }

    if tool_name == "list_video_summary_channels":
        rows = result.get("candidate_channels") if isinstance(result.get("candidate_channels"), list) else []
        errors = result.get("errors") if isinstance(result.get("errors"), list) else []
        return {
            "depth": result.get("depth"),
            "time_window": result.get("time_window"),
            "requested_count": result.get("requested_count"),
            "unchecked_count": result.get("unchecked_count"),
            "unchecked_channel_ids": result.get("unchecked_channel_ids"),
            "active_count": result.get("active_count"),
            "inactive_count": result.get("inactive_count"),
            "error_count": result.get("error_count"),
            "total_channels_checked": result.get("total_channels_checked"),
            "returned": result.get("returned"),
            "deferred_count": result.get("deferred_count"),
            "deferred_channel_ids": result.get("deferred_channel_ids"),
            "per_turn_channel_limit": result.get("per_turn_channel_limit"),
            "requires_confirmation": result.get("requires_confirmation"),
            "full_research_note": result.get("full_research_note"),
            "errors": [
                {
                    "channel_id": row.get("channel_id"),
                    "title": row.get("title"),
                    "error": row.get("error"),
                }
                for row in errors[:8]
                if isinstance(row, dict)
            ],
            "candidate_channels": [
                {
                    "channel_id": row.get("channel_id"),
                    "title": row.get("title"),
                    "summary_count": row.get("summary_count"),
                    "first_time": row.get("first_time"),
                    "latest_time": row.get("latest_time"),
                    "alert_total": row.get("alert_total"),
                    "running": row.get("running"),
                }
                for row in rows[:AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "get_video_summaries":
        entries = result.get("entries") if isinstance(result.get("entries"), list) else []
        evidence_frames = result.get("evidence_frames") if isinstance(result.get("evidence_frames"), list) else []
        return {
            "channel_id": result.get("channel_id"),
            "depth": result.get("depth"),
            "display_limit": result.get("display_limit"),
            "level_limit_applied": result.get("level_limit_applied"),
            "backend_truncated": result.get("backend_truncated"),
            "source_counts": result.get("source_counts"),
            "time_window": result.get("time_window"),
            "coverage": result.get("coverage"),
            "count": result.get("count"),
            "total_in_window": result.get("total_in_window"),
            "truncated": result.get("truncated"),
            "selection_strategy": result.get("selection_strategy"),
            "running": result.get("running"),
            "evidence_frame_query": result.get("evidence_frame_query"),
            "evidence_frame_queries": result.get("evidence_frame_queries"),
            "evidence_frame_attempted_sources": result.get("evidence_frame_attempted_sources"),
            "attempted_sources": result.get("attempted_sources"),
            "evidence_frame_totals": result.get("evidence_frame_totals"),
            "totals": result.get("totals"),
            "evidence_frames": [
                _compact_detection_for_model(row)
                for row in evidence_frames[:8]
                if isinstance(row, dict)
            ],
            "entries": [
                {
                    "time": row.get("time"),
                    "window_start": row.get("window_start"),
                    "window_end": row.get("window_end"),
                    "window_end_time": row.get("window_end_time"),
                    "level": row.get("level"),
                    "frame_count": row.get("frame_count"),
                    "item_count": row.get("item_count"),
                    "alert_total": row.get("alert_total"),
                    "alert_counts": row.get("alert_counts"),
                    "alert_severities": row.get("alert_severities"),
                    "summary": row.get("summary"),
                }
                for row in entries[:20]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "count_video_summary_events":
        events = result.get("transition_events") if isinstance(result.get("transition_events"), list) else []
        timeline = result.get("timeline") if isinstance(result.get("timeline"), list) else []
        return {
            "channel_id": result.get("channel_id"),
            "depth": result.get("depth"),
            "event_kind": result.get("event_kind"),
            "entity_query": result.get("entity_query"),
            "anchor_query": result.get("anchor_query"),
            "score_semantics": result.get("score_semantics"),
            "source_counts": result.get("source_counts"),
            "coverage": result.get("coverage"),
            "total_in_window": result.get("total_in_window"),
            "backend_truncated": result.get("backend_truncated"),
            "counts": result.get("counts"),
            "notes": result.get("notes"),
            "transition_events": [
                {
                    "type": row.get("type"),
                    "basis": row.get("basis"),
                    "time": row.get("time"),
                    "window_start": row.get("window_start"),
                    "window_end": row.get("window_end"),
                    "window_end_time": row.get("window_end_time"),
                    "previous_time": row.get("previous_time"),
                    "previous_state": row.get("previous_state"),
                    "summary": row.get("summary"),
                }
                for row in events[:24]
                if isinstance(row, dict)
            ],
            "timeline_samples": [
                {
                    "time": row.get("time"),
                    "window_start": row.get("window_start"),
                    "window_end": row.get("window_end"),
                    "state": row.get("state"),
                    "explicit_appearance": row.get("explicit_appearance"),
                    "explicit_disappearance": row.get("explicit_disappearance"),
                    "summary": row.get("summary"),
                }
                for row in timeline[:24]
                if isinstance(row, dict)
            ],
            "timeline_returned": result.get("timeline_returned"),
            "timeline_total": result.get("timeline_total"),
            "event_returned": result.get("event_returned"),
            "event_total": result.get("event_total"),
        }

    if tool_name == "track_visual_state_transitions":
        transitions = result.get("transitions") if isinstance(result.get("transitions"), list) else []
        segments = result.get("segments") if isinstance(result.get("segments"), list) else []
        boundary_frames = result.get("boundary_frames") if isinstance(result.get("boundary_frames"), list) else []
        candidate_frames = result.get("candidate_frames") if isinstance(result.get("candidate_frames"), list) else []
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        return {
            "channel_id": result.get("channel_id"),
            "subject_query": result.get("subject_query"),
            "positive_state_query": result.get("positive_state_query"),
            "negative_state_query": result.get("negative_state_query"),
            "negative_state_query_effective": result.get("negative_state_query_effective"),
            "alternate_state_query": result.get("alternate_state_query"),
            "positive_label": result.get("positive_label"),
            "negative_label": result.get("negative_label"),
            "alternate_label": result.get("alternate_label"),
            "sources": result.get("sources"),
            "time_window": result.get("time_window"),
            "coverage": result.get("coverage"),
            "score_semantics": result.get("score_semantics"),
            "thresholds": result.get("thresholds"),
            "counts": result.get("counts"),
            "frame_count": result.get("frame_count"),
            "source_totals": result.get("source_totals"),
            "source_returned": result.get("source_returned"),
            "warnings": warnings[:8],
            "operator_note": result.get("operator_note"),
            "transitions": [
                {
                    "type": row.get("type"),
                    "from_state": row.get("from_state"),
                    "to_state": row.get("to_state"),
                    "from_time": row.get("from_time"),
                    "to_time": row.get("to_time"),
                    "from_ms": row.get("from_ms"),
                    "to_ms": row.get("to_ms"),
                    "gap_sec": row.get("gap_sec"),
                    "before_frame": row.get("before_frame"),
                    "after_frame": row.get("after_frame"),
                }
                for row in transitions[:24]
                if isinstance(row, dict)
            ],
            "segments": [
                {
                    "state": row.get("state"),
                    "stability": row.get("stability"),
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "start_ms": row.get("start_ms"),
                    "end_ms": row.get("end_ms"),
                    "duration_sec": row.get("duration_sec"),
                    "sample_count": row.get("sample_count"),
                    "first_frame": row.get("first_frame"),
                    "last_frame": row.get("last_frame"),
                }
                for row in segments[:24]
                if isinstance(row, dict)
            ],
            "boundary_frames": [
                _compact_detection_for_model(row)
                for row in boundary_frames[:12]
                if isinstance(row, dict)
            ],
            "candidate_frames": [
                _compact_detection_for_model(row)
                for row in candidate_frames[:12]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "list_probes":
        rows = result.get("probes") if isinstance(result.get("probes"), list) else []
        return {
            "count": result.get("count"),
            "since_hours": result.get("since_hours"),
            "probes": [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "channel_id": row.get("channel_id"),
                    "enabled": row.get("enabled"),
                    "pos_floor": row.get("pos_floor"),
                    "margin": row.get("margin"),
                    "severity": row.get("severity"),
                    "bookmark": row.get("bookmark"),
                    "hit_count_24h": row.get("hit_count_24h"),
                    "latest_timestamp_ms": row.get("latest_timestamp_ms"),
                }
                for row in rows[:12]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "survey_channels":
        rows = result.get("channels") if isinstance(result.get("channels"), list) else []
        return {
            "fast_mode": result.get("fast_mode"),
            "duration_sec": result.get("duration_sec"),
            "sample_count": result.get("sample_count"),
            "channels": [
                {
                    "channel_id": row.get("channel_id"),
                    "title": row.get("title"),
                    "sample_count": row.get("sample_count"),
                    "survey": str(row.get("survey") or "")[:500],
                    "error": row.get("error"),
                }
                for row in rows[:8]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "create_probe":
        probe = result.get("probe") if isinstance(result.get("probe"), dict) else result.get("proposed")
        conflicts = result.get("conflicts") if isinstance(result.get("conflicts"), list) else []
        return {
            "status": result.get("status"),
            "action": result.get("action"),
            "exists": result.get("exists"),
            "probe_id": result.get("probe_id"),
            "probe_name": result.get("probe_name") or (probe or {}).get("name"),
            "channel_id": (probe or {}).get("channel_id"),
            "conflicts": [
                {"id": row.get("id"), "name": row.get("name"), "channel_id": row.get("channel_id")}
                for row in conflicts[:8]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "delete_probes":
        rows = result.get("targets") if isinstance(result.get("targets"), list) else []
        return {
            "status": result.get("status"),
            "delete_all": result.get("delete_all"),
            "deleted": result.get("deleted"),
            "count": result.get("count"),
            "targets": [
                {"id": row.get("id"), "name": row.get("name"), "channel_id": row.get("channel_id")}
                for row in rows[:12]
                if isinstance(row, dict)
            ],
        }

    if tool_name == "describe_frame":
        return {
            "description": result.get("description"),
            "source": result.get("source"),
            "channel_id": result.get("channel_id"),
            "image_path": result.get("image_path"),
            "note": result.get("note"),
        }

    if tool_name == "get_prompt_settings":
        return _compact_prompt_settings_for_model(result)

    if tool_name == "update_prompt_settings":
        compact = dict(result)
        compact.pop("approval", None)
        if isinstance(compact.get("current"), dict):
            compact["current"] = _compact_prompt_settings_for_model({"current": compact["current"]})
        if isinstance(compact.get("proposed"), dict):
            compact["proposed"] = _compact_prompt_settings_for_model({"current": compact["proposed"]})
        return compact

    if tool_name == "deploy_summary":
        return {
            "mode": result.get("mode"),
            "wipe": result.get("wipe"),
            "elapsed_sec": result.get("elapsed_sec"),
            "overview": result.get("overview"),
            "channels": result.get("channels"),
            "probes": result.get("probes"),
            "prompt_targets": result.get("prompt_targets"),
            "notes": result.get("notes"),
        }

    return _strip_thumbnails_deep(result)
