"""
agent.py — AI agent backend for evo-ssearch.

Provides:
  AgentStore   — SQLite session/message persistence
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
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_MAX_HISTORY_MESSAGES = 20       # last N messages kept as context
AGENT_MAX_TOOL_ITERATIONS  = 8        # runaway loop guard
AGENT_HEARTBEAT_INTERVAL   = 15       # seconds between SSE heartbeats
AGENT_SESSION_TTL_DAYS     = 30       # sessions older than this are GC'd
AGENT_MAX_SESSIONS         = 100      # sessions kept per store (GC oldest)
AGENT_MAX_MESSAGES_PER_SESSION = 200  # messages kept per session (prune oldest)
AGENT_MAX_RUNTIME_SKILLS_CHARS = 3_500
AGENT_MAX_ACTIVE_SKILL_CHARS   = 6_000
AGENT_MAX_PROBES_IN_PROMPT     = 12

_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_archive",
            "description": (
                "Semantic text search over either an indexed folder (CLIP/FAISS) or "
                "the detections archive (vector search over stored detections). "
                "Returns ranked results with file paths, similarity scores, and metadata. "
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
                            "detections: searches the detections archive via stored CLIP vectors."
                        ),
                    },
                    "folder": {
                        "type": "string",
                        "description": "Required when scope=indexed_folder. Absolute path to an indexed folder.",
                    },
                    "probe_id": {
                        "type": "string",
                        "description": "Optional. When scope=detections, restrict to detections from this probe.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict detections search to this channel.",
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
                        "description": "Detections search only. clip: fast/default. dino/fusion: richer visual matching when available.",
                    },
                    "candidate_limit": {
                        "type": "integer",
                        "description": "Detections search only. Candidate pool before final ranking. Default: 20000.",
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
            "name": "get_detections",
            "description": (
                "Retrieve recent detection events from the archive for one or more probes. "
                "Returns a list with timestamps, confidence scores, and image paths. "
                "Use to inspect what a probe has been triggering on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {
                        "type": "string",
                        "description": "Probe ID to filter by. If omitted, returns detections across all probes.",
                    },
                    "probe_name": {
                        "type": "string",
                        "description": "Human-readable probe name. Resolved to probe_id internally.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict to detections from this channel.",
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
                "Aggregate detection statistics grouped by probe. Returns hit counts and "
                "latest activity timestamps. Use for an overview before drilling into specific events."
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
                        "description": "Max nodes per rollup level to inspect before slicing the requested depth. Defaults to limit.",
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

    def call_with_tools(self, messages: List[Dict[str, Any]]) -> _LMResponse:
        """Blocking non-streaming call with tools. Returns parsed response."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": _TOOL_SCHEMAS,
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
# Session store
# ---------------------------------------------------------------------------

class AgentStore:
    """SQLite-backed session and message store."""

    def __init__(
        self,
        path: str = "agent_sessions.sqlite3",
        max_sessions: int = AGENT_MAX_SESSIONS,
        max_messages_per_session: int = AGENT_MAX_MESSAGES_PER_SESSION,
        session_ttl_days: int = AGENT_SESSION_TTL_DAYS,
    ) -> None:
        self.path = Path(path)
        self.max_sessions = max(10, int(max_sessions))
        self.max_msg = max(20, int(max_messages_per_session))
        self.ttl_ms = int(session_ttl_days) * 86_400_000
        self._lock = threading.RLock()
        self._init_db()
        self._gc()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        id          TEXT PRIMARY KEY,
                        title       TEXT,
                        created_at  INTEGER NOT NULL,
                        updated_at  INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS agent_messages (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id      TEXT NOT NULL
                                            REFERENCES agent_sessions(id) ON DELETE CASCADE,
                        role            TEXT NOT NULL,
                        content         TEXT,
                        tool_calls      TEXT,
                        tool_call_id    TEXT,
                        tool_name       TEXT,
                        tool_result     TEXT,
                        created_at      INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_msg_session
                        ON agent_messages (session_id, id ASC);
                    CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated
                        ON agent_sessions (updated_at DESC);
                """)
                conn.commit()
            finally:
                conn.close()

    # ── Session CRUD ────────────────────────────────────────────────────────

    def create_session(self) -> str:
        sid = uuid.uuid4().hex
        now = _now_ms()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO agent_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                    (sid, None, now, now),
                )
                conn.commit()
            finally:
                conn.close()
        return sid

    def touch_session(self, session_id: str, title: Optional[str] = None) -> None:
        now = _now_ms()
        with self._lock:
            conn = self._connect()
            try:
                if title is not None:
                    conn.execute(
                        "UPDATE agent_sessions SET updated_at=?, title=COALESCE(title,?) WHERE id=?",
                        (now, title, session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE agent_sessions SET updated_at=? WHERE id=?",
                        (now, session_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id FROM agent_sessions WHERE id=?", (session_id,)
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("""
                    SELECT s.id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM agent_sessions s
                    LEFT JOIN agent_messages m ON m.session_id = s.id
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                """).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, title, created_at, updated_at FROM agent_sessions WHERE id=?",
                    (session_id,),
                ).fetchone()
                if not row:
                    return None
                messages = conn.execute(
                    "SELECT * FROM agent_messages WHERE session_id=? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
                return {
                    "session_id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "messages": [_msg_row_to_dict(m) for m in messages],
                }
            finally:
                conn.close()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM agent_sessions WHERE id=?", (session_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ── Message CRUD ────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_result: Optional[Any] = None,
    ) -> int:
        now = _now_ms()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        tr_json = json.dumps(tool_result, default=str) if tool_result is not None else None
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO agent_messages
                       (session_id, role, content, tool_calls, tool_call_id, tool_name, tool_result, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (session_id, role, content, tc_json, tool_call_id, tool_name, tr_json, now),
                )
                msg_id = cur.lastrowid
                self._prune_messages_locked(conn, session_id)
                conn.commit()
            finally:
                conn.close()
        self.touch_session(session_id)
        return msg_id or 0

    def _prune_messages_locked(self, conn: sqlite3.Connection, session_id: str) -> None:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM agent_messages WHERE session_id=?", (session_id,)
        ).fetchone()
        total = int(row["c"]) if row else 0
        if total <= self.max_msg:
            return
        excess = total - self.max_msg
        conn.execute(
            """DELETE FROM agent_messages WHERE id IN (
               SELECT id FROM agent_messages WHERE session_id=? ORDER BY id ASC LIMIT ?
            )""",
            (session_id, excess),
        )

    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Return the last AGENT_MAX_HISTORY_MESSAGES messages for this session
        as a compact list of OpenAI-compatible dialog messages.

        We intentionally do not replay persisted tool-call / tool-result traffic
        across turns. LM Studio prompt templates are often brittle around older
        tool traces, while the operator-facing assistant text already summarizes
        the important outcome ("preview generated", "report ready", etc.).
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM agent_messages WHERE session_id=?
                       ORDER BY id DESC LIMIT ?""",
                    (session_id, AGENT_MAX_HISTORY_MESSAGES * 4),
                ).fetchall()
            finally:
                conn.close()

        rows = list(reversed(rows))  # restore chronological order
        messages: List[Dict[str, Any]] = []

        for row in rows:
            msg = _msg_row_to_openai(row)
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "").strip()

            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue

            if messages:
                prev = messages[-1]
                prev_role = str(prev.get("role") or "")
                prev_content = str(prev.get("content") or "")
                if prev_role == role:
                    if prev_content.strip() == content:
                        continue
                    prev["content"] = f"{prev_content}\n\n{content}".strip()
                    continue

            messages.append({"role": role, "content": content})

        messages = messages[-AGENT_MAX_HISTORY_MESSAGES:]
        while messages and messages[0].get("role") != "user":
            messages.pop(0)
        return messages

    # ── GC ──────────────────────────────────────────────────────────────────

    def _gc(self) -> None:
        cutoff = _now_ms() - self.ttl_ms
        with self._lock:
            conn = self._connect()
            try:
                # Delete expired sessions (cascade removes their messages)
                conn.execute(
                    "DELETE FROM agent_sessions WHERE updated_at < ?", (cutoff,)
                )
                # If still too many sessions, remove the oldest ones
                conn.execute(
                    """DELETE FROM agent_sessions WHERE id IN (
                       SELECT id FROM agent_sessions
                       ORDER BY updated_at DESC
                       LIMIT -1 OFFSET ?
                    )""",
                    (self.max_sessions,),
                )
                conn.commit()
            finally:
                conn.close()


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
            "get_detections":       self._get_detections,
            "get_detection_summary": self._get_detection_summary,
            "list_channels":        self._list_channels,
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
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            sort_by=sort_by,
            candidate_limit=candidate_limit,
            mode=mode,
        )
        return {"scope": scope, "count": len(results), "results": _strip_thumbnails(results)}

    # ── get_detections ──────────────────────────────────────────────────────

    def _get_detections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        probe_id = str(args.get("probe_id") or "").strip() or None

        # Resolve probe_name → probe_id if needed
        probe_name_raw = str(args.get("probe_name") or "").strip()
        if not probe_id and probe_name_raw:
            probe_id = self._resolve_probe_id_by_name(probe_name_raw)

        channel_id  = _opt_int(args.get("channel_id"))
        since_ms, until_ms = self._resolve_time_window(args, default_since_hours=24.0)
        limit       = max(1, min(100, int(args.get("limit") or 20)))
        offset      = max(0, int(args.get("offset") or 0))
        sort_by     = str(args.get("sort_by") or "newest").strip().lower()
        if sort_by not in {"newest", "oldest"}:
            sort_by = "newest"

        rows, total = self._list_detection_window(
            probe_id=probe_id,
            channel_id=channel_id,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
        )

        return {
            "probe_id": probe_id,
            "channel_id": channel_id,
            "since_ms": since_ms,
            "until_ms": until_ms,
            "total_in_window": total,
            "returned": len(rows),
            "offset": offset,
            "sort_by": sort_by,
            "detections": [_safe_detection(r) for r in rows],
        }

    # ── get_detection_summary ───────────────────────────────────────────────

    def _get_detection_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id  = _opt_int(args.get("channel_id"))
        since_ms, until_ms = self._resolve_time_window(args, default_since_hours=24.0)

        if until_ms is None:
            rows = self._ds.summarize_by_probe(since_ms=since_ms, channel_id=channel_id)
        else:
            grouped: Dict[Tuple[str, int], Dict[str, Any]] = {}
            offset = 0
            while True:
                batch, _total = self._ds.list_detections(
                    probe_id=None,
                    channel_id=channel_id,
                    since_ms=since_ms,
                    until_ms=until_ms,
                    limit=500,
                    offset=offset,
                )
                if not batch:
                    break
                for row in batch:
                    key = (str(row.get("probe_id") or ""), int(row.get("channel_id") or 0))
                    slot = grouped.setdefault(key, {
                        "probe_id": row.get("probe_id"),
                        "probe_name": row.get("probe_name"),
                        "channel_id": row.get("channel_id"),
                        "hit_count": 0,
                        "latest_timestamp_ms": 0,
                    })
                    slot["hit_count"] += 1
                    slot["latest_timestamp_ms"] = max(
                        int(slot["latest_timestamp_ms"] or 0),
                        int(row.get("event_timestamp_ms") or 0),
                    )
                offset += len(batch)
                if len(batch) < 500:
                    break
            rows = sorted(
                grouped.values(),
                key=lambda row: int(row.get("latest_timestamp_ms") or 0),
                reverse=True,
            )
        return {
            "since_ms": since_ms,
            "until_ms": until_ms,
            "probe_count": len(rows),
            "total_detections": sum(r.get("hit_count", 0) for r in rows),
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

    # ── list_probes ────────────────────────────────────────────────────────

    def _list_probes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        since_hours = float(args.get("since_hours") or 24)
        since_ms = int(time.time() * 1000 - since_hours * 3_600_000)
        probes = self._ps.list_probes()
        summary_rows = self._ds.summarize_by_probe(since_ms=since_ms)
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
            "sort_by": sort_by,
            "periods": period_reports,
            "bands": band_reports,
            "batch_size": len(selected),
            "detections": [_safe_detection(row) for row in selected],
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

        # Live snapshot path
        if channel_id is not None:
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
            resolved_path = rec.get("image_path")
            if not resolved_path:
                # Fall back to thumbnail
                thumb = rec.get("thumbnail")
                if not thumb:
                    raise ToolError(f"Detection {detection_id} has no image_path or thumbnail.")
                return self._describe_from_thumb_b64(thumb, prompt)

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
            "snapshot_b64": encoded,
        }

    def _describe_from_thumb_b64(self, thumb_b64: str, prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are an expert visual analyst. Be concise and factual."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Task: {prompt}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{thumb_b64}",
                    "detail": "high",
                }},
            ]},
        ]
        description = self._lm(messages)
        return {
            "description": description,
            "source": "thumbnail",
            "image_path": None,
            "snapshot_b64": thumb_b64,
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
        depth       = str(args.get("depth") or "L1").strip().upper()
        if depth == "LIVE":
            depth = "L0"
        limit       = max(1, min(100, int(args.get("limit") or 20)))
        level_limit = max(limit, min(500, int(args.get("level_limit") or limit)))
        run_selector = str(args.get("run") or "").strip() or None

        if not hasattr(self._lxm, "summary_rollups"):
            raise ToolError("Luxriot manager is not available or not configured.")

        from_ts = _opt_float(args.get("from_ts"))
        to_ts = _opt_float(args.get("to_ts"))
        if from_ts is None and to_ts is None:
            since_hours = float(args.get("since_hours") or 6)
            to_ts = time.time()
            from_ts = to_ts - since_hours * 3600.0
        else:
            if to_ts is None:
                to_ts = time.time()
            if from_ts is None:
                from_ts = max(0.0, float(to_ts) - 6 * 3600.0)

        try:
            rollups = self._lxm.summary_rollups(
                channel_id=channel_id,
                run_selector=run_selector,
                start_ts=from_ts,
                end_ts=to_ts,
                level_limit=level_limit,
            )
        except Exception as exc:
            raise ToolError(f"Could not fetch summaries: {exc}") from exc

        nodes = (rollups.get("levels") or {}).get(depth) or []

        # Flatten to a clean list for the LLM
        import datetime as _dt
        entries = []
        for node in nodes[-limit:]:
            entry: Dict[str, Any] = {}
            ts = node.get("window_start") or node.get("created_at")
            if ts:
                entry["time"] = _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
            text = str(node.get("summary") or "").strip()
            if text:
                entry["summary"] = text[:800]
            if entry.get("summary"):
                entries.append(entry)

        return {
            "channel_id": channel_id,
            "depth": depth,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "run": run_selector,
            "count": len(entries),
            "selected_run": rollups.get("selected_run"),
            "run_filter_id": rollups.get("run_filter_id"),
            "running": bool(rollups.get("running")),
            "entries": entries,
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

        summary_rows = self._ds.summarize_by_probe(since_ms=since_ms, channel_id=channel_id)

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
                since_ms=since_ms,
                until_ms=until_ms,
                limit=top_events,
                offset=0,
            )

            # Accumulate hourly buckets
            all_rows, _ = self._ds.list_detections(
                probe_id=pid,
                channel_id=channel_id,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=500,
                offset=0,
            )
            for det in all_rows:
                ts = det.get("timestamp_ms") or 0
                hour_key = time.strftime("%Y-%m-%d %H:00", time.localtime(ts / 1000))
                activity_by_hour[hour_key] += 1

            probes_data.append({
                "probe_id": pid,
                "probe_name": row.get("probe_name"),
                "channel_id": row.get("channel_id"),
                "hit_count": row.get("hit_count", 0),
                "latest_ts": row.get("latest_timestamp_ms"),
                "top_events": [_safe_detection(r) for r in top_rows],
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

    def _list_detection_window(
        self,
        *,
        probe_id: Optional[str],
        channel_id: Optional[int],
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
) -> str:
    now_str = time.strftime("%Y-%m-%d %H:%M")

    # Active probes
    try:
        probes = probes_store.list_probes()
    except Exception:
        probes = []

    # Recent detection counts (last 24h)
    since_ms = int(time.time() * 1000 - 86_400_000)
    try:
        probe_summary = detections_store.summarize_by_probe(since_ms=since_ms)
        hit_by_probe  = {r["probe_id"]: r["hit_count"] for r in probe_summary}
    except Exception:
        hit_by_probe  = {}

    # Available channels
    try:
        raw_channels = getattr(luxriot_manager, "channels", None) or []
        channels_str = ", ".join(
            f"{c.get('id')} ({c.get('title', 'unknown')})"
            for c in (raw_channels if isinstance(raw_channels, list) else [])
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
        f"- Do not claim support for PDF export, CSV export, emails, file links, async report queues, or background jobs unless a tool explicitly returns that artifact.\n"
        f"- If an operator asks for an unsupported export, say so plainly and offer the closest available format, such as a structured chat report.\n"
        f"- Prefer absolute time windows (since_ms/until_ms or from_ts/to_ts) when the operator asks about a specific date or period.\n"
        f"- Use the repository playbooks index below as routing hints; load-bearing details are provided only for explicitly activated playbooks.\n"
        f"- For probe tuning or archive research, follow the matching playbook when one is activated or clearly implied.\n"
        f"- When returning search results or detections, summarize; don't dump raw lists.\n"
        f"- Ask the operator a clarifying question if the available data is too sparse to make a safe change.\n"
        f"- Use markdown for structure."
        f"{skills_block}"
        f"{active_skills_block}"
    )


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
        store_path: str = "agent_sessions.sqlite3",
    ) -> None:
        self._ps  = probes_store
        self._ds  = detections_store
        self._lxm = luxriot_manager
        self.store = AgentStore(path=store_path)
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

    def stream_chat(
        self,
        session_id: Optional[str],
        message: str,
        image_b64: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Main entry point. Yields SSE-formatted strings.
        Each event is:  'data: <json>\\n\\n'
        """
        # ── session setup ──────────────────────────────────────────────────
        if not session_id or not self.store.session_exists(session_id):
            session_id = self.store.create_session()

        title = message[:60].strip() or "Chat"
        self.store.touch_session(session_id, title=title)

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

        self.store.add_message(session_id, role="user", content=message)

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
            if len(mentioned_probes) != 1 and not wants_all_probes:
                probe_names = [
                    str(probe.get("name") or "").strip()
                    for probe in self._ps.list_probes()
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
                self.store.add_message(session_id, role="assistant", content=clarification)
                yield _sse({"type": "text", "content": clarification})
                yield _sse({"type": "done", "session_id": session_id})
                return
        system_prompt = build_system_prompt(
            self._ps,
            self._ds,
            self._lxm,
            active_skill_slugs=requested_skill_slugs,
        )
        history = self.store.load_history(session_id)

        # Replace the stored user content with the full (possibly image-bearing) one
        in_flight: List[Dict[str, Any]] = (
            [{"role": "system", "content": system_prompt}]
            + history[:-1]                          # all but the just-added user msg
            + [{"role": "user", "content": user_content}]
        )

        # Accumulated messages from this turn (to persist after streaming)
        new_assistant_messages: List[Dict[str, Any]] = []

        # ── tool loop ──────────────────────────────────────────────────────
        for iteration in range(AGENT_MAX_TOOL_ITERATIONS):
            # Run the blocking LM call in a thread so we can emit heartbeats
            lm_response: _LMResponse
            try:
                lm_response = yield from _run_with_heartbeats(
                    fn=lambda: self._lm_client.call_with_tools(in_flight),
                    heartbeat_interval=AGENT_HEARTBEAT_INTERVAL,
                )
            except Exception as exc:
                yield _sse({"type": "error", "message": f"LM error: {exc}"})
                yield _sse({"type": "done", "session_id": session_id})
                return

            if lm_response.finish_reason != "tool_calls" or not lm_response.tool_calls:
                # Model wants to respond with text — break out to streaming phase
                break

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

                try:
                    result = yield from _run_with_heartbeats(
                        fn=lambda tc=tc, progress_queue=progress_queue: self._tools.execute(
                            tc.name,
                            tc.args,
                            progress_cb=lambda event: progress_queue.put(event),
                        ),
                        heartbeat_interval=AGENT_HEARTBEAT_INTERVAL,
                        progress_queue=progress_queue,
                    )
                    result_for_model = _compact_tool_result_for_model(tc.name, result)
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(result_for_model, default=str)}
                    yield _sse({"type": "tool_result", "name": tc.name, "result": result})
                except ToolError as exc:
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

        else:
            # Hit iteration limit
            yield _sse({"type": "error",
                        "message": f"Tool loop exceeded {AGENT_MAX_TOOL_ITERATIONS} iterations."})
            yield _sse({"type": "done", "session_id": session_id})
            return

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
                )

        # Persist final assistant text
        if final_text:
            self.store.add_message(session_id, role="assistant", content=final_text)

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


def _sort_detection_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    sort_key: Callable[[Dict[str, Any]], Any]
    reverse = True
    if sort_by == "oldest":
        sort_key = lambda r: (int(r.get("event_timestamp_ms") or 0), int(r.get("id") or 0))
        reverse = False
    elif sort_by == "highest_pos":
        sort_key = lambda r: (float(r.get("pos_score") or 0.0), int(r.get("event_timestamp_ms") or 0))
    elif sort_by == "lowest_pos":
        sort_key = lambda r: (float(r.get("pos_score") or 0.0), int(r.get("event_timestamp_ms") or 0))
        reverse = False
    elif sort_by == "highest_margin":
        sort_key = lambda r: (float(r.get("margin") or 0.0), int(r.get("event_timestamp_ms") or 0))
    elif sort_by == "lowest_margin":
        sort_key = lambda r: (float(r.get("margin") or 0.0), int(r.get("event_timestamp_ms") or 0))
        reverse = False
    else:
        sort_key = lambda r: (int(r.get("event_timestamp_ms") or 0), int(r.get("id") or 0))
    return sorted(rows, key=sort_key, reverse=reverse)


def _detection_image_url(r: Dict[str, Any]) -> Optional[str]:
    """Return a URL the frontend can use to load the detection's image."""
    ip = str(r.get("image_path") or r.get("path") or "").strip()
    if ip:
        from urllib.parse import quote
        return f"/detections/image?image_path={quote(ip, safe='')}"
    if r.get("thumbnail"):
        # inline data-URI from stored thumbnail
        return f"data:image/jpeg;base64,{r['thumbnail']}"
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


def _compact_detection_for_model(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": r.get("id"),
        "timestamp_ms": r.get("timestamp_ms") or r.get("event_timestamp_ms") or r.get("recorded_at_ms"),
        "probe_id": r.get("probe_id"),
        "probe_name": r.get("probe_name"),
        "channel_id": r.get("channel_id"),
        "severity": r.get("severity"),
        "bookmark_sent": r.get("bookmark_sent"),
        "pos_score": r.get("pos_score"),
        "neg_score": r.get("neg_score"),
        "margin": r.get("margin"),
        "image_url": r.get("image_url") or _detection_image_url(r),
    }


def _compact_search_result_for_model(r: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": r.get("filepath") or r.get("path") or r.get("image_path"),
        "score": r.get("score"),
        "timestamp_ms": r.get("timestamp_ms") or r.get("event_timestamp_ms"),
        "probe_name": r.get("probe_name"),
        "channel_id": r.get("channel_id"),
        "image_url": r.get("image_url"),
    }
    if r.get("detection_id") is not None:
        row["detection_id"] = r.get("detection_id")
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

    if tool_name == "get_detections":
        detections = result.get("detections") if isinstance(result.get("detections"), list) else []
        return {
            "probe_id": result.get("probe_id"),
            "channel_id": result.get("channel_id"),
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
            "count": result.get("count"),
            "results": [_compact_search_result_for_model(r) for r in rows[:8] if isinstance(r, dict)],
        }

    if tool_name == "build_research_batch":
        rows = result.get("detections") if isinstance(result.get("detections"), list) else []
        return {
            "probe_id": result.get("probe_id"),
            "channel_id": result.get("channel_id"),
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
                "hit_count": row.get("hit_count"),
                "latest_timestamp_ms": row.get("latest_timestamp_ms"),
            })
        return {
            "since_ms": result.get("since_ms"),
            "until_ms": result.get("until_ms"),
            "probe_count": result.get("probe_count"),
            "total_detections": result.get("total_detections"),
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
                    "hit_count_24h": row.get("hit_count_24h"),
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


def _msg_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for field_name in ("tool_calls", "tool_result"):
        raw = d.get(field_name)
        if raw:
            try:
                d[field_name] = json.loads(raw)
            except Exception:
                pass
    return d


def _msg_row_to_openai(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a DB row to an OpenAI-compatible message dict."""
    d = dict(row)
    role = d.get("role", "user")

    if role == "assistant":
        msg: Dict[str, Any] = {"role": "assistant"}
        if d.get("content"):
            msg["content"] = d["content"]
        tcs_raw = d.get("tool_calls")
        if tcs_raw:
            try:
                msg["tool_calls"] = json.loads(tcs_raw)
            except Exception:
                pass
        return msg

    if role == "tool":
        raw_result = d.get("tool_result") or "{}"
        return {
            "role": "tool",
            "tool_call_id": d.get("tool_call_id", ""),
            "name": d.get("tool_name", ""),
            "content": raw_result if isinstance(raw_result, str) else json.dumps(raw_result),
        }

    # user
    return {"role": "user", "content": d.get("content") or ""}
