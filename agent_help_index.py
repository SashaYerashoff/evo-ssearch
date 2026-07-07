"""In-repo BM25 self-help index over allowlisted operator/admin docs.

Read-only retrieval so the agent can explain UI steps, workflows, scenario
meaning, and product capabilities/limits from EVA's own vetted documentation.

No external dependency (closed-network + license-audited deployment): a small
Okapi BM25 over a handful of markdown files. Role-gating is applied by the caller
using each chunk's ``required_permission`` (None = any agent:use user) via
``partition_by_permission``; the security boundary remains the tool gateway, not
this index. See docs/architecture/agent_self_help_design.md.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allowlist ONLY. (path relative to repo root, required_permission for the file
# [None = operator/any agent:use], audience label). Never add history/, gtm/,
# legal/, field_rollout, or anything with secrets/[FIELD] values.
_ALLOWLIST: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("docs/operator/operator_guide.md", None, "operator"),
    ("docs/operator/operator_scenarios.md", None, "operator"),
    ("docs/operator/agent_capabilities.md", None, "operator"),
    ("docs/operator/demo_runbook.md", None, "operator"),
    ("docs/00_CANON/glossary.md", None, "operator"),
    ("docs/known_limitations.md", None, "operator"),
    ("docs/admin/observability.md", None, "operator"),
    ("docs/admin/admin_guide.md", "users:manage", "admin"),
    ("docs/admin/backup_recovery.md", "settings:manage", "admin"),
    ("docs/install/deployment_guide.md", "settings:manage", "engineer"),
    ("docs/install/inference_topology.md", "settings:manage", "engineer"),
)

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the of to for and or in on at is are be with you your this that it as "
    "by from can do does how what when where which who not no into over per use "
    "using if then than via i we our".split()
)
_K1 = 1.5
_B = 0.75
_SNIPPET_MAX = 600


def _tokenize(text: str) -> List[str]:
    cleaned = _CODE_FENCE_RE.sub(" ", text)
    cleaned = cleaned.replace("`", " ").replace("#", " ").replace("*", " ")
    return [
        t
        for t in _TOKEN_RE.findall(cleaned.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


def _split_sections(body: str) -> List[Tuple[str, str]]:
    """(heading, text) per ##/### section; text before the first heading -> 'Overview'."""
    current_heading = "Overview"
    current: List[str] = []
    sections: List[Tuple[str, List[str]]] = []
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current:
                sections.append((current_heading, current))
            current_heading = match.group(2).strip() or current_heading
            current = []
        else:
            current.append(line)
    if current:
        sections.append((current_heading, current))
    out: List[Tuple[str, str]] = []
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        if text:
            out.append((heading, text))
    return out


@dataclass
class _Chunk:
    doc_path: str
    heading: str
    section: str
    audience: str
    required_permission: Optional[str]
    text: str
    tokens: List[str]
    tf: Dict[str, int]


class HelpIndex:
    """Deterministic BM25 index over the allowlisted docs under ``base_dir``."""

    def __init__(
        self,
        base_dir: Path,
        allowlist: Sequence[Tuple[str, Optional[str], str]] = _ALLOWLIST,
    ) -> None:
        self.base_dir = Path(base_dir)
        self._chunks: List[_Chunk] = []
        self._doc_freq: Dict[str, int] = {}
        self._avg_len: float = 0.0
        self._indexed_docs: List[str] = []
        self._build(allowlist)

    def _build(self, allowlist: Sequence[Tuple[str, Optional[str], str]]) -> None:
        for rel_path, required_permission, audience in allowlist:
            try:
                body = (self.base_dir / rel_path).read_text(encoding="utf-8")
            except OSError:
                continue
            self._indexed_docs.append(rel_path)
            title = Path(rel_path).stem.replace("_", " ").title()
            for heading, text in _split_sections(body):
                tokens = _tokenize(f"{heading} {text}")
                if not tokens:
                    continue
                tf: Dict[str, int] = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1
                self._chunks.append(
                    _Chunk(
                        doc_path=rel_path,
                        heading=heading,
                        section=f"{title} § {heading}",
                        audience=audience,
                        required_permission=required_permission,
                        text=text,
                        tokens=tokens,
                        tf=tf,
                    )
                )
        df: Dict[str, int] = {}
        total_len = 0
        for chunk in self._chunks:
            total_len += len(chunk.tokens)
            for term in chunk.tf:
                df[term] = df.get(term, 0) + 1
        self._doc_freq = df
        self._avg_len = (total_len / len(self._chunks)) if self._chunks else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens: Sequence[str], chunk: _Chunk) -> float:
        if not self._avg_len:
            return 0.0
        dl = len(chunk.tokens)
        score = 0.0
        for term in set(query_tokens):
            tf = chunk.tf.get(term, 0)
            if not tf:
                continue
            denom = tf + _K1 * (1 - _B + _B * dl / self._avg_len)
            score += self._idf(term) * (tf * (_K1 + 1)) / denom
        return score

    @property
    def indexed_docs(self) -> List[str]:
        return list(self._indexed_docs)

    def query(self, text: str, pool: int = 24) -> List[Dict[str, Any]]:
        q = _tokenize(text)
        if not q or not self._chunks:
            return []
        scored: List[Tuple[float, _Chunk]] = []
        for chunk in self._chunks:
            s = self._score(q, chunk)
            if s > 0:
                scored.append((s, chunk))
        # Stable order: score desc, then doc/heading for determinism on ties.
        scored.sort(key=lambda item: (-item[0], item[1].doc_path, item[1].heading))
        out: List[Dict[str, Any]] = []
        for s, chunk in scored[: max(1, pool)]:
            snippet = chunk.text.strip()
            if len(snippet) > _SNIPPET_MAX:
                snippet = snippet[:_SNIPPET_MAX].rstrip() + " …"
            out.append(
                {
                    "doc": chunk.doc_path,
                    "section": chunk.section,
                    "heading": chunk.heading,
                    "audience": chunk.audience,
                    "required_permission": chunk.required_permission,
                    "snippet": snippet,
                    "score": round(float(s), 4),
                }
            )
        return out


def partition_by_permission(
    candidates: Sequence[Dict[str, Any]],
    granted: Optional[Sequence[str]] = None,
    top_k: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split candidates into allowed ``results`` and redacted ``restricted_matches``.

    A candidate is allowed if its ``required_permission`` is None or present in
    ``granted``. Restricted entries carry no snippet/procedure text.
    """
    granted_set = {str(p) for p in (granted or ())}
    results: List[Dict[str, Any]] = []
    restricted: List[Dict[str, Any]] = []
    for cand in candidates:
        req = cand.get("required_permission")
        if req is None or req in granted_set:
            if len(results) < top_k:
                results.append(
                    {
                        "doc": cand["doc"],
                        "section": cand["section"],
                        "heading": cand["heading"],
                        "snippet": cand["snippet"],
                        "score": cand["score"],
                    }
                )
        elif len(restricted) < top_k:
            restricted.append(
                {
                    "section": cand["section"],
                    "heading": cand["heading"],
                    "required_permission": req,
                    "score": cand["score"],
                }
            )
    return results, restricted


def build_help_response(
    query: str,
    candidates: Sequence[Dict[str, Any]],
    granted: Optional[Sequence[str]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Shape the agent-facing help response with role gating + a redirect signal.

    ``candidates`` must be globally score-ordered (as returned by HelpIndex.query),
    so candidates[0] is the best overall match. ``best_match_restricted`` is set
    when that best match is not allowed for the caller — so the agent redirects to
    the required permission even if weaker allowed matches exist.
    """
    granted_set = {str(p) for p in (granted or ())}
    results, restricted = partition_by_permission(candidates, granted_set, top_k=top_k)
    best_match_restricted = False
    best_required_permission: Optional[str] = None
    best_restricted_section: Optional[str] = None
    if candidates:
        top = candidates[0]
        req = top.get("required_permission")
        if req is not None and req not in granted_set:
            best_match_restricted = True
            best_required_permission = req
            best_restricted_section = top.get("section")
    return {
        "query": query,
        "results": results,
        "restricted_matches": restricted,
        "best_match_restricted": best_match_restricted,
        "best_required_permission": best_required_permission,
        "best_restricted_section": best_restricted_section,
    }


_lock = threading.Lock()
_singleton: Optional[HelpIndex] = None


def get_help_index() -> HelpIndex:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = HelpIndex(Path(__file__).resolve().parent)
        return _singleton
