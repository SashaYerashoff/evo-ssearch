from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = REPO_ROOT / "docs/tuktuk/grammar_pin.md"


class TuktukGrammarPinTests(unittest.TestCase):
    def test_pinned_artifact_hashes_match_checked_in_files(self) -> None:
        pin = PIN_PATH.read_text(encoding="utf-8")
        pinned = dict(
            re.findall(
                r"\| `(docs/tuktuk/[^`]+)` \| `([0-9a-f]{64})` \|",
                pin,
            )
        )

        self.assertGreaterEqual(len(pinned), 6)
        for relative_path, expected_hash in pinned.items():
            path = REPO_ROOT / relative_path
            self.assertTrue(path.exists(), relative_path)
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, relative_path)

    def test_agent_instructions_point_to_pin_and_review_questions(self) -> None:
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

        for text in (agents, claude):
            self.assertIn("docs/tuktuk/grammar_pin.md", text)
            self.assertIn("docs/tuktuk/grammar_review_questions.md", text)


if __name__ == "__main__":
    unittest.main()
