import json
import unittest

from incident_attention import (
    ALERT_CONTRACT_BLOCK,
    BATCH_STATE_BLOCK,
    AttentionTier,
    ContextCompactionTier,
    IncidentAttentionCandidate,
    IncidentPromptEnvelopePlanner,
    PromptBudgetError,
    PromptEnvelopeBudget,
    ProtectedPromptBlock,
    allocate_incident_attention,
    estimate_text_tokens,
)


class IncidentAttentionPolicyTests(unittest.TestCase):
    def test_normal_uses_two_foreground_and_hard_priority_uses_four(self):
        normal = [
            IncidentAttentionCandidate(
                incident_id=f"normal-{index}",
                operator_selected=False,
                updated_at_ms=index,
            )
            for index in range(5)
        ]

        normal_plan = allocate_incident_attention(normal)
        self.assertEqual(normal_plan.foreground_limit, 2)
        self.assertEqual(
            normal_plan.foreground_incident_ids,
            ("normal-4", "normal-3"),
        )

        hard_plan = allocate_incident_attention(
            normal
            + [
                IncidentAttentionCandidate(
                    incident_id="critical",
                    level="critical",
                    operator_selected=False,
                )
            ]
        )
        self.assertEqual(hard_plan.effective_level, "critical")
        self.assertEqual(hard_plan.foreground_limit, 4)
        self.assertEqual(len(hard_plan.foreground_incident_ids), 4)
        self.assertEqual(hard_plan.foreground_incident_ids[0], "critical")

    def test_hot_overflow_is_parked_and_never_implies_resolution(self):
        candidates = [
            IncidentAttentionCandidate(
                incident_id=f"incident-{index}",
                updated_at_ms=index,
                unresolved=True,
            )
            for index in range(10)
        ]

        allocation = allocate_incident_attention(candidates)

        self.assertEqual(len(allocation.foreground_incident_ids), 2)
        self.assertEqual(len(allocation.hot_incident_ids), 8)
        self.assertEqual(len(allocation.parked_incident_ids), 2)
        self.assertEqual(
            set(allocation.hot_incident_ids) | set(allocation.parked_incident_ids),
            {candidate.incident_id for candidate in candidates},
        )
        for decision in allocation.decisions:
            self.assertFalse(decision.resolution_inferred)
            if decision.tier is AttentionTier.PARKED:
                self.assertTrue(decision.unresolved)
                self.assertIn("attention_capacity_only", decision.reasons)

    def test_incumbency_and_resolution_debt_are_deterministic_rank_inputs(self):
        allocation = allocate_incident_attention(
            [
                IncidentAttentionCandidate(
                    incident_id="plain",
                    updated_at_ms=10,
                ),
                IncidentAttentionCandidate(
                    incident_id="debt",
                    resolution_debt=3,
                    updated_at_ms=10,
                ),
                IncidentAttentionCandidate(
                    incident_id="incumbent",
                    incumbent_tier=AttentionTier.FOREGROUND,
                    updated_at_ms=10,
                ),
            ]
        )

        self.assertEqual(
            allocation.all_incident_ids,
            ("incumbent", "debt", "plain"),
        )
        self.assertIn("incumbent_foreground", allocation.decisions[0].reasons)
        self.assertIn("resolution_debt_3", allocation.decisions[1].reasons)


class IncidentPromptEnvelopePlannerTests(unittest.TestCase):
    @staticmethod
    def _protected_blocks():
        return (
            ProtectedPromptBlock(
                ALERT_CONTRACT_BLOCK,
                "ALERT JSON CONTRACT -- preserve exactly",
            ),
            ProtectedPromptBlock(
                BATCH_STATE_BLOCK,
                "BATCH_STATE_JSON={\"routine\":false}",
            ),
        )

    def test_prompt_envelope_preserves_contract_and_accounts_every_budget(self):
        planner = IncidentPromptEnvelopePlanner()
        candidates = [
            IncidentAttentionCandidate(
                incident_id=f"incident-{index}",
                context=json.dumps(
                    {
                        "title": f"incident {index}",
                        "possible_start_ms": 100 + index,
                        "timeline": [
                            {"timestamp_ms": 100 + index, "label": "observed"}
                        ],
                    }
                ),
                updated_at_ms=index,
            )
            for index in range(3)
        ]
        budget = PromptEnvelopeBudget(
            context_window_tokens=1_400,
            max_text_tokens=1_000,
            max_vision_tokens=200,
            max_output_tokens=100,
            max_incident_tokens=500,
        )

        plan = planner.plan(
            candidates,
            protected_blocks=self._protected_blocks(),
            budget=budget,
            vision_tokens=180,
            output_tokens=90,
        )

        self.assertEqual(
            tuple(block.text for block in plan.protected_blocks),
            tuple(block.text for block in self._protected_blocks()),
        )
        self.assertLessEqual(plan.text_tokens_used, budget.max_text_tokens)
        self.assertLessEqual(plan.incident_tokens_used, budget.max_incident_tokens)
        self.assertLessEqual(plan.vision_tokens, budget.max_vision_tokens)
        self.assertLessEqual(plan.output_tokens, budget.max_output_tokens)
        self.assertLessEqual(
            plan.text_tokens_used + plan.vision_tokens + plan.output_tokens,
            budget.context_window_tokens,
        )
        self.assertEqual(
            plan.incident_tokens_used,
            sum(context.token_estimate for context in plan.incident_contexts),
        )
        for block in plan.protected_blocks:
            self.assertEqual(block.token_estimate, estimate_text_tokens(block.text))

    def test_large_context_is_semantically_compacted_not_blindly_truncated(self):
        planner = IncidentPromptEnvelopePlanner()
        marker = "SECRET-RAW-MARKER-"
        candidates = [
            IncidentAttentionCandidate(
                incident_id=f"incident-{index}",
                context=marker * 500,
                updated_at_ms=index,
            )
            for index in range(2)
        ]
        budget = PromptEnvelopeBudget(
            context_window_tokens=600,
            max_text_tokens=400,
            max_vision_tokens=100,
            max_output_tokens=50,
            max_incident_tokens=240,
        )

        plan = planner.plan(
            candidates,
            protected_blocks=self._protected_blocks(),
            budget=budget,
        )

        self.assertEqual(
            set(context.incident_id for context in plan.incident_contexts)
            | set(plan.omitted_incident_ids),
            {"incident-0", "incident-1"},
        )
        self.assertTrue(plan.incident_contexts)
        for context in plan.incident_contexts:
            self.assertNotEqual(context.compaction_tier, ContextCompactionTier.FULL)
            self.assertNotIn(marker, context.text)
            payload = json.loads(context.text)
            self.assertEqual(payload["incident_id"], context.incident_id)
            self.assertEqual(payload["attention_tier"], context.attention_tier.value)
            self.assertFalse(payload["resolution_inferred"])

    def test_parallel_incident_stubs_are_reserved_before_context_is_enriched(self):
        planner = IncidentPromptEnvelopePlanner(token_estimator=len)
        candidates = [
            IncidentAttentionCandidate(
                incident_id=f"incident-{index}",
                context=json.dumps(
                    {"title": f"incident {index}", "summary": "x" * 160}
                ),
                updated_at_ms=index,
            )
            for index in range(2)
        ]
        budget = PromptEnvelopeBudget(
            context_window_tokens=1_000,
            max_text_tokens=700,
            max_vision_tokens=100,
            max_output_tokens=100,
            max_incident_tokens=480,
        )

        plan = planner.plan(
            candidates,
            protected_blocks=(
                ProtectedPromptBlock(ALERT_CONTRACT_BLOCK, "A"),
                ProtectedPromptBlock(BATCH_STATE_BLOCK, "B"),
            ),
            budget=budget,
        )

        self.assertEqual(
            tuple(context.incident_id for context in plan.incident_contexts),
            ("incident-1", "incident-0"),
        )
        self.assertEqual(plan.omitted_incident_ids, ())
        self.assertTrue(
            all(
                context.compaction_tier is not ContextCompactionTier.FULL
                for context in plan.incident_contexts
            )
        )

    def test_only_four_incidents_enter_prompt_while_eight_remain_hot(self):
        planner = IncidentPromptEnvelopePlanner()
        candidates = [
            IncidentAttentionCandidate(
                incident_id=f"incident-{index}",
                context=json.dumps({"summary": f"incident {index}"}),
                updated_at_ms=index,
            )
            for index in range(10)
        ]

        plan = planner.plan(
            candidates,
            protected_blocks=self._protected_blocks(),
            budget=PromptEnvelopeBudget(
                context_window_tokens=4_000,
                max_text_tokens=3_000,
                max_vision_tokens=100,
                max_output_tokens=100,
                max_incident_tokens=2_000,
            ),
        )

        self.assertEqual(len(plan.allocation.hot_incident_ids), 8)
        self.assertEqual(len(plan.incident_contexts), 4)
        self.assertEqual(len(plan.omitted_incident_ids), 6)
        self.assertEqual(
            tuple(item.incident_id for item in plan.incident_contexts),
            plan.allocation.all_incident_ids[:4],
        )

    def test_protected_blocks_fail_atomically_instead_of_being_truncated(self):
        planner = IncidentPromptEnvelopePlanner(token_estimator=len)
        budget = PromptEnvelopeBudget(
            context_window_tokens=100,
            max_text_tokens=50,
            max_vision_tokens=20,
            max_output_tokens=20,
            max_incident_tokens=10,
        )
        protected = (
            ProtectedPromptBlock(ALERT_CONTRACT_BLOCK, "A" * 30),
            ProtectedPromptBlock(BATCH_STATE_BLOCK, "B" * 30),
        )

        with self.assertRaisesRegex(PromptBudgetError, "refusing truncation"):
            planner.plan([], protected_blocks=protected, budget=budget)

    def test_vision_and_output_requests_are_bounded(self):
        planner = IncidentPromptEnvelopePlanner()
        budget = PromptEnvelopeBudget(
            context_window_tokens=200,
            max_text_tokens=100,
            max_vision_tokens=50,
            max_output_tokens=40,
            max_incident_tokens=50,
        )

        with self.assertRaisesRegex(PromptBudgetError, "vision"):
            planner.plan(
                [],
                protected_blocks=self._protected_blocks(),
                budget=budget,
                vision_tokens=51,
            )
        with self.assertRaisesRegex(PromptBudgetError, "output"):
            planner.plan(
                [],
                protected_blocks=self._protected_blocks(),
                budget=budget,
                output_tokens=41,
            )


if __name__ == "__main__":
    unittest.main()
