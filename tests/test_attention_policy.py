import json
import unittest

from attention_policy import (
    AggregationConfig,
    ApexMarker,
    AttentionCandidate,
    AttentionCoordinator,
    AttentionMode,
    AttentionPolicyConfig,
    AttentionVector,
    CostBudgetConfig,
    CostBudgetState,
    CostedAttentionCandidate,
    CoordinatorConfig,
    CvSample,
    EmbeddingSnapshot,
    EpisodeRole,
    FrameSelectorConfig,
    GlobalBudgetConfig,
    HomeostaticAttentionPolicy,
    InferenceCost,
    ModelFrameCandidate,
    PORT_EIGHT_CHANNEL_PRESET,
    ProbeScore,
    aggregate_cv_intervals,
    allocate_cost_aware_attention,
    allocate_global_attention,
    build_attention_episode,
    profile_for_mode,
    select_model_frames,
)


def _snapshot(channel: str, second: int, *, probe: bool = False) -> EmbeddingSnapshot:
    scores = (
        (ProbeScore("person", 0.72, 0.21, 0.51, "v1"),)
        if probe
        else ()
    )
    return EmbeddingSnapshot(
        channel_id=channel,
        snapshot_id=f"{channel}-{second}",
        timestamp_ms=second * 1_000,
        embedding_ref=f"embedding://{channel}/{second}",
        frame_hash=f"hash-{second}",
        probe_scores=scores,
    )


class CvAggregationTests(unittest.TestCase):
    def test_dense_cv_is_reduced_and_linked_to_saved_snapshots_and_apex(self):
        samples = [
            CvSample(0, 0.05),
            CvSample(250, 0.10),
            CvSample(500, 0.12),
            CvSample(750, 0.08),
            CvSample(1_000, 0.42),
            CvSample(1_250, 0.61),
            CvSample(1_500, 0.88),
            CvSample(1_750, 0.55),
            CvSample(2_000, 0.15),
            CvSample(2_250, 0.08),
            CvSample(2_500, 0.05),
        ]
        snapshots = [_snapshot("7", second) for second in range(3)]

        intervals = aggregate_cv_intervals(
            "7",
            samples,
            snapshots,
            [ApexMarker("burst-1", 1_520)],
            AggregationConfig(
                motion_enter=0.35,
                motion_exit=0.20,
                snapshot_link_tolerance_ms=0,
                apex_link_tolerance_ms=600,
            ),
        )

        self.assertEqual([item.kind.value for item in intervals], ["quiet", "motion", "quiet"])
        self.assertEqual([item.sample_count for item in intervals], [4, 4, 3])
        self.assertEqual(intervals[1].peak_timestamp_ms, 1_500)
        self.assertEqual(intervals[1].linked_snapshot_ids, ("7-1",))
        self.assertEqual(intervals[1].apex_links[0].snapshot_id, "7-1")
        self.assertEqual(intervals[1].apex_links[0].distance_ms, 520)

        payload = json.loads(intervals[1].printable())
        self.assertNotIn("image", payload)
        self.assertNotIn("jpeg", payload)
        self.assertEqual(payload["apex_links"][0]["apex_id"], "burst-1")

    def test_large_cv_gap_is_not_misrepresented_as_quiet(self):
        intervals = aggregate_cv_intervals(
            "7",
            [CvSample(0, 0.8), CvSample(250, 0.7), CvSample(5_000, 0.1)],
            config=AggregationConfig(max_sample_gap_ms=1_000),
        )
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0].kind.value, "motion")
        self.assertEqual(intervals[0].end_ms, 250)
        self.assertEqual(intervals[1].start_ms, 5_000)

    def test_snapshot_type_has_no_image_payload_escape_hatch(self):
        with self.assertRaises(TypeError):
            EmbeddingSnapshot(
                channel_id="7",
                snapshot_id="7-1",
                timestamp_ms=1_000,
                embedding_ref="embedding://7/1",
                image=b"pixels",
            )


class HomeostaticPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = HomeostaticAttentionPolicy(
            AttentionPolicyConfig(
                quiet_target_interval_ms=100_000,
                watch_target_interval_ms=50_000,
                active_target_interval_ms=10_000,
                min_mode_dwell_ms=2_000,
                burst_min_dwell_ms=1_000,
                burst_cooldown_ms=10_000,
            )
        )

    def test_mode_hysteresis_burst_cooldown_and_degraded_recovery(self):
        first = self.policy.evaluate(
            "7", AttentionVector(0, motion_intensity=0.05), last_vlm_ms=0
        )
        self.assertEqual(first.mode, AttentionMode.QUIET)

        active = self.policy.evaluate(
            "7",
            AttentionVector(
                3_000,
                motion_intensity=0.9,
                motion_persistence=0.8,
                probe_positive=0.8,
            ),
            first.state,
        )
        self.assertEqual(active.mode, AttentionMode.ACTIVE)

        held = self.policy.evaluate(
            "7",
            AttentionVector(4_000, motion_intensity=0.45, motion_persistence=0.4),
            active.state,
        )
        self.assertEqual(held.mode, AttentionMode.ACTIVE)
        self.assertIn("hold_hysteresis", held.reasons[0])

        burst = self.policy.evaluate(
            "7", AttentionVector(6_000, burst=0.95), held.state
        )
        self.assertEqual(burst.mode, AttentionMode.BURST)
        self.assertEqual(burst.state.cooldown_until_ms, 16_000)

        after_burst = self.policy.evaluate(
            "7", AttentionVector(8_000, motion_intensity=0.65), burst.state
        )
        self.assertEqual(after_burst.mode, AttentionMode.WATCH)

        refractory = self.policy.evaluate(
            "7", AttentionVector(9_000, burst=0.95), after_burst.state
        )
        self.assertNotEqual(refractory.mode, AttentionMode.BURST)

        degraded = self.policy.evaluate(
            "7", AttentionVector(20_000, source_health=0.2), refractory.state
        )
        self.assertEqual(degraded.mode, AttentionMode.DEGRADED)
        self.assertEqual(degraded.priority, 0.0)

        held_degraded = self.policy.evaluate(
            "7", AttentionVector(25_000, source_health=0.9), degraded.state
        )
        self.assertEqual(held_degraded.mode, AttentionMode.DEGRADED)

        recovered = self.policy.evaluate(
            "7", AttentionVector(31_000, source_health=0.9), held_degraded.state
        )
        self.assertEqual(recovered.mode, AttentionMode.QUIET)

    def test_coverage_debt_and_printing_are_replayable(self):
        decision = self.policy.evaluate(
            "7",
            AttentionVector(
                250_000,
                signal_staleness=0.9,
                uncertainty=0.4,
            ),
            last_vlm_ms=0,
        )
        self.assertEqual(decision.coverage_debt, 2.0)
        self.assertEqual(decision.printable(), decision.printable())
        payload = json.loads(decision.printable())
        self.assertEqual(payload["coverage_debt"], 2.0)
        self.assertEqual(payload["vector"]["signal_staleness"], 0.9)
        self.assertTrue(payload["reasons"])

        dispatched = self.policy.record_dispatch(decision.state, 250_100)
        self.assertEqual(dispatched.last_vlm_ms, 250_100)
        self.assertEqual(dispatched.coverage_debt, 0.0)


class GlobalBudgetTests(unittest.TestCase):
    def _decision(
        self,
        channel: int,
        *,
        now: int,
        last_vlm: int,
        motion: float = 0.0,
        burst: float = 0.0,
    ):
        return HomeostaticAttentionPolicy(
            AttentionPolicyConfig(
                quiet_target_interval_ms=100_000,
                active_target_interval_ms=10_000,
            )
        ).evaluate(
            str(channel),
            AttentionVector(now, motion_intensity=motion, burst=burst),
            last_vlm_ms=last_vlm,
        )

    def test_global_budget_reserves_fairness_across_sixteen_channels(self):
        now = 1_000_000
        candidates = []
        for channel in range(16):
            if channel == 15:
                decision = self._decision(
                    channel, now=now, last_vlm=0, motion=0.0
                )
            elif channel == 0:
                decision = self._decision(
                    channel, now=now, last_vlm=995_000, burst=1.0
                )
            else:
                decision = self._decision(
                    channel, now=now, last_vlm=995_000, motion=0.7
                )
            candidates.append(AttentionCandidate(str(channel), decision))

        allocation = allocate_global_attention(
            candidates,
            now,
            GlobalBudgetConfig(
                total_units=2.0,
                max_jobs=2,
                fairness_fraction=0.5,
                urgent_priority=0.55,
            ),
        )

        self.assertEqual(
            [(item.channel_id, item.phase) for item in allocation.selected],
            [("0", "urgent"), ("15", "fairness")],
        )
        self.assertEqual(allocation.used_units, 2.0)
        self.assertEqual(len(allocation.rejected), 14)
        self.assertEqual(allocation.printable(), allocation.printable())

    def test_degraded_and_not_ready_candidates_are_excluded_with_reasons(self):
        policy = HomeostaticAttentionPolicy()
        degraded = policy.evaluate(
            "bad", AttentionVector(10_000, source_health=0.1)
        )
        ready_later = policy.evaluate("later", AttentionVector(10_000))
        allocation = allocate_global_attention(
            [
                AttentionCandidate("bad", degraded),
                AttentionCandidate("later", ready_later, ready_at_ms=20_000),
            ],
            10_000,
            GlobalBudgetConfig(total_units=2),
        )
        self.assertEqual(allocation.selected, ())
        self.assertEqual(
            dict(allocation.rejected),
            {
                "bad": "degraded_source",
                "later": "not_ready_until:20000",
            },
        )


class EpisodeSelectionTests(unittest.TestCase):
    def test_episode_roles_reference_only_persisted_embedding_snapshots(self):
        snapshots = [_snapshot("7", second, probe=second == 6) for second in range(11)]
        samples = []
        for index in range(41):
            timestamp = index * 250
            motion = (
                0.05
                if timestamp < 4_000
                else 0.85
                if timestamp < 7_000
                else 0.10
            )
            samples.append(CvSample(timestamp, motion))
        intervals = aggregate_cv_intervals(
            "7",
            samples,
            snapshots,
            [ApexMarker("event-apex", 6_100)],
            AggregationConfig(
                snapshot_link_tolerance_ms=0,
                apex_link_tolerance_ms=1_000,
            ),
        )

        episode = build_attention_episode("7", 6_000, intervals, snapshots)
        roles = {role for frame in episode.frames for role in frame.roles}
        self.assertEqual(
            roles,
            {
                EpisodeRole.CONTROL,
                EpisodeRole.PRE,
                EpisodeRole.ONSET,
                EpisodeRole.APEX,
                EpisodeRole.POST,
            },
        )
        stored_ids = {snapshot.snapshot_id for snapshot in snapshots}
        self.assertTrue(
            all(frame.snapshot_id in stored_ids for frame in episode.frames)
        )
        apex = next(
            frame for frame in episode.frames if EpisodeRole.APEX in frame.roles
        )
        self.assertEqual(apex.snapshot_id, "7-6")
        self.assertEqual(apex.probe_scores[0].margin, 0.51)

        payload = json.loads(episode.printable())
        self.assertTrue(payload["frames"])
        self.assertTrue(
            all("embedding_ref" in frame for frame in payload["frames"])
        )
        self.assertFalse(any("image" in frame for frame in payload["frames"]))

    def test_episode_without_nearby_motion_is_explicitly_empty(self):
        snapshots = [_snapshot("7", second) for second in range(3)]
        intervals = aggregate_cv_intervals(
            "7",
            [CvSample(0, 0.8), CvSample(250, 0.7)],
            snapshots,
        )
        episode = build_attention_episode(
            "7", 20_000, intervals, snapshots
        )
        self.assertEqual(episode.frames, ())
        self.assertEqual(
            episode.reasons, ("no_motion_interval_near_trigger",)
        )


class CoordinatorApiTests(unittest.TestCase):
    def test_high_level_observe_plan_dispatch_complete_and_status_flow(self):
        coordinator = AttentionCoordinator(
            CoordinatorConfig(
                attention=AttentionPolicyConfig(
                    quiet_target_interval_ms=5_000,
                    burst_min_dwell_ms=500,
                ),
                burst_postroll_wait_ms=1_000,
                minimum_dispatch_gap_ms=500,
                control_audit_lookback_ms=10_000,
                budget=GlobalBudgetConfig(
                    total_units=2,
                    max_jobs=2,
                    fairness_fraction=0.5,
                ),
            )
        )
        for timestamp in range(0, 6_001, 250):
            if timestamp % 1_000 == 0:
                coordinator.link_snapshot(
                    "7",
                    timestamp,
                    f"embedding://7/{timestamp}",
                    meta={
                        "snapshot_id": f"s-{timestamp}",
                        "frame_hash": f"h-{timestamp}",
                        "probe_scores": [
                            {
                                "probe_id": "person",
                                "positive": 0.7,
                                "negative": 0.2,
                                "margin": 0.5,
                            }
                        ],
                    },
                )
            moving = 2_000 <= timestamp < 4_000
            coordinator.observe_cv(
                "7",
                timestamp,
                0.85 if moving else 0.05,
                4.0 if moving else 0.7,
                "burst" if moving else "quiet",
            )
            if timestamp == 3_000:
                coordinator.link_apex("7", timestamp, "apex-7")

        readiness = coordinator.should_dispatch("7", 6_000)
        self.assertTrue(readiness.due)
        plan = coordinator.plan_due(6_000)
        self.assertEqual(len(plan.jobs), 1)
        job = plan.jobs[0]
        roles = {role for frame in job.episode.frames for role in frame.roles}
        self.assertIn(EpisodeRole.APEX, roles)
        self.assertTrue(
            all(frame.embedding_ref.startswith("embedding://") for frame in job.episode.frames)
        )

        job_id = coordinator.mark_dispatched("7", 6_000, job.episode_id)
        self.assertFalse(coordinator.should_dispatch("7", 6_100).due)
        coordinator.mark_completed("7", 6_500, job_id=job_id)
        status = coordinator.status("7")
        self.assertEqual(status["snapshots"], 7)
        self.assertEqual(status["inflight_job_id"], None)
        self.assertTrue(status["last_completion_success"])
        self.assertEqual(plan.printable(), plan.printable())

    def test_coordinator_rejects_image_payload_metadata(self):
        coordinator = AttentionCoordinator()
        with self.assertRaisesRegex(ValueError, "image payload"):
            coordinator.link_snapshot(
                "7",
                1_000,
                "embedding://7/1",
                meta={"jpeg": "base64-data"},
            )

    def test_coordinator_enforces_sixteen_channel_bound(self):
        coordinator = AttentionCoordinator(CoordinatorConfig(max_channels=16))
        for channel in range(16):
            coordinator.observe_cv(str(channel), 0, 0.0, 0.0, "quiet")
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            coordinator.observe_cv("16", 0, 0.0, 0.0, "quiet")
        self.assertEqual(coordinator.status()["channel_count"], 16)


class PortPresetTests(unittest.TestCase):
    def test_exact_mode_contract_and_independent_one_hz_embeddings(self):
        expected = {
            AttentionMode.QUIET: (10_000, 120_000, 6, 8, 8, True),
            AttentionMode.WATCH: (5_000, 90_000, 6, 8, 10, True),
            AttentionMode.ACTIVE: (2_500, 60_000, 8, 12, 12, True),
            AttentionMode.BURST: (1_000, 30_000, 10, 16, 16, True),
            AttentionMode.DEGRADED: (15_000, 120_000, 4, 6, 6, True),
        }
        self.assertEqual(PORT_EIGHT_CHANNEL_PRESET.channel_limit, 8)
        self.assertEqual(PORT_EIGHT_CHANNEL_PRESET.steady_l0_per_minute, 6.0)
        for mode, values in expected.items():
            profile = profile_for_mode(mode)
            self.assertEqual(
                (
                    profile.cadence_ms,
                    profile.deadline_ms,
                    profile.min_frames,
                    profile.target_frames,
                    profile.max_frames,
                    profile.dispatch_enabled,
                ),
                values,
            )
            self.assertEqual(profile.embedding_cadence_ms, 1_000)
            self.assertEqual(profile.hard_accumulator_cap, 16)

    def test_hard_cap_and_deadline_force_dispatch_in_every_mode(self):
        for mode in AttentionMode:
            profile = profile_for_mode(mode)
            self.assertTrue(
                profile.should_force_dispatch(
                    accumulator_size=16,
                    now_ms=1_001,
                    last_dispatch_ms=1_000,
                )
            )
            self.assertFalse(
                profile.should_force_dispatch(
                    accumulator_size=15,
                    now_ms=profile.deadline_ms - 1,
                    last_dispatch_ms=0,
                )
            )
            self.assertTrue(
                profile.should_force_dispatch(
                    accumulator_size=4,
                    now_ms=profile.deadline_ms,
                    last_dispatch_ms=0,
                )
            )


def _model_candidate(
    index: int,
    *,
    roles=(),
    vector=None,
    tokens: int = 500,
    salience: float = 0.5,
    motion: float = 0.3,
    probe: float = 0.2,
) -> ModelFrameCandidate:
    embedding = vector if vector is not None else (1.0, index + 1.0, 0.5)
    return ModelFrameCandidate(
        channel_id="7",
        snapshot_id=f"s-{index:02d}",
        timestamp_ms=index * 1_000,
        embedding_ref=f"embedding://7/{index}",
        frame_hash=f"hash-{index}",
        roles=roles,
        motion_score=motion,
        probe_score=probe,
        salience=salience,
        sharpness_score=0.5,
        estimated_tokens=tokens,
        embedding=embedding,
    )


def _event_candidates(count: int = 20, *, tokens: int = 500):
    role_map = {
        0: (EpisodeRole.CONTROL,),
        4: (EpisodeRole.PRE,),
        5: (EpisodeRole.ONSET,),
        8: (EpisodeRole.APEX,),
        12: (EpisodeRole.POST,),
    }
    return [
        _model_candidate(
            index,
            roles=role_map.get(index, ()),
            tokens=tokens,
            vector=(
                1.0 if index % 3 == 0 else 0.0,
                1.0 if index % 3 == 1 else 0.0,
                1.0 if index % 3 == 2 else 0.0,
                index / 20.0,
            ),
            salience=min(1.0, 0.3 + index / 30.0),
            motion=min(1.0, index / 20.0),
        )
        for index in range(count)
    ]


class ModelFrameSelectorTests(unittest.TestCase):
    def test_burst_selects_mandatory_anchors_chronologically_and_trims_tokens(self):
        selection = select_model_frames(
            "7",
            _event_candidates(),
            mode=AttentionMode.BURST,
            token_budget=6_000,
        )
        self.assertTrue(selection.preflight_ok, selection.reasons)
        self.assertEqual(len(selection.frames), 12)
        self.assertEqual(selection.estimated_tokens, 6_000)
        self.assertEqual(len(selection.trimmed_snapshot_ids), 4)
        self.assertEqual(
            [frame.timestamp_ms for frame in selection.frames],
            sorted(frame.timestamp_ms for frame in selection.frames),
        )
        roles = {role for frame in selection.frames for role in frame.roles}
        self.assertTrue(
            {
                EpisodeRole.CONTROL,
                EpisodeRole.PRE,
                EpisodeRole.ONSET,
                EpisodeRole.APEX,
                EpisodeRole.POST,
                EpisodeRole.CURRENT,
            }.issubset(roles)
        )

    def test_quiet_and_degraded_need_control_current_not_fake_event_roles(self):
        candidates = [
            _model_candidate(
                index,
                roles=(EpisodeRole.CONTROL,) if index == 0 else (),
            )
            for index in range(8)
        ]
        quiet = select_model_frames(
            "7", candidates, mode="quiet", token_budget=4_000
        )
        self.assertTrue(quiet.preflight_ok, quiet.reasons)
        self.assertEqual(quiet.missing_roles, ())
        self.assertEqual(len(quiet.frames), 8)

        degraded = select_model_frames(
            "7", candidates, mode="degraded", token_budget=2_000
        )
        self.assertTrue(degraded.preflight_ok, degraded.reasons)
        self.assertEqual(len(degraded.frames), 4)

    def test_watch_preserves_present_event_anchors_without_requiring_missing_ones(self):
        candidates = [
            _model_candidate(
                index,
                roles=(EpisodeRole.CONTROL,)
                if index == 0
                else (EpisodeRole.PRE,)
                if index == 3
                else (EpisodeRole.POST,)
                if index == 6
                else (),
            )
            for index in range(10)
        ]
        selection = select_model_frames(
            "7", candidates, mode="watch", token_budget=5_000
        )
        self.assertTrue(selection.preflight_ok, selection.reasons)
        roles = {role for frame in selection.frames for role in frame.roles}
        self.assertIn(EpisodeRole.PRE, roles)
        self.assertIn(EpisodeRole.POST, roles)
        self.assertNotIn(EpisodeRole.APEX, selection.missing_roles)

    def test_active_missing_apex_fails_preflight_instead_of_fabricating_it(self):
        candidates = [
            candidate
            for candidate in _event_candidates(14)
            if EpisodeRole.APEX not in candidate.roles
        ]
        selection = select_model_frames(
            "7", candidates, mode="active", token_budget=7_000
        )
        self.assertFalse(selection.preflight_ok)
        self.assertIn(EpisodeRole.APEX, selection.missing_roles)

    def test_redundancy_penalty_prefers_a_diverse_optional_frame(self):
        candidates = _event_candidates(13)
        apex = next(
            candidate for candidate in candidates if EpisodeRole.APEX in candidate.roles
        )
        duplicate = _model_candidate(
            6,
            vector=apex.embedding,
            salience=0.75,
            motion=0.4,
        )
        diverse = _model_candidate(
            7,
            vector=(0.0, 0.0, 0.0, 1.0),
            salience=1.0,
            motion=1.0,
        )
        candidates[6] = duplicate
        candidates[7] = diverse
        selection = select_model_frames(
            "7",
            candidates,
            mode="quiet",
            token_budget=4_000,
            config=FrameSelectorConfig(redundancy_weight=1.0),
        )
        ids = {frame.snapshot_id for frame in selection.frames}
        self.assertIn(diverse.snapshot_id, ids)
        self.assertNotIn(duplicate.snapshot_id, ids)

    def test_mandatory_anchor_token_overflow_is_explicit(self):
        selection = select_model_frames(
            "7",
            _event_candidates(14, tokens=1_000),
            mode="active",
            token_budget=5_000,
        )
        self.assertFalse(selection.preflight_ok)
        self.assertGreater(selection.estimated_tokens, selection.token_budget)
        self.assertIn(
            "mandatory_anchors_exceed_token_budget", selection.reasons
        )


class CostAwareBudgetTests(unittest.TestCase):
    def _decision(self, channel: str, mode: AttentionMode, debt: float = 0.0):
        policy = HomeostaticAttentionPolicy(
            AttentionPolicyConfig(
                quiet_target_interval_ms=100_000,
                active_target_interval_ms=10_000,
            )
        )
        now = 1_000_000
        if mode is AttentionMode.BURST:
            vector = AttentionVector(now, burst=1.0)
        elif mode is AttentionMode.ACTIVE:
            vector = AttentionVector(
                now,
                motion_intensity=1.0,
                motion_persistence=1.0,
                probe_positive=1.0,
            )
        else:
            vector = AttentionVector(now)
        target = (
            5_000
            if mode is AttentionMode.BURST
            else 10_000
            if mode is AttentionMode.ACTIVE
            else 100_000
        )
        last_vlm = int(now - debt * target)
        return policy.evaluate(channel, vector, last_vlm_ms=last_vlm)

    def test_steady_bucket_admits_six_reference_l0_requests_per_minute(self):
        config = CostBudgetConfig(
            reference_l0_tokens=100,
            reference_l0_slot_seconds=10,
        )
        state = CostBudgetState(1_000_000, 600, 60)
        candidates = [
            CostedAttentionCandidate(
                str(index),
                self._decision(str(index), AttentionMode.QUIET, debt=0.5),
                InferenceCost(40, 40, 20, 10),
            )
            for index in range(7)
        ]
        allocation = allocate_cost_aware_attention(
            candidates, 1_000_000, state, config
        )
        self.assertEqual(len(allocation.selected), 6)
        self.assertEqual(len(allocation.rejected), 1)
        self.assertEqual(allocation.state_after.available_tokens, 0)
        self.assertEqual(allocation.state_after.available_slot_seconds, 0)

    def test_coverage_fairness_is_reserved_before_high_priority_work(self):
        config = CostBudgetConfig(
            reference_l0_tokens=100,
            reference_l0_slot_seconds=10,
            fairness_fraction=0.5,
            max_jobs_per_cycle=2,
        )
        overdue = CostedAttentionCandidate(
            "quiet-overdue",
            self._decision("quiet-overdue", AttentionMode.QUIET, debt=2.0),
            InferenceCost(20, 20, 10, 5),
        )
        active = CostedAttentionCandidate(
            "active",
            self._decision("active", AttentionMode.ACTIVE, debt=0.2),
            InferenceCost(20, 20, 10, 5),
        )
        allocation = allocate_cost_aware_attention(
            [active, overdue],
            1_000_000,
            CostBudgetState(1_000_000, 100, 10),
            config,
        )
        self.assertEqual(
            {entry.channel_id for entry in allocation.selected},
            {"quiet-overdue", "active"},
        )
        phases = {entry.channel_id: entry.phase for entry in allocation.selected}
        self.assertEqual(phases["quiet-overdue"], "fairness")

    def test_burst_borrows_then_refill_repays_token_and_slot_debt(self):
        config = CostBudgetConfig(
            reference_l0_tokens=100,
            reference_l0_slot_seconds=10,
            burst_borrow_l0=2,
            fairness_fraction=0.5,
        )
        overdue = CostedAttentionCandidate(
            "quiet",
            self._decision("quiet", AttentionMode.QUIET, debt=2),
            InferenceCost(20, 20, 10, 5),
        )
        burst = CostedAttentionCandidate(
            "burst",
            self._decision("burst", AttentionMode.BURST, debt=0.2),
            InferenceCost(80, 80, 40, 20),
        )
        allocation = allocate_cost_aware_attention(
            [burst, overdue],
            1_000_000,
            CostBudgetState(1_000_000, 100, 10),
            config,
        )
        self.assertEqual(
            [entry.phase for entry in allocation.selected],
            ["urgent", "fairness"],
        )
        self.assertEqual(allocation.state_after.available_tokens, -150)
        self.assertEqual(allocation.state_after.available_slot_seconds, -15)
        self.assertEqual(allocation.state_after.debt_l0(config), 1.5)

        repayment = allocate_cost_aware_attention(
            [],
            1_015_000,
            allocation.state_after,
            config,
        )
        self.assertEqual(repayment.repaid_tokens, 150)
        self.assertEqual(repayment.repaid_slot_seconds, 15)
        self.assertEqual(repayment.state_after.debt_l0(config), 0)

    def test_slot_seconds_can_be_limiting_even_when_tokens_fit(self):
        config = CostBudgetConfig(
            reference_l0_tokens=100,
            reference_l0_slot_seconds=10,
        )
        candidates = [
            CostedAttentionCandidate(
                str(index),
                self._decision(str(index), AttentionMode.QUIET, debt=0.2),
                InferenceCost(20, 20, 10, 20),
            )
            for index in range(4)
        ]
        allocation = allocate_cost_aware_attention(
            candidates,
            1_000_000,
            CostBudgetState(1_000_000, 600, 60),
            config,
        )
        self.assertEqual(len(allocation.selected), 3)
        self.assertEqual(
            dict(allocation.rejected)["3"], "slot_seconds_budget_exhausted"
        )


if __name__ == "__main__":
    unittest.main()
