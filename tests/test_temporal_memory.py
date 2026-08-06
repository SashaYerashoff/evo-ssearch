from __future__ import annotations

import pytest

from temporal_memory import (
    AttentionBudget,
    ChildDisposition,
    ChildDispositionKind,
    EpisodeStatus,
    ObservationKind,
    ObservationState,
    build_series,
    complete_child_dispositions,
    make_observation,
    segment_observations,
    select_attention_observations,
)


BASE_MS = 1_800_000_000_000
MINUTE_MS = 60_000


def event(
    batch: str,
    timestamp_ms: int,
    semantic_key: str,
    label: str,
    state: ObservationState = ObservationState.CONTINUING,
    *,
    ordinal: int = 0,
    evidence_refs: tuple[str, ...] = (),
):
    return make_observation(
        channel_id=7,
        source_batch_id=batch,
        ordinal=ordinal,
        kind=ObservationKind.EVENT,
        state=state,
        semantic_key=semantic_key,
        label=label,
        start_ms=timestamp_ms,
        evidence_refs=evidence_refs,
    )


def gap(
    batch: str,
    kind: ObservationKind,
    start_ms: int,
    end_ms: int,
    *,
    applies_to: tuple[str, ...] = (),
):
    return make_observation(
        channel_id=7,
        source_batch_id=batch,
        ordinal=0,
        kind=kind,
        start_ms=start_ms,
        end_ms=end_ms,
        applies_to=applies_to,
    )


def test_long_fire_and_two_frame_pickpocket_remain_parallel_episodes():
    observations = [
        event(
            "fire-00",
            BASE_MS,
            "fire:north-hall",
            "Fire in north hall",
            ObservationState.NEW,
        ),
        event(
            "fire-10",
            BASE_MS + 10 * MINUTE_MS,
            "fire:north-hall",
            "Fire in north hall",
        ),
        event(
            "pick-1",
            BASE_MS + 12 * MINUTE_MS,
            "pickpocket:entrance",
            "Pickpocket takes a bag",
            ObservationState.NEW,
        ),
        event(
            "pick-2",
            BASE_MS + 12 * MINUTE_MS + 2_000,
            "pickpocket:entrance",
            "Pickpocket takes a bag",
            ObservationState.RESOLVED,
        ),
        event(
            "fire-20",
            BASE_MS + 20 * MINUTE_MS,
            "fire:north-hall",
            "Fire in north hall",
        ),
        event(
            "fire-30",
            BASE_MS + 30 * MINUTE_MS,
            "fire:north-hall",
            "Fire in north hall",
        ),
        event(
            "fire-40",
            BASE_MS + 40 * MINUTE_MS,
            "fire:north-hall",
            "Fire in north hall",
        ),
        event(
            "fire-47",
            BASE_MS + 47 * MINUTE_MS,
            "fire:north-hall",
            "Fire in north hall",
        ),
    ]

    result = segment_observations(reversed(observations))

    assert len(result.episodes) == 2
    assert result.to_dict() == segment_observations(observations).to_dict()
    by_key = {episode.semantic_key: episode for episode in result.episodes}
    fire = by_key["fire:north-hall"]
    pickpocket = by_key["pickpocket:entrance"]
    assert fire.status is EpisodeStatus.OPEN
    assert fire.observed_span_ms == 47 * MINUTE_MS
    assert len(fire.observation_ids) == 6
    assert pickpocket.status is EpisodeStatus.RESOLVED
    assert pickpocket.observed_span_ms == 2_000
    assert len(pickpocket.observation_ids) == 2
    assert fire.episode_id != pickpocket.episode_id
    assert len(result.dispositions) == len(observations)


def test_new_resolved_routine_new_produces_two_episodes():
    first = event(
        "first-new",
        BASE_MS,
        "fire:north-hall",
        "Fire in north hall",
        ObservationState.NEW,
    )
    resolved = event(
        "first-resolved",
        BASE_MS + MINUTE_MS,
        "fire:north-hall",
        "Fire is no longer visible",
        ObservationState.RESOLVED,
    )
    routine = gap(
        "routine",
        ObservationKind.ROUTINE_GAP,
        BASE_MS + 2 * MINUTE_MS,
        BASE_MS + 2 * MINUTE_MS,
        applies_to=("fire:north-hall",),
    )
    second = event(
        "second-new",
        BASE_MS + 3 * MINUTE_MS,
        "fire:north-hall",
        "Fire in north hall",
        ObservationState.NEW,
    )

    result = segment_observations([first, resolved, routine, second])

    assert len(result.episodes) == 2
    assert result.episodes[0].status is EpisodeStatus.RESOLVED
    assert result.episodes[1].status is EpisodeStatus.OPEN
    assert result.episodes[0].episode_id != result.episodes[1].episode_id
    dispositions = {item.child_id: item.disposition for item in result.dispositions}
    assert dispositions[resolved.observation_id] is ChildDispositionKind.RESOLVED_EPISODE
    assert dispositions[routine.observation_id] is ChildDispositionKind.ROUTINE_GAP
    assert dispositions[second.observation_id] is ChildDispositionKind.STARTED_EPISODE


def test_coverage_gap_does_not_become_a_routine_boundary():
    first = event(
        "new",
        BASE_MS,
        "person-down:lobby",
        "Person lies on the floor",
        ObservationState.NEW,
    )
    unavailable = gap(
        "camera-offline",
        ObservationKind.COVERAGE_GAP,
        BASE_MS + 5_000,
        BASE_MS + 120_000,
        applies_to=("person-down:lobby",),
    )
    continuing = event(
        "visible-again",
        BASE_MS + 121_000,
        "person-down:lobby",
        "Person lies on the floor",
    )

    result = segment_observations(
        [first, unavailable, continuing], max_observed_gap_ms=10_000
    )

    assert len(result.episodes) == 1
    episode = result.episodes[0]
    assert episode.status is EpisodeStatus.OPEN
    assert episode.observation_ids == (
        first.observation_id,
        continuing.observation_id,
    )
    assert episode.coverage_gap_ids == (unavailable.observation_id,)
    disposition = next(
        item for item in result.dispositions if item.child_id == unavailable.observation_id
    )
    assert disposition.disposition is ChildDispositionKind.COVERAGE_GAP_KEEP

    observed_routine = gap(
        "observed-routine",
        ObservationKind.ROUTINE_GAP,
        BASE_MS + 5_000,
        BASE_MS + 120_000,
        applies_to=("person-down:lobby",),
    )
    routine_result = segment_observations(
        [first, observed_routine, continuing], max_observed_gap_ms=10_000
    )
    assert len(routine_result.episodes) == 2
    assert routine_result.episodes[0].status is EpisodeStatus.ENDED_BY_ROUTINE
    assert routine_result.episodes[1].status is EpisodeStatus.OPEN


def test_observed_gap_splits_an_episode_without_calling_it_routine():
    first = event(
        "door-first",
        BASE_MS,
        "forced-door:west",
        "Forced door opening",
        ObservationState.NEW,
    )
    much_later = event(
        "door-later",
        BASE_MS + 30_000,
        "forced-door:west",
        "Forced door opening",
    )

    result = segment_observations(
        [first, much_later], max_observed_gap_ms=10_000
    )

    assert len(result.episodes) == 2
    assert result.episodes[0].status is EpisodeStatus.ENDED_BY_OBSERVED_GAP
    assert result.episodes[0].boundary_observation_id is None
    assert result.episodes[1].status is EpisodeStatus.OPEN


def test_same_label_can_have_distinct_episode_ids_and_one_explicit_series():
    first = event(
        "bag-1",
        BASE_MS,
        "unattended-bag:lobby",
        "Unattended bag",
        ObservationState.NEW,
    )
    routine = gap(
        "bag-routine",
        ObservationKind.ROUTINE_GAP,
        BASE_MS + MINUTE_MS,
        BASE_MS + MINUTE_MS,
        applies_to=("unattended-bag:lobby",),
    )
    second = event(
        "bag-2",
        BASE_MS + 2 * MINUTE_MS,
        "unattended-bag:lobby",
        "Unattended bag",
        ObservationState.NEW,
    )
    episodes = segment_observations([first, routine, second]).episodes

    series = build_series(episodes)
    replayed = build_series(reversed(episodes))

    assert len(episodes) == 2
    assert episodes[0].label == episodes[1].label
    assert episodes[0].episode_id != episodes[1].episode_id
    assert series.episode_ids == tuple(item.episode_id for item in episodes)
    assert replayed == series


def test_server_owned_ids_are_deterministic_and_ignore_model_wording():
    original = event(
        "stable-batch",
        BASE_MS,
        "smoke:north-hall",
        "Smoke in north hall",
        ObservationState.NEW,
    )
    reworded = event(
        "stable-batch",
        BASE_MS,
        "smoke:north-hall",
        "Visible smoke near the north hall ceiling",
        ObservationState.NEW,
    )

    original_result = segment_observations([original])
    reworded_result = segment_observations([reworded])

    assert original.observation_id == reworded.observation_id
    assert original_result.episodes[0].episode_id == reworded_result.episodes[0].episode_id
    assert original_result.to_dict() == segment_observations([original]).to_dict()


def test_total_dispositions_fill_unknown_children_with_unclassified_keep():
    proposed = ChildDisposition(
        child_id="child-1",
        disposition=ChildDispositionKind.CONTINUED_EPISODE,
        episode_id="ep-1",
    )

    completed = complete_child_dispositions(
        ["child-1", "child-2", "child-3"], [proposed]
    )

    assert [item.child_id for item in completed] == ["child-1", "child-2", "child-3"]
    assert completed[0] == proposed
    assert completed[1].disposition is ChildDispositionKind.UNCLASSIFIED_KEEP
    assert completed[2].disposition is ChildDispositionKind.UNCLASSIFIED_KEEP


def test_uncertain_event_is_retained_as_unclassified_keep_inside_episode():
    uncertain = event(
        "uncertain",
        BASE_MS,
        "possible-fall:lobby",
        "Possible fall",
        ObservationState.UNCERTAIN,
    )

    result = segment_observations([uncertain])

    assert len(result.episodes) == 1
    assert result.dispositions[0].disposition is ChildDispositionKind.UNCLASSIFIED_KEEP
    assert result.dispositions[0].episode_id == result.episodes[0].episode_id


def test_attention_selection_has_pure_deterministic_2_4_8_semantics():
    observations = [
        event(
            f"attention-{index}",
            BASE_MS + index * 1_000,
            "fire:north-hall",
            "Fire in north hall",
            ObservationState.NEW if index == 0 else ObservationState.CONTINUING,
        )
        for index in range(10)
    ]

    selected_two = select_attention_observations(
        reversed(observations), AttentionBudget.BOUNDARIES
    )
    selected_four = select_attention_observations(observations, 4)
    selected_eight = select_attention_observations(observations, 8)

    assert selected_two == (observations[0], observations[-1])
    assert len(selected_four) == 4
    assert len(selected_eight) == 8
    assert selected_four[0] == selected_eight[0] == observations[0]
    assert selected_four[-1] == selected_eight[-1] == observations[-1]
    with pytest.raises(ValueError, match="exactly 2, 4, or 8"):
        select_attention_observations(observations, 3)


def test_temporal_dtos_reject_unbounded_or_ambiguous_input():
    with pytest.raises(ValueError, match="at most 240"):
        event(
            "too-long",
            BASE_MS,
            "event:key",
            "x" * 241,
            ObservationState.NEW,
        )

    observation = event(
        "duplicate",
        BASE_MS,
        "event:key",
        "Event",
        ObservationState.NEW,
    )
    with pytest.raises(ValueError, match="observation_id values must be unique"):
        segment_observations([observation, observation])
    with pytest.raises(ValueError, match="child_ids must be unique"):
        complete_child_dispositions(["same", "same"])
