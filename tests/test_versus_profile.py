from __future__ import annotations

import pytest

from minoflux.versus_neural_cli import build_parser
from minoflux_ai.reachability import clear_reachability_cache, reachable_placements
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_search import (
    VersusSearchConfig,
    choose_versus_action,
)
from minoflux_ai.versus_profile import (
    VERSUS_PROFILE_PHASES,
    VersusSearchProfile,
    active_versus_profile,
    collect_versus_profile,
    profile_timer_start,
    record_profile_elapsed,
)
from minoflux_engine import Game, VersusMatch


class _ProfileScorer:
    @staticmethod
    def score_placements(game, placements):
        return tuple(float(placement.x) for placement in placements)

    def score_placement_groups(self, groups):
        return tuple(self.score_placements(game, placements) for game, placements in groups)

    def score_many(self, game, evaluations):
        return self.score_placements(game, tuple(item.placement for item in evaluations))


def test_profile_rows_have_stable_order_and_derived_values() -> None:
    profile = VersusSearchProfile()
    profile.record("total_search", 2_000_000, calls=2)
    profile.record("clone_game", 500_000, calls=4)

    rows = profile.phase_rows()
    assert tuple(row["name"] for row in rows) == VERSUS_PROFILE_PHASES
    clone = next(row for row in rows if row["name"] == "clone_game")
    assert clone == {
        "name": "clone_game",
        "calls": 4,
        "totalMs": 0.5,
        "meanUsPerCall": 125.0,
        "percentOfSearch": 25.0,
    }
    assert profile.to_dict()["timingKind"] == "inclusive"


def test_collector_activates_profile_and_collects_reachability() -> None:
    profile = VersusSearchProfile()
    assert active_versus_profile() is None

    with collect_versus_profile(profile) as collected:
        assert collected is profile
        assert active_versus_profile() is profile
        placements = reachable_placements(Game(20260904), include_paths=False)

    assert active_versus_profile() is None
    assert placements
    srs = next(
        row for row in profile.phase_rows() if row["name"] == "srs_reachability"
    )
    assert srs["calls"] == 1
    assert float(srs["totalMs"]) >= 0.0
    assert profile.to_dict()["reachability"]["placements"] == len(placements)


def test_disabled_timer_does_not_read_clock(monkeypatch) -> None:
    def fail_clock() -> int:
        raise AssertionError("disabled profiling read the clock")

    monkeypatch.setattr("minoflux_ai.versus_profile._clock_ns", fail_clock)
    assert profile_timer_start(None) == 0
    record_profile_elapsed(None, "total_search", 0)


def test_table_and_unknown_phase_handling() -> None:
    profile = VersusSearchProfile()
    profile.record("total_search", 1_000_000)
    profile.record("resolve_lock", 125_000, calls=5)

    table = profile.format_table()
    assert "Versus profile (inclusive timings)" in table
    assert "total_search" in table
    assert "resolve_lock" in table
    assert "board_copy" not in table
    assert "board_copy" in profile.format_table(include_zero=True)
    with pytest.raises(ValueError, match="Unknown versus profile phase"):
        profile.record("not_a_phase", 1)


def test_real_versus_search_populates_profile_without_changing_choice() -> None:
    match = VersusMatch(4101)
    scorer = _ProfileScorer()
    config = VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=1,
            srs_reachable=True,
        ),
        candidate_width=3,
        opponent_reply_width=1,
    )
    expected = choose_versus_action(
        match,
        "player",
        config=config,
        scorer=scorer,
        opponent_scorer=scorer,
    )
    clear_reachability_cache()
    with collect_versus_profile() as profile:
        actual = choose_versus_action(
            match,
            "player",
            config=config,
            scorer=scorer,
            opponent_scorer=scorer,
        )

    assert actual == expected
    rows = {row["name"]: row for row in profile.phase_rows()}
    for phase in (
        "total_search",
        "root_placement_generation",
        "opponent_placement_generation",
        "srs_reachability",
        "branch_groups",
        "neural_placement_scoring",
        "root_simulate_action",
        "reply_simulate_action",
        "clone_versus_match",
        "clone_game",
        "board_copy",
        "bag_rng_state_copy",
        "garbage_queue_copy",
        "garbage_rng_state_copy",
        "apply_search_action",
        "resolve_lock",
        "score_versus_state",
        "max_height_and_holes",
        "path_materialization",
        "python_aggregation_tie_breaking",
    ):
        assert rows[phase]["calls"] > 0


def test_cli_profile_flag_and_cpu_reference_batch_default() -> None:
    parser = build_parser()
    benchmark = parser.parse_args(["benchmark"])
    selfplay = parser.parse_args(["selfplay"])
    profiled = parser.parse_args(["benchmark", "--profile", "--game-batch", "4"])

    assert benchmark.game_batch == 1
    assert selfplay.game_batch == 1
    assert not benchmark.profile
    assert not selfplay.profile
    assert profiled.profile
    assert profiled.game_batch == 4
