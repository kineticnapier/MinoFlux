from __future__ import annotations

import json

from minoflux_ai import run_versus_benchmark
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_batch_fast import (
    _ORIGINAL_CHOOSE_VERSUS_ACTIONS_BATCH,
    choose_versus_actions_batch,
)
from minoflux_ai.versus_neural import (
    VersusSelfPlayConfig,
    generate_versus_selfplay_dataset,
)
from minoflux_ai.versus_search import VersusSearchConfig, VersusSearchRequest
from minoflux_engine import VersusMatch


class _PlacementScorer:
    def __init__(self) -> None:
        self.group_calls = 0
        self.batch_sizes: list[int] = []

    @staticmethod
    def _scores(placements):
        return tuple(
            float(placement.x)
            + 0.01 * float(placement.rotation)
            + 0.0001 * float(placement.y)
            for placement in placements
        )

    def score_placements(self, game, placements):
        return self._scores(placements)

    def score_placement_groups(self, groups):
        self.group_calls += 1
        self.batch_sizes.append(sum(len(placements) for _game, placements in groups))
        return tuple(self._scores(placements) for _game, placements in groups)

    def score_many(self, game, evaluations):
        return self._scores(tuple(evaluation.placement for evaluation in evaluations))


class _StateScorer:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    @staticmethod
    def score_match(match, root_side, to_move=None):
        own = match.side(root_side)
        opponent = match.opponent(root_side)
        turn_term = 0.125 if to_move == root_side else -0.125
        return (
            0.01 * float(own.sent - opponent.sent)
            + 0.001 * float(opponent.pending.pending_lines - own.pending.pending_lines)
            + turn_term
        )

    def score_matches(self, entries):
        self.calls += 1
        self.batch_sizes.append(len(entries))
        return tuple(self.score_match(*entry) for entry in entries)


def _config() -> VersusSearchConfig:
    return VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=4,
            srs_reachable=True,
            allow_180=False,
            reachability_node_limit=8000,
        ),
        candidate_width=16,
        opponent_reply_width=4,
    )


def _requests(seeds, scorer, state_scorer):
    return tuple(
        VersusSearchRequest(
            match=VersusMatch(seed),
            side_name="player",
            config=_config(),
            scorer=scorer,
            opponent_scorer=scorer,
            state_scorer=state_scorer,
        )
        for seed in seeds
    )


def _signature(choice):
    if choice is None:
        return None
    return (
        choice.action.to_dict(),
        choice.score,
        choice.immediate,
        choice.resolution,
        None if choice.opponent_reply is None else choice.opponent_reply.to_dict(),
    )


def test_fused_root_reply_batches_match_reference_and_reduce_forward_calls() -> None:
    seeds = (811, 829, 857)
    old_placement = _PlacementScorer()
    old_state = _StateScorer()
    old = _ORIGINAL_CHOOSE_VERSUS_ACTIONS_BATCH(
        _requests(seeds, old_placement, old_state)
    )

    fused_placement = _PlacementScorer()
    fused_state = _StateScorer()
    fused = choose_versus_actions_batch(
        _requests(seeds, fused_placement, fused_state)
    )

    assert tuple(map(_signature, fused)) == tuple(map(_signature, old))
    assert old_placement.group_calls == 2
    assert fused_placement.group_calls == 1
    assert old_state.calls == 2
    assert fused_state.calls == 1
    assert fused_placement.batch_sizes[0] == sum(old_placement.batch_sizes)
    assert fused_state.batch_sizes[0] == sum(old_state.batch_sizes)


def test_fused_benchmark_per_game_and_aggregate_match_reference(monkeypatch) -> None:
    config = _config()
    common = dict(
        games=4,
        max_turns=8,
        seed_base=81001,
        seed_step=31,
        player_config=config,
        ai_config=config,
        game_batch=4,
        progress=False,
    )
    monkeypatch.setenv("MINOFLUX_DISABLE_VERSUS_BATCH_FUSION", "1")
    reference = run_versus_benchmark(
        **common,
        player_scorer=_PlacementScorer(),
        ai_scorer=_PlacementScorer(),
        player_state_scorer=_StateScorer(),
        ai_state_scorer=_StateScorer(),
    )
    monkeypatch.delenv("MINOFLUX_DISABLE_VERSUS_BATCH_FUSION")
    fused = run_versus_benchmark(
        **common,
        player_scorer=_PlacementScorer(),
        ai_scorer=_PlacementScorer(),
        player_state_scorer=_StateScorer(),
        ai_state_scorer=_StateScorer(),
    )
    assert fused == reference
    assert fused.to_dict() == reference.to_dict()


def test_fused_selfplay_jsonl_is_byte_identical_to_reference(tmp_path, monkeypatch) -> None:
    config = VersusSelfPlayConfig(
        games=4,
        max_turns=6,
        seed_base=63001,
        seed_step=17,
        search_config=_config(),
        game_batch=4,
        max_records_per_game=0,
    )
    reference_path = tmp_path / "reference.jsonl"
    fused_path = tmp_path / "fused.jsonl"

    monkeypatch.setenv("MINOFLUX_DISABLE_VERSUS_BATCH_FUSION", "1")
    reference = generate_versus_selfplay_dataset(
        reference_path,
        _PlacementScorer(),
        config,
        value_scorer=_StateScorer(),
    )
    monkeypatch.delenv("MINOFLUX_DISABLE_VERSUS_BATCH_FUSION")
    fused = generate_versus_selfplay_dataset(
        fused_path,
        _PlacementScorer(),
        config,
        value_scorer=_StateScorer(),
    )

    assert fused_path.read_bytes() == reference_path.read_bytes()
    ignored = {"path"}
    assert {key: value for key, value in fused.items() if key not in ignored} == {
        key: value for key, value in reference.items() if key not in ignored
    }
    assert all(
        json.loads(line)["format"] == "minoflux_versus_selfplay_v1"
        for line in fused_path.read_text(encoding="utf-8").splitlines()
    )


def test_fused_asymmetric_scorers_do_not_mix_policy_batches() -> None:
    player = _PlacementScorer()
    ai = _PlacementScorer()
    state = _StateScorer()
    requests = tuple(
        VersusSearchRequest(
            match=VersusMatch(seed),
            side_name="player" if index % 2 == 0 else "ai",
            config=_config(),
            scorer=player if index % 2 == 0 else ai,
            opponent_scorer=ai if index % 2 == 0 else player,
            state_scorer=state,
        )
        for index, seed in enumerate((901, 902, 903, 904))
    )
    choices = choose_versus_actions_batch(requests)
    assert all(choice is not None for choice in choices)
    assert player.group_calls == 1
    assert ai.group_calls == 1
    assert state.calls == 1
