from __future__ import annotations

import json

from minoflux_ai import run_versus_benchmark
from minoflux_ai.search import SearchConfig, choose_search_action, choose_search_actions_batch
from minoflux_ai.versus_neural import VersusSelfPlayConfig, generate_versus_selfplay_dataset
from minoflux_ai.versus_search import VersusSearchConfig
from minoflux_engine import Game


class _PlacementBatchScorer:
    def __init__(self) -> None:
        self.group_calls = 0

    def score_placements(self, game, placements):
        return tuple(
            float(placement.x)
            + 0.01 * float(placement.rotation)
            + 0.0001 * float(placement.y)
            for placement in placements
        )

    def score_placement_groups(self, groups):
        self.group_calls += 1
        return tuple(
            self.score_placements(game, placements)
            for game, placements in groups
        )

    def score_many(self, game, evaluations):
        return self.score_placements(
            game,
            tuple(evaluation.placement for evaluation in evaluations),
        )


class _MatchBatchScorer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def score_matches(self, entries):
        self.batch_sizes.append(len(entries))
        return (0.0,) * len(entries)

    def score_match(self, match, root_side, to_move=None):
        return 0.0


def _versus_config(*, candidate_width: int = 3, reply_width: int = 2) -> VersusSearchConfig:
    return VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=False,
            lookahead_pieces=0,
            beam_width=1,
            srs_reachable=False,
        ),
        candidate_width=candidate_width,
        opponent_reply_width=reply_width,
    )


def test_multi_game_batch_matches_individual_root_choices() -> None:
    config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=4,
        srs_reachable=False,
    )
    seeds = (7001, 7002, 7003, 7004)

    expected = []
    for seed in seeds:
        scorer = _PlacementBatchScorer()
        expected.append(choose_search_action(Game(seed), config=config, scorer=scorer))

    scorer = _PlacementBatchScorer()
    actual = choose_search_actions_batch(
        tuple(Game(seed) for seed in seeds),
        config=config,
        scorer=scorer,
    )

    assert scorer.group_calls == 1
    assert len(actual) == len(expected)
    for single, batched in zip(expected, actual):
        assert single is not None
        assert batched is not None
        assert batched.action.to_dict() == single.action.to_dict()
        assert batched.score == single.score


def test_serial_and_batched_versus_benchmarks_match_with_mirrored_sides() -> None:
    config = _versus_config()
    serial = run_versus_benchmark(
        6,
        max_turns=8,
        seed_base=81001,
        seed_step=31,
        player_config=config,
        ai_config=config,
        player_scorer=_PlacementBatchScorer(),
        ai_scorer=_PlacementBatchScorer(),
        player_state_scorer=_MatchBatchScorer(),
        ai_state_scorer=_MatchBatchScorer(),
        game_batch=1,
    )
    batched_state = _MatchBatchScorer()
    batched = run_versus_benchmark(
        6,
        max_turns=8,
        seed_base=81001,
        seed_step=31,
        player_config=config,
        ai_config=config,
        player_scorer=_PlacementBatchScorer(),
        ai_scorer=_PlacementBatchScorer(),
        player_state_scorer=batched_state,
        ai_state_scorer=batched_state,
        game_batch=4,
    )

    assert batched == serial
    assert [game.seed for game in batched.per_game] == [81001, 81001, 81032, 81032, 81063, 81063]
    assert [game.models_swapped for game in batched.per_game] == [False, True, False, True, False, True]
    assert max(batched_state.batch_sizes) > config.candidate_width


def test_game_batch_larger_than_remaining_games_matches_serial() -> None:
    config = _versus_config(candidate_width=2, reply_width=1)
    serial = run_versus_benchmark(
        3,
        max_turns=5,
        seed_base=92001,
        player_config=config,
        ai_config=config,
        game_batch=1,
    )
    batched = run_versus_benchmark(
        3,
        max_turns=5,
        seed_base=92001,
        player_config=config,
        ai_config=config,
        game_batch=8,
    )
    assert batched == serial


def test_rolling_batch_removes_early_finishes_and_refills_slots() -> None:
    config = _versus_config(candidate_width=1, reply_width=0)
    kwargs = dict(
        games=5,
        max_turns=100,
        seed_base=555,
        seed_step=31,
        player_config=config,
        ai_config=config,
    )
    serial = run_versus_benchmark(
        **kwargs,
        player_scorer=_PlacementBatchScorer(),
        ai_scorer=_PlacementBatchScorer(),
        game_batch=1,
    )
    batched = run_versus_benchmark(
        **kwargs,
        player_scorer=_PlacementBatchScorer(),
        ai_scorer=_PlacementBatchScorer(),
        game_batch=3,
    )

    assert batched == serial
    assert all(game.turns < 100 for game in batched.per_game)
    assert len({game.turns for game in batched.per_game}) > 1


def test_batched_selfplay_preserves_record_order_seed_mapping_and_outcomes(tmp_path) -> None:
    config = _versus_config(candidate_width=2, reply_width=1)
    serial_path = tmp_path / "serial.jsonl"
    batched_path = tmp_path / "batched.jsonl"
    common = dict(
        games=5,
        max_turns=6,
        seed_base=63001,
        seed_step=17,
        search_config=config,
    )
    serial = generate_versus_selfplay_dataset(
        serial_path,
        _PlacementBatchScorer(),
        VersusSelfPlayConfig(**common, game_batch=1),
        value_scorer=_MatchBatchScorer(),
    )
    batched_state = _MatchBatchScorer()
    batched = generate_versus_selfplay_dataset(
        batched_path,
        _PlacementBatchScorer(),
        VersusSelfPlayConfig(**common, game_batch=3),
        value_scorer=batched_state,
    )

    assert batched_path.read_bytes() == serial_path.read_bytes()
    ignored_summary_keys = {"gameBatch", "path"}
    assert {key: value for key, value in batched.items() if key not in ignored_summary_keys} == {
        key: value for key, value in serial.items() if key not in ignored_summary_keys
    }
    records = [json.loads(line) for line in batched_path.read_text(encoding="utf-8").splitlines()]
    assert [record["game"] for record in records] == sorted(record["game"] for record in records)
    for game_index in range(5):
        game_records = [record for record in records if record["game"] == game_index]
        assert game_records
        assert {record["seed"] for record in game_records} == {63001 + game_index * 17}
        assert len({record["outcome"] for record in game_records if record["side"] == "player"}) == 1
        assert len({record["outcome"] for record in game_records if record["side"] == "ai"}) == 1
        assert sum(record["terminal"] for record in game_records) == 2
    assert max(batched_state.batch_sizes) > config.candidate_width
