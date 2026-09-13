from __future__ import annotations

from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_benchmark import (
    VersusGameResult,
    _summarize_benchmark,
    run_versus_benchmark,
)
from minoflux_ai.versus_search import VersusSearchConfig


def _game(*, seed: int, winner: str, swapped: bool) -> VersusGameResult:
    return VersusGameResult(
        seed=seed,
        winner=winner,
        turns=1,
        player_pieces=1,
        ai_pieces=1,
        player_attack=0,
        ai_attack=0,
        player_sent=0,
        ai_sent=0,
        player_canceled=0,
        ai_canceled=0,
        player_received=0,
        ai_received=0,
        player_garbage_applied=0,
        ai_garbage_applied=0,
        player_pending=0,
        ai_pending=0,
        player_final_height=0,
        ai_final_height=0,
        player_final_holes=0,
        ai_final_holes=0,
        player_max_b2b=0,
        ai_max_b2b=0,
        player_max_surge=0,
        ai_max_surge=0,
        models_swapped=swapped,
    )


def test_policy_and_physical_side_aggregates_are_explicitly_distinct() -> None:
    # winner is already in logical-policy orientation, exactly like benchmark
    # perGame after _remap_swapped_result().
    results = (
        _game(seed=100, winner="player", swapped=False),
        _game(seed=100, winner="player", swapped=True),
        _game(seed=131, winner="ai", swapped=False),
        _game(seed=131, winner="ai", swapped=True),
        _game(seed=162, winner="draw", swapped=False),
        _game(seed=162, winner="player", swapped=True),
    )
    payload = _summarize_benchmark(
        results,
        max_turns=10,
        seed_base=100,
        seed_step=31,
    ).to_dict()

    # Legacy fields keep their existing logical-policy meaning.
    assert payload["playerWins"] == 3
    assert payload["aiWins"] == 2
    assert payload["draws"] == 1

    # New names make that meaning explicit.
    assert payload["playerPolicyWins"] == 3
    assert payload["aiPolicyWins"] == 2
    assert payload["playerPolicyWinRate"] == 3 / 6
    assert payload["aiPolicyWinRate"] == 2 / 6
    assert payload["unswappedPlayerPolicyWins"] == 1
    assert payload["unswappedAiPolicyWins"] == 1
    assert payload["swappedPlayerPolicyWins"] == 2
    assert payload["swappedAiPolicyWins"] == 1

    # Physical-side wins require reversing logical winners on swapped legs.
    assert payload["physicalPlayerWins"] == 2
    assert payload["physicalAiWins"] == 3
    assert payload["physicalPlayerWinRate"] == 2 / 6
    assert payload["physicalAiWinRate"] == 3 / 6

    assert payload["seedCount"] == 3
    assert payload["mirroredGameCount"] == 3
    assert payload["unswappedGameCount"] == 3
    assert payload["seedStepUnit"] == "mirroredPair"
    assert payload["perGameOrientation"] == "policy"


def test_symmetric_policy_mirroring_cancels_physical_side_bias() -> None:
    config = VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=False,
            lookahead_pieces=0,
            beam_width=1,
            srs_reachable=False,
        ),
        candidate_width=2,
        opponent_reply_width=1,
    )
    result = run_versus_benchmark(
        6,
        max_turns=8,
        seed_base=7001,
        seed_step=31,
        player_config=config,
        ai_config=config,
    )
    payload = result.to_dict()

    assert payload["seedCount"] == 3
    assert payload["mirroredGameCount"] == 3
    assert payload["playerPolicyWins"] == payload["aiPolicyWins"]

    for unswapped, swapped in zip(result.per_game[::2], result.per_game[1::2]):
        assert unswapped.seed == swapped.seed
        assert not unswapped.models_swapped
        assert swapped.models_swapped
        if unswapped.winner == "draw":
            assert swapped.winner == "draw"
        else:
            assert swapped.winner == ("ai" if unswapped.winner == "player" else "player")


def test_odd_benchmark_game_count_uses_one_seed_per_mirrored_pair() -> None:
    config = VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=False,
            lookahead_pieces=0,
            beam_width=1,
            srs_reachable=False,
        ),
        candidate_width=1,
        opponent_reply_width=0,
    )
    result = run_versus_benchmark(
        5,
        max_turns=1,
        seed_base=9001,
        seed_step=17,
        player_config=config,
        ai_config=config,
    )
    assert [game.seed for game in result.per_game] == [9001, 9001, 9018, 9018, 9035]
    payload = result.to_dict()
    assert payload["seedCount"] == 3
    assert payload["mirroredGameCount"] == 2
    assert payload["unswappedGameCount"] == 3
