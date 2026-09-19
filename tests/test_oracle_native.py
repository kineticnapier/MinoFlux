from __future__ import annotations

from minoflux_ai.oracle import OracleConfig, oracle_native_available, search_oracle
from minoflux_ai.search import SearchConfig, rank_search_actions
from minoflux_engine import Game


def _action_key(action):
    placement = action.placement
    return (
        bool(action.use_hold),
        placement.piece,
        int(placement.x),
        int(placement.y),
        int(placement.rotation),
        bool(placement.last_move_was_rotation),
        placement.rotation_kick_index,
    )


def test_native_oracle_returns_exact_reachable_action() -> None:
    assert oracle_native_available()

    game = Game(20260919)
    config = OracleConfig(
        beam_width=64,
        depth=1,
        allow_180=True,
        reachability_node_limit=8_000,
    )
    choice = search_oracle(game, config)

    assert choice is not None

    legal = rank_search_actions(
        game,
        config=SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=4,
            srs_reachable=True,
            allow_180=True,
            reachability_node_limit=8_000,
        ),
        limit=None,
    )
    assert _action_key(choice.action) in {_action_key(action) for action, _ in legal}


def test_native_oracle_is_deterministic() -> None:
    config = OracleConfig(beam_width=64, depth=2, allow_180=True)

    left = search_oracle(Game(8100001), config)
    right = search_oracle(Game(8100001), config)

    assert left is not None
    assert right is not None
    assert _action_key(left.action) == _action_key(right.action)
    assert left.score == right.score
