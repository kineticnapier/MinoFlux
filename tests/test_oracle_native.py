from __future__ import annotations

import pytest

import minoflux_ai.oracle as oracle_module
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


def test_native_oracle_uses_shared_table_reachability_backend() -> None:
    assert oracle_native_available()
    assert oracle_module._native.reachability_backend() == "shared-table-v1"


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


def test_search_does_not_call_python_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("Python search must not run inside the native oracle")

    monkeypatch.setattr(oracle_module, "choose_search_action", fail)

    choice = search_oracle(
        Game(314159),
        OracleConfig(beam_width=32, depth=2, allow_180=True),
    )

    assert choice is not None
