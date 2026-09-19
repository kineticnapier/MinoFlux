from __future__ import annotations

import random

import pytest

from minoflux_ai import _oracle_native
from minoflux_ai.bitboard import board_row_masks
from minoflux_ai.reachability_native import reachable_placements_pathless_native
from minoflux_engine import Game


def _native_key(value) -> tuple[int, int, int, bool, int | None, int | None, int | None]:
    def optional(raw: object) -> int | None:
        value = int(raw)
        return None if value < 0 else value

    return (
        int(value["x"]),
        int(value["y"]),
        int(value["rotation"]),
        bool(value["lastMoveWasRotation"]),
        optional(value["kickIndex"]),
        optional(value["rotationFrom"]),
        optional(value["rotationTo"]),
    )


def _placement_key(placement) -> tuple[int, int, int, bool, int | None, int | None, int | None]:
    return (
        int(placement.x),
        int(placement.y),
        int(placement.rotation),
        bool(placement.last_move_was_rotation),
        placement.rotation_kick_index,
        placement.rotation_from,
        placement.rotation_to,
    )


def _assert_parity(game: Game, *, allow_180: bool, max_nodes: int = 8_000) -> None:
    expected = reachable_placements_pathless_native(
        game,
        allow_180=allow_180,
        max_nodes=max_nodes,
    )
    actual = _oracle_native.reachable(
        board_row_masks(game.board),
        game.current,
        allow_180,
        max_nodes,
    )
    assert tuple(_native_key(item) for item in actual) == tuple(
        _placement_key(placement) for placement in expected
    )


@pytest.mark.parametrize("piece", tuple("IJLOSTZ"))
@pytest.mark.parametrize("allow_180", (False, True))
def test_oracle_reachability_matches_existing_native_on_empty_board(
    piece: str,
    allow_180: bool,
) -> None:
    game = Game(1)
    game.board = [[None] * game.width for _ in range(game.height)]
    game.current = piece
    game.x, game.y, game.rotation = 3, 1, 0
    game.game_over = False
    _assert_parity(game, allow_180=allow_180)


def test_oracle_reachability_matches_existing_native_on_deterministic_boards() -> None:
    rng = random.Random(20260919)
    for case in range(24):
        game = Game(case + 100)
        game.board = [[None] * game.width for _ in range(game.height)]
        game.current = rng.choice(tuple("IJLOSTZ"))
        game.x, game.y, game.rotation = 3, 1, 0
        for y in range(rng.randrange(12, 20), game.height):
            for x in range(game.width):
                if rng.random() < 0.38:
                    game.board[y][x] = "J"
        game.game_over = game._collides(game.current, 3, 1, 0)
        _assert_parity(
            game,
            allow_180=bool(case & 1),
            max_nodes=(64, 512, 8_000)[case % 3],
        )
