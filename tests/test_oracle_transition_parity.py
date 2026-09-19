from __future__ import annotations

from collections import deque

import pytest

from minoflux_ai import _oracle_native
from minoflux_ai.bitboard import board_row_masks
from minoflux_engine import Game, Placement
from minoflux_engine.pieces import SHAPES


QUEUE = ("I", "O", "T", "S", "Z", "J", "L", "I")


def _placement(
    piece: str,
    x: int,
    y: int,
    rotation: int,
    *,
    last_rotation: bool = False,
    kick_index: int | None = None,
    rotation_from: int | None = None,
    rotation_to: int | None = None,
) -> Placement:
    cells = tuple((x + dx, y + dy) for dx, dy in SHAPES[piece][rotation % 4])
    return Placement(
        piece,
        x,
        y,
        rotation,
        cells,
        (),
        last_rotation,
        kick_index,
        rotation_from,
        rotation_to,
    )


def _empty_game(piece: str) -> Game:
    game = Game(20260919)
    game.board = [[None] * game.width for _ in range(game.height)]
    game.current = piece
    game.x, game.y, game.rotation = 3, 1, 0
    game.queue = deque(QUEUE)
    game.hold_piece = None
    game.hold_used = False
    game.combo = -1
    game.back_to_back = False
    game.b2b_chain = 0
    game.surge_charge = 0
    game.game_over = False
    game.last_lock = None
    return game


def _fill(game: Game, y: int, xs: range | tuple[int, ...] | list[int]) -> None:
    for x in xs:
        game.board[y][x] = "G"


def _native_transition(game: Game, placement: Placement, *, use_hold: bool) -> dict[str, object]:
    return _oracle_native.transition(
        board_row_masks(game.board),
        game.current,
        game.hold_piece,
        tuple(game.queue),
        int(game.combo),
        bool(game.back_to_back),
        int(game.b2b_chain),
        not bool(game.hold_used),
        {
            "piece": placement.piece,
            "x": placement.x,
            "y": placement.y,
            "rotation": placement.rotation,
            "holdUsed": use_hold,
            "lastMoveWasRotation": placement.last_move_was_rotation,
            "kickIndex": -1 if placement.rotation_kick_index is None else placement.rotation_kick_index,
            "rotationFrom": -1 if placement.rotation_from is None else placement.rotation_from,
            "rotationTo": -1 if placement.rotation_to is None else placement.rotation_to,
        },
    )


def _assert_transition_parity(game: Game, placement: Placement, *, use_hold: bool = False) -> None:
    initial_hold = game.hold_piece
    native = _native_transition(game, placement, use_hold=use_hold)

    if use_hold:
        assert game.hold()
    lock = game.place(placement)

    assert tuple(native["rows"]) == board_row_masks(game.board)
    assert native["current"] == game.current
    assert native["hold"] == game.hold_piece
    assert bool(native["canHold"]) == (not game.hold_used)
    assert int(native["combo"]) == game.combo
    assert bool(native["backToBack"]) == game.back_to_back
    assert int(native["b2bChain"]) == game.b2b_chain
    assert int(native["surgeCharge"]) == game.surge_charge
    assert bool(native["gameOver"]) == game.game_over

    assert int(native["lines"]) == lock.lines
    assert int(native["attack"]) == lock.attack
    assert native["spin"] == lock.spin
    assert bool(native["perfectClear"]) == lock.perfect_clear
    assert int(native["surgeReleased"]) == lock.surge_released

    expected_consumed = 0
    if use_hold and initial_hold is None:
        expected_consumed += 1
    if not game.game_over:
        expected_consumed += 1
    assert int(native["queueIndex"]) == expected_consumed


def test_transition_matches_engine_perfect_clear() -> None:
    game = _empty_game("I")
    _fill(game, 23, tuple(x for x in range(10) if x not in (3, 4, 5, 6)))

    _assert_transition_parity(game, _placement("I", 3, 22, 0))


def test_transition_matches_engine_b2b_release_and_combo() -> None:
    game = _empty_game("O")
    _fill(game, 23, tuple(x for x in range(10) if x not in (4, 5)))
    game.combo = 3
    game.back_to_back = True
    game.b2b_chain = 4
    game.surge_charge = 4

    _assert_transition_parity(game, _placement("O", 3, 22, 0))


def test_transition_matches_engine_full_tspin_single() -> None:
    game = _empty_game("T")
    x = 3
    _fill(game, 23, tuple(col for col in range(10) if col not in (x, x + 1, x + 2)))
    _fill(game, 22, (x, x + 2))

    _assert_transition_parity(
        game,
        _placement(
            "T",
            x,
            22,
            0,
            last_rotation=True,
            kick_index=0,
            rotation_from=3,
            rotation_to=0,
        ),
    )


def test_transition_matches_engine_mini_tspin_double() -> None:
    game = _empty_game("T")
    x = 3
    y = 21
    _fill(game, 22, tuple(col for col in range(10) if col not in (x + 1, x + 2)))
    _fill(game, 23, tuple(col for col in range(10) if col != x + 1))
    _fill(game, 21, (x,))

    _assert_transition_parity(
        game,
        _placement(
            "T",
            x,
            y,
            1,
            last_rotation=True,
            kick_index=0,
            rotation_from=0,
            rotation_to=1,
        ),
    )


def test_transition_matches_engine_hidden_row_topout() -> None:
    game = _empty_game("I")
    _fill(game, 4, (3, 4, 5, 6))

    _assert_transition_parity(game, _placement("I", 3, 2, 0))


@pytest.mark.parametrize("occupied_hold", (False, True))
def test_transition_matches_engine_hold_queue_consumption(occupied_hold: bool) -> None:
    game = _empty_game("T")
    if occupied_hold:
        game.hold_piece = "I"
        placement = _placement("I", 3, 22, 0)
    else:
        placement = _placement(QUEUE[0], 3, 22, 0)

    _assert_transition_parity(game, placement, use_hold=True)
