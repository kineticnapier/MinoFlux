from __future__ import annotations

from minoflux_ai.bitboard import board_row_masks
from minoflux_ai.reachability import _rows_to_board_bits, _state_tables
from minoflux_engine import Game
from minoflux_engine.pieces import SHAPES


def _board(seed: int) -> Game:
    game = Game(seed)
    game.board = [[None] * game.width for _ in range(game.height)]
    for y in range(game.height - 8, game.height):
        for x in range(game.width):
            if ((x * 17 + y * 11 + seed) % 5) in (0, 1):
                game.board[y][x] = "G"
    return game


def test_precomputed_collision_masks_match_engine_for_all_packed_states() -> None:
    for seed in (7, 20260905):
        game = _board(seed)
        rows = board_row_masks(game.board)
        board_bits = _rows_to_board_bits(rows, game.width)

        for piece in SHAPES:
            tables = _state_tables(piece, game.width, game.height)
            for state_id, shape_mask in enumerate(tables.collision_mask):
                x = tables.state_x[state_id]
                y = tables.state_y[state_id]
                rotation = state_id & 3
                expected = game._collides(piece, x, y, rotation)
                actual = shape_mask < 0 or bool(board_bits & shape_mask)
                assert actual is expected
