from __future__ import annotations

from array import array
import os
import unittest
from unittest.mock import patch

from minoflux_ai.bitboard import ROW_OCCUPANCY_BYTES, board_row_masks, placement_cells
from minoflux_ai.neural import NeuralValueConfig, _context_prefix, _encode_placement_result_compact
from minoflux_ai.neural_fast import native_neural_available
from minoflux_ai.reachability import clear_reachability_cache
from minoflux_ai.search import SearchAction, SearchConfig, _branch_groups, apply_search_action
from minoflux_engine import Game, Placement
from minoflux_engine.pieces import SHAPES

try:
    from minoflux_ai import _neural_native
except ImportError:
    _neural_native = None


_CONFIG = NeuralValueConfig().normalized()
_SEARCH_CONFIG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
).normalized()


def _prefixes(game: Game):
    source_queue = tuple(game.queue)
    locked = _context_prefix(
        current=game.current,
        hold_piece=game.hold_piece,
        queue=source_queue,
        config=_CONFIG,
    )
    normal = (
        _context_prefix(
            current=source_queue[0],
            hold_piece=game.hold_piece,
            queue=source_queue[1:],
            config=_CONFIG,
        )
        if len(source_queue) >= _CONFIG.queue_length + 1
        else None
    )
    return source_queue, normal, locked


def _python_buffers(game: Game, placements):
    source_rows = board_row_masks(game.board)
    source_queue, normal, locked = _prefixes(game)
    states = []
    for placement in placements:
        encoded = _encode_placement_result_compact(
            game,
            placement,
            _CONFIG,
            source_rows=source_rows,
            source_queue=source_queue,
            normal_context_prefix=normal,
            locked_context_prefix=locked,
        )
        if encoded is None:
            raise AssertionError("test corpus unexpectedly requires clone fallback")
        states.append(encoded)
    board = bytearray().join(
        ROW_OCCUPANCY_BYTES[mask]
        for state in states
        for mask in state.rows
    )
    contexts = array("f")
    for state in states:
        contexts.extend(state.context)
    return bytes(board), contexts.tobytes()


def _native_buffers(game: Game, placements):
    source_rows = board_row_masks(game.board)
    source_queue, normal, locked = _prefixes(game)
    if normal is None:
        raise AssertionError("test corpus unexpectedly has a short queue")
    return _neural_native.encode_placement_group(
        source_rows,
        placements,
        game.current,
        source_queue[0],
        game.width,
        game.height,
        game.hidden_rows,
        game.combo,
        game.back_to_back,
        game.b2b_chain,
        normal,
        locked,
    )


@unittest.skipUnless(native_neural_available(), "native neural extension unavailable")
class NativeNeuralEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _neural_native.register_shapes(SHAPES)

    def assert_buffers_equal(self, game: Game, placements) -> None:
        expected_board, expected_context = _python_buffers(game, placements)
        actual_board, actual_context = _native_buffers(game, placements)
        self.assertEqual(actual_board, expected_board)
        self.assertEqual(actual_context, expected_context)

    def test_direct_and_hold_deterministic_gameplay_corpus(self) -> None:
        games = [Game(8100001 + index * 97) for index in range(8)]
        compared = 0
        for _step in range(40):
            for game in games:
                if game.game_over:
                    continue
                clear_reachability_cache()
                direct, hold_game, hold = _branch_groups(
                    game,
                    _SEARCH_CONFIG,
                    include_paths=False,
                )
                if direct:
                    self.assert_buffers_equal(game, direct)
                    compared += len(direct)
                if hold_game is not None and hold:
                    self.assert_buffers_equal(hold_game, hold)
                    compared += len(hold)
                placement = direct[0] if direct else (hold[0] if hold else None)
                if placement is not None:
                    apply_search_action(
                        game,
                        SearchAction(False if direct else True, placement),
                    )
        self.assertGreater(compared, 2_000)

    def test_t_spin_kick_metadata_and_b2b_context(self) -> None:
        game = Game(1234)
        game.board = [[None] * game.width for _ in range(game.height)]
        game.current = "T"
        game.x = 3
        game.y = 20
        game.rotation = 0
        game.combo = 7
        game.back_to_back = True
        game.b2b_chain = 5
        for x in range(game.width):
            game.board[23][x] = "J"
        for x in (3, 4, 5):
            game.board[23][x] = None
        game.board[21][3] = "J"
        game.board[21][5] = "J"
        placement = Placement(
            piece="T",
            x=3,
            y=20,
            rotation=0,
            cells=placement_cells("T", 3, 20, 0),
            path=(),
            last_move_was_rotation=True,
            rotation_kick_index=4,
            rotation_from=3,
            rotation_to=0,
        )
        self.assert_buffers_equal(game, (placement,))

    def test_native_neural_can_be_disabled_without_affecting_import(self) -> None:
        from minoflux_ai import neural_fast

        with patch.dict(os.environ, {"MINOFLUX_DISABLE_NATIVE_NEURAL": "1"}):
            self.assertTrue(neural_fast.native_neural_available())


if __name__ == "__main__":
    unittest.main()
