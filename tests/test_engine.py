from __future__ import annotations

import unittest

from minoflux_engine import Game
from minoflux.simulation import run_smoke


class EngineTests(unittest.TestCase):
    def test_seed_is_deterministic(self) -> None:
        first = Game(123)
        second = Game(123)
        self.assertEqual(first.current, second.current)
        self.assertEqual(tuple(first.queue), tuple(second.queue))

    def test_hard_drop_locks_piece(self) -> None:
        game = Game(1)
        result = game.hard_drop()
        self.assertEqual(game.pieces_placed, 1)
        self.assertFalse(result.game_over)
        self.assertTrue(any(cell is not None for row in game.board for cell in row))

    def test_hold_once_per_piece(self) -> None:
        game = Game(2)
        self.assertTrue(game.hold())
        self.assertFalse(game.hold())
        game.hard_drop()
        self.assertTrue(game.hold())

    def test_legal_placements_are_resting(self) -> None:
        game = Game(3)
        placements = game.legal_placements()
        self.assertGreater(len(placements), 0)
        placement = placements[0]
        self.assertFalse(game._collides(placement.piece, placement.x, placement.y, placement.rotation))
        self.assertTrue(game._collides(placement.piece, placement.x, placement.y + 1, placement.rotation))

    def test_place_advances_game(self) -> None:
        game = Game(4)
        game.place(game.legal_placements()[0])
        self.assertEqual(game.pieces_placed, 1)

    def test_smoke_is_deterministic(self) -> None:
        self.assertEqual(run_smoke(2, 30, 7), run_smoke(2, 30, 7))


if __name__ == "__main__":
    unittest.main()
