from __future__ import annotations

import unittest

from minoflux_engine import Game
from minoflux_engine.pieces import I_KICK_TABLE, JLSTZ_KICK_TABLE, kick_tests


class LockDelayTests(unittest.TestCase):
    @staticmethod
    def grounded_game(*, delay: float = 500.0, limit: int = 15, piece: str = "T") -> Game:
        game = Game(1, lock_delay_ms=delay, lock_reset_limit=limit)
        game.current = piece
        game.rotation = 0
        game.x = 3
        game.y = game.ghost_y()
        game.lock_elapsed_ms = 0.0
        game.lock_resets = 0
        return game

    def test_ground_contact_waits_for_lock_delay(self) -> None:
        game = self.grounded_game()
        self.assertIsNone(game.gravity_step())
        self.assertIsNone(game.advance_time(499.0))
        self.assertEqual(game.pieces_placed, 0)
        result = game.advance_time(1.0)
        self.assertIsNotNone(result)
        self.assertEqual(game.pieces_placed, 1)

    def test_successful_grounded_move_resets_timer(self) -> None:
        game = self.grounded_game()
        game.advance_time(400.0)
        self.assertTrue(game.move_left())
        self.assertEqual(game.lock_elapsed_ms, 0.0)
        self.assertEqual(game.lock_resets, 1)
        self.assertIsNone(game.advance_time(499.0))
        self.assertIsNotNone(game.advance_time(1.0))

    def test_reset_limit_prevents_infinite_stalling(self) -> None:
        game = self.grounded_game(limit=2, piece="O")
        game.advance_time(400.0)
        self.assertTrue(game.move_left())
        self.assertEqual(game.lock_resets, 1)
        game.advance_time(400.0)
        self.assertTrue(game.move_right())
        self.assertEqual(game.lock_resets, 2)
        game.advance_time(400.0)
        self.assertTrue(game.move_left())
        self.assertEqual(game.lock_elapsed_ms, 400.0)
        self.assertEqual(game.lock_resets, 2)
        self.assertIsNotNone(game.advance_time(100.0))

    def test_grounded_rotation_resets_timer(self) -> None:
        game = self.grounded_game(piece="T")
        game.advance_time(400.0)
        self.assertTrue(game.rotate_cw())
        self.assertEqual(game.lock_elapsed_ms, 0.0)
        self.assertEqual(game.lock_resets, 1)

    def test_hard_drop_still_locks_immediately(self) -> None:
        game = Game(2, lock_delay_ms=10_000)
        result = game.hard_drop()
        self.assertFalse(result.game_over)
        self.assertEqual(game.pieces_placed, 1)


class SrsTests(unittest.TestCase):
    def test_jlstz_srs_table_uses_engine_y_axis(self) -> None:
        self.assertEqual(
            JLSTZ_KICK_TABLE[(0, 1)],
            ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        )
        self.assertEqual(
            JLSTZ_KICK_TABLE[(3, 0)],
            ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        )

    def test_i_piece_has_separate_srs_table(self) -> None:
        self.assertEqual(
            I_KICK_TABLE[(0, 1)],
            ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
        )
        self.assertEqual(kick_tests("I", 0, 1), I_KICK_TABLE[(0, 1)])
        self.assertEqual(kick_tests("T", 0, 1), JLSTZ_KICK_TABLE[(0, 1)])

    def test_srs_wall_kick_is_applied_in_order(self) -> None:
        game = Game(3)
        game.current = "T"
        game.rotation = 1
        game.x = -1
        game.y = 5
        self.assertFalse(game._collides(game.current, game.x, game.y, game.rotation))
        self.assertTrue(game.rotate_ccw())
        self.assertEqual(game.rotation, 0)
        self.assertEqual(game.x, 0)


if __name__ == "__main__":
    unittest.main()
