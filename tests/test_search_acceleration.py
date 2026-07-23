from __future__ import annotations

from collections import deque
from copy import copy
import random
import unittest

from minoflux_ai import (
    DEFAULT_WEIGHTS,
    SearchConfig,
    apply_search_action,
    choose_search_action,
    rank_placements,
    rank_search_actions,
    run_heuristic_benchmark,
)
from minoflux_ai.search import clone_game
from minoflux_engine import Game


def exact_clone(game: Game) -> Game:
    cloned = copy(game)
    cloned.board = [row.copy() for row in game.board]
    cloned.queue = deque(game.queue)
    cloned._bag = copy(game._bag)
    cloned._bag._queue = deque(game._bag._queue)
    cloned_rng = random.Random()
    cloned_rng.setstate(game._bag._rng.getstate())
    cloned._bag._rng = cloned_rng
    return cloned


class BoundedRankingTests(unittest.TestCase):
    def test_limited_placement_ranking_matches_full_prefix(self) -> None:
        game = Game(123)
        full = rank_placements(game)
        self.assertGreater(len(full), 4)
        self.assertEqual(rank_placements(game, limit=4), full[:4])

    def test_limited_hold_ranking_matches_full_prefix(self) -> None:
        game = Game(321)
        config = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4)
        full = rank_search_actions(game, DEFAULT_WEIGHTS, config)
        self.assertGreater(len(full), 4)
        self.assertEqual(rank_search_actions(game, DEFAULT_WEIGHTS, config, limit=4), full[:4])

    def test_discount_is_applied_to_future_score(self) -> None:
        game = Game(9)
        config = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=1, discount=0.9)
        choice = choose_search_action(game, DEFAULT_WEIGHTS, config)
        self.assertIsNotNone(choice)
        assert choice is not None
        child = clone_game(game)
        apply_search_action(child, choice.action)
        next_ranked = rank_search_actions(child, DEFAULT_WEIGHTS, config, limit=1)
        self.assertTrue(next_ranked)
        expected = choice.immediate.score + 0.9 * next_ranked[0][1].score
        self.assertAlmostEqual(choice.score, expected)


class PreviewCloneTests(unittest.TestCase):
    def test_preview_clone_matches_exact_stream_within_search_horizon(self) -> None:
        source = Game(2026)
        fast = clone_game(source)
        exact = exact_clone(source)
        config = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=1)

        for _ in range(4):
            ranked = rank_search_actions(exact, DEFAULT_WEIGHTS, config, limit=1)
            self.assertTrue(ranked)
            action = ranked[0][0]
            apply_search_action(exact, action)
            apply_search_action(fast, action)
            self.assertEqual(fast.board, exact.board)
            self.assertEqual(fast.current, exact.current)
            self.assertEqual(fast.hold_piece, exact.hold_piece)
            self.assertEqual(tuple(fast.queue)[:3], tuple(exact.queue)[:3])
            self.assertEqual(fast.lines, exact.lines)
            self.assertEqual(fast.attack, exact.attack)
            self.assertEqual(fast.game_over, exact.game_over)
            if exact.game_over:
                break


class ParallelBenchmarkTests(unittest.TestCase):
    def test_parallel_and_serial_benchmarks_match(self) -> None:
        config = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=2)
        serial = run_heuristic_benchmark(
            games=2,
            max_pieces=16,
            seed_base=77,
            seed_step=31,
            search_config=config,
            workers=1,
            record_best_replay=True,
        )
        parallel = run_heuristic_benchmark(
            games=2,
            max_pieces=16,
            seed_base=77,
            seed_step=31,
            search_config=config,
            workers=2,
            record_best_replay=True,
        )
        self.assertEqual(serial.per_game, parallel.per_game)
        self.assertEqual(serial.best_game, parallel.best_game)
        self.assertEqual(serial.best_replay, parallel.best_replay)
        self.assertEqual(serial.workers, 1)
        self.assertEqual(parallel.workers, 2)


if __name__ == "__main__":
    unittest.main()
