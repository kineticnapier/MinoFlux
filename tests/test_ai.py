from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from minoflux_ai import (
    DEFAULT_WEIGHTS,
    HeuristicWeights,
    choose_placement,
    column_heights,
    extract_board_features,
    load_weights,
    run_heuristic_benchmark,
    save_weights,
)
from minoflux_engine import BOARD_HEIGHT, BOARD_WIDTH, Game


class FeatureTests(unittest.TestCase):
    def test_empty_board_features(self) -> None:
        board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.assertEqual(column_heights(board), (0,) * BOARD_WIDTH)
        features = extract_board_features(board)
        self.assertEqual(features.aggregate_height, 0)
        self.assertEqual(features.holes, 0)
        self.assertEqual(features.bumpiness, 0)

    def test_hole_and_depth_are_counted(self) -> None:
        board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        board[-3][0] = "T"
        board[-1][0] = "T"
        board[-1][1] = "I"
        features = extract_board_features(board)
        self.assertEqual(column_heights(board)[:2], (3, 1))
        self.assertEqual(features.aggregate_height, 4)
        self.assertEqual(features.max_height, 3)
        self.assertEqual(features.holes, 1)
        self.assertEqual(features.hole_depth, 1)
        self.assertEqual(features.bumpiness, 3)

    def test_ragged_board_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_board_features([[None], [None, None]])


class HeuristicTests(unittest.TestCase):
    def test_choice_is_legal_and_does_not_mutate_game(self) -> None:
        game = Game(123)
        before = game.snapshot()
        choice = choose_placement(game)
        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertIn(choice.placement, game.legal_placements())
        self.assertEqual(game.snapshot(), before)

    def test_weights_round_trip(self) -> None:
        custom = HeuristicWeights(holes=-4.5, attack=2.25)
        with TemporaryDirectory() as directory:
            path = save_weights(f"{directory}/weights.json", custom)
            self.assertEqual(load_weights(path), custom)

    def test_unknown_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HeuristicWeights.from_mapping({"not_a_feature": 1})


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_is_deterministic(self) -> None:
        first = run_heuristic_benchmark(games=2, max_pieces=30, seed_base=5)
        second = run_heuristic_benchmark(games=2, max_pieces=30, seed_base=5)
        self.assertEqual(first, second)
        self.assertGreater(first.pieces, 0)

    def test_default_bot_reaches_small_piece_limit(self) -> None:
        result = run_heuristic_benchmark(games=1, max_pieces=40, seed_base=7, weights=DEFAULT_WEIGHTS)
        self.assertEqual(result.per_game[0].pieces, 40)
        self.assertTrue(result.per_game[0].completed)


if __name__ == "__main__":
    unittest.main()
