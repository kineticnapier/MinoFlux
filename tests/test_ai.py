from __future__ import annotations

from collections import deque
from tempfile import TemporaryDirectory
import unittest

from minoflux_ai import (
    DEFAULT_WEIGHTS,
    HeuristicWeights,
    choose_placement,
    column_heights,
    extract_board_features,
    load_weights,
    rank_placements,
    run_heuristic_benchmark,
    save_weights,
)
from minoflux_ai.features import BoardFeatures
from minoflux_ai.heuristic import (
    PlacementFeatures,
    _hold_t_supply_balance_score,
    _t_spin_slot_queue_match_score,
    _t_spin_slot_supply_match,
    score_features,
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
        self.assertEqual(features.t_spin_slots, 0)
        self.assertEqual(features.t_spin_slot_density, 0.0)

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

    def test_three_corner_t_cavity_is_counted_as_t_spin_slot(self) -> None:
        board = [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        pivot_x = 4
        pivot_y = BOARD_HEIGHT - 2
        board[pivot_y - 1][pivot_x - 1] = "L"
        board[pivot_y - 1][pivot_x + 1] = "J"
        board[pivot_y + 1][pivot_x - 1] = "S"
        features = extract_board_features(board)
        self.assertGreaterEqual(features.t_spin_slots, 1)
        self.assertAlmostEqual(
            features.t_spin_slot_density,
            features.t_spin_slots / (1 + features.holes),
        )

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

    def test_ranking_nonempty_board_does_not_mutate_game(self) -> None:
        game = Game(321)
        for _ in range(8):
            placement = game.legal_placements()[0]
            game.place(placement)
        before = game.snapshot()
        ranked = rank_placements(game)
        self.assertTrue(ranked)
        self.assertEqual(game.snapshot(), before)

    def test_slot_height_quality_formula(self) -> None:
        board = BoardFeatures(
            aggregate_height=0,
            max_height=6,
            holes=0,
            hole_depth=0,
            bumpiness=0,
            wells=0,
            t_spin_slots=2,
            occupied_cells=0,
        )
        features = PlacementFeatures(
            board=board,
            new_holes=0,
            lines=0,
            attack=0,
            spin_lines=0,
            perfect_clear=False,
            game_over=False,
        )
        weights = HeuristicWeights(
            aggregate_height=0,
            max_height=0,
            holes=0,
            hole_depth=0,
            bumpiness=0,
            wells=0,
            t_spin_slots=0,
            t_spin_slot_density=0,
            t_spin_slot_delta=0,
            t_spin_slot_height_quality=1,
            t_spin_slot_low_clean=0,
            t_spin_slot_supply_match=0,
            new_holes=0,
            lines=0,
            attack=0,
            spin_lines=0,
            perfect_clear=0,
            game_over=0,
        )
        self.assertAlmostEqual(score_features(features, weights), 1.0)

    def test_slot_low_clean_formula_penalizes_holes_and_height_together(self) -> None:
        board = BoardFeatures(
            aggregate_height=0,
            max_height=6,
            holes=1,
            hole_depth=0,
            bumpiness=0,
            wells=0,
            t_spin_slots=3,
            occupied_cells=0,
        )
        features = PlacementFeatures(
            board=board,
            new_holes=0,
            lines=0,
            attack=0,
            spin_lines=0,
            perfect_clear=False,
            game_over=False,
        )
        weights = HeuristicWeights(
            aggregate_height=0,
            max_height=0,
            holes=0,
            hole_depth=0,
            bumpiness=0,
            wells=0,
            t_spin_slots=0,
            t_spin_slot_density=0,
            t_spin_slot_delta=0,
            t_spin_slot_height_quality=0,
            t_spin_slot_low_clean=1,
            t_spin_slot_supply_match=0,
            new_holes=0,
            lines=0,
            attack=0,
            spin_lines=0,
            perfect_clear=0,
            game_over=0,
        )
        self.assertAlmostEqual(score_features(features, weights), 1.0)

    def test_slot_supply_match_tracks_t_availability_and_excess_slots(self) -> None:
        self.assertAlmostEqual(_t_spin_slot_supply_match(1, 0), 1.0)
        self.assertAlmostEqual(_t_spin_slot_supply_match(1, 3), 0.5)
        self.assertAlmostEqual(_t_spin_slot_supply_match(0, 0), 0.0)
        self.assertAlmostEqual(_t_spin_slot_supply_match(2, 6), -0.35)
        self.assertAlmostEqual(_t_spin_slot_supply_match(2, 7), -0.35)

    def test_queue_slot_match_tracks_exact_near_t_supply_without_mutation(self) -> None:
        game = Game(123)
        game.current = "T"
        game.queue = deque(["I", "T", "O", "S", "T", "Z", "T"])
        game.hold_piece = "T"
        before = game.snapshot()
        self.assertAlmostEqual(_t_spin_slot_queue_match_score(game, 4), 0.96)
        self.assertAlmostEqual(_t_spin_slot_queue_match_score(game, 5), 0.74)
        self.assertAlmostEqual(_t_spin_slot_queue_match_score(game, 2), 0.04)
        self.assertEqual(game.snapshot(), before)

    def test_hold_t_supply_balance_only_applies_after_hold(self) -> None:
        game = Game(123)
        game.current = "I"
        game.queue = deque(["I"] * 7)
        game.hold_piece = "T"
        game.hold_used = False
        self.assertEqual(_hold_t_supply_balance_score(game, 1), 0.0)

        game.hold_used = True
        before = game.snapshot()
        self.assertAlmostEqual(_hold_t_supply_balance_score(game, 1), 0.12)
        self.assertAlmostEqual(_hold_t_supply_balance_score(game, 3), -0.44)
        self.assertEqual(game.snapshot(), before)

    def test_weights_round_trip(self) -> None:
        custom = HeuristicWeights(
            holes=-4.5,
            attack=2.25,
            t_spin_slots=0.9,
            t_spin_slot_density=0.4,
            t_spin_slot_height_quality=0.65,
            t_spin_slot_low_clean=0.75,
            t_spin_slot_supply_match=0.6,
            t_spin_slot_queue_match=0.7,
            hold_t_supply_balance=0.8,
        )
        with TemporaryDirectory() as directory:
            path = save_weights(f"{directory}/weights.json", custom)
            self.assertEqual(load_weights(path), custom)

    def test_old_weight_mapping_uses_density_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("t_spin_slot_density")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(loaded.t_spin_slot_density, DEFAULT_WEIGHTS.t_spin_slot_density)

    def test_old_weight_mapping_uses_slot_height_quality_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("t_spin_slot_height_quality")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(
            loaded.t_spin_slot_height_quality,
            DEFAULT_WEIGHTS.t_spin_slot_height_quality,
        )

    def test_old_weight_mapping_uses_slot_low_clean_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("t_spin_slot_low_clean")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(
            loaded.t_spin_slot_low_clean,
            DEFAULT_WEIGHTS.t_spin_slot_low_clean,
        )

    def test_old_weight_mapping_uses_slot_supply_match_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("t_spin_slot_supply_match")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(
            loaded.t_spin_slot_supply_match,
            DEFAULT_WEIGHTS.t_spin_slot_supply_match,
        )

    def test_old_weight_mapping_uses_queue_match_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("t_spin_slot_queue_match")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(
            loaded.t_spin_slot_queue_match,
            DEFAULT_WEIGHTS.t_spin_slot_queue_match,
        )

    def test_old_weight_mapping_uses_hold_t_supply_balance_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("hold_t_supply_balance")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(
            loaded.hold_t_supply_balance,
            DEFAULT_WEIGHTS.hold_t_supply_balance,
        )

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
