from __future__ import annotations

from copy import deepcopy
import unittest

from minoflux_ai import CEMConfig, DEFAULT_WEIGHTS, choose_placement, evaluate_placement, train_cem
from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import HeuristicWeights, PlacementFeatures, score_features
from minoflux_engine import Game


class FastPlacementEvaluationTests(unittest.TestCase):
    @staticmethod
    def reference_features(game: Game, placement):
        before = extract_board_features(game.board)
        simulation = deepcopy(game)
        result = simulation.place(placement)
        after = extract_board_features(simulation.board)
        features = PlacementFeatures(
            board=after,
            new_holes=max(0, after.holes - before.holes),
            lines=result.lines,
            attack=result.attack,
            spin_lines=result.lines if result.spin is not None else 0,
            perfect_clear=result.perfect_clear,
            game_over=result.game_over,
            spin=result.spin,
            t_spin_slot_delta=after.t_spin_slots - before.t_spin_slots,
        )
        return features, score_features(features, DEFAULT_WEIGHTS)

    def test_board_only_simulation_matches_game_place(self) -> None:
        game = Game(123)
        checked = 0
        for _ in range(20):
            placements = game.legal_placements()
            self.assertTrue(placements)
            for placement in placements:
                actual = evaluate_placement(game, placement)
                expected_features, expected_score = self.reference_features(game, placement)
                self.assertEqual(actual.features, expected_features)
                self.assertAlmostEqual(actual.score, expected_score)
                checked += 1
            choice = choose_placement(game)
            self.assertIsNotNone(choice)
            game.place(choice.placement)
            if game.game_over:
                break
        self.assertGreater(checked, 250)

    def test_evaluation_does_not_mutate_game_state(self) -> None:
        game = Game(321)
        for _ in range(8):
            choice = choose_placement(game)
            self.assertIsNotNone(choice)
            game.place(choice.placement)
        placement = game.legal_placements()[0]
        board_before = deepcopy(game.board)
        state_before = (
            game.current_piece,
            game.hold_piece,
            game.pieces_placed,
            game.lines,
            game.attack,
            game.combo,
            game.back_to_back,
            game.b2b_chain,
        )
        evaluate_placement(game, placement)
        self.assertEqual(game.board, board_before)
        self.assertEqual(
            (
                game.current_piece,
                game.hold_piece,
                game.pieces_placed,
                game.lines,
                game.attack,
                game.combo,
                game.back_to_back,
                game.b2b_chain,
            ),
            state_before,
        )

    def test_old_weight_mapping_gets_new_slot_delta_default(self) -> None:
        weights = HeuristicWeights.from_mapping({"attack": 0.85, "t_spin_slots": 1.2})
        self.assertEqual(weights.attack, 0.85)
        self.assertEqual(weights.t_spin_slots, 1.2)
        self.assertEqual(weights.t_spin_slot_delta, 0.25)


class ParallelCemTests(unittest.TestCase):
    @staticmethod
    def config(workers: int) -> CEMConfig:
        return CEMConfig(
            generations=1,
            population=4,
            elite_fraction=0.25,
            games_per_candidate=1,
            max_pieces=10,
            validation_games=1,
            workers=workers,
            screen_games=1,
            screen_max_pieces=4,
            screen_fraction=0.5,
            random_seed=77,
        )

    def test_screening_reduces_full_evaluations(self) -> None:
        result = train_cem(self.config(1))
        generation = result.history[0]
        self.assertEqual(generation.evaluated_candidates, 2)
        self.assertEqual(generation.screened_out_candidates, 2)
        self.assertGreaterEqual(generation.elapsed_seconds, 0)

    def test_parallel_and_serial_results_are_deterministic(self) -> None:
        serial = train_cem(self.config(1))
        parallel = train_cem(self.config(2))
        self.assertEqual(serial.best_weights, parallel.best_weights)
        self.assertEqual(serial.best_training_fitness, parallel.best_training_fitness)
        self.assertEqual(serial.validation_fitness, parallel.validation_fitness)
        self.assertEqual(parallel.workers, 2)


if __name__ == "__main__":
    unittest.main()