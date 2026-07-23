from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from minoflux_ai import (
    CEMConfig,
    REPLAY_FORMAT,
    TRAINABLE_WEIGHT_NAMES,
    load_replay,
    replay_to_game,
    run_heuristic_benchmark,
    save_replay,
    train_cem,
)


class ReplayTests(unittest.TestCase):
    def test_best_benchmark_replay_round_trip(self) -> None:
        benchmark = run_heuristic_benchmark(
            games=2,
            max_pieces=20,
            seed_base=9,
            record_best_replay=True,
        )
        replay = benchmark.best_replay
        self.assertIsNotNone(replay)
        assert replay is not None
        self.assertEqual(replay.format, REPLAY_FORMAT)
        self.assertEqual(replay.seed, benchmark.best_game.seed)
        self.assertEqual(len(replay.steps), benchmark.best_game.pieces)

        final_game = replay_to_game(replay)
        self.assertEqual(final_game.pieces_placed, replay.final.pieces)
        self.assertEqual(final_game.lines, replay.final.lines)
        self.assertEqual(final_game.attack, replay.final.attack)
        self.assertEqual(final_game.score, replay.final.score)
        self.assertEqual(final_game.game_over, replay.final.topout)

        with TemporaryDirectory() as directory:
            path = save_replay(f"{directory}/best.json", replay)
            self.assertEqual(load_replay(path), replay)

    def test_partial_replay_rebuilds_requested_step(self) -> None:
        benchmark = run_heuristic_benchmark(games=1, max_pieces=12, record_best_replay=True)
        replay = benchmark.best_replay
        assert replay is not None
        game = replay_to_game(replay, steps=5)
        self.assertEqual(game.pieces_placed, 5)


class CEMTests(unittest.TestCase):
    def test_game_over_weight_is_not_sampled(self) -> None:
        self.assertNotIn("game_over", TRAINABLE_WEIGHT_NAMES)

    def test_tiny_cem_run_is_deterministic_and_has_replay(self) -> None:
        config = CEMConfig(
            generations=1,
            population=3,
            elite_fraction=0.34,
            games_per_candidate=1,
            max_pieces=8,
            validation_games=1,
            random_seed=77,
        )
        first = train_cem(config)
        second = train_cem(config)
        self.assertEqual(first.best_weights, second.best_weights)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.validation.to_dict(), second.validation.to_dict())
        self.assertEqual(len(first.history), 1)
        self.assertIsNotNone(first.validation.best_replay)


if __name__ == "__main__":
    unittest.main()
