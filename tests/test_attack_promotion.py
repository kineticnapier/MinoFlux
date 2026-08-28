from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minoflux_ai import (
    ATTACK_SPIN_FITNESS,
    BALANCED_FITNESS,
    CLEAN_ATTACK_FITNESS,
    BenchmarkGame,
    BenchmarkResult,
    HeuristicWeights,
    PromotionConfig,
    SearchConfig,
    benchmark_fitness,
    bootstrap_champion,
    compare_candidate_to_champion,
    load_weights,
    run_heuristic_game,
    save_weights,
)


def make_benchmark(
    *,
    attack: int,
    spins: int,
    spin_lines: int,
    completed: int = 2,
    mean_holes: float = 0.0,
    mean_hole_depth: float = 0.0,
    mean_bumpiness: float = 0.0,
    mean_max_height: float = 0.0,
    high_stack_fraction: float = 0.0,
) -> BenchmarkResult:
    game = BenchmarkGame(
        seed=1,
        pieces=100,
        lines=40,
        attack=attack // 2,
        spins=spins // 2,
        spin_lines=spin_lines // 2,
        perfect_clears=0,
        score=1000,
        topout=False,
        completed=True,
        mean_holes=mean_holes,
        mean_hole_depth=mean_hole_depth,
        mean_bumpiness=mean_bumpiness,
        mean_max_height=mean_max_height,
        peak_max_height=int(mean_max_height),
        high_stack_steps=int(high_stack_fraction * 100),
        high_stack_fraction=high_stack_fraction,
    )
    return BenchmarkResult(
        games=2,
        max_pieces=100,
        seed_base=1,
        seed_step=31,
        search_config=SearchConfig(),
        workers=1,
        pieces=200,
        mean_pieces=100,
        lines=80,
        mean_lines=40,
        attack=attack,
        mean_attack=attack / 2,
        spins=spins,
        mean_spins=spins / 2,
        spin_lines=spin_lines,
        mean_spin_lines=spin_lines / 2,
        perfect_clears=0,
        mean_perfect_clears=0,
        mean_holes=mean_holes,
        mean_hole_depth=mean_hole_depth,
        mean_bumpiness=mean_bumpiness,
        mean_max_height=mean_max_height,
        peak_max_height=int(mean_max_height),
        high_stack_steps=int(high_stack_fraction * 200),
        high_stack_fraction=high_stack_fraction,
        topouts=2 - completed,
        completed=completed,
        per_game=(game, game),
        best_game=game,
    )


class AttackFitnessTests(unittest.TestCase):
    def test_attack_spin_profile_values_offense_more_than_balanced(self) -> None:
        defensive = make_benchmark(attack=100, spins=0, spin_lines=0)
        offensive = make_benchmark(attack=120, spins=8, spin_lines=12)
        attack_gain = benchmark_fitness(offensive, ATTACK_SPIN_FITNESS) - benchmark_fitness(defensive, ATTACK_SPIN_FITNESS)
        balanced_gain = benchmark_fitness(offensive, BALANCED_FITNESS) - benchmark_fitness(defensive, BALANCED_FITNESS)
        self.assertGreater(attack_gain, balanced_gain)

    def test_clean_attack_profile_penalizes_dirty_high_stack(self) -> None:
        clean = make_benchmark(
            attack=120,
            spins=8,
            spin_lines=12,
            mean_holes=0.5,
            mean_hole_depth=0.8,
            mean_bumpiness=5.0,
            mean_max_height=7.0,
            high_stack_fraction=0.0,
        )
        dirty = make_benchmark(
            attack=120,
            spins=8,
            spin_lines=12,
            mean_holes=4.0,
            mean_hole_depth=9.0,
            mean_bumpiness=18.0,
            mean_max_height=14.0,
            high_stack_fraction=0.25,
        )
        self.assertGreater(
            benchmark_fitness(clean, CLEAN_ATTACK_FITNESS),
            benchmark_fitness(dirty, CLEAN_ATTACK_FITNESS),
        )

    def test_clean_attack_profile_penalizes_topout_more_than_attack_spin(self) -> None:
        safe = make_benchmark(attack=120, spins=8, spin_lines=12, completed=2)
        fragile = make_benchmark(attack=120, spins=8, spin_lines=12, completed=1)
        clean_loss = benchmark_fitness(safe, CLEAN_ATTACK_FITNESS) - benchmark_fitness(fragile, CLEAN_ATTACK_FITNESS)
        attack_spin_loss = benchmark_fitness(safe, ATTACK_SPIN_FITNESS) - benchmark_fitness(fragile, ATTACK_SPIN_FITNESS)
        self.assertGreater(clean_loss, attack_spin_loss)

    def test_benchmark_collects_stack_quality_telemetry(self) -> None:
        game = run_heuristic_game(
            7,
            max_pieces=12,
            search_config=SearchConfig(lookahead_pieces=0, beam_width=1),
        )
        self.assertGreater(game.pieces, 0)
        self.assertGreaterEqual(game.mean_holes, 0)
        self.assertGreaterEqual(game.mean_hole_depth, 0)
        self.assertGreaterEqual(game.mean_bumpiness, 0)
        self.assertGreaterEqual(game.mean_max_height, 0)
        self.assertGreaterEqual(game.peak_max_height, game.mean_max_height)
        self.assertLessEqual(game.high_stack_steps, game.pieces)
        self.assertGreaterEqual(game.high_stack_fraction, 0)
        self.assertLessEqual(game.high_stack_fraction, 1)


class PromotionTests(unittest.TestCase):
    def test_candidate_must_beat_champion(self) -> None:
        candidate = make_benchmark(attack=120, spins=6, spin_lines=8)
        champion = make_benchmark(attack=100, spins=2, spin_lines=2)
        with patch("minoflux_ai.promotion.run_heuristic_benchmark", side_effect=[candidate, champion]):
            result = compare_candidate_to_champion(
                HeuristicWeights(attack=2),
                HeuristicWeights(attack=1),
                SearchConfig(),
                config=PromotionConfig(games=2, max_pieces=100),
            )
        self.assertTrue(result.promoted)
        self.assertGreater(result.fitness_gain or 0, 0)

    def test_completion_guard_rejects_fragile_candidate(self) -> None:
        candidate = make_benchmark(attack=200, spins=20, spin_lines=30, completed=0)
        champion = make_benchmark(attack=100, spins=0, spin_lines=0, completed=2)
        with patch("minoflux_ai.promotion.run_heuristic_benchmark", side_effect=[candidate, champion]):
            result = compare_candidate_to_champion(
                HeuristicWeights(attack=2),
                HeuristicWeights(attack=1),
                SearchConfig(),
                config=PromotionConfig(games=2, max_pieces=100, max_completion_loss=1),
            )
        self.assertFalse(result.promoted)
        self.assertIn("completed", result.reason)

    def test_recovery_bootstraps_missing_champion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery = root / "recovery.json"
            champion = root / "champion.json"
            expected = HeuristicWeights(spin_lines=4.5)
            save_weights(recovery, expected)
            result = bootstrap_champion(champion, recovery_path=recovery)
            self.assertEqual(result, champion)
            self.assertEqual(load_weights(champion), expected)


if __name__ == "__main__":
    unittest.main()
