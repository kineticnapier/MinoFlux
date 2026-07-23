from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minoflux_ai import (
    ATTACK_SPIN_FITNESS,
    BALANCED_FITNESS,
    BenchmarkGame,
    BenchmarkResult,
    HeuristicWeights,
    PromotionConfig,
    SearchConfig,
    benchmark_fitness,
    bootstrap_champion,
    compare_candidate_to_champion,
    load_weights,
    save_weights,
)


def make_benchmark(*, attack: int, spins: int, spin_lines: int, completed: int = 2) -> BenchmarkResult:
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
