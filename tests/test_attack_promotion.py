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
    VersusBenchmarkResult,
    benchmark_fitness,
    bootstrap_champion,
    compare_candidate_to_champion,
    load_weights,
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
    high_stack_rate: float = 0.0,
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
        high_stack_rate=high_stack_rate,
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
        high_stack_rate=high_stack_rate,
        topouts=2 - completed,
        completed=completed,
        per_game=(game, game),
        best_game=game,
    )


def make_versus(
    *,
    candidate_wins: int,
    champion_wins: int,
    draws: int,
    candidate_sent: float = 12.0,
    champion_sent: float = 10.0,
) -> VersusBenchmarkResult:
    games = candidate_wins + champion_wins + draws
    return VersusBenchmarkResult(
        games=games,
        max_turns=120,
        seed_base=12_000_052,
        seed_step=193,
        player_wins=candidate_wins,
        ai_wins=champion_wins,
        draws=draws,
        mean_turns=80.0,
        player_mean_attack=candidate_sent + 1.0,
        ai_mean_attack=champion_sent + 1.0,
        player_mean_sent=candidate_sent,
        ai_mean_sent=champion_sent,
        per_game=(),
    )


class AttackFitnessTests(unittest.TestCase):
    def test_attack_spin_profile_values_offense_more_than_balanced(self) -> None:
        defensive = make_benchmark(attack=100, spins=0, spin_lines=0)
        offensive = make_benchmark(attack=120, spins=8, spin_lines=12)
        attack_gain = benchmark_fitness(offensive, ATTACK_SPIN_FITNESS) - benchmark_fitness(defensive, ATTACK_SPIN_FITNESS)
        balanced_gain = benchmark_fitness(offensive, BALANCED_FITNESS) - benchmark_fitness(defensive, BALANCED_FITNESS)
        self.assertGreater(attack_gain, balanced_gain)

    def test_clean_attack_penalizes_dirty_stack(self) -> None:
        clean = make_benchmark(attack=120, spins=8, spin_lines=12)
        dirty = make_benchmark(
            attack=120,
            spins=8,
            spin_lines=12,
            mean_holes=3.0,
            mean_hole_depth=8.0,
            mean_bumpiness=12.0,
            mean_max_height=13.0,
            high_stack_rate=0.25,
        )
        self.assertGreater(
            benchmark_fitness(clean, CLEAN_ATTACK_FITNESS),
            benchmark_fitness(dirty, CLEAN_ATTACK_FITNESS),
        )

    def test_clean_attack_punishes_topout_more_than_attack_spin(self) -> None:
        safe = make_benchmark(attack=120, spins=8, spin_lines=12, completed=2)
        fragile = make_benchmark(attack=120, spins=8, spin_lines=12, completed=1)
        clean_loss = benchmark_fitness(safe, CLEAN_ATTACK_FITNESS) - benchmark_fitness(fragile, CLEAN_ATTACK_FITNESS)
        attack_loss = benchmark_fitness(safe, ATTACK_SPIN_FITNESS) - benchmark_fitness(fragile, ATTACK_SPIN_FITNESS)
        self.assertGreater(clean_loss, attack_loss)


class PromotionTests(unittest.TestCase):
    def test_candidate_must_beat_champion_in_solo_and_versus(self) -> None:
        candidate = make_benchmark(attack=120, spins=6, spin_lines=8)
        champion = make_benchmark(attack=100, spins=2, spin_lines=2)
        versus = make_versus(candidate_wins=5, champion_wins=2, draws=1)
        with (
            patch("minoflux_ai.promotion.run_heuristic_benchmark", side_effect=[candidate, champion]),
            patch("minoflux_ai.promotion.run_versus_benchmark", return_value=versus) as versus_run,
        ):
            result = compare_candidate_to_champion(
                HeuristicWeights(attack=2),
                HeuristicWeights(attack=1),
                SearchConfig(),
                config=PromotionConfig(games=2, max_pieces=100),
            )
        self.assertTrue(result.promoted)
        self.assertGreater(result.fitness_gain or 0, 0)
        self.assertEqual(result.versus_win_margin, 3)
        self.assertIs(result.versus, versus)
        versus_run.assert_called_once()

    def test_completion_guard_rejects_fragile_candidate_before_versus(self) -> None:
        candidate = make_benchmark(attack=200, spins=20, spin_lines=30, completed=0)
        champion = make_benchmark(attack=100, spins=0, spin_lines=0, completed=2)
        with (
            patch("minoflux_ai.promotion.run_heuristic_benchmark", side_effect=[candidate, champion]),
            patch("minoflux_ai.promotion.run_versus_benchmark") as versus_run,
        ):
            result = compare_candidate_to_champion(
                HeuristicWeights(attack=2),
                HeuristicWeights(attack=1),
                SearchConfig(),
                config=PromotionConfig(games=2, max_pieces=100, max_completion_loss=1),
            )
        self.assertFalse(result.promoted)
        self.assertIn("completed", result.reason)
        self.assertIsNone(result.versus)
        versus_run.assert_not_called()

    def test_fitness_guard_rejects_weaker_candidate_before_versus(self) -> None:
        candidate = make_benchmark(attack=90, spins=0, spin_lines=0)
        champion = make_benchmark(attack=100, spins=0, spin_lines=0)
        with (
            patch("minoflux_ai.promotion.run_heuristic_benchmark", side_effect=[candidate, champion]),
            patch("minoflux_ai.promotion.run_versus_benchmark") as versus_run,
        ):
            result = compare_candidate_to_champion(
                HeuristicWeights(attack=2),
                HeuristicWeights(attack=1),
                SearchConfig(),
                config=PromotionConfig(games=2, max_pieces=100),
            )
        self.assertFalse(result.promoted)
        self.assertIn("fitness gain", result.reason)
        versus_run.assert_not_called()

    def test_versus_gate_rejects_solo_winner_that_does_not_win_head_to_head(self) -> None:
        candidate = make_benchmark(attack=130, spins=8, spin_lines=10)
        champion = make_benchmark(attack=100, spins=2, spin_lines=2)
        versus = make_versus(candidate_wins=3, champion_wins=3, draws=2, candidate_sent=14.0, champion_sent=10.0)
        with (
            patch("minoflux_ai.promotion.run_heuristic_benchmark", side_effect=[candidate, champion]),
            patch("minoflux_ai.promotion.run_versus_benchmark", return_value=versus),
        ):
            result = compare_candidate_to_champion(
                HeuristicWeights(attack=2),
                HeuristicWeights(attack=1),
                SearchConfig(),
                config=PromotionConfig(games=2, max_pieces=100),
            )
        self.assertFalse(result.promoted)
        self.assertEqual(result.versus_win_margin, 0)
        self.assertIn("mirrored versus", result.reason)

    def test_versus_game_count_is_normalized_to_complete_mirrored_pairs(self) -> None:
        cfg = PromotionConfig(versus_games=7, minimum_versus_win_margin=0).normalized()
        self.assertEqual(cfg.versus_games, 8)
        self.assertEqual(cfg.minimum_versus_win_margin, 1)

    def test_first_champion_does_not_need_versus_opponent(self) -> None:
        candidate = make_benchmark(attack=100, spins=2, spin_lines=2)
        with (
            patch("minoflux_ai.promotion.run_heuristic_benchmark", return_value=candidate),
            patch("minoflux_ai.promotion.run_versus_benchmark") as versus_run,
        ):
            result = compare_candidate_to_champion(
                HeuristicWeights(attack=1),
                None,
                SearchConfig(),
                config=PromotionConfig(games=2, max_pieces=100),
            )
        self.assertTrue(result.promoted)
        self.assertIsNone(result.versus)
        versus_run.assert_not_called()

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
