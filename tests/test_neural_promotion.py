from __future__ import annotations

import unittest

from minoflux.versus_neural_cli import build_parser
from minoflux_ai.neural_promotion import (
    NEURAL_PROMOTION_FORMAT,
    NeuralModelSpec,
    NeuralSoloBenchmarkResult,
    NeuralSoloGameResult,
    build_neural_promotion_report,
    summarize_paired_versus,
)
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_benchmark import VersusBenchmarkResult, VersusGameResult


def solo(model: str, *, attack: int, completed: bool) -> NeuralSoloBenchmarkResult:
    game = NeuralSoloGameResult(
        seed=1,
        pieces=100 if completed else 70,
        lines=40,
        attack=attack,
        topout=not completed,
        completed=completed,
        t_spin_locks=3,
        t_spin_lines=5,
        t_spin_attack=10,
        difficult_clears=8,
        b2b_active_clears=7,
        max_b2b_chain=4,
        surge_released=4,
        surge_release_events=1,
        combo_clears=8,
        max_combo=5,
        perfect_clears=1,
    )
    return NeuralSoloBenchmarkResult(
        model=model,
        games=1,
        max_pieces=100,
        seed_base=1,
        seed_step=31,
        game_batch_size=1,
        search_config=SearchConfig(),
        elapsed_seconds=1.0,
        per_game=(game,),
    )


def versus_game(seed: int, winner: str, swapped: bool) -> VersusGameResult:
    return VersusGameResult(
        seed=seed,
        winner=winner,
        turns=20,
        player_pieces=10,
        ai_pieces=10,
        player_attack=4,
        ai_attack=3,
        player_sent=3,
        ai_sent=2,
        player_canceled=1,
        ai_canceled=1,
        player_received=2,
        ai_received=3,
        player_garbage_applied=1,
        ai_garbage_applied=2,
        player_pending=0,
        ai_pending=0,
        player_final_height=4,
        ai_final_height=6,
        player_final_holes=0,
        ai_final_holes=1,
        player_max_b2b=2,
        ai_max_b2b=1,
        player_max_surge=0,
        ai_max_surge=0,
        models_swapped=swapped,
    )


class NeuralPromotionTests(unittest.TestCase):
    def test_promotion_cli_defaults_to_reference_models_and_paired_versus(self) -> None:
        args = build_parser().parse_args(
            ["promotion", "--candidate", "candidate.pt", "--no-progress"]
        )
        self.assertEqual(args.candidate, "candidate.pt")
        self.assertEqual(args.champion, "data/models/placement-v2-human3-home.pt")
        self.assertEqual(args.reference, "data/models/placement-v2-human4-e5.pt")
        self.assertEqual(args.versus_pairs, 50)
        self.assertEqual(args.game_batch, 1)
        self.assertTrue(args.hold)

    def test_mirrored_pair_summary_uses_same_seed_twice(self) -> None:
        per_game = (
            versus_game(100, "player", False),
            versus_game(100, "player", True),
            versus_game(131, "ai", False),
            versus_game(131, "draw", True),
        )
        result = VersusBenchmarkResult(
            games=4,
            max_turns=100,
            seed_base=100,
            seed_step=31,
            player_wins=2,
            ai_wins=1,
            draws=1,
            mean_turns=20.0,
            player_mean_attack=4.0,
            ai_mean_attack=3.0,
            player_mean_sent=3.0,
            ai_mean_sent=2.0,
            player_mean_canceled=1.0,
            ai_mean_canceled=1.0,
            player_mean_received=2.0,
            ai_mean_received=3.0,
            player_mean_pieces=10.0,
            ai_mean_pieces=10.0,
            per_game=per_game,
        )
        summary = summarize_paired_versus(result, model_a="A", model_b="B")
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["pairWinsA"], 1)
        self.assertEqual(summary["pairWinsB"], 1)
        self.assertEqual(summary["pairTies"], 0)
        self.assertEqual(summary["aWinsAsLeft"], 1)
        self.assertEqual(summary["aWinsAsRight"], 1)

    def test_incomplete_pair_is_rejected(self) -> None:
        result = VersusBenchmarkResult(
            games=1,
            max_turns=100,
            seed_base=100,
            seed_step=31,
            player_wins=1,
            ai_wins=0,
            draws=0,
            mean_turns=20.0,
            player_mean_attack=4.0,
            ai_mean_attack=3.0,
            player_mean_sent=3.0,
            ai_mean_sent=2.0,
            per_game=(versus_game(100, "player", False),),
        )
        with self.assertRaises(ValueError):
            summarize_paired_versus(result, model_a="A", model_b="B")

    def test_report_has_no_automatic_promotion_threshold(self) -> None:
        report = build_neural_promotion_report(
            candidate=NeuralModelSpec("candidate", "candidate.pt"),
            champion=NeuralModelSpec("human3", "human3.pt"),
            reference=NeuralModelSpec("e5", "e5.pt"),
            solo={
                "candidate": solo("candidate.pt", attack=30, completed=True),
                "champion": solo("human3.pt", attack=25, completed=True),
                "reference": solo("e5.pt", attack=35, completed=False),
            },
            versus={},
            conditions={"solo": {"games": 1}},
            created_at="2026-09-09T00:00:00Z",
        )
        self.assertEqual(report["format"], NEURAL_PROMOTION_FORMAT)
        self.assertIsNone(report["decision"])
        self.assertIn("distribution-only", report["decisionPolicy"])
        comparison = report["comparisons"]["candidateMinusChampionSolo"]
        self.assertGreater(comparison["attackPerPiece"], 0.0)

    def test_solo_result_exposes_survival_and_attack_metrics(self) -> None:
        payload = solo("model.pt", attack=25, completed=True).to_dict()
        self.assertEqual(payload["meanPiecesSurvived"], 100.0)
        self.assertEqual(payload["completionRate"], 1.0)
        self.assertEqual(payload["attackPerPiece"], 0.25)
        self.assertEqual(payload["tSpinAttack"], 10)
        self.assertEqual(payload["maxB2bChain"], 4)
        self.assertEqual(payload["maxCombo"], 5)


if __name__ == "__main__":
    unittest.main()
