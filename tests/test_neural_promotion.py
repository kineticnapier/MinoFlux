from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minoflux.versus_neural_cli import _promotion, build_parser
from minoflux_ai.neural_promotion import (
    NEURAL_PROMOTION_FORMAT,
    NeuralModelSpec,
    NeuralSoloBenchmarkResult,
    NeuralSoloGameResult,
    build_neural_promotion_report,
    reverse_versus_benchmark_result,
    run_neural_solo_benchmark,
    same_neural_model,
    summarize_paired_versus,
)
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_benchmark import (
    VersusBenchmarkResult,
    VersusGameResult,
    run_versus_benchmark,
)


def solo(model: str, *, attack: int = 25, completed: bool = True) -> NeuralSoloBenchmarkResult:
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


def versus_game(
    seed: int,
    winner: str,
    swapped: bool,
    *,
    player_attack: int = 4,
    ai_attack: int = 3,
) -> VersusGameResult:
    return VersusGameResult(
        seed=seed,
        winner=winner,
        turns=20,
        player_pieces=11,
        ai_pieces=9,
        player_attack=player_attack,
        ai_attack=ai_attack,
        player_sent=3,
        ai_sent=2,
        player_canceled=1,
        ai_canceled=2,
        player_received=2,
        ai_received=3,
        player_garbage_applied=1,
        ai_garbage_applied=2,
        player_pending=0,
        ai_pending=1,
        player_final_height=4,
        ai_final_height=6,
        player_final_holes=0,
        ai_final_holes=1,
        player_max_b2b=2,
        ai_max_b2b=1,
        player_max_surge=3,
        ai_max_surge=1,
        models_swapped=swapped,
    )


def versus_result() -> VersusBenchmarkResult:
    per_game = (
        versus_game(100, "player", False, player_attack=8, ai_attack=3),
        versus_game(100, "player", True, player_attack=7, ai_attack=2),
    )
    return VersusBenchmarkResult(
        games=2,
        max_turns=100,
        seed_base=100,
        seed_step=31,
        player_wins=2,
        ai_wins=0,
        draws=0,
        mean_turns=20.0,
        player_mean_attack=7.5,
        ai_mean_attack=2.5,
        player_mean_sent=3.0,
        ai_mean_sent=2.0,
        player_mean_canceled=1.0,
        ai_mean_canceled=2.0,
        player_mean_received=2.0,
        ai_mean_received=3.0,
        player_mean_pieces=11.0,
        ai_mean_pieces=9.0,
        per_game=per_game,
    )


class ZeroScorer:
    def score_placement_groups(self, groups):
        return tuple(tuple(0.0 for _ in placements) for _game, placements in groups)

    def score_placements(self, game, placements):
        return tuple(0.0 for _ in placements)


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
        self.assertFalse(args.same_model_match)

    def test_same_sha_is_preferred_for_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / "a.pt"
            b = root / "b.pt"
            a.write_bytes(b"same-checkpoint")
            b.write_bytes(b"same-checkpoint")
            self.assertTrue(
                same_neural_model(
                    NeuralModelSpec.from_path(a, name="a"),
                    NeuralModelSpec.from_path(b, name="b"),
                )
            )

    def test_same_path_fallback_reuses_when_sha_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "missing.pt"
            alias = root / "nested" / ".." / "missing.pt"
            a = NeuralModelSpec.from_path(direct, name="a")
            b = NeuralModelSpec.from_path(alias, name="b")
            self.assertIsNone(a.sha256)
            self.assertIsNone(b.sha256)
            self.assertTrue(same_neural_model(a, b))

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

    def test_reversed_reuse_remaps_labels_metrics_and_side_flags(self) -> None:
        original = versus_result()
        reversed_result = reverse_versus_benchmark_result(original)
        summary = summarize_paired_versus(reversed_result, model_a="B", model_b="A")
        self.assertEqual(summary["modelA"], "B")
        self.assertEqual(summary["modelB"], "A")
        self.assertEqual(summary["aWins"], 0)
        self.assertEqual(summary["bWins"], 2)
        self.assertEqual(summary["meanAttackA"], 2.5)
        self.assertEqual(summary["meanAttackB"], 7.5)
        self.assertEqual(
            {game.models_swapped for game in reversed_result.per_game},
            {False, True},
        )

    def test_candidate_equal_champion_reuses_solo_and_duplicate_baseline_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.pt"
            champion = root / "champion.pt"
            reference = root / "e5.pt"
            candidate.write_bytes(b"human3")
            champion.write_bytes(b"human3")
            reference.write_bytes(b"e5")
            output = root / "report.json"
            args = build_parser().parse_args(
                [
                    "promotion",
                    "--candidate", str(candidate),
                    "--champion", str(champion),
                    "--reference", str(reference),
                    "--solo-games", "1",
                    "--versus-pairs", "1",
                    "--no-progress",
                    "--output", str(output),
                ]
            )

            def fake_solo(_scorer, *, model, **_kwargs):
                return solo(model)

            with (
                patch("minoflux.versus_neural_cli._load_solo", return_value=object()) as load_model,
                patch("minoflux.versus_neural_cli.run_neural_solo_benchmark", side_effect=fake_solo) as solo_run,
                patch("minoflux.versus_neural_cli.run_versus_benchmark", return_value=versus_result()) as versus_run,
                patch("minoflux.versus_neural_cli._print"),
            ):
                self.assertEqual(_promotion(args), 0)

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(load_model.call_count, 2)
            self.assertEqual(solo_run.call_count, 2)
            self.assertEqual(versus_run.call_count, 1)
            self.assertTrue(report["solo"]["candidate"]["executed"])
            self.assertTrue(report["solo"]["champion"]["reused"])
            self.assertEqual(report["solo"]["champion"]["reusedFrom"], "candidate")
            same_model = report["versus"]["candidateVsChampion"]
            self.assertTrue(same_model["skipped"])
            self.assertEqual(same_model["reason"], "same-model matchup")
            duplicate = report["versus"]["championVsReference"]
            self.assertTrue(duplicate["reused"])
            self.assertEqual(duplicate["reusedFrom"], "candidateVsReference")
            self.assertFalse(duplicate["reversedFromSource"])
            self.assertEqual(duplicate["modelA"], "human3")
            self.assertEqual(duplicate["modelB"], "e5")

    def test_duplicate_pair_reuse_reverses_model_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "human3-a.pt"
            champion = root / "e5.pt"
            reference = root / "human3-b.pt"
            candidate.write_bytes(b"human3")
            champion.write_bytes(b"e5")
            reference.write_bytes(b"human3")
            output = root / "report.json"
            args = build_parser().parse_args(
                [
                    "promotion",
                    "--candidate", str(candidate),
                    "--candidate-name", "human3-candidate",
                    "--champion", str(champion),
                    "--champion-name", "e5",
                    "--reference", str(reference),
                    "--reference-name", "human3-reference",
                    "--solo-games", "1",
                    "--versus-pairs", "1",
                    "--no-progress",
                    "--output", str(output),
                ]
            )

            with (
                patch("minoflux.versus_neural_cli._load_solo", return_value=object()),
                patch(
                    "minoflux.versus_neural_cli.run_neural_solo_benchmark",
                    side_effect=lambda _scorer, *, model, **_kwargs: solo(model),
                ),
                patch("minoflux.versus_neural_cli.run_versus_benchmark", return_value=versus_result()) as versus_run,
                patch("minoflux.versus_neural_cli._print"),
            ):
                self.assertEqual(_promotion(args), 0)

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(versus_run.call_count, 1)
            reversed_match = report["versus"]["championVsReference"]
            self.assertTrue(reversed_match["reused"])
            self.assertTrue(reversed_match["reversedFromSource"])
            self.assertEqual(reversed_match["modelA"], "e5")
            self.assertEqual(reversed_match["modelB"], "human3-reference")
            self.assertEqual(reversed_match["aWins"], 0)
            self.assertEqual(reversed_match["bWins"], 2)
            self.assertEqual(reversed_match["meanAttackA"], 2.5)
            self.assertEqual(reversed_match["meanAttackB"], 7.5)

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

    def test_small_deterministic_smoke_uses_real_search_and_mirrored_versus(self) -> None:
        scorer = ZeroScorer()
        solo_result = run_neural_solo_benchmark(
            scorer,
            model="zero-scorer",
            games=1,
            max_pieces=3,
            seed_base=901,
            seed_step=31,
            game_batch_size=1,
            search_config=SearchConfig(lookahead_pieces=0),
            progress=False,
        )
        self.assertEqual(solo_result.games, 1)
        self.assertGreater(solo_result.per_game[0].pieces, 0)

        versus = run_versus_benchmark(
            2,
            max_turns=4,
            seed_base=1201,
            seed_step=31,
            player_scorer=scorer,
            ai_scorer=scorer,
            progress=False,
            game_batch=1,
        )
        summary = summarize_paired_versus(versus, model_a="zero-a", model_b="zero-b")
        self.assertEqual(summary["pairs"], 1)
        self.assertEqual(len(summary["pairedSeeds"]), 1)
        self.assertEqual(summary["pairedSeeds"][0]["seed"], 1201)


if __name__ == "__main__":
    unittest.main()
