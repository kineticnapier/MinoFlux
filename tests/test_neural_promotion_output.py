from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest

from minoflux.versus_neural_cli import _print_promotion_report, build_parser


def sample_report() -> dict[str, object]:
    return {
        "format": "minoflux_neural_promotion_benchmark_v1",
        "decision": None,
        "models": {
            "candidate": {"name": "candidate", "path": "candidate.pt", "sha256": "abc"},
            "champion": {"name": "human3", "path": "human3.pt", "sha256": "def"},
            "reference": {"name": "e5", "path": "e5.pt", "sha256": "ghi"},
        },
        "solo": {
            "candidate": {
                "modelName": "candidate",
                "games": 10,
                "attackPerPiece": 0.25,
                "meanPiecesSurvived": 280.5,
                "completionRate": 0.8,
                "topouts": 2,
                "executed": True,
                "reused": False,
                "skipped": False,
                "perGame": [{"seed": 1}],
            },
            "champion": {
                "modelName": "human3",
                "games": 10,
                "attackPerPiece": 0.22,
                "meanPiecesSurvived": 295.0,
                "completionRate": 0.9,
                "topouts": 1,
                "executed": True,
                "reused": False,
                "skipped": False,
                "perGame": [{"seed": 1}],
            },
            "reference": {
                "modelName": "e5",
                "games": 10,
                "attackPerPiece": 0.29,
                "meanPiecesSurvived": 250.0,
                "completionRate": 0.6,
                "topouts": 4,
                "executed": False,
                "reused": True,
                "skipped": False,
                "reusedFrom": "candidate",
                "perGame": [{"seed": 1}],
            },
        },
        "versus": {
            "candidateVsChampion": {
                "modelA": "candidate",
                "modelB": "human3",
                "games": 20,
                "pairs": 10,
                "aWins": 11,
                "bWins": 8,
                "draws": 1,
                "aWinRate": 0.55,
                "executed": True,
                "reused": False,
                "skipped": False,
                "pairedSeeds": [{"seed": 1}],
            },
            "candidateVsReference": {
                "modelA": "candidate",
                "modelB": "e5",
                "games": 0,
                "pairs": 0,
                "executed": False,
                "reused": False,
                "skipped": True,
                "reason": "same-model matchup",
            },
        },
        "outputPath": "data/benchmarks/report.json",
    }


class NeuralPromotionOutputTests(unittest.TestCase):
    def capture(self, report: dict[str, object], **kwargs: bool) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _print_promotion_report(report, **kwargs)
        return buffer.getvalue()

    def test_promotion_output_flags_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["promotion", "--candidate", "candidate.pt"])
        self.assertFalse(default.print_json)
        self.assertFalse(default.pretty_json)
        compact = parser.parse_args(["promotion", "--candidate", "candidate.pt", "--print-json"])
        self.assertTrue(compact.print_json)
        pretty = parser.parse_args(["promotion", "--candidate", "candidate.pt", "--pretty-json"])
        self.assertTrue(pretty.pretty_json)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["promotion", "--candidate", "candidate.pt", "--print-json", "--pretty-json"]
            )

    def test_default_output_is_short_human_readable_summary(self) -> None:
        text = self.capture(sample_report())
        self.assertIn("Neural promotion benchmark", text)
        self.assertIn("candidate (candidate): APP 0.2500", text)
        self.assertIn("candidateVsChampion: candidate vs human3", text)
        self.assertIn("candidateVsReference: skipped (same-model matchup)", text)
        self.assertIn("Report: data/benchmarks/report.json", text)
        self.assertNotIn('"perGame"', text)
        self.assertLess(len(text.splitlines()), 12)

    def test_print_json_is_compact_single_line(self) -> None:
        report = sample_report()
        text = self.capture(report, print_json=True)
        expected = json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.assertEqual(text, expected)
        self.assertEqual(len(text.rstrip("\n").splitlines()), 1)

    def test_pretty_json_preserves_legacy_stdout_format(self) -> None:
        report = sample_report()
        text = self.capture(report, pretty_json=True)
        self.assertEqual(text, json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
