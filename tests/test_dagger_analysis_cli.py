from __future__ import annotations

import unittest

from minoflux.dagger_analysis_cli import _expert_snapshot, _summarize_records


def _context(*, lines: int = 0, attack: int = 0, spin: bool = False) -> list[float]:
    return [0.0] * 50 + [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        lines / 4.0,
        attack / 20.0,
        float(spin),
        0.0,
    ]


def _candidate(*, lines: int = 0, attack: int = 0, spin: bool = False) -> dict[str, object]:
    rows = [0] * 24
    rows[-1] = 1
    return {
        "rows": rows,
        "context": _context(lines=lines, attack=attack, spin=spin),
        "move": [0, "T", 3, 20, 1],
    }


def _record(
    *,
    seed: int,
    piece: int,
    margin: float,
    matched: bool,
    reasons: list[str],
    lines: int = 0,
    attack: int = 0,
    spin: bool = False,
) -> dict[str, object]:
    return {
        "seed": seed,
        "pieceIndex": piece,
        "expertIndex": 0,
        "candidates": [_candidate(lines=lines, attack=attack, spin=spin)],
        "learnerMargin": margin,
        "learnerMatchedOracle": matched,
        "daggerReasons": reasons,
    }


class DaggerAnalysisCliTests(unittest.TestCase):
    def test_expert_snapshot_decodes_board_and_context(self) -> None:
        snapshot = _expert_snapshot(
            _record(
                seed=1,
                piece=10,
                margin=0.2,
                matched=False,
                reasons=["random_control", "nn_oracle_disagree"],
                lines=2,
                attack=4,
                spin=True,
            )
        )

        self.assertEqual(snapshot["max_height"], 1)
        self.assertEqual(snapshot["holes"], 0)
        self.assertEqual(snapshot["lines"], 2)
        self.assertEqual(snapshot["attack"], 4)
        self.assertTrue(snapshot["spin"])
        self.assertEqual(snapshot["move"], "T@(3,20)r1")

    def test_summary_counts_confident_disagreements_and_reasons(self) -> None:
        records = [
            _record(
                seed=1,
                piece=10,
                margin=0.20,
                matched=False,
                reasons=["random_control", "nn_oracle_disagree"],
                attack=4,
            ),
            _record(
                seed=1,
                piece=110,
                margin=0.04,
                matched=True,
                reasons=["low_margin"],
            ),
        ]

        summary = _summarize_records(records)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["disagreements"], 1)
        self.assertAlmostEqual(summary["disagreement_rate"], 50.0)
        self.assertEqual(summary["reason_counts"]["random_control"], 1)
        self.assertEqual(summary["reason_counts"]["low_margin"], 1)
        self.assertEqual(summary["confident_disagreements"][0.15], 1)
        self.assertEqual(summary["confident_disagreements"][0.30], 0)
        self.assertEqual(summary["piece_buckets"]["000-099"], [1, 1])
        self.assertEqual(summary["piece_buckets"]["100-199"], [1, 0])


if __name__ == "__main__":
    unittest.main()
