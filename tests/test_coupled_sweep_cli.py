from __future__ import annotations

from pathlib import Path
import unittest

from minoflux.coupled_sweep_cli import (
    _duel_command,
    _parse_duel_summary,
    _parse_seeds,
    _train_command,
)


class CoupledSweepCliTests(unittest.TestCase):
    def test_parse_seeds_preserves_order_and_rejects_duplicates(self) -> None:
        self.assertEqual(_parse_seeds("12345, 23456,34567"), (12345, 23456, 34567))
        with self.assertRaisesRegex(ValueError, "Duplicate training seed"):
            _parse_seeds("1,2,1")
        with self.assertRaisesRegex(ValueError, "At least one training seed"):
            _parse_seeds(" , ")

    def test_parse_duel_summary(self) -> None:
        parsed = _parse_duel_summary(
            "data/models/a.pt 540 - 484 data/models/b.pt (draw 0)\n"
            "win 52.7% - 47.3% | APP 0.2816 - 0.2747 | sent/piece 0.2427 - 0.2349\n"
        )
        self.assertEqual(parsed.left_wins, 540)
        self.assertEqual(parsed.right_wins, 484)
        self.assertEqual(parsed.draws, 0)
        self.assertAlmostEqual(parsed.left_win_rate, 52.7)
        self.assertAlmostEqual(parsed.right_app, 0.2747)
        self.assertAlmostEqual(parsed.left_sent, 0.2427)

    def test_train_command_uses_same_seed_and_deterministic_coupled_path(self) -> None:
        command = _train_command(
            dataset="full.jsonl",
            base_dataset="base.jsonl",
            output=Path("model.pt"),
            seed=23456,
            epochs=8,
            batch_size=64,
            samples_per_epoch=2304,
            device="auto",
        )
        self.assertIn("train-coupled", command)
        self.assertIn("--deterministic", command)
        self.assertEqual(command[command.index("--seed") + 1], "23456")
        self.assertEqual(command[command.index("--validation-fraction") + 1], "0")

    def test_duel_command_keeps_evaluation_seed_fixed(self) -> None:
        command = _duel_command(
            left=Path("left.pt"),
            right=Path("right.pt"),
            games=1024,
            max_turns=1000,
            duel_seed_base=10_000_001,
            game_batch=16,
            beam=4,
            device="auto",
            allow_180=True,
        )
        self.assertIn("duel", command)
        self.assertEqual(command[command.index("--seed-base") + 1], "10000001")
        self.assertNotIn("--no-180", command)


if __name__ == "__main__":
    unittest.main()
