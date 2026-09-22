from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT
from minoflux_ai.neural_mix import merge_neural_datasets


def _record(seed: int, piece: int, marker: str) -> dict[str, object]:
    return {
        "format": NEURAL_DATASET_FORMAT,
        "seed": seed,
        "pieceIndex": piece,
        "expertIndex": 0,
        "candidates": [
            {
                "rows": [0] * 24,
                "context": [0.0] * 59,
                "move": [0, "I", 3, 20, 0],
            }
        ],
        "marker": marker,
    }


class NeuralMixTests(unittest.TestCase):
    def test_later_dataset_replaces_duplicate_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "mixed.jsonl"
            first.write_text(
                json.dumps(_record(1, 2, "old")) + "\n" + json.dumps(_record(2, 3, "keep")) + "\n",
                encoding="utf-8",
            )
            second.write_text(json.dumps(_record(1, 2, "new")) + "\n", encoding="utf-8")
            result = merge_neural_datasets(output, [first, second])
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["readRecords"], 3)
            self.assertEqual(result["writtenRecords"], 2)
            by_key = {(record["seed"], record["pieceIndex"]): record for record in records}
            self.assertEqual(by_key[(1, 2)]["marker"], "new")
            self.assertEqual(by_key[(2, 3)]["marker"], "keep")

    def test_input_weights_are_written_and_compose_with_existing_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "weighted.jsonl"
            first.write_text(json.dumps(_record(11, 0, "gold")) + "\n", encoding="utf-8")
            dagger = _record(22, 0, "dagger")
            dagger["trainingWeight"] = 0.5
            second.write_text(json.dumps(dagger) + "\n", encoding="utf-8")

            result = merge_neural_datasets(
                output,
                [first, second],
                input_weights=[1.0, 0.25],
            )
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            by_marker = {record["marker"]: record for record in records}
            self.assertEqual(result["inputWeights"], [1.0, 0.25])
            self.assertEqual(by_marker["gold"]["trainingWeight"], 1.0)
            self.assertEqual(by_marker["dagger"]["trainingWeight"], 0.125)

            metadata = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["inputWeights"], [1.0, 0.25])


if __name__ == "__main__":
    unittest.main()
