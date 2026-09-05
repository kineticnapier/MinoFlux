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


if __name__ == "__main__":
    unittest.main()
