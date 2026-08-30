from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai import DEFAULT_WEIGHTS
from minoflux_ai.neural_dagger import write_neural_dagger_dataset
from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT, NeuralDatasetConfig


class _HeuristicLikeScorer:
    def score_many(self, game, evaluations):
        return [evaluation.score for evaluation in evaluations]


class NeuralDaggerTests(unittest.TestCase):
    def test_dagger_writes_teacher_labels_on_learner_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dagger.jsonl"
            result = write_neural_dagger_dataset(
                path,
                _HeuristicLikeScorer(),
                NeuralDatasetConfig(
                    games=1,
                    max_pieces=2,
                    max_candidates=4,
                    teacher_lookahead=0,
                    rollout_horizon=0,
                    rollout_candidates=0,
                ),
                DEFAULT_WEIGHTS,
                sample_rate=1.0,
                max_samples=1,
            )
            self.assertEqual(result["samples"], 1)
            self.assertGreaterEqual(result["visitedStates"], 1)
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["format"], NEURAL_DATASET_FORMAT)
            self.assertEqual(record["source"], "dagger_teacher")
            self.assertIn("expertIndices", record)
            self.assertTrue(record["daggerReasons"])


if __name__ == "__main__":
    unittest.main()
