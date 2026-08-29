from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai import SearchConfig
from minoflux_ai.neural import NeuralValueConfig, NeuralValueEvaluator
from minoflux_ai.neural_dataset import (
    NEURAL_DATASET_FORMAT,
    NeuralDatasetConfig,
    generate_neural_ranking_samples,
    pack_board_rows,
    unpack_board_rows,
    write_neural_ranking_dataset,
)
from minoflux_ai.neural_train import NeuralTrainConfig, train_neural_value_model


class NeuralDatasetTests(unittest.TestCase):
    def test_board_pack_round_trip(self) -> None:
        config = NeuralValueConfig()
        board = tuple(float(index % 7 == 0) for index in range(240))
        self.assertEqual(unpack_board_rows(pack_board_rows(board, config), config), board)

    def test_small_dataset_contains_expert_and_compact_states(self) -> None:
        config = NeuralDatasetConfig(
            games=1,
            max_pieces=2,
            max_candidates=4,
            search_config=SearchConfig(
                allow_hold=True,
                lookahead_pieces=0,
                beam_width=2,
                srs_reachable=True,
                reachability_node_limit=8000,
            ),
        )
        samples = list(generate_neural_ranking_samples(config))
        self.assertTrue(samples)
        self.assertLessEqual(len(samples), 2)
        for sample in samples:
            self.assertGreaterEqual(sample.expert_index, 0)
            self.assertLess(sample.expert_index, len(sample.candidates))
            self.assertLessEqual(len(sample.candidates), 4)
            for candidate in sample.candidates:
                self.assertEqual(len(candidate.board_rows), 24)
                self.assertEqual(len(candidate.context), NeuralValueConfig().context_size)

    def test_dataset_writer_produces_jsonl_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.jsonl"
            result = write_neural_ranking_dataset(
                path,
                NeuralDatasetConfig(games=1, max_pieces=1, max_candidates=3),
            )
            self.assertGreater(result["samples"], 0)
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["format"], NEURAL_DATASET_FORMAT)
            self.assertTrue(path.with_suffix(".jsonl.meta.json").exists())


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch optional dependency not installed")
class NeuralTrainingTests(unittest.TestCase):
    def test_tiny_training_saves_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny.jsonl"
            checkpoint = Path(directory) / "tiny.pt"
            write_neural_ranking_dataset(
                dataset,
                NeuralDatasetConfig(games=2, max_pieces=2, max_candidates=4),
            )
            result = train_neural_value_model(
                dataset,
                checkpoint,
                NeuralTrainConfig(
                    epochs=1,
                    batch_size=2,
                    validation_fraction=0.5,
                    device="cpu",
                ),
            )
            self.assertTrue(checkpoint.exists())
            self.assertEqual(result.checkpoint_path, str(checkpoint))
            evaluator = NeuralValueEvaluator.from_checkpoint(checkpoint, device="cpu")
            self.assertEqual(evaluator.device, "cpu")


if __name__ == "__main__":
    unittest.main()
